"""Training and replica observables of parity learning at a fixed training set.

One self-contained file for the paper "Replica Fragmentation and Glassy Dynamics
in Parity Learning".  It needs NumPy, SciPy and PyTorch and reads no data file.

    python replica_overlaps.py                  # self-checks, then a 16-replica ensemble
                                                # (a replica takes about 1.5 minutes on one
                                                # core; replicas train side by side)
    python replica_overlaps.py --replicas 48    # a larger ensemble
    python replica_overlaps.py --check          # self-checks only
    python replica_overlaps.py --intervention   # the two learning-rate arms of Sec. 5.2
    python replica_overlaps.py --ensemble 'data/large_data_ensembles/L12_N1280/*.npz' --L 12
                                                # the observables of a stored ensemble

===========================================================================
THE TASK
===========================================================================
An Ising chain of L spins whose first spin points up.  The input is the L-1
domain walls, s_i = 1 where spins i-1 and i differ; the target is the spin
configuration, sigma_j = (-1)^(s_1 + ... + s_j).  Reconstructing the spins is
cumulative parity: output position j computes the parity of the first j walls,
so the tail of the chain is the hard part.  For L = 12 there are 2^11 = 2048
inputs.

Each wall is fed to the two positions it separates (two input channels), and the
network returns one logit per position.  The soft spin u_j = tanh(logit_j / 2) is
the magnetization of the Bernoulli distribution the logit defines.  The block
accuracy A_block is the fraction of inputs with every position right.

===========================================================================
REPLICAS AT FIXED DISORDER
===========================================================================
The training set is the quenched disorder.  It is drawn once (N inputs from
seed TRAIN_SEED) and shared by every replica.  Replicas differ only in their
initialization (seed r) and minibatch order (seed r+1), and all of them are
evaluated on the same 2048 inputs drawn from seed EVAL_SEED, so overlaps between
replicas compare functions on one common set.

A_block on the evaluation inputs is recorded after every epoch.  At
(L, N) = (12, 1280) most replicas reach A_block >= 0.8 and many then fall back.
The endpoint class is read off that curve at epoch 150:

    never      A_block < 0.8 throughout
    stable     reaches 0.8 and never falls below 0.5 afterwards
    retreat    reaches 0.8, falls below 0.5, and is below 0.5 at epoch 150
    recovery   reaches 0.8, falls below 0.5, and is at or above 0.8 at epoch 150
    neither    reaches 0.8, falls below 0.5, and ends between 0.5 and 0.8

The paper's (12, 1280) ensemble has 1200 replicas: 255 never, 145 stable,
269 retreat, 416 recovery and 115 neither.

===========================================================================
THE OVERLAPS
===========================================================================
With truth spins y_j(b) = +-1 on evaluation input b, and averages over b and
the output positions j of a window,

    m        = E[u_r y]                truth overlap, a Mattis magnetization
    q_self   = E[u_r^2]                self-overlap
    q_cross  = E[u_r u_s], r != s      cross-replica overlap
    Delta_N  = q_self - m              Nishimori gap
    chi_SG   = q_self - q_cross        self-cross gap, the variance across replicas

Replicas drawn from the Bayes posterior at the true prior sit on the Nishimori
line, q_self = m.  These replicas are optimization endpoints, so Delta_N ~ 0 is
an observation about them, not an identity.

Write a_r = u_r y.  Since y^2 = 1, Q^raw_{rs} = E[u_r u_s] = E[a_r a_s].  Split a
into its mean over inputs, the profile mu_{r,j}, and the input-dependent
residual.  The cross terms vanish under the joint average, so exactly

    Q^raw = m m^T + C^prof + W
    C^prof_{rs} = Cov_j(mu_{r,j}, mu_{s,j})
    W_{rs}      = E_j[ Cov_b(a_{r,j}(b), a_{s,j}(b)) ]

and Q^c = Q^raw - m m^T = C^prof + W is the connected overlap.  For a replica
matrix A, gap(A) = mean diagonal - mean off-diagonal.  gap is linear, so
gap(Q^raw) is the sum of the gaps of the three terms.  The share
gap(W) / gap(Q^c) is the part of the connected self-cross gap that is
input-dependent rather than carried by the mean profile.

R^W_{rs} = W_rs / sqrt(W_rr W_ss) is the cosine between two replicas'
residuals and d = 1 - R^W their distance.  Average-linkage clustering on d gives
a tree, and three numbers say how tree-like d is:

    R2_tree      1 - sum (d - d_tree)^2 / sum (d - mean d)^2 over pairs
    P(d_um<0.1)  the fraction of triples whose sorted similarities
                 q1 <= q2 <= q3 have (q2 - q1)/(q3 - q1) < 0.1; an exact
                 ultrametric triangle has 0
    I_C, f_max   the normalized Colless imbalance of the tree, and the largest
                 cluster of its four-cluster cut as a fraction of the replicas

All of them depend on the number of replicas, so they are measured on
SUBSET_DRAWS random subsets of SUBSET_SIZE replicas and averaged.  P(R^W < 0)
is the fraction of pairs whose residuals are more than a right angle apart.
A finite evaluation set makes a few small cosines negative by noise alone; the
split-half control re-estimates R^W on each half of the inputs and asks whether
the pairs that are obtuse on the full set are obtuse on both halves.

===========================================================================
THE LEARNING-RATE INTERVENTION
===========================================================================
One network on a training set of N inputs drawn from a fixed split of the
complete input space, with 512 inputs held out.  Both arms share the split,
the initialization, the minibatch stream, the optimizer and the 2400-update
budget.  The intervention arm holds eta_0 = 2e-3 until held-out A_block first
reaches 0.8 at update t*, then lowers it along a cosine to 0.02 eta_0 at the end
of the budget; the constant arm keeps eta_0.  The two arms are one run up to
t*, so any later difference is the lowered rate alone.

===========================================================================
WHAT REPRODUCES
===========================================================================
Applied to the stored ensembles of the paper's data release, summarize,
obtuse_pair_fraction and split_half return the values of the paper.  The
training values of an individual replica depend on the machine, the thread
count and the PyTorch version.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform

TRAIN_SEED = 12345            # the shared training set
EVAL_SEED = 777               # the shared evaluation inputs
EVAL_SAMPLES = 2048
EPOCHS = 150                  # observation epoch of the endpoint classes
BATCH = 128
LR, WEIGHT_DECAY = 2e-3, 1e-4
ARCH = dict(channels=16, depth=8, num_heads=4, ffn_dim=64, dropout=0.0)

HI, LO = 0.8, 0.5             # endpoint thresholds on A_block
SUBSET_SIZE, SUBSET_DRAWS = 47, 400   # replicas per subset, subsets of the tree statistics
TAIL = {12: slice(8, 12), 16: slice(11, 16), 20: slice(14, 20)}   # j >= 2L/3


# ===========================================================================
# The task
# ===========================================================================

def encode(walls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs [n, 2, L] and target bits [n, L] for domain walls [n, L-1] in {0, 1}.

    Target bit 1 is spin down.  Channel 0 carries wall i at position i, channel 1 at
    position i-1, so each position sees the walls on both of its sides.
    """
    n, L = walls.shape[0], walls.shape[1] + 1
    targets = torch.zeros(n, L)
    targets[:, 1:] = torch.cumsum(walls, 1).remainder(2.0)
    inputs = torch.zeros(n, 2, L)
    inputs[:, 0, 1:] = walls
    inputs[:, 1, :-1] = walls
    return inputs, targets


def draw_set(num: int, L: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """num independent uniform inputs of length L from the given seed."""
    g = torch.Generator().manual_seed(seed)
    return encode(torch.randint(0, 2, (num, L - 1), generator=g).float())


def complete_set(L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All 2^(L-1) inputs; input k has wall i equal to bit i of k."""
    walls = ((torch.arange(2 ** (L - 1))[:, None] >> torch.arange(L - 1)[None, :]) & 1).float()
    return encode(walls)


@torch.no_grad()
def block_accuracy(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    return float(((model(inputs) > 0).float() == targets).all(1).float().mean())


# ===========================================================================
# The model
# ===========================================================================

class TransformerBlock(nn.Module):
    """Pre-norm encoder block: LayerNorm, attention, residual; LayerNorm, MLP, residual."""

    def __init__(self, channels: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm_attn(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + self.dropout(attn_out)
        return h + self.ffn(self.norm_ffn(h))


class ChainTransformer(nn.Module):
    """Encoder-only Transformer over the L positions, bidirectional, with a learned
    positional embedding: [B, 2, L] -> one logit per position [B, L].

    The order in which parameters are created fixes how a seed initializes them.
    """

    def __init__(self, L: int, channels: int, depth: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(2, channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, L, channels))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(channels, num_heads, ffn_dim, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.transpose(1, 2)) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.norm_out(h)).squeeze(-1)


# ===========================================================================
# One replica
# ===========================================================================

def train_replica(seed: int, L: int = 12, N: int = 1280, epochs: int = EPOCHS):
    """Train replica `seed` on the shared training set.

    Returns (seed, A_block after each epoch, soft spins on the evaluation inputs
    after the last epoch).  One intra-op thread, so replicas can run side by side.
    """
    torch.set_num_threads(1)
    x_train, z_train = draw_set(N, L, TRAIN_SEED)
    x_eval, z_eval = draw_set(EVAL_SAMPLES, L, EVAL_SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ChainTransformer(L=L, **ARCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = torch.Generator().manual_seed(seed + 1)
    curve = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(N, generator=order)
        for k in range(math.ceil(N / BATCH)):
            idx = perm[k * BATCH:(k + 1) * BATCH]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x_train[idx]), z_train[idx])
            loss.backward()
            opt.step()
        curve.append(block_accuracy(model, x_eval, z_eval))
    with torch.no_grad():
        soft = torch.tanh(model(x_eval) / 2.0).numpy().astype(np.float32)
    return seed, np.array(curve), soft


def train_ensemble(seeds, L: int, N: int, workers: int):
    """Train the given replicas in parallel processes; returns seeds, block_acc, xhat, yspin."""
    started = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(train_replica, s, L, N) for s in seeds]
        for f in futures:
            s, curve, soft = f.result()
            results[s] = (curve, soft)
            label = endpoint_class(curve[None])[0]
            print(f"  replica {s:3d}: peak A_block {curve.max():.2f}, at epoch {len(curve)} "
                  f"{curve[-1]:.2f}  -> {label}   ({time.time() - started:.0f} s)", flush=True)
    order = sorted(results)
    _, z_eval = draw_set(EVAL_SAMPLES, L, EVAL_SEED)
    return (np.array(order), np.stack([results[s][0] for s in order]),
            np.stack([results[s][1] for s in order]),
            (2 * z_eval - 1).numpy().astype(np.float64))


# ===========================================================================
# Endpoint classes
# ===========================================================================

CLASSES = ("never", "stable", "retreat", "recovery", "neither")


def endpoint_class(block_acc: np.ndarray, epoch: int = EPOCHS) -> list[str]:
    """The class of each A_block curve (rows of block_acc) at the observation epoch."""
    labels = []
    for curve in block_acc[:, :epoch]:
        if curve.max() < HI:
            labels.append("never")
            continue
        first = int((curve >= HI).argmax())
        if not (curve[first:] < LO).any():
            labels.append("stable")
        elif curve[-1] < LO:
            labels.append("retreat")
        elif curve[-1] >= HI:
            labels.append("recovery")
        else:
            labels.append("neither")
    return labels


# ===========================================================================
# Overlaps
# ===========================================================================

def decompose(xhat: np.ndarray, yspin: np.ndarray, window: slice) -> dict:
    """m, Q^raw, C^prof and W of the replicas xhat [R, inputs, L] on one window.

    The definitions of the docstring, written out; summarize computes the same
    matrices inside its own arithmetic.
    """
    a = xhat[:, :, window].astype(np.float64) * yspin[None, :, window]
    mu = a.mean(axis=1)                                   # profile [R, positions]
    m = mu.mean(axis=1)
    raw = np.einsum("rbj,sbj->rs", a, a) / (a.shape[1] * a.shape[2])
    dmu = mu - m[:, None]
    prof = dmu @ dmu.T / mu.shape[1]
    eps = a - mu[:, None, :]
    W = np.einsum("rbj,sbj->rs", eps, eps) / (a.shape[1] * a.shape[2])
    return dict(m=m, raw=raw, prof=prof, W=W)


def gap(A: np.ndarray) -> float:
    """Mean diagonal minus mean off-diagonal entry."""
    n = A.shape[0]
    return float(np.mean(np.diag(A)) - (A.sum() - np.trace(A)) / (n * n - n))


def normalized_residual(xhat: np.ndarray, yspin: np.ndarray, window: slice) -> np.ndarray:
    """R^W on one window: the residual Gram matrix normalized by its diagonal."""
    v = xhat[:, :, window].astype(np.float64)
    t = yspin[:, window]
    n = v.shape[0]
    al = v * t[None]
    cf = (al - al.mean(axis=1)[:, None, :]).reshape(n, -1)
    W = cf @ cf.T / cf.shape[1]
    return W / np.sqrt(np.outer(np.diag(W), np.diag(W)))


def obtuse_pair_fraction(R: np.ndarray) -> tuple[float, float]:
    """P(R^W < 0) over distinct pairs, and its leave-one-replica jackknife standard error
    (spread of up to 120 leave-one-out values, scaled by sqrt(n-1))."""
    n = R.shape[0]
    p = float((R[np.triu_indices(n, 1)] < 0).mean())
    deleted = np.arange(n) if n <= 120 else np.random.default_rng(3).choice(n, 120, replace=False)
    lo = np.array([(np.delete(np.delete(R, i, 0), i, 1)[np.triu_indices(n - 1, 1)] < 0).mean()
                   for i in deleted])
    return p, float(np.sqrt(n - 1) * lo.std())


def _subset_tree_fit(R, n, m, subset_draws):
    """Mean tree fit and near-ultrametric fraction of R over `subset_draws` random subsets of
    size m."""
    rg = np.random.default_rng(23)
    tri = np.array(list(itertools.combinations(range(m), 3)))
    r2s, pus = [], []
    for _ in range(subset_draws):
        k = rg.choice(n, m, replace=False)
        r = R[np.ix_(k, k)]
        d = np.maximum(1.0 - r, 0.0)
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2
        dc = squareform(d, checks=False)
        Z = linkage(dc, method="average")
        _, co = cophenet(Z, dc)
        r2s.append(1.0 - float(np.sum((dc - co) ** 2) / np.sum((dc - dc.mean()) ** 2)))
        q = np.sort(np.stack([r[tri[:, 0], tri[:, 1]], r[tri[:, 0], tri[:, 2]],
                              r[tri[:, 1], tri[:, 2]]], axis=1), axis=1)
        den = q[:, 2] - q[:, 0]
        ok = den > 0
        pus.append(float((((q[ok, 1] - q[ok, 0]) / den[ok]) < 0.1).mean()))
    return float(np.mean(r2s)), float(np.mean(pus))


def summarize(xhat, yspin, window, subset_size=SUBSET_SIZE, subset_draws=SUBSET_DRAWS):
    """Gaps, overlaps and tree statistics of one replica set on one window.

    Returns a dict with
      n                          number of replicas
      gap_raw, gap_Qc, gap_W     self-cross gaps of Q^raw, Q^c and W; share = gap_W/gap_Qc
      m, q_self, q_cross         the raw overlaps; Delta_N = q_self - m, chi_SG = q_self - q_cross
      I_C, f_max, r2_tree, p_um  Colless index, largest-cluster fraction of the K=4 cut, tree
                                 fit and near-ultrametric fraction, each the mean over
                                 `subset_draws` random subsets of `subset_size` replicas; *_sd
                                 is the standard deviation over those draws
      *_se                       standard errors, see the comments below
      trim5_n, trim5_k4          the K=4 cut of the full set after the residual-norm filter
    """
    v = xhat[:, :, window].astype(np.float64)
    t = yspin[:, window]
    n = v.shape[0]
    al = v * t[None]
    m_r = al.mean(axis=(1, 2))
    fl = v.reshape(n, -1)
    Qc = fl @ fl.T / fl.shape[1] - np.outer(m_r, m_r)
    cf = (al - al.mean(axis=1)[:, None, :]).reshape(n, -1)
    W = cf @ cf.T / cf.shape[1]
    g = lambda M: float(np.mean(np.diag(M)) - (M.sum() - np.trace(M)) / (n * n - n))
    rw = W / np.sqrt(np.outer(np.diag(W), np.diag(W)))

    # The illustrative K=4 cuts use the same filter as the retreat tree of Fig. 7: retain
    # floor(0.95 n) replicas with the largest residual norms W_rr, then build the tree.
    n_filtered = int(0.95 * n)
    keep_filtered = np.sort(np.argsort(np.diag(W))[n - n_filtered:])
    rw_filtered = rw[np.ix_(keep_filtered, keep_filtered)]
    d_filtered = np.maximum(1.0 - rw_filtered, 0.0)
    np.fill_diagonal(d_filtered, 0.0)
    d_filtered = (d_filtered + d_filtered.T) / 2
    z_filtered = linkage(squareform(d_filtered, checks=False), method="average")
    lab_filtered = fcluster(z_filtered, t=4, criterion="maxclust")
    filtered_k4 = sorted((int((lab_filtered == c).sum()) for c in set(lab_filtered)),
                         reverse=True)

    # The four tree statistics on random subsets of `subset_size` replicas, all at one count.
    rng = np.random.default_rng(23)
    ics, fms, r2s, pus = [], [], [], []
    tri = np.array(list(itertools.combinations(range(subset_size), 3)))
    for _ in range(subset_draws):
        k = rng.choice(n, subset_size, replace=False)
        r = rw[np.ix_(k, k)]
        d = np.maximum(1.0 - r, 0.0)
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2
        dc = squareform(d, checks=False)
        Z = linkage(dc, method="average")
        sz = lambda q: 1 if q < subset_size else int(Z[int(q) - subset_size, 3])
        ics.append(2.0 * sum(abs(sz(int(Z[i, 0])) - sz(int(Z[i, 1])))
                             for i in range(subset_size - 1))
                   / ((subset_size - 1) * (subset_size - 2)))
        lab = fcluster(Z, t=4, criterion="maxclust")
        fms.append(max(int((lab == c).sum()) for c in set(lab)) / subset_size)
        _, co = cophenet(Z, dc)
        r2s.append(1.0 - float(np.sum((dc - co) ** 2) / np.sum((dc - dc.mean()) ** 2)))
        q = np.sort(np.stack([r[tri[:, 0], tri[:, 1]], r[tri[:, 0], tri[:, 2]],
                              r[tri[:, 1], tri[:, 2]]], axis=1), axis=1)
        den = q[:, 2] - q[:, 0]
        ok = den > 0
        pus.append(float((((q[ok, 1] - q[ok, 0]) / den[ok]) < 0.1).mean()))

    # Leave-one-replica jackknife on the tree fit and the near-ultrametric fraction.  The draw
    # size is min(subset_size, n-1), and the same subset pattern is used for every deletion, so
    # the Monte-Carlo noise of the inner draws largely cancels in the spread.  At most 60
    # deletions.
    m_eff = min(subset_size, n - 1)
    deleted = np.arange(n) if n <= 60 else np.random.default_rng(7).choice(n, 60, replace=False)
    jk = np.array([_subset_tree_fit(np.delete(np.delete(rw, i, 0), i, 1), n - 1, m_eff,
                                    subset_draws) for i in deleted])
    r2_tree_se = float(np.sqrt(n - 1) * jk[:, 0].std())
    p_um_se = float(np.sqrt(n - 1) * jk[:, 1].std())

    iu = np.triu_indices(n, 1)
    Qraw = fl @ fl.T / fl.shape[1]          # = E[u_r u_s] because the truth spins square to one
    gap_raw = g(Qraw)
    q_self, q_cross, m = float(np.diag(Qraw).mean()), float(Qraw[iu].mean()), float(m_r.mean())
    # m, q_self and Delta_N are means of per-replica quantities, so SE = sd/sqrt(n).  q_cross
    # and chi_SG are degree-2 U-statistics whose SE follows the Hajek projection on the
    # per-replica row means; pair entries that share a replica are dependent.
    q_self_r = np.diag(Qraw)
    offQ = Qraw.copy()
    np.fill_diagonal(offQ, 0.0)
    qbar = offQ.sum(1) / (n - 1)
    se = lambda u: float(np.std(u, ddof=1) / np.sqrt(n))

    # Jackknife standard errors on the connected gap and the residual share (at most 60
    # deletions, spread scaled by sqrt(n-1)).
    def gap_and_share(u):
        uu = u.astype(np.float64)
        nn = uu.shape[0]
        aa = uu * t[None]
        mm = aa.mean(axis=(1, 2))
        ff = uu.reshape(nn, -1)
        Qc2 = ff @ ff.T / ff.shape[1] - np.outer(mm, mm)
        cc = (aa - aa.mean(axis=1)[:, None, :]).reshape(nn, -1)
        W2 = cc @ cc.T / cc.shape[1]
        gg = lambda Mx: float(np.mean(np.diag(Mx)) - (Mx.sum() - np.trace(Mx)) / (nn * nn - nn))
        return gg(Qc2), gg(W2) / gg(Qc2)
    idx = np.arange(n) if n <= 60 else np.random.default_rng(3).choice(n, 60, replace=False)
    lo = np.array([gap_and_share(np.delete(v, i, 0)) for i in idx])
    gap_Qc_se = np.sqrt(n - 1) * lo[:, 0].std()
    share_se = np.sqrt(n - 1) * lo[:, 1].std()

    return dict(n=n, gap_raw=gap_raw, gap_Qc=g(Qc), gap_W=g(W), share=g(W) / g(Qc),
                gap_Qc_se=float(gap_Qc_se), share_se=float(share_se),
                m=m, q_self=q_self, q_cross=q_cross,
                Delta_N=q_self - m, chi_SG=q_self - q_cross,
                m_se=se(m_r), q_self_se=se(q_self_r), q_cross_se=2 * se(qbar),
                Delta_N_se=se(q_self_r - m_r), chi_SG_se=se(q_self_r - 2 * qbar),
                I_C=float(np.mean(ics)), I_C_sd=float(np.std(ics, ddof=1)),
                f_max=float(np.mean(fms)), f_max_sd=float(np.std(fms, ddof=1)),
                r2_tree=float(np.mean(r2s)), r2_tree_sd=float(np.std(r2s, ddof=1)),
                r2_tree_se=r2_tree_se, p_um_se=p_um_se,
                p_um=float(np.mean(pus)), p_um_sd=float(np.std(pus, ddof=1)),
                trim5_n=n_filtered, trim5_k4=filtered_k4)


def _rw_on(aligned: np.ndarray, index: np.ndarray) -> np.ndarray:
    """R^W estimated from the evaluation inputs in `index` alone, profile included."""
    sub = aligned[:, index, :]
    flat = (sub - sub.mean(axis=1)[:, None, :]).reshape(sub.shape[0], -1)
    W = flat @ flat.T / flat.shape[1]
    return W / np.sqrt(np.outer(np.diag(W), np.diag(W)))


def split_half(xhat: np.ndarray, yspin: np.ndarray, window: slice) -> dict:
    """Split-half control on P(R^W < 0).

    agree: of the pairs obtuse on the full set, the share obtuse on both halves.  Under noise
    alone a different set of pairs comes out negative on each half and the fraction rises
    with less data; genuinely opposed pairs stay obtuse.  The contiguous halves are the
    reported ones; a seeded random split is given for comparison.
    """
    aligned = xhat[:, :, window].astype(np.float64) * yspin[None, :, window]
    n, nb = aligned.shape[0], aligned.shape[1]
    iu = np.triu_indices(n, 1)
    full = _rw_on(aligned, np.arange(nb))[iu]
    cut = nb // 2
    con = [_rw_on(aligned, np.arange(cut))[iu], _rw_on(aligned, np.arange(cut, nb))[iu]]
    perm = np.random.default_rng(0).permutation(nb)
    ran = [_rw_on(aligned, perm[:cut])[iu], _rw_on(aligned, perm[cut:])[iu]]
    obtuse = full < 0
    both = lambda h: float(((h[0] < 0) & (h[1] < 0))[obtuse].mean()) if obtuse.any() else float("nan")
    return dict(p_full=float(obtuse.mean()), p_half=[float((h < 0).mean()) for h in con],
                agree=both(con), agree_random=both(ran))


def report(name: str, xhat, yspin, window, subset_size=SUBSET_SIZE,
           subset_draws=SUBSET_DRAWS) -> dict:
    """Print the observables of one replica set."""
    # numpy can raise spurious divide and overflow flags on the BLAS matmul path; the
    # results are checked for finiteness instead.
    with np.errstate(all="ignore"):
        r = summarize(xhat, yspin, window, subset_size, subset_draws)
        p, p_se = obtuse_pair_fraction(normalized_residual(xhat, yspin, window))
        sh = split_half(xhat, yspin, window)
    assert all(np.isfinite(v) for v in r.values() if isinstance(v, float))
    print(f"  {name:<9} n={r['n']:<4} m={r['m']:.4f}  q_self={r['q_self']:.4f}  "
          f"q_cross={r['q_cross']:.4f}  Delta_N={r['Delta_N']:+.4f}  chi_SG={r['chi_SG']:.4f}")
    print(f"  {'':<9} gap(Q^c)={r['gap_Qc']:.4f} +- {r['gap_Qc_se']:.4f}  "
          f"share={r['share']:.4f} +- {r['share_se']:.4f}")
    print(f"  {'':<9} R2_tree={r['r2_tree']:.3f}  P(d_um<0.1)={r['p_um']:.3f}  "
          f"I_C={r['I_C']:.3f}  f_max={r['f_max']:.3f}   "
          f"({subset_size} replicas, {subset_draws} draws)")
    print(f"  {'':<9} P(R^W<0)={p:.4f} +- {p_se:.4f}   split-half: halves "
          f"{sh['p_half'][0]:.4f}/{sh['p_half'][1]:.4f}, obtuse on both {sh['agree']:.3f}")
    return dict(r, p_obtuse=p, p_obtuse_se=p_se, **{f"split_{k}": v for k, v in sh.items()})


def load_ensemble(pattern: str):
    """Pool the stored parts (seeds<a>-<b>.npz files) that match `pattern`, each seed once."""
    s, a, x, ys = [], [], [], None
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        s.append(z["seeds"])
        a.append(z["block_acc"])
        x.append(z["final_xhat"].astype(np.float32))
        ys = z["yspin"]
    if not s:
        raise SystemExit(f"no ensemble part matches {pattern}")
    s = np.concatenate(s).astype(int)
    a = np.concatenate(a)
    x = np.concatenate(x)
    _, i = np.unique(s, return_index=True)
    return s[i], a[i], x[i], ys.astype(np.float64)


# ===========================================================================
# The learning-rate intervention
# ===========================================================================

def intervention(N: int, data_seed: int, model_seed: int, schedule: str, L: int = 12,
                 steps: int = 2400, lr: float = LR, min_factor: float = 0.02,
                 trigger: float = HI, batch: int = BATCH, every: int = 100) -> dict:
    """One arm: schedule is "constant" or "postgen-cosine".  Returns the decay onset t* and
    the held-out A_block and learning rate every `every` updates."""
    torch.set_num_threads(1)
    x_all, z_all = complete_set(L)
    perm = torch.randperm(x_all.shape[0], generator=torch.Generator().manual_seed(data_seed))
    test, pool = perm[:512], perm[512:]
    x_test, z_test = x_all[test], z_all[test]
    x_train, z_train = x_all[pool][:N], z_all[pool][:N]

    seed = 100_000 + 1_003 * data_seed + model_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ChainTransformer(L=L, **ARCH)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    minibatches = torch.Generator().manual_seed(200_000 + 1_009 * data_seed + model_seed)
    eta_min, onset, history = lr * min_factor, None, []

    def evaluate(step):
        nonlocal onset
        acc = block_accuracy(model, x_test, z_test)
        history.append(dict(step=step, lr=opt.param_groups[0]["lr"], test_block_acc=acc))
        if schedule == "postgen-cosine" and onset is None and step > 0 and acc >= trigger:
            onset = step

    evaluate(0)
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(N, (batch,), generator=minibatches)
        loss = F.binary_cross_entropy_with_logits(model(x_train[idx]), z_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # the decay starts at the update after the checkpoint that saw A_block >= trigger
        if onset is not None and steps > onset:
            progress = min((step - onset) / (steps - onset), 1.0)
            for group in opt.param_groups:
                group["lr"] = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * progress)) / 2
        if step % every == 0 or step == steps:
            evaluate(step)
            model.train()
    return dict(decay_onset_step=onset, history=history)


# ===========================================================================
# Self-checks
# ===========================================================================

def self_check() -> None:
    rng = np.random.default_rng(0)

    # The task: every input once, spins flip exactly at the walls.
    x, z = complete_set(12)
    walls = x[:, 0, 1:]
    assert len({tuple(w) for w in walls.int().tolist()}) == 2048
    spins = 1 - 2 * z
    assert torch.equal(spins[:, 0], torch.ones(2048))
    assert torch.equal((spins[:, 1:] != spins[:, :-1]).float(), walls)
    assert torch.equal(x[:, 1, :-1], walls)

    # The model: [B, 2, L] -> [B, L], with the paper's parameter count.
    model = ChainTransformer(L=12, **ARCH)
    assert model(x[:5]).shape == (5, 12)
    assert sum(p.numel() for p in model.parameters()) == 26529

    # Q^raw = m m^T + C^prof + W, exactly, and gap is additive over the three terms.
    xhat = np.tanh(rng.normal(size=(9, 300, 12)))
    yspin = np.where(rng.random((300, 12)) < 0.5, -1.0, 1.0)
    for window in (slice(0, 12), TAIL[12]):
        d = decompose(xhat, yspin, window)
        total = np.outer(d["m"], d["m"]) + d["prof"] + d["W"]
        assert np.abs(d["raw"] - total).max() < 1e-12
        assert abs(gap(d["raw"]) - gap(np.outer(d["m"], d["m"])) - gap(d["prof"])
                   - gap(d["W"])) < 1e-12
        r = summarize(xhat, yspin, window, subset_size=6, subset_draws=5)
        assert abs(r["gap_Qc"] - gap(d["prof"] + d["W"])) < 1e-12
        assert abs(r["gap_W"] - gap(d["W"])) < 1e-12
        assert abs(r["q_self"] - np.mean(np.diag(d["raw"]))) < 1e-12
        assert abs(r["m"] - d["m"].mean()) < 1e-12

    # The overlaps of a replica set that outputs the truth with confidence c: m = c and
    # q_self = q_cross = c^2, so the Nishimori gap is c^2 - c < 0 below full confidence.
    c = 0.7
    same = np.repeat((c * yspin)[None], 5, axis=0)
    d = decompose(same, yspin, slice(0, 12))
    assert np.allclose(d["m"], c) and np.allclose(d["raw"], c * c)

    # Tree statistics: an exactly ultrametric R (two superclusters of two clusters each)
    # gives a perfect tree fit and only isosceles-narrow triangles.
    labels = np.repeat(np.arange(4), 12)
    R = np.where(labels[:, None] // 2 == labels[None] // 2, 0.3, -0.1)
    R = np.where(labels[:, None] == labels[None], 0.8, R)
    np.fill_diagonal(R, 1.0)
    r2, pum = _subset_tree_fit(R, 48, 47, 3)
    assert r2 > 1 - 1e-12 and pum == 1.0
    p, _ = obtuse_pair_fraction(R)
    assert abs(p - 24 * 24 / (48 * 47 / 2)) < 1e-12      # the pairs across superclusters

    # Endpoint classes.
    curves = np.array([np.full(150, 0.3),                                    # never
                       np.r_[np.zeros(20), np.full(130, 0.9)],               # stable
                       np.r_[np.zeros(20), np.full(30, 0.9), np.full(100, 0.1)],   # retreat
                       np.r_[np.zeros(20), np.full(30, 0.9), np.full(50, 0.1),
                             np.full(50, 0.95)],                             # recovery
                       np.r_[np.zeros(20), np.full(30, 0.9), np.full(50, 0.1),
                             np.full(50, 0.6)]])                             # neither
    assert endpoint_class(curves) == list(CLASSES)
    print("self-checks passed")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="self-checks only")
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--N", type=int, default=None,
                   help="training-set size (default 1280 for the ensemble, 512 for --intervention)")
    p.add_argument("--replicas", type=int, default=16, help="replicas to train")
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    p.add_argument("--save", default="", metavar="FILE",
                   help="write the trained ensemble as one part file FILE (.npz)")
    p.add_argument("--ensemble", default="", metavar="GLOB",
                   help="part files of a stored ensemble; report their observables")
    p.add_argument("--window", choices=("full", "tail"), default="full")
    p.add_argument("--intervention", action="store_true",
                   help="run the constant and the post-generalization cosine arm")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--model-seed", type=int, default=0)
    a = p.parse_args()

    self_check()
    if a.check:
        return 0
    window = slice(0, a.L) if a.window == "full" else TAIL[a.L]

    if a.intervention:
        N = a.N or 512
        arms = {}
        for schedule in ("constant", "postgen-cosine"):
            print(f"\n{schedule} arm, N={N}, data seed {a.data_seed}, model seed {a.model_seed}",
                  flush=True)
            arms[schedule] = intervention(N, a.data_seed, a.model_seed, schedule, L=a.L)
        onset = arms["postgen-cosine"]["decay_onset_step"]
        print(f"\ndecay onset t* = {onset}\n  update   A_block constant   A_block decayed   lr decayed")
        for h0, h1 in zip(arms["constant"]["history"], arms["postgen-cosine"]["history"]):
            print(f"  {h0['step']:6d}   {h0['test_block_acc']:16.3f}   "
                  f"{h1['test_block_acc']:15.3f}   {h1['lr']:.2e}")
        return 0

    if a.ensemble:
        seeds, block_acc, xhat, yspin = load_ensemble(a.ensemble)
        names, subset_draws = ("retreat", "recovery"), SUBSET_DRAWS
    else:
        N = a.N or 1280
        seeds = range(a.first_seed, a.first_seed + a.replicas)
        print(f"\ntraining {a.replicas} replicas at (L, N) = ({a.L}, {N}), "
              f"{a.workers} at a time, {EPOCHS} epochs each", flush=True)
        seeds, block_acc, xhat, yspin = train_ensemble(seeds, a.L, N, a.workers)
        if a.save:
            os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
            np.savez_compressed(a.save,
                                final_xhat=xhat, block_acc=block_acc, seeds=seeds,
                                yspin=yspin.astype(np.float32))
        names, subset_draws = ("all", "retreat", "recovery"), 100

    labels = np.array(endpoint_class(block_acc))
    print(f"\n{len(seeds)} replicas: " + ", ".join(f"{(labels == c).sum()} {c}" for c in CLASSES))
    print(f"observables on output positions {window.start}..{window.stop - 1}")
    for name in names:
        keep = np.ones(len(seeds), bool) if name == "all" else labels == name
        n = int(keep.sum())
        if n < 5:
            print(f"  {name:<9} {n} replicas, too few")
            continue
        report(name, xhat[keep], yspin, window, min(SUBSET_SIZE, n), subset_draws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
