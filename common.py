"""The task, the network, the data layout and the replica sets shared by analysis.py,
figures.py and train.py of "Replica Fragmentation and Glassy Dynamics in Parity Learning".

The file is imported, not run.  It holds what more than one of the three files needs:
the directories, the input encoding and the input sets of the task, the Transformer, the
loaders of the stored replica ensembles, the reach and retreat thresholds with the endpoint
selection of Sec. 4, the overlap matrices with the two normalized-residual measurements built
on them, the table writers, and the command line of analysis.py and figures.py.

===========================================================================
THE TASK
===========================================================================
An Ising chain of L spins whose first spin points up.  The input is the L-1 domain walls,
s_i = 1 where spins i and i+1 differ (i = 0, ..., L-2).  The target bit at position j is

    z_j = s_0 + s_1 + ... + s_{j-1}  mod 2,         z_0 = 0,

so output j is the parity of the first j walls; the truth spin is y_j = 2 z_j - 1.  Each wall
is fed to the two positions it separates: channel 0 of position j carries the wall on its left,
s_{j-1}, and channel 1 the wall on its right, s_j.  A network returns one logit per position.
The soft spin u_j = tanh(logit_j / 2) = 2 sigmoid(logit_j) - 1 is the magnetization of the
Bernoulli distribution the logit defines, so u_j y_j > 0 means position j is right.  The block
accuracy A_block is the fraction of inputs with every position right.

There are 2^(L-1) inputs.  The training set of a shared-training-set ensemble is the
quenched disorder: N uniform draws with the seed TRAIN_SEED, shared by every replica, which
covers fewer than N distinct configurations because draws repeat.  Three input sets recur:

    sample_set(num, L, seed)   num independent uniform draws
    complete_set(L)            all 2^(L-1) configurations; row k has s_i = bit i of k
    held_out_set(L, N)         the complete set minus the configurations of the training
                               set, in the order of complete_set; at (12, 1280) it holds
                               2048 - 963 = 1085 configurations, at (12, 64) 1984

===========================================================================
THE NETWORK
===========================================================================
ChainTransformer is a bidirectional pre-norm encoder over the L positions: a linear embedding of
the two input channels plus a learned positional embedding, `depth` blocks of
(LayerNorm, multi-head attention, residual; LayerNorm, two-layer GELU feed-forward,
residual), a final LayerNorm and one logit per position.  Every network of the paper uses
ARCHITECTURE: 16 channels, depth 8, 4 heads, feed-forward width 64, no dropout.  The order in
which the constructor creates parameters fixes how a seed initializes them.

===========================================================================
THE REPLICA SETS OF SEC. 4
===========================================================================
A replica is one network trained on the shared training set from its own initialization
seed.  Two kinds of ensemble enter Table 3:

    MEMORIZATION   four small-data ensembles (label, L, N, interpolating_only); the networks
                   are stored as weights and evaluated here
    LARGE_DATA     five large-data ensembles (label, L, N); each stored replica carries its
                   soft spins on 2048 evaluation draws (seed EVAL_SEED) after epoch 150 and
                   its A_block after every epoch on the same draws

A label is the string "(L,N)".  The endpoint class of a large-data replica is read off its
A_block curve at the observation epoch EPOCHS = 150 with the thresholds HI = 0.8, LO = 0.5:

    retreat     A_block >= HI at some epoch, and A_block < LO at epoch 150
    recovery    after its first crossing of HI it falls below LO, and A_block >= HI at
                epoch 150

The same two thresholds define every crossing in the paper: a run reaches, or acquires the
rule, at its first A_block >= HI (first_crossing), and retreats at its first A_block < LO
after that (first_fall), in the training-set-size scans, the trajectories and the
learning-rate intervention alike.

Each large-data ensemble therefore contributes a retreat set and a recovery set (regime
names "retreat" and "recovered" in the keys "<regime>|(L,N)"), which with the four
memorization sets gives the fourteen sets of Sec. 4.

Windows of output positions: all positions, the tabulated tails of Table 3 (TAIL for the
large-data ensembles, TAIL_MEMORIZATION for the memorization ensembles), and the common tail
j >= 2L/3 (common_tail), the one rule applied to every set when regimes are compared on a
tail.

===========================================================================
THE OVERLAP MATRICES
===========================================================================
For replica r, evaluation input b and output position j in a window, a_rj(b) = u_rj(b) y_j(b)
is the truth-aligned soft spin.  With E the average over b and j,

    m_r        = E[a_r]                            truth overlap (Mattis magnetization)
    Q^raw_rs   = E[u_r u_s] = E[a_r a_s]           raw overlap, since y^2 = 1
    Q^c_rs     = Q^raw_rs - m_r m_s                connected overlap
    mu_rj      = E_b[a_rj(b)]                      truth-alignment profile
    W_rs       = E_bj[(a_rj(b) - mu_rj)(a_sj(b) - mu_sj)]   profile-centered residual
    C^prof     = Q^c - W                           covariance of the mean profiles
    R^W_rs     = W_rs / sqrt(W_rr W_ss)            normalized residual: the cosine of the
                                                   angle between two replicas' residuals
    gap(A)     = mean diagonal - mean off-diagonal entry of a replica matrix A

q_self and q_cross are the mean diagonal and off-diagonal entries of Q^raw; the Nishimori gap
is Delta_N = q_self - m and the self-cross gap chi_SG = q_self - q_cross = gap(Q^raw).  The
share gap(W) / gap(Q^c) is the part of the connected self-cross gap that depends on the input
rather than on the mean profile.  P(R^W < 0), the obtuse-pair fraction of Sec. 4.3, is the
share of replica pairs whose residuals are more than a right angle apart.

===========================================================================
PAPER LOCATION -> NAME
===========================================================================
    Sec. 2.1, App. A.1           encode, sample_walls, sample_set, complete_set, wall_codes,
                                 held_out_set, TRAIN_SEED, EVAL_SEED
    Sec. 2.3                     TransformerBlock, ChainTransformer, ARCHITECTURE,
                                 load_chain_transformer
    Fig. 1, Sec. 3.1, App. B     INDEPENDENT_TRAINING_SETS, TRANSFORMER_SIZE_SCAN, CNN_SIZE_SCAN,
                                 REPRESENTATIVE_TRANSFORMER_RUNS, REPRESENTATIVE_CNN_RUNS,
                                 load_history, history_crossing
    Table 3                      MEMORIZATION, LARGE_DATA, TAIL, TAIL_MEMORIZATION,
                                 SMALL_DATA_ENSEMBLES, LARGE_DATA_ENSEMBLES,
                                 load_memorization, load_large_data
    Table 2, Secs. 3.2-3.4       TRAJECTORY_ENSEMBLES, TRAJECTORY_EPOCHS
    Sec. 5.2                     LEARNING_RATE_INTERVENTION
    App. A                       CHECKPOINTED_TRAJECTORIES, FIXED_EPOCH_REPLICAS, RETREAT_EVENTS
    Sec. 4.1                     overlap_matrices, gap, normalize_by_diagonal,
                                 normalized_residual, common_tail
    Sec. 4.2, App. C.1           HI, LO, EPOCHS, reached, retreat, recovery, ends_high,
                                 endpoint_sets
    Secs. 3, 4, 5.2, App. C.1    first_crossing, crossing_epoch, first_fall, crossing_outcome
    Sec. 4.2, Figs. 7, C1, C2    load_l12_n1280_in_storage_order
    Sec. 4.3, Figs. 8, 9         obtuse_fraction, obtuse_pair_fraction

write_json and write_rows write the tables of all three files, and run_items is the command
line of analysis.py and figures.py.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ===========================================================================
# Directories
# ===========================================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"          # trained networks and training records; only train.py writes it
RESULTS = ROOT / "results"    # tables and arrays written by analysis.py (and a few by figures.py)
FIGURE = ROOT / "figure"      # the figures of the paper, and nothing else

# Runs with independently sampled fixed training sets (Sec. 3.1, App. B): every run draws its
# own N strings from its seed, so two runs never share a training set.  The training-set-size
# scans hold one directory L<L>_N<N>_seed<s> per run (120 epochs); the representative runs of
# Fig. 1(a) hold one directory seed<s> per run (600 epochs).
INDEPENDENT_TRAINING_SETS = DATA / "independent_training_sets"
TRANSFORMER_SIZE_SCAN = {length: INDEPENDENT_TRAINING_SETS / "size_scan" / f"transformer_L{length}"
                         for length in (12, 16, 20)}
CNN_SIZE_SCAN = INDEPENDENT_TRAINING_SETS / "size_scan" / "cnn_L12"
REPRESENTATIVE_TRANSFORMER_RUNS = (INDEPENDENT_TRAINING_SETS / "representative_runs"
                                   / "transformer_L12_N1280")
REPRESENTATIVE_CNN_RUNS = INDEPENDENT_TRAINING_SETS / "representative_runs" / "cnn_L12_N2048"
# Runs with a fresh draw for every minibatch; the scans above take their configurations from them.
FRESH_MINIBATCH_RUNS = DATA / "fresh_minibatch_runs"

# Replicas on the shared training set.
TRAJECTORY_ENSEMBLES = DATA / "trajectory_ensembles"      # recorded every epoch (Table 2, Fig. 3)
TRAJECTORY_EPOCHS = 600       # epochs of each trajectory of L12_N1280_overlaps (Figs. 2, 3)
SMALL_DATA_ENSEMBLES = DATA / "small_data_ensembles"      # L<L>_N<N>/seed<s>/ (Table 3, Fig. 5)
LARGE_DATA_ENSEMBLES = DATA / "large_data_ensembles"      # L<L>_N<N>/seeds<a>-<b>.npz (Table 3)
CHECKPOINTED_TRAJECTORIES = DATA / "checkpointed_trajectories"   # weights at epochs 20-220 (App. A)
FIXED_EPOCH_REPLICAS = DATA / "fixed_epoch_replicas"      # weights at epoch 150 (App. A.1)
RETREAT_EVENTS = DATA / "retreat_events"                  # events.csv (App. A.2)
LEARNING_RATE_INTERVENTION = DATA / "learning_rate_intervention"   # the two arms of Sec. 5.2


def load_history(path: Path) -> list[dict]:
    """The per-epoch rows of one fixed-training-set run's metrics.json.

    A row carries the epoch ``epoch`` (counted from 1), the held-out accuracies ``bit_acc`` and
    ``block_acc`` and, in the scans, the accuracies on the run's own training set
    (``train_block_acc``).  Used for Fig. 1, Fig. B1 and the crossing epochs of Secs. 1 and 3.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["train_history"]


def history_crossing(history: list[dict], key: str = "block_acc") -> int | None:
    """The epoch of the first row of a load_history history whose `key` is at or above HI, or
    None (crossing_epoch)."""
    return crossing_epoch([int(row["epoch"]) for row in history],
                          [float(row[key]) for row in history])


def write_json(path: Path, payload) -> None:
    """json.dump with indent 2 and no trailing newline, the format of most tables here."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_rows(path: Path, rows: list[dict], fieldnames=None) -> None:
    """A CSV file with one row per dict; the columns are the keys of the first row unless
    `fieldnames` gives them."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# The task
# ===========================================================================

TRAIN_SEED = 12345            # the shared training set of every shared-training-set ensemble
EVAL_SEED = 777               # the evaluation draws of the large-data and the L = 24 ensembles


def encode(walls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Network inputs [n, 2, L] and target bits [n, L] of domain walls [n, L-1] in {0, 1}.

    Channel 0 carries wall i at position i+1 and channel 1 at position i, so each position sees
    the walls on both of its sides; the target bit z_j is the parity of the first j walls (1 =
    spin down relative to the first spin).  All operations are exact on 0/1 floats.
    """
    n, L = walls.shape[0], walls.shape[1] + 1
    targets = torch.zeros(n, L)
    targets[:, 1:] = torch.cumsum(walls, 1).remainder(2.0)
    inputs = torch.zeros(n, 2, L)
    inputs[:, 0, 1:] = walls
    inputs[:, 1, :-1] = walls
    return inputs, targets


def sample_walls(num: int, L: int, seed: int) -> torch.Tensor:
    """num independent uniform domain-wall configurations [num, L-1] (int64) from `seed`.

    Every set of the paper is drawn with this generator call, so a seed names a set.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (num, L - 1), generator=generator)


def sample_set(num: int, L: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and targets of num uniform draws from `seed` (TRAIN_SEED: the training set)."""
    return encode(sample_walls(num, L, seed).float())


def complete_set(L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All 2^(L-1) inputs; row k has wall i equal to bit i of k (binary-index order)."""
    index = torch.arange(2 ** (L - 1), dtype=torch.long)
    shifts = torch.arange(L - 1, dtype=torch.long)
    return encode(((index[:, None] >> shifts[None, :]) & 1).float())


def wall_codes(inputs: torch.Tensor) -> torch.Tensor:
    """The integer sum_i s_i 2^i of each input, which is its row index in complete_set."""
    walls = inputs[:, 0, 1:].to(torch.long)
    weights = 2 ** torch.arange(inputs.shape[2] - 1, dtype=torch.long)
    return walls @ weights


def held_out_set(L: int, N: int, seed: int = TRAIN_SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """The complete set minus the configurations of the training set (N draws from `seed`).

    No replica of a shared-training-set ensemble has seen any of these inputs, and averages
    over them are exact averages over the unseen part of the input space.  The rows keep the
    order of complete_set.
    """
    inputs, targets = complete_set(L)
    train_inputs, _ = sample_set(N, L, seed)
    keep = torch.ones(len(inputs), dtype=torch.bool)
    keep[torch.unique(wall_codes(train_inputs))] = False
    return inputs[keep], targets[keep]


# ===========================================================================
# The network
# ===========================================================================

class TransformerBlock(nn.Module):
    """Pre-norm encoder block: LayerNorm, attention, residual; LayerNorm, GELU MLP, residual."""

    def __init__(self, channels: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, channels] -> [B, L, channels]."""
        h_norm = self.norm_attn(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + self.dropout(attn_out)
        h = h + self.ffn(self.norm_ffn(h))
        return h


class ChainTransformer(nn.Module):
    """Encoder-only Transformer over the L positions, without a causal mask: [B, 2, L] -> [B, L]
    logits.

    The target at position j depends on every wall to its left, so attention has to pool across
    positions.  The parameters are created in the order in_proj, pos_embed (initialized
    N(0, 0.02^2)), the blocks, norm_out, out_proj; that order fixes the initialization a seed
    gives.  State-dict keys follow the attribute names, e.g. blocks.7.ffn.3.weight is the
    down-projection of the last block's feed-forward layer.
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
        """x: [B, 2, L] -> logits [B, L]."""
        h = self.in_proj(x.transpose(1, 2)) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.out_proj(h).squeeze(-1)


ARCHITECTURE = dict(channels=16, depth=8, num_heads=4, ffn_dim=64, dropout=0.0)


def load_chain_transformer(path, L: int) -> ChainTransformer:
    """A ChainTransformer of ARCHITECTURE at chain length L with the weights stored at `path`.

    The caller chooses the mode.  With dropout zero both modes give the same function, but
    eval mode lets the attention layer take its fused inference kernel, train mode keeps it on
    the path written in elementary operations, and the two differ in the last bits.
    """
    model = ChainTransformer(L=L, **ARCHITECTURE)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


# ===========================================================================
# The replica sets of Sec. 4
# ===========================================================================

MEMORIZATION = [("(12,64)", 12, 64, False),
                ("(24,64)", 24, 64, False),
                ("(24,128)", 24, 128, False),
                ("(24,192)", 24, 192, True)]
LARGE_DATA = [("(12,1280)", 12, 1280),
              ("(16,2048)", 16, 2048),
              ("(16,2560)", 16, 2560),
              ("(16,3072)", 16, 3072),
              ("(20,24576)", 20, 24576)]

# Endpoint thresholds on the held-out block accuracy and the observation epoch (Sec. 4.2).
HI, LO, EPOCHS = 0.8, 0.5, 150

# The tabulated tails of Table 3 are j >= 2L/3 for the large-data ensembles (8 at L = 12, 11 at
# L = 16, 14 at L = 20) but j >= 18 for the L = 24 memorization ensembles.
TAIL = {12: slice(8, 12), 16: slice(11, 16), 20: slice(14, 20)}
TAIL_MEMORIZATION = {(12, 64): slice(8, 12), (24, 64): slice(18, 24),
                     (24, 128): slice(18, 24), (24, 192): slice(18, 24)}


def common_tail(L: int) -> slice:
    """The window j >= 2L/3 (ceiling), the same rule for every set."""
    return slice(-(-2 * L // 3), L)


def load_memorization(L: int, N: int, interpolating_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """Soft spins xhat [replicas, inputs, L] and truth spins yspin [inputs, L] of the (L, N)
    memorization ensemble.

    Every trained network of the ensemble (data/small_data_ensembles/L<L>_N<N>/seed*/) is
    evaluated in ascending seed order.  The evaluation set is the held-out set at (12, 64) and
    4000 uniform draws with EVAL_SEED at L = 24.  With interpolating_only only the replicas whose
    recorded final training bit accuracy is exactly one are kept, which is what a
    memorization ensemble requires at (24, 192).
    """
    # one intra-op thread for every network evaluation of the ensemble
    torch.set_num_threads(1)
    dirs = {}
    for q in glob.glob(str(SMALL_DATA_ENSEMBLES / f"L{L}_N{N}" / "seed*")):
        if os.path.exists(f"{q}/model.pt"):
            dirs.setdefault(int(re.search(r"seed(\d+)$", q).group(1)), q)
    if interpolating_only:
        def train_bit_acc(q):
            m = json.load(open(f"{q}/metrics.json"))
            return m.get("final_train_bit_acc", m.get("final_train_acc"))
        dirs = {s: q for s, q in dirs.items() if train_bit_acc(q) == 1.0}
    if L == 12 and N == 64:
        x_te, z_te = held_out_set(L, N)
    else:
        x_te, z_te = sample_set(4000, L, EVAL_SEED)
    xh = np.empty((len(dirs), x_te.shape[0], L))
    for i, s in enumerate(sorted(dirs)):
        m = load_chain_transformer(f"{dirs[s]}/model.pt", L).eval()
        with torch.no_grad():
            xh[i] = torch.tanh(m(x_te) / 2.0).numpy()
    return xh, (2 * z_te - 1).numpy().astype(np.float64)


def load_large_data(L: int, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """seeds, block_acc [replicas, epochs], xhat [replicas, 2048, L], yspin [2048, L] of the
    (L, N) large-data ensemble, in ascending seed order.

    The ensemble is stored in parts, data/large_data_ensembles/L<L>_N<N>/seeds<a>-<b>.npz with
    the replicas of seeds a to b; the parts are read in sorted order of their names and a seed
    stored in more than one part is counted once.  xhat is float32, yspin
    float64; block_acc is the A_block after each epoch on the 2048 evaluation draws.
    """
    s, a, x, ys = [], [], [], None
    for f in sorted(glob.glob(str(LARGE_DATA_ENSEMBLES / f"L{L}_N{N}" / "seeds*.npz"))):
        z = np.load(f)
        s.append(z["seeds"])
        a.append(z["block_acc"])
        x.append(z["final_xhat"].astype(np.float32))
        ys = z["yspin"]
    s = np.concatenate(s).astype(int)
    a = np.concatenate(a)
    x = np.concatenate(x)
    _, i = np.unique(s, return_index=True)
    return s[i], a[i], x[i], ys.astype(np.float64)


def load_l12_n1280_in_storage_order() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """xhat, block, seeds, yspin of the (12, 1280) ensemble in storage order.

    The same 1200 replicas as load_large_data(12, 1280), with yspin kept float32: the parts
    data/large_data_ensembles/L12_N1280/seeds<a>-<b>.npz are concatenated in sorted order of
    their names, and a seed stored more than once is kept at its first occurrence.  Fig. 7,
    Figs. C1 and C2 and the numbers of Sec. 4.2 that come from them use this loader.
    """
    parts = sorted((LARGE_DATA_ENSEMBLES / "L12_N1280").glob("seeds*.npz"))
    if not parts:
        raise FileNotFoundError(f"No seeds*.npz found under {LARGE_DATA_ENSEMBLES / 'L12_N1280'}")
    loaded = [np.load(path) for path in parts]
    xhat = np.concatenate([data["final_xhat"] for data in loaded], axis=0)
    block = np.concatenate([data["block_acc"] for data in loaded], axis=0)
    seeds = np.concatenate([data["seeds"] for data in loaded], axis=0)
    yspin = loaded[0]["yspin"]
    _, first = np.unique(seeds, return_index=True)
    first = np.sort(first)
    return xhat[first], block[first], seeds[first], yspin


def first_crossing(curve, level: float = HI) -> int | None:
    """Index of the first value of the curve at or above `level`, or None if there is none."""
    hits = np.flatnonzero(np.asarray(curve) >= level)
    return int(hits[0]) if hits.size else None


def crossing_epoch(epochs, curve, level: float = HI) -> int | None:
    """The entry of `epochs` at the first crossing of `level` by the curve, or None."""
    index = first_crossing(curve, level)
    return None if index is None else int(np.asarray(epochs)[index])


def first_fall(curve, hi: float = HI, lo: float = LO) -> int | None:
    """Index of the first value below `lo` at or after the first crossing of `hi`, or None: the
    retreat event of a trajectory."""
    first = first_crossing(curve, hi)
    if first is None:
        return None
    below = np.flatnonzero(np.asarray(curve)[first:] < lo)
    return first + int(below[0]) if below.size else None


def crossing_outcome(curve, hi: float = HI, lo: float = LO) -> tuple[bool, bool, bool]:
    """(reaches hi, falls below lo at or after its first crossing of hi, ends at or above hi)."""
    return (first_crossing(curve, hi) is not None, first_fall(curve, hi, lo) is not None,
            bool(curve[-1] >= hi))


def reached(block_acc: np.ndarray, hi: float = HI, epoch: int = EPOCHS) -> np.ndarray:
    """Replicas whose block accuracy is at least `hi` at some epoch up to `epoch`."""
    return block_acc[:, :epoch].max(axis=1) >= hi


def retreat(block_acc: np.ndarray, hi: float = HI, lo: float = LO,
            epoch: int = EPOCHS) -> np.ndarray:
    """Retreat endpoints: reached `hi` by `epoch`, and below `lo` at `epoch` (by default HI, LO
    and the observation epoch)."""
    return reached(block_acc, hi, epoch) & (block_acc[:, epoch - 1] < lo)


def recovery(block_acc: np.ndarray) -> np.ndarray:
    """Recovery endpoints: after the first crossing of HI the replica falls below LO, and it is
    at or above HI at the observation epoch.  A replica that never fell is stable and is not
    selected."""
    fell = np.array([first_fall(curve[:EPOCHS]) is not None for curve in block_acc], dtype=bool)
    return fell & (block_acc[:, EPOCHS - 1] >= HI)


def ends_high(block_acc: np.ndarray) -> np.ndarray:
    """Replicas that reached HI and are at or above HI at the observation epoch: the stable and
    the recovery endpoints."""
    return reached(block_acc) & (block_acc[:, EPOCHS - 1] >= HI)


def endpoint_sets(block_acc: np.ndarray):
    """The two endpoint sets of one large-data ensemble, as (regime, mask) pairs; the regime
    names "retreat" and "recovered" are the keys of every table of Sec. 4."""
    return (("retreat", retreat(block_acc)), ("recovered", recovery(block_acc)))


# ===========================================================================
# Overlap matrices and the normalized residual
# ===========================================================================

def overlap_matrices(xhat: np.ndarray, yspin: np.ndarray, window: slice):
    """m_r, Q^raw, Q^c and W of the replicas xhat [R, inputs, L] on one window of positions.

    The soft spins are taken in float64.  W is the Gram matrix of the truth-aligned outputs
    after each replica's mean over inputs has been removed at every position, averaged over
    inputs and positions; C^prof = Q^c - W.
    """
    v = xhat[:, :, window].astype(np.float64)
    t = yspin[:, window]
    n = v.shape[0]
    al = v * t[None]
    m_r = al.mean(axis=(1, 2))
    fl = v.reshape(n, -1)
    Qraw = fl @ fl.T / fl.shape[1]
    Qc = Qraw - np.outer(m_r, m_r)
    cf = (al - al.mean(axis=1)[:, None, :]).reshape(n, -1)
    W = cf @ cf.T / cf.shape[1]
    return m_r, Qraw, Qc, W


def gap(A: np.ndarray) -> float:
    """gap(A): the mean diagonal minus the mean off-diagonal entry of a replica matrix."""
    n = A.shape[0]
    return float(np.mean(np.diag(A)) - (A.sum() - np.trace(A)) / (n * n - n))


def normalize_by_diagonal(A: np.ndarray) -> np.ndarray:
    """A_rs / sqrt(A_rr A_ss)."""
    return A / np.sqrt(np.outer(np.diag(A), np.diag(A)))


def normalized_residual(xhat: np.ndarray, yspin: np.ndarray, window: slice) -> np.ndarray:
    """R^W on one window: W normalized by its diagonal, the cosine between two residuals."""
    return normalize_by_diagonal(overlap_matrices(xhat, yspin, window)[3])


def obtuse_fraction(R: np.ndarray) -> float:
    """P(R^W < 0): the share of distinct replica pairs whose residuals are obtuse."""
    return float((R[np.triu_indices(R.shape[0], 1)] < 0).mean())


def obtuse_pair_fraction(R: np.ndarray) -> tuple[float, float]:
    """P(R^W < 0) and its leave-one-replica jackknife standard error (Sec. 4.3, Figs. 8, 9).

    The fraction is a property of the pair distribution and needs no equal replica count.
    The error is the spread of the leave-one-out fractions scaled by sqrt(n-1); for n > 120
    the deletions are 120 replicas drawn with seed 3.
    """
    n = R.shape[0]
    p = obtuse_fraction(R)
    deleted = np.arange(n) if n <= 120 else np.random.default_rng(3).choice(n, 120, replace=False)
    lo = np.array([obtuse_fraction(np.delete(np.delete(R, i, 0), i, 1)) for i in deleted])
    return p, float(np.sqrt(n - 1) * lo.std())


# ===========================================================================
# The command line of analysis.py and figures.py
# ===========================================================================

def run_items(items: list, script: str, description: str) -> int:
    """The command line of analysis.py and figures.py over their (function, writes, paper
    location) items.

        NAME ...   run the named items one after the other in this process; the intra-op thread
                   count of torch is set back after every item, so a count pinned by one item
                   does not carry over to the next
        --all      every item in the order of `items`, each as `python3 <script> NAME` in its own
                   process
        --list     name, paper location and output of every item
    """
    by_name = {function.__name__: function for function, _, _ in items}
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="NAME", help="items to run in this process")
    parser.add_argument("--list", action="store_true", help="list the items and stop")
    parser.add_argument("--all", action="store_true",
                        help="run every item in the listed order, each in its own process")
    args = parser.parse_args()
    if args.list:
        for function, writes, location in items:
            print(f"{function.__name__:42s} {location:40s} {writes}")
        return 0
    if args.all:
        if args.names:
            parser.error("--all takes no item names")
        for function, writes, location in items:
            name = function.__name__
            print(f"== {name}   ({location}) -> {writes}", flush=True)
            start = time.time()
            status = subprocess.run([sys.executable, script, name], cwd=ROOT).returncode
            if status:
                print(f"{name} exited with status {status}", file=sys.stderr)
                return status
            print(f"== {name} done in {time.time() - start:.0f} s", flush=True)
        return 0
    if not args.names:
        parser.error("name one or more items, or use --list or --all")
    unknown = [name for name in args.names if name not in by_name]
    if unknown:
        parser.error(f"unknown item {unknown[0]!r}; see --list")
    for name in args.names:
        threads = torch.get_num_threads()
        start = time.time()
        print(f"[{name}]", flush=True)
        by_name[name]()
        torch.set_num_threads(threads)
        print(f"[{name}] done in {time.time() - start:.0f} s", flush=True)
    return 0
