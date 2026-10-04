#!/usr/bin/env python3
"""The analyses of "Replica Fragmentation and Glassy Dynamics in Parity Learning": every table
and array under results/ that a number or a figure of the paper is read from.

One function per analysis.  Each reads data/ (one of them reads the table another item
writes) and writes the files that --list names under results/.  Nothing here trains a
network.  The task, the network, the loaders, the endpoint selection and the overlap
matrices are defined in common.py, whose docstring gives the notation used below.

    python3 analysis.py --list               every item, what it writes, where the paper uses it
    python3 analysis.py NAME [NAME ...]      run the named items in this process
    python3 analysis.py --all                every item in dependency order, each in its own
                                             process (about two hours)

The analyses run with one BLAS thread: the order of floating-point accumulation in a matrix
product depends on the thread count, and with one thread the tables come out the same on
every run.  The thread variables (OMP, MKL, OPENBLAS, VECLIB, NUMEXPR) default to 1 before
NumPy is imported.  Three items set the intra-op thread count of torch themselves, because
their last digits depend on it: fixed_epoch_comparison and retreat_event_contrasts (two),
final_ffn_peak_to_retreat (three).  --all starts every item as a separate process so that
neither these settings nor any other state reaches the next item.

===========================================================================
ENDPOINT CLASSES AND REPLICA SETS (Secs. 3, 4; App. C.1)
===========================================================================
A large-data replica is classified by its 150-epoch held-out A_block curve with HI = 0.8 and
LO = 0.5: never (peak below HI), stable (reaches HI, never below LO afterwards), and, for the
replicas that fall below LO after reaching HI, retreat (below LO at epoch 150), recovery (at
or above HI at epoch 150) or neither.  The retreat and recovery sets of the five large-data
ensembles and the four memorization ensembles are the fourteen replica sets of Sec. 4.

On one replica set and one window of output positions, summarize measures the overlaps of
common.py (m, q_self, q_cross, Delta_N, chi_SG, the gaps of Q^raw, Q^c and W, the share
gap(W)/gap(Q^c)) and four statistics of the average-linkage tree built on d = 1 - R^W:

    R2_tree      1 - sum_pairs (d - d_tree)^2 / sum_pairs (d - mean d)^2, d_tree the
                 cophenetic distance; r_coph is the Pearson correlation of d and d_tree
    p_um         P(delta_um < 0.1): over replica triples with sorted similarities
                 q1 <= q2 <= q3, the fraction with (q2 - q1)/(q3 - q1) < 0.1; an exact
                 ultrametric triangle (two equal smallest similarities) has 0
    I_C          normalized Colless imbalance of the tree
    f_max        largest cluster of the four-cluster cut, as a fraction of the replicas

The tree statistics depend on the number of replicas, so they are averages over
SUBSET_DRAWS = 400 random subsets of SUBSET_SIZE = 47 replicas, the largest count every set
reaches.  Standard errors are closed-form for means of per-replica quantities, Hajek projections
for the pair averages q_cross and chi_SG, and leave-one-replica jackknifes for the gap, the share
and the tree statistics.

===========================================================================
LOSS LANDSCAPE AT FIXED EPOCH AND AT RETREAT (App. A)
===========================================================================
For one network and one fixed diagnostic set, L(theta) is the mean binary cross-entropy over
all inputs and all L output bits.  The diagnostics are its gradient norm |grad L|_2 over all
trainable parameters and the two algebraic extrema lambda_min, lambda_max of its Hessian.
The Hessian is never formed: Hessian-vector products come from a second automatic
differentiation and feed an implicitly restarted Lanczos iteration (ARPACK, relative
tolerance 2e-3, at most 250 iterations).  A stationary saddle needs a small gradient norm and
lambda_min < 0.  At a retreat the contrast is the difference of differences
DD X = (X_retreat - X_last-high) - (X_stable,end - X_stable,start) against a stable interval
of the same seed and lag, so a drift common to both intervals cancels.

===========================================================================
PAPER LOCATION -> FUNCTION
===========================================================================
    Secs. 1, 3; Fig. 1            train_heldout_crossing_epochs
    Sec. 3.3                      nishimori_gap_small_data
    Sec. 3.4; Sec. 4.2            trajectory_ensemble_retreat
    Fig. 5, Sec. 3.5              tail_observables
    Table 3, Figs. 6, 8, 9        replica_set_geometry
    Figs. 8, 9, Sec. 4.3          normalized_residuals
    Fig. 7                        retreat_matrices_l12_first400, retreat_matrices_norm_filter,
                                  retreat_matrices_l16_n2048
    Sec. 4.2                      gf2_identifiability, tree_permutation_references,
                                  retreat_definition_controls, k4_cut_equal_count,
                                  truth_scalar_correlation, evaluation_outside_training_set
    Sec. 4.3                      retreat_obtuse_pair_fraction, split_half_obtuse
    Sec. 5.2                      intervention_pair_outcomes
    App. A.1, A.2                 fixed_epoch_comparison
    App. A.2                      retreat_event_contrasts
    Fig. A1, App. A.3             final_ffn_peak_to_retreat
    App. B                        block_vs_bit_centered_correlation
    Table C1                      endpoint_prevalence
    Table C3                      tail_geometry
"""
from __future__ import annotations

import os

# One BLAS thread unless the caller sets otherwise; this has to precede the NumPy import.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")

import csv
import itertools
import json
import re
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.spatial.distance import squareform

from common import (
    CHECKPOINTED_TRAJECTORIES, CNN_SIZE_SCAN, DATA, EPOCHS, EVAL_SEED, FIXED_EPOCH_REPLICAS, HI,
    LARGE_DATA, LEARNING_RATE_INTERVENTION, LO, MEMORIZATION, RESULTS, RETREAT_EVENTS, ROOT,
    SMALL_DATA_ENSEMBLES, TAIL, TAIL_MEMORIZATION, TRAIN_SEED, TRAJECTORY_ENSEMBLES,
    TRANSFORMER_SIZE_SCAN,
    ChainTransformer, ARCHITECTURE, common_tail, complete_set, crossing_outcome, encode,
    endpoint_sets, ends_high, first_crossing, first_fall, gap, held_out_set, history_crossing,
    load_chain_transformer, load_history, load_l12_n1280_in_storage_order, load_large_data,
    load_memorization, normalized_residual, obtuse_fraction, overlap_matrices, reached, recovery,
    retreat, run_items, sample_set, sample_walls, wall_codes, write_json,
)

# Fitted crossovers N_c of Sec. 3.1, rounded as the paper quotes them.
CROSSOVER_N = {12: 1111, 16: 2600, 20: 9300}


# ===========================================================================
# Secs. 1, 3 and App. B: the fixed-training-set scans of Fig. 1
# ===========================================================================

CROSSING_SCANS = (("cnn", 12, CNN_SIZE_SCAN),
                  ("transformer", 12, TRANSFORMER_SIZE_SCAN[12]),
                  ("transformer", 16, TRANSFORMER_SIZE_SCAN[16]),
                  ("transformer", 20, TRANSFORMER_SIZE_SCAN[20]))


def scan_lags(root: Path, length: int):
    """Number of runs, the lags t_heldout - t_train of the successful runs, and the number of
    successful runs whose training accuracy stays below the threshold."""
    pattern = re.compile(rf"L{length}_N(\d+)_seed(\d+)$")
    runs, lags, train_below = 0, [], 0
    for path in sorted(root.glob(f"L{length}_N*_seed*/metrics.json")):
        if pattern.fullmatch(path.parent.name) is None:
            continue
        history = load_history(path)
        runs += 1
        t_heldout = history_crossing(history, "block_acc")
        if t_heldout is None:
            continue
        t_train = history_crossing(history, "train_block_acc")
        if t_train is None:
            train_below += 1
        else:
            lags.append(t_heldout - t_train)
    return runs, np.asarray(lags), train_below


def train_heldout_crossing_epochs():
    """Lag between the training and the held-out crossing of 0.8 in the four scans of Fig. 1.

    Secs. 1 and 3 state that a successful run fits its training set and generalizes at
    essentially the same epoch.  Every run of a training-set-size scan logs after
    each of its 120 epochs the block accuracy on its own fixed training set and on the held-out
    set.  A run is successful if the held-out block accuracy reaches 0.8 (Sec. 3.1), and then

        t_train = first epoch with training block accuracy >= 0.8
        t_heldout = first epoch with held-out block accuracy >= 0.8
        lag = t_heldout - t_train          (positive: the training set is fitted first)

    Reads   the metrics.json of every run of the CNN scan at L = 12 and of the Transformer
            scans at L = 12, 16, 20 (common.CNN_SIZE_SCAN, common.TRANSFORMER_SIZE_SCAN)
    Writes  results/train_heldout_crossing_epochs.csv, one row per scan: runs, successful runs,
            successful runs whose training accuracy stays below 0.8, median, mean, quartiles
            and extremes of the lag, and the shares of successful runs with lag 0, |lag| <= 1
            and |lag| <= 2.
    """
    rows = []
    for architecture, length, root in CROSSING_SCANS:
        runs, lags, train_below = scan_lags(root, length)
        successful = len(lags) + train_below
        q25, q50, q75 = np.quantile(lags, [0.25, 0.5, 0.75])
        row = dict(architecture=architecture, L=length, n_runs=runs, n_successful=successful,
                   n_train_below_threshold=train_below,
                   lag_median=float(q50), lag_mean=float(lags.mean()),
                   lag_q25=float(q25), lag_q75=float(q75),
                   lag_min=int(lags.min()), lag_max=int(lags.max()),
                   share_lag_0=float((lags == 0).sum() / successful),
                   share_abs_lag_le_1=float((np.abs(lags) <= 1).sum() / successful),
                   share_abs_lag_le_2=float((np.abs(lags) <= 2).sum() / successful))
        rows.append(row)
        print(f"{architecture:11s} L={length}: {successful} of {runs} runs successful;  "
              f"lag median {q50:.0f}, mean {lags.mean():.2f}, quartiles {q25:.0f}..{q75:.0f}, "
              f"range {lags.min()}..{lags.max()};  lag 0 in {row['share_lag_0']:.0%}, "
              f"|lag| <= 1 in {row['share_abs_lag_le_1']:.0%}, "
              f"|lag| <= 2 in {row['share_abs_lag_le_2']:.0%}"
              + (f";  training accuracy below the threshold throughout in {train_below}"
                 if train_below else ""))

    output = RESULTS / "train_heldout_crossing_epochs.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", output.relative_to(ROOT))


def block_vs_bit_centered_correlation():
    """Block accuracy against the independent-position estimate, pooled and centered (App. B).

    Every logged epoch of every run of the L = 12 Transformer training-set-size scan gives one
    pair: the held-out block accuracy Y and the estimate X = (A_bit)^12 that treats the twelve
    positions as independent.  The pooled Pearson correlation of X and Y is large partly
    because both rise with the training-set size N and with the epoch.  To remove that common
    drift, X and Y are centered within each group of checkpoints sharing N and the epoch (the
    runs of one N at one epoch), and the correlation is taken over the concatenated residuals.
    The figure function of Fig. B1 draws the pooled cloud.

    Reads   common.TRANSFORMER_SIZE_SCAN[12]/L12_N<N>_seed<s>/metrics.json
    Writes  results/block_vs_bit_centered_correlation.json: runs, the number of values of N
            and the runs at each, the epoch range, checkpoints, groups, the pooled and the
            centered correlation.
    """
    length = 12
    scan = TRANSFORMER_SIZE_SCAN[length]
    pattern = re.compile(rf"L{length}_N(\d+)_seed(\d+)$")
    runs_at = Counter()
    groups = defaultdict(list)          # (N, epoch) -> [(X, Y) of each run]
    for path in sorted(scan.glob(f"L{length}_N*_seed*/metrics.json")):
        parsed = pattern.fullmatch(path.parent.name)
        if parsed is None:
            continue
        n_train = int(parsed.group(1))
        runs_at[n_train] += 1
        for row in load_history(path):
            groups[(n_train, int(row["epoch"]))].append(
                (float(row["bit_acc"]) ** length, float(row["block_acc"])))

    # The groups are concatenated in insertion order, which fixes the order of the sums.
    pooled = np.concatenate([np.asarray(v) for v in groups.values()])
    centered = np.concatenate([np.asarray(v) - np.asarray(v).mean(axis=0)
                               for v in groups.values()])
    r_pooled = float(np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1])
    r_centered = float(np.corrcoef(centered[:, 0], centered[:, 1])[0, 1])
    epochs = sorted({epoch for _, epoch in groups})

    out = dict(L=length, runs=int(sum(runs_at.values())), training_set_sizes=len(runs_at),
               runs_per_training_set_size={str(n): runs_at[n] for n in sorted(runs_at)},
               epochs=[epochs[0], epochs[-1]], checkpoints=int(len(pooled)),
               groups=len(groups), corr_pooled=r_pooled, corr_centered=r_centered)
    print(f"L={length} Transformer scan: {out['runs']} runs over {out['training_set_sizes']} "
          f"values of N, epochs {epochs[0]}-{epochs[-1]}, {out['checkpoints']} checkpoints")
    print("  runs per N: " + ", ".join(f"{n}: {runs_at[n]}" for n in sorted(runs_at)))
    print(f"  pooled correlation of A_block and (A_bit)^{length}: {r_pooled:.4f}")
    print(f"  centered within each (N, epoch) group, {len(groups)} groups: {r_centered:.4f}")

    output = RESULTS / "block_vs_bit_centered_correlation.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(output, out)
    print("wrote", output.relative_to(ROOT))


# ===========================================================================
# Sec. 3: shared-training-set ensembles followed in time
# ===========================================================================

def nishimori_gap_small_data():
    """Persistence of the Nishimori gap Delta_N = q_self - m in the two N = 64 runs of Fig. 3.

    The right column of Fig. 3 follows two independently initialized Transformers trained on
    one shared training set of 64 strings at L = 12, evaluated after every epoch on the
    complete set of 2048 inputs.  Sec. 3.3 quotes how persistent a positive gap is: over the
    300 replica-epoch observations of epochs 201-350 it counts those with Delta_N > 0.05, and at
    epoch 350 both gaps are large while both block accuracies are below 0.1.  The largest block
    accuracy over epochs 1-350 is reported as well, because the value at epoch 350 alone does
    not say whether the accuracy stays below 0.1 on the way.

    Reads   data/trajectory_ensembles/L12_N64_overlaps/per_seed_epoch_metrics.csv
    Writes  results/nishimori_gap_small_data.json
    """
    metrics = TRAJECTORY_ENSEMBLES / "L12_N64_overlaps" / "per_seed_epoch_metrics.csv"
    seeds = (0, 6)
    first_epoch, last_epoch = 201, 350
    threshold = 0.05

    rows = {}
    with open(metrics, newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["seed"]), int(row["epoch"]))] = (float(row["delta_N"]),
                                                           float(row["block_acc"]))

    window = [rows[(s, e)][0] for s in seeds for e in range(first_epoch, last_epoch + 1)]
    above = sum(d > threshold for d in window)
    out = dict(L=12, N_train=64, seeds=list(seeds),
               window_epochs=[first_epoch, last_epoch], gap_threshold=threshold,
               observations=len(window), observations_above_threshold=above,
               fraction_above_threshold=above / len(window), runs=[])
    print(f"N=64, seeds {seeds}, epochs {first_epoch}-{last_epoch}: {len(window)} observations, "
          f"{above} with Delta_N > {threshold} ({above / len(window):.1%})")
    for s in seeds:
        delta, block = rows[(s, last_epoch)]
        peak = max(rows[(s, e)][1] for e in range(1, last_epoch + 1))
        in_window = sum(rows[(s, e)][0] > threshold for e in range(first_epoch, last_epoch + 1))
        out["runs"].append(dict(seed=s, observations_above_threshold=in_window,
                                delta_N_at_last_epoch=delta, block_acc_at_last_epoch=block,
                                max_block_acc_through_last_epoch=peak))
        print(f"  seed {s}: {in_window} of {last_epoch - first_epoch + 1} above; at epoch "
              f"{last_epoch} Delta_N = {delta:.3f}, A_block = {block:.3f}; "
              f"largest A_block over epochs 1-{last_epoch} = {peak:.3f}")

    output = RESULTS / "nishimori_gap_small_data.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(output, out)
    print("wrote", output.relative_to(ROOT))


def first_lag_at_or_below(curve, level):
    """The smallest lag at which the curve is at or below the level, or None."""
    return next((lag for lag, value in enumerate(curve) if value <= level), None)


def trajectory_ensemble_retreat():
    """Reach and retreat counts of the 16-replica trajectory ensemble at (12, 1280) (Sec. 3.4),
    and the two-time and cross-replica tail overlap of replicas in retreat (Sec. 4.2).

    The ensemble is the second row of Table 2: 16 replicas on the shared training set, with
    the soft spins of every replica stored after each of 150 epochs on 4096 evaluation draws;
    A_block at an epoch is measured on those draws.

    Counts.  A replica reaches if A_block >= 0.8 at some epoch, and retreats if A_block falls
    below 0.5 at a later epoch; the first such epoch is its retreat event (the event on which
    Fig. 4 aligns the same replicas).

    Overlaps.  A replica is in retreat at epoch t if it reached 0.8 before t and A_block(t) <
    0.5, the selection of Table 3 with the observation epoch moved to t.  On the tail
    j = 8, ..., 11 the overlap of two output vectors is

        C(u, v) = E_{b, j in tail}[u_j(b) v_j(b)],      c(u, v) = C(u, v) / sqrt(C(u, u) C(v, v)),

    C in the units of q_self and q_cross, c the cosine of the angle between the two outputs,
    which removes the self-overlaps (they differ between epochs and replicas) so that c = 1
    means the same function up to scale.  The two-time overlap pairs the outputs of one
    replica at t and t + lag, averaged over every replica-epoch (r, t) in retreat with t + lag
    inside the 150 epochs, once without condition on t + lag and once with the replica in
    retreat at t + lag as well.  The cross overlap pairs two different replicas in retreat at
    one epoch, pooled over all such pairs and epochs; its normalized form is the overlap
    between different replicas quoted in Sec. 4.2.  For each two-time curve the first lag at or
    below the cross level of the same form is reported.

    Reads   data/trajectory_ensembles/L12_N1280_soft_outputs/xhat_history.npz
    Writes  results/trajectory_ensemble_retreat.json
    """
    source = TRAJECTORY_ENSEMBLES / "L12_N1280_soft_outputs" / "xhat_history.npz"
    L = 12
    max_lag = 50
    print_lags = (1, 5, 10, 15, 20, 25, 30, 40, 50)
    with np.load(source) as data:
        xhat = data["xhat"]
        yspin = data["yspin"]
        seeds = data["seeds"].astype(int)
    block = (np.sign(xhat) == yspin[None, None]).all(axis=-1).mean(axis=-1)
    R, E = block.shape

    # Reaching, the retreat event, and the replica-epochs in retreat.
    in_retreat = np.zeros((R, E), bool)
    first_reach, event = {}, {}
    for r in range(R):
        first = first_crossing(block[r])
        if first is None:
            continue
        first_reach[r] = first
        in_retreat[r, first + 1:] = block[r, first + 1:] < LO
        fall = first_fall(block[r])
        if fall is not None:
            event[r] = fall
    print(f"{R} replicas, {E} epochs, {xhat.shape[2]} evaluation draws")
    print(f"  reach A_block >= {HI}: {len(first_reach)} replicas;  "
          f"later below {LO}: {len(event)} replicas")
    print("  retreat events (seed: epoch): "
          + ", ".join(f"{seeds[r]}: {event[r] + 1}" for r in sorted(event)))
    print(f"  replica-epochs in retreat: {int(in_retreat.sum())}")

    # Tail outputs and their epoch-by-epoch overlap matrix, one replica at a time.
    tail = xhat[:, :, :, TAIL[L]].astype(np.float64).reshape(R, E, -1)
    size = tail.shape[2]
    C = np.stack([tail[r] @ tail[r].T for r in range(R)]) / size
    norm = np.sqrt(np.einsum("rtt->rt", C))
    c = C / (norm[:, :, None] * norm[:, None, :])

    def two_time(matrix, both):
        """Mean of matrix[r, t, t+lag] over the replica-epochs (r, t) in retreat, per lag."""
        means, counts = [], []
        for lag in range(max_lag + 1):
            ok = in_retreat[:, :E - lag].copy()
            if both:
                ok &= in_retreat[:, lag:]
            r, t = np.nonzero(ok)
            means.append(float(matrix[r, t, t + lag].mean()))
            counts.append(int(ok.sum()))
        return means, counts

    # Pairs of different replicas that are in retreat at the same epoch.
    cross_raw, cross_normalized = [], []
    for t in range(E):
        idx = np.flatnonzero(in_retreat[:, t])
        if idx.size < 2:
            continue
        g = tail[idx, t] @ tail[idx, t].T / size
        iu = np.triu_indices(idx.size, 1)
        cross_raw.append(g[iu])
        cross_normalized.append((g / np.sqrt(np.outer(np.diag(g), np.diag(g))))[iu])
    cross_raw = np.concatenate(cross_raw)
    cross_normalized = np.concatenate(cross_normalized)
    level_raw, level = float(cross_raw.mean()), float(cross_normalized.mean())
    print(f"  cross-replica tail overlap over {cross_raw.size} same-epoch pairs in retreat: "
          f"normalized {level:.3f}, raw {level_raw:.3f}")

    out = dict(L=L, N_train=1280, replicas=R, epochs=E, evaluation_draws=int(xhat.shape[2]),
               high=HI, low=LO, tail_positions=[TAIL[L].start, TAIL[L].stop - 1],
               reached=len(first_reach), retreated=len(event),
               retreat_event_epoch={str(seeds[r]): event[r] + 1 for r in sorted(event)},
               replica_epochs_in_retreat=int(in_retreat.sum()),
               cross_replica=dict(pairs=int(cross_raw.size), normalized=level, raw=level_raw),
               lags=list(range(max_lag + 1)), two_time={})
    for key, both, text in (("reference_in_retreat", False, "in retreat at t"),
                            ("both_in_retreat", True, "in retreat at t and at t + lag")):
        normalized, counts = two_time(c, both)
        raw, _ = two_time(C, both)
        first = dict(normalized=first_lag_at_or_below(normalized, level),
                     raw=first_lag_at_or_below(raw, level_raw))
        out["two_time"][key] = dict(normalized=normalized, raw=raw, pairs=counts,
                                    first_lag_at_cross_level=first)
        print(f"  two-time tail overlap, replica {text}:")
        print("    lag        " + "".join(f"{lag:7d}" for lag in print_lags))
        print("    normalized " + "".join(f"{normalized[lag]:7.3f}" for lag in print_lags))
        print("    raw        " + "".join(f"{raw[lag]:7.3f}" for lag in print_lags))
        print("    pairs      " + "".join(f"{counts[lag]:7d}" for lag in print_lags))
        print(f"    first lag at or below the cross-replica level: normalized "
              f"{first['normalized']}, raw {first['raw']}")

    output = RESULTS / "trajectory_ensemble_retreat.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(output, out)
    print("wrote", output.relative_to(ROOT))


# ---------------------------------------------------------------- Fig. 5: tail observables

TAIL_SCAN_SEEDS = tuple(range(6))   # the replicas of each training-set size


def tail_scan_replica(run_dir: Path) -> ChainTransformer:
    """One replica of the L = 12 scan over N, built from the configuration it records.

    Raises unless the run records L = 12, the training seed TRAIN_SEED and at least 4000
    steps.
    """
    with (run_dir / "metrics.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload["config"]
    if (int(config["L"]) != 12 or int(payload["train_seed"]) != TRAIN_SEED
            or int(payload["steps"]) < 4000):
        raise ValueError(f"incompatible shared-disorder run: {run_dir}")
    model = ChainTransformer(L=12, channels=int(config["channels"]), depth=int(config["depth"]),
                             num_heads=int(config["heads"]), ffn_dim=int(config["ffn"]),
                             dropout=0.0)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def tail_scan_predictions(run_dirs: list[Path], inputs: torch.Tensor):
    """Soft spins tanh(logit/2) and hard bits (logit > 0) of every replica, [R, inputs, L]."""
    soft_spins, hard_predictions = [], []
    for run_dir in run_dirs:
        logits = tail_scan_replica(run_dir)(inputs).numpy()
        soft_spins.append(np.tanh(logits / 2.0))
        hard_predictions.append((logits > 0.0).astype(float))
    return np.stack(soft_spins), np.stack(hard_predictions)


def position_overlaps(soft_spins: np.ndarray, truth_pm: np.ndarray):
    """q_self(j), m(j) and q_cross(j) per output position, averaged over replicas and inputs
    (replica pairs r != s for q_cross)."""
    replicas, n_inputs, length = soft_spins.shape
    q_self = np.mean(soft_spins**2, axis=(0, 1))
    m_truth = np.mean(soft_spins * truth_pm[None, :, :], axis=(0, 1))
    q_cross = np.empty(length)
    for j in range(length):
        values = soft_spins[:, :, j]
        gram = values @ values.T / n_inputs
        q_cross[j] = (gram.sum() - np.trace(gram)) / (replicas * (replicas - 1))
    return q_self, m_truth, q_cross


def tail_summary(test_means, test_hard, truth_pm, held_out_targets,
                 train_means, train_hard, train_targets) -> dict[str, float]:
    """The tail averages (j = 8, ..., 11) of one row of the Fig. 5 table."""
    q_self, m_truth, q_cross = position_overlaps(test_means, truth_pm)
    train_q_self = np.mean(train_means**2, axis=(0, 1))
    train_accuracy = np.mean(train_hard == train_targets.numpy()[None, :, :], axis=(0, 1))
    test_accuracy = np.mean(test_hard == held_out_targets.numpy()[None, :, :], axis=(0, 1))
    tail = TAIL[12]
    return {
        "tail_q_self": float(np.mean(q_self[tail])),
        "tail_m_truth": float(np.mean(m_truth[tail])),
        "tail_q_cross": float(np.mean(q_cross[tail])),
        "tail_chi_sg": float(np.mean((q_self - q_cross)[tail])),
        "tail_train_q_self": float(np.mean(train_q_self[tail])),
        "tail_train_acc": float(np.mean(train_accuracy[tail])),
        "tail_test_acc": float(np.mean(test_accuracy[tail])),
    }


def jackknife_se(values: list[float]) -> float:
    """Leave-one-out jackknife error sqrt((n-1)/n sum (x_i - mean)^2) of n leave-one-out values."""
    array = np.asarray(values, dtype=float)
    n = len(array)
    center = array.mean()
    return float(np.sqrt((n - 1) / n * np.sum((array - center) ** 2)))


def tail_observables():
    """Tail observables of the L = 12 scan over the training-set size N (Fig. 5, Sec. 3.5).

    At each N six replicas (seeds 0-5) share the training set of N draws with TRAIN_SEED and
    differ only in initialization and minibatch order.  The held-out observables are exact
    averages over the held-out set (the 2048 inputs minus the distinct training inputs).  With
    the soft spin tanh(logit/2) the table gives, averaged over the tail j = 8, ..., 11,

        q_self, m (m_truth), q_cross, chi_SG = q_self - q_cross     on the held-out inputs
        q_self on the training inputs, and the bit accuracy on both input sets

    with leave-one-replica jackknife errors <key>_se, the number of distinct training inputs
    N_unique and the number of held-out inputs.  The sizes N are those whose directory holds
    all six replicas with stored weights.

    Reads   data/small_data_ensembles/L12_N<N>/seed<s>/{metrics.json, model.pt}
    Writes  results/tail_observables.csv, one row per N
    """
    pattern = re.compile(r"L12_N(\d+)/seed(\d+)")
    available: dict[int, set[int]] = {}
    for metrics_path in SMALL_DATA_ENSEMBLES.glob("L12_N*/seed*/metrics.json"):
        run_dir = metrics_path.parent
        parsed = pattern.fullmatch(f"{run_dir.parent.name}/{run_dir.name}")
        if parsed is None or not (run_dir / "model.pt").exists():
            continue
        n, seed = (int(value) for value in parsed.groups())
        available.setdefault(n, set()).add(seed)
    ns = sorted(n for n, found in available.items() if set(TAIL_SCAN_SEEDS).issubset(found))
    if not ns:
        raise SystemExit("no complete multi-replica N points found")

    rows = []
    for n in ns:
        run_dirs = [SMALL_DATA_ENSEMBLES / f"L12_N{n}" / f"seed{seed}" for seed in TAIL_SCAN_SEEDS]
        train_inputs, train_targets = sample_set(n, 12, TRAIN_SEED)
        n_unique = len(torch.unique(wall_codes(train_inputs)))
        held_out_inputs, held_out_targets = held_out_set(12, n)
        test_means, test_hard = tail_scan_predictions(run_dirs, held_out_inputs)
        truth_pm = (2.0 * held_out_targets - 1.0).numpy()
        train_means, train_hard = tail_scan_predictions(run_dirs, train_inputs)
        arrays = (test_means, test_hard, truth_pm, held_out_targets,
                  train_means, train_hard, train_targets)
        summary = tail_summary(*arrays)
        replicas = test_means.shape[0]
        leave_one = []
        for idx in range(replicas):
            keep = np.arange(replicas) != idx
            leave_one.append(tail_summary(test_means[keep], test_hard[keep], truth_pm,
                                          held_out_targets, train_means[keep], train_hard[keep],
                                          train_targets))
        errors = {f"{key}_se": jackknife_se([row[key] for row in leave_one])
                  for key in leave_one[0]}
        rows.append({"N": n, "N_unique": n_unique, "held_out_inputs": len(held_out_inputs),
                     "replicas": len(TAIL_SCAN_SEEDS), **summary, **errors})

    output = RESULTS / "tail_observables.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"N={row['N']:4d} S={row['replicas']}: "
              f"qself={row['tail_q_self']:.3f}, m={row['tail_m_truth']:.3f}, "
              f"qcross={row['tail_q_cross']:.3f}, chi={row['tail_chi_sg']:.3f}, "
              f"train_acc={row['tail_train_acc']:.3f}, test_acc={row['tail_test_acc']:.3f}")
    print(output)


# ===========================================================================
# Sec. 4: the fourteen replica sets
# ===========================================================================

SUBSET_SIZE, SUBSET_DRAWS = 47, 400   # replicas per random subset, subsets of the tree statistics


def tree_distance(r: np.ndarray) -> np.ndarray:
    """Condensed distances d = max(1 - r, 0) of a replica similarity matrix r, with zero
    diagonal and symmetrized, in the form scipy's linkage takes."""
    d = np.maximum(1.0 - r, 0.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2
    return squareform(d, checks=False)


def tree_fit(dc: np.ndarray, co: np.ndarray) -> float:
    """R^2_tree of the pair distances dc against the cophenetic distances co of their tree."""
    return 1.0 - float(np.sum((dc - co) ** 2) / np.sum((dc - dc.mean()) ** 2))


def near_ultrametric_fraction(r: np.ndarray, tri: np.ndarray) -> float:
    """P(delta_um < 0.1) of the similarity matrix r over the replica triples `tri`.

    With q1 <= q2 <= q3 the three sorted pair similarities of a triple,
    delta_um = (q2 - q1)/(q3 - q1); triples with q3 = q1 are left out.
    """
    q = np.sort(np.stack([r[tri[:, 0], tri[:, 1]], r[tri[:, 0], tri[:, 2]],
                          r[tri[:, 1], tri[:, 2]]], axis=1), axis=1)
    den = q[:, 2] - q[:, 0]
    ok = den > 0
    return float((((q[ok, 1] - q[ok, 0]) / den[ok]) < 0.1).mean())


def cluster_sizes(Z: np.ndarray, k: int) -> list[int]:
    """Sizes of the clusters of the k-cluster cut of the tree Z, largest first."""
    lab = fcluster(Z, t=k, criterion="maxclust")
    return sorted((int((lab == c).sum()) for c in set(lab)), reverse=True)


def colless_index(Z: np.ndarray, m: int) -> float:
    """Normalized Colless imbalance of a tree over m leaves: 2 sum_nodes |n_left - n_right| /
    ((m-1)(m-2)), 0 for a balanced tree and 1 for a caterpillar."""
    sz = lambda q: 1 if q < m else int(Z[int(q) - m, 3])
    return 2.0 * sum(abs(sz(int(Z[i, 0])) - sz(int(Z[i, 1]))) for i in range(m - 1)) \
        / ((m - 1) * (m - 2))


def subset_tree_fit(R: np.ndarray, n: int, m: int, draws: int):
    """Mean R^2_tree and P(delta_um < 0.1) of R over `draws` random subsets of size m (seed 23)."""
    rg = np.random.default_rng(23)
    tri = np.array(list(itertools.combinations(range(m), 3)))
    r2s, pus = [], []
    for _ in range(draws):
        k = rg.choice(n, m, replace=False)
        r = R[np.ix_(k, k)]
        dc = tree_distance(r)
        Z = linkage(dc, method="average")
        _, co = cophenet(Z, dc)
        r2s.append(tree_fit(dc, co))
        pus.append(near_ultrametric_fraction(r, tri))
    return float(np.mean(r2s)), float(np.mean(pus))


def summarize(xhat: np.ndarray, yspin: np.ndarray, window: slice,
              subset_size: int = SUBSET_SIZE, subset_draws: int = SUBSET_DRAWS) -> dict:
    """Gaps, overlaps and tree statistics of one replica set on one window (Table 3).

    Returns a dict with
      n                          number of replicas
      gap_raw, gap_Qc, gap_W     self-cross gaps of Q^raw, Q^c and W; share = gap_W / gap_Qc
      m, q_self, q_cross         the raw overlaps; Delta_N = q_self - m, chi_SG = q_self - q_cross
      I_C, f_max, r2_tree, p_um  Colless index, largest-cluster fraction of the K = 4 cut, tree
                                 fit and near-ultrametric fraction, each the mean over
                                 `subset_draws` random subsets of `subset_size` replicas
                                 (generator seed 23); *_sd
                                 is the standard deviation over those draws
      *_se                       standard errors, see the comments below
      trim5_n, trim5_k4          replica count and K = 4 cut of the whole set after the
                                 residual-norm filter of Fig. 7
    """
    m_r, Qraw, Qc, W = overlap_matrices(xhat, yspin, window)
    v = xhat[:, :, window].astype(np.float64)
    t = yspin[:, window]
    n = v.shape[0]
    rw = W / np.sqrt(np.outer(np.diag(W), np.diag(W)))

    # The illustrative K = 4 cuts of Sec. 4.2 use the filter of the retreat tree of Fig. 7:
    # retain floor(0.95 n) replicas with the largest residual norms W_rr, then build the tree.
    n_filtered = int(0.95 * n)
    keep_filtered = np.sort(np.argsort(np.diag(W))[n - n_filtered:])
    rw_filtered = rw[np.ix_(keep_filtered, keep_filtered)]
    filtered_k4 = cluster_sizes(linkage(tree_distance(rw_filtered), method="average"), 4)

    # The four tree statistics on random subsets of `subset_size` replicas.  All four are carried
    # at the same subset size, so every geometry column of a figure that spans the three regimes
    # rests on one replica count.
    rng = np.random.default_rng(23)
    ics, fms, r2s, pus = [], [], [], []
    tri = np.array(list(itertools.combinations(range(subset_size), 3)))
    for _ in range(subset_draws):
        k = rng.choice(n, subset_size, replace=False)
        r = rw[np.ix_(k, k)]
        dc = tree_distance(r)
        Z = linkage(dc, method="average")
        ics.append(colless_index(Z, subset_size))
        lab = fcluster(Z, t=4, criterion="maxclust")
        fms.append(max(int((lab == c).sum()) for c in set(lab)) / subset_size)
        _, co = cophenet(Z, dc)
        r2s.append(tree_fit(dc, co))
        pus.append(near_ultrametric_fraction(r, tri))

    # Leave-one-replica jackknife on the tree fit and the near-ultrametric fraction, so that
    # every error bar of Fig. 8 is a standard error of the same kind.  The draw size is
    # min(subset_size, n-1): deleting a replica leaves n-1, which drops below `subset_size` only
    # for the set whose replica count equals it.  The same subset pattern is used for every
    # deletion, so the Monte-Carlo noise of the inner draws largely cancels in the spread.  At
    # most 60 deletions are evaluated.
    m_eff = min(subset_size, n - 1)
    deleted = np.arange(n) if n <= 60 else np.random.default_rng(7).choice(n, 60, replace=False)
    jk = np.array([subset_tree_fit(np.delete(np.delete(rw, i, 0), i, 1), n - 1, m_eff,
                                   subset_draws)
                   for i in deleted])
    r2_tree_se = float(np.sqrt(n - 1) * jk[:, 0].std())
    p_um_se = float(np.sqrt(n - 1) * jk[:, 1].std())

    iu = np.triu_indices(n, 1)
    # Q^raw = E[u_r u_s] because the truth spins square to one.  The raw gap is carried so that
    # the three terms of the decomposition can be quoted on the same window for every regime:
    # gap(Q^raw) = gap(m m^T) + gap(C^prof) + gap(W).
    gap_raw = gap(Qraw)
    q_self, q_cross, m = float(np.diag(Qraw).mean()), float(Qraw[iu].mean()), float(m_r.mean())
    # Closed-form standard errors.  m, q_self and Delta_N are means of per-replica quantities,
    # so SE = sd/sqrt(n).  q_cross and chi_SG are degree-2 U-statistics whose SE follows the
    # Hajek projection on the per-replica row means.
    q_self_r = np.diag(Qraw)
    offQ = Qraw.copy()
    np.fill_diagonal(offQ, 0.0)
    qbar = offQ.sum(1) / (n - 1)
    se = lambda u: float(np.std(u, ddof=1) / np.sqrt(n))

    # Jackknife standard errors on the connected gap and the residual share.  Both are pair
    # averages, so a replica is dropped in turn and the spread of the leave-one-out values is
    # scaled by sqrt(n-1); at most 60 deletions (seed 3) are evaluated.
    def gap_and_share(u):
        _, _, Qc_u, W_u = overlap_matrices(u, t, slice(None))
        return gap(Qc_u), gap(W_u) / gap(Qc_u)
    idx = np.arange(n) if n <= 60 else np.random.default_rng(3).choice(n, 60, replace=False)
    lo = np.array([gap_and_share(np.delete(v, i, 0)) for i in idx])
    gap_Qc_se = np.sqrt(n - 1) * lo[:, 0].std()
    share_se = np.sqrt(n - 1) * lo[:, 1].std()

    return dict(n=n, gap_raw=gap_raw, gap_Qc=gap(Qc), gap_W=gap(W), share=gap(W) / gap(Qc),
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


def replica_set_geometry():
    """Gaps, overlaps and tree statistics of the fourteen replica sets (Table 3; Figs.
    6, 8, 9; Secs. 4.2, 4.3).

    Every set is measured by summarize on three windows: all positions, the tabulated tail of
    Table 3 ("tail") and the common tail j >= 2L/3 ("tail_u").  Every comparison across
    regimes uses all positions (Sec. 4.1), and Sec. 4.3 repeats it on the common tail because
    the tabulated tails are not one rule.  The paper uses, on all positions and under
    "tail_u": n, gap_raw, gap_Qc, gap_W, share, m, q_self, q_cross, Delta_N, chi_SG, r2_tree and
    p_um with their *_se, and p_um_sd; under "tail" the gap decomposition of the (12,64)
    memorization set, whose tabulated tail is the common tail; trim5_n and trim5_k4 for the
    (16,2048) retreat set.  I_C, f_max, their *_sd and r2_tree_sd are not quoted.

    Reads   the memorization networks (common.load_memorization) and the large-data
            ensembles (common.load_large_data)
    Writes  results/replica_set_geometry.json: {"matched_n", "ndraw", "sets"} (the subset size
            and the number of subsets of the tree statistics, and the sets); sets maps
            "<regime>|(L,N)" to the all-position measurement plus L, N, regime, and the same
            measurement on the two tails under "tail" and "tail_u".
    About twenty minutes.
    """
    def all_windows(v, ys, L, tail):
        r = dict(summarize(v, ys, slice(0, L)), L=L)
        r["tail"] = summarize(v, ys, tail)
        r["tail_u"] = summarize(v, ys, common_tail(L))
        return r

    out = {}
    for lab, L, N, interpolating_only in MEMORIZATION:
        xh, ys = load_memorization(L, N, interpolating_only)
        out[f"memorization|{lab}"] = dict(all_windows(xh, ys, L, TAIL_MEMORIZATION[(L, N)]),
                                          N=N, regime="memorization")
        print("memorization", lab, out[f"memorization|{lab}"]["n"], flush=True)
    for lab, L, N in LARGE_DATA:
        s, a, x, ys = load_large_data(L, N)
        for reg, k in endpoint_sets(a):
            out[f"{reg}|{lab}"] = dict(all_windows(x[k], ys, L, TAIL[L]), N=N, regime=reg)
            print(reg, lab, out[f"{reg}|{lab}"]["n"], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "replica_set_geometry.json"
    write_json(path, dict(matched_n=SUBSET_SIZE, ndraw=SUBSET_DRAWS, sets=out))
    print("wrote", path.relative_to(ROOT))


def normalized_residuals():
    """The normalized residual matrices R^W of the fourteen replica sets (Figs. 8, 9; Sec. 4.3).

    The obtuse-pair fraction and the pair distributions of Figs. 8 and 9 are read off these
    matrices, so they are stored once, for every set and for the two windows the figures use:
    "full" (all positions) and "tail_u" (the common tail j >= 2L/3).

    Reads   the memorization networks and the large-data ensembles
    Writes  results/normalized_residuals.npz, one square float64 matrix per key
            "<regime>|(L,N)|<window>", e.g. "retreat|(20,24576)|tail_u"; rows and columns are
            the selected replicas in ascending seed order.
    Several minutes, most of them spent evaluating the memorization networks.
    """
    def windows(L):
        return (("full", slice(0, L)), ("tail_u", common_tail(L)))

    out = {}
    for lab, L, N, interpolating_only in MEMORIZATION:
        xh, ys = load_memorization(L, N, interpolating_only)
        for name, sl in windows(L):
            out[f"memorization|{lab}|{name}"] = normalized_residual(xh, ys, sl)
        print("memorization", lab, xh.shape[0], flush=True)
    for lab, L, N in LARGE_DATA:
        s, a, x, ys = load_large_data(L, N)
        for reg, k in endpoint_sets(a):
            for name, sl in windows(L):
                out[f"{reg}|{lab}|{name}"] = normalized_residual(x[k], ys, sl)
            print(reg, lab, int(k.sum()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "normalized_residuals.npz"
    np.savez_compressed(path, **out)
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------- Fig. 7: the retreat trees

RETREAT_MATRICES = RESULTS / "retreat_matrices"


def retreat_matrices_l12_first400():
    """Q^c, C^prof and W of the retreat replicas among the 400 lowest seeds at (12, 1280) (Fig. 7).

    The (12, 1280) replicas are read in storage order (common.load_l12_n1280_in_storage_order),
    the 400 smallest seeds are kept, and inside that subset the retreat replicas are selected:
    peak A_block >= 0.8 over the 150 saved epochs and A_block < 0.5 at the last one.  The cut is
    on rank in seed order.  On all positions the item computes m_r, the connected overlap Q^c,
    the profile covariance C^prof = E_j[mu_rj mu_sj] - m_r m_s (from the profiles, not as
    Q^c - W) and the residual W, in the replica order of the subset, and then orders the
    replicas by m_r.

    Reads   data/large_data_ensembles/L12_N1280/seeds*.npz
    Writes  results/retreat_matrices/L12_N1280_first400.npz with
                plotted_index            0 .. R-1
                retreat_set_index        position of each replica among the retreat replicas
                                         of the subset before the ordering by m_r
                storage_index            position of each replica in the storage order
                seed, m_r                seed and truth overlap of each replica
                Qc, Cprof, W             the three matrices, replicas ordered by m_r
            results/retreat_matrices/L12_N1280_first400.json: how many replicas of the subset
            retreat, end at or above 0.8, never reach 0.8, or end between the thresholds.
    """
    first_seeds = 400
    out = RETREAT_MATRICES / "L12_N1280_first400.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    xhat, block, seeds, yspin = load_l12_n1280_in_storage_order()
    seed_order = np.argsort(seeds)
    selected = seed_order[:first_seeds]
    selected_mask = np.zeros(len(seeds), dtype=bool)
    selected_mask[selected] = True

    learned = reached(block)
    retreat_index = np.flatnonzero(selected_mask & retreat(block))
    retreat_xhat = xhat[retreat_index]
    retreat_seeds = seeds[retreat_index]

    # The matrices are built in the storage order of the selected replicas.
    m_r, _, Qc, W = overlap_matrices(retreat_xhat, yspin, slice(0, xhat.shape[2]))
    aligned = retreat_xhat.astype(np.float64) * yspin[None]
    profile = aligned.mean(axis=1)
    Cprof = (profile @ profile.T) / profile.shape[1] - np.outer(m_r, m_r)

    order = np.argsort(m_r)
    np.savez_compressed(
        out,
        plotted_index=np.arange(len(order), dtype=np.int64),
        retreat_set_index=order.astype(np.int64),
        storage_index=retreat_index[order].astype(np.int64),
        seed=np.asarray(retreat_seeds, dtype=np.int64)[order],
        m_r=m_r[order],
        Qc=Qc[np.ix_(order, order)],
        Cprof=Cprof[np.ix_(order, order)],
        W=W[np.ix_(order, order)],
    )

    final = block[:, EPOCHS - 1]
    summary = {
        "selected_replicas": first_seeds,
        "replicas": int(len(seeds)),
        "selected_seed_min": int(seeds[selected].min()),
        "selected_seed_max": int(seeds[selected].max()),
        "selected_retreat_replicas": int(len(retreat_index)),
        "selected_ends_high_replicas": int(np.count_nonzero(selected_mask & ends_high(block))),
        "selected_never_replicas": int(np.count_nonzero(selected_mask & ~learned)),
        "selected_other_replicas": int(np.count_nonzero(selected_mask & learned & (final >= LO)
                                                        & (final < HI))),
    }
    with out.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def json_clean(value: Any) -> Any:
    """NumPy arrays and scalars inside nested dicts and lists as plain Python values."""
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    return value


def retreat_matrices_norm_filter():
    """The bottom-5% residual-norm filter of Fig. 7 applied to the (12, 1280) retreat matrices.

    The filter takes the diagonal of W, finds its 5% quantile and keeps the replicas whose W_rr
    lies strictly above it.  The dropped replicas have the smallest input-dependent
    fluctuation; R^W_rs = W_rs / sqrt(W_rr W_ss) divides their rows by the smallest norms, so
    they are the noisiest input to the clustering.  Of the 84 retreat replicas 79 remain.
    Every per-replica vector of the input is sliced and every replica-by-replica matrix is
    indexed on both axes to the kept replicas.

    Reads   results/retreat_matrices/L12_N1280_first400.npz (retreat_matrices_l12_first400)
    Writes  results/retreat_matrices/L12_N1280_first400_norm_filtered.npz: the arrays of the input
            restricted to the kept replicas, plus trim_keep_index and trim_drop_index
            (positions in the input), trim_diag_threshold (the quantile value of W_rr) and
            trim_diag_quantile (0.05)
            results/retreat_matrices/L12_N1280_first400_norm_filtered.json: the input file name,
            the threshold, the counts, and the indices, seeds and W_rr of the dropped replicas
    """
    source = RETREAT_MATRICES / "L12_N1280_first400.npz"
    out = RETREAT_MATRICES / "L12_N1280_first400_norm_filtered.npz"
    quantile = 0.05
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(source)
    matrix = np.asarray(data["W"], dtype=np.float64)
    n = matrix.shape[0]
    diagonal = np.diag(matrix)
    threshold = float(np.quantile(diagonal, quantile))
    keep = diagonal > threshold
    keep_index = np.flatnonzero(keep)
    drop_index = np.flatnonzero(~keep)

    payload: dict[str, np.ndarray] = {}
    for key in data.files:
        value = np.asarray(data[key])
        if value.ndim == 1 and value.shape[0] == n:
            payload[key] = value[keep_index]
        elif value.ndim == 2 and value.shape == (n, n):
            payload[key] = value[np.ix_(keep_index, keep_index)]
        else:
            payload[key] = value
    payload["trim_keep_index"] = keep_index.astype(np.int64)
    payload["trim_drop_index"] = drop_index.astype(np.int64)
    payload["trim_diag_threshold"] = np.asarray(threshold, dtype=np.float64)
    payload["trim_diag_quantile"] = np.asarray(quantile, dtype=np.float64)
    np.savez_compressed(out, **payload)

    seeds = np.asarray(data["seed"], dtype=np.int64)
    summary = {
        "input": source.name,
        "matrix": "W",
        "quantile": quantile,
        "threshold": threshold,
        "n_input": int(n),
        "n_kept": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "dropped_indices": drop_index,
        "dropped_diagonal": diagonal[drop_index],
        "dropped_seeds": seeds[drop_index],
    }
    with out.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(json_clean(summary), handle, indent=2)
    print(json.dumps(json_clean(summary), indent=2))


def retreat_matrices_l16_n2048():
    """The overlap matrices of the (16, 2048) retreat set, drawn as the lower tree of Fig. 7.

    On all positions: m_r, Q^c, W, C^prof = Q^c - W and the diagonal-normalized R^Qc and R^W,
    replicas in ascending seed order.  Prints the share gap(W)/gap(Q^c) of the self-cross gap
    that survives the subtraction of the mean profiles.

    Reads   data/large_data_ensembles/L16_N2048/seeds*.npz
    Writes  results/retreat_matrices/L16_N2048.npz with seed, m_r, Qc, Cprof, W, RQc, RW
    """
    L, N = 16, 2048
    out = RETREAT_MATRICES / "L16_N2048.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds, block_acc, xhat, yspin = load_large_data(L, N)
    keep = retreat(block_acc)
    m_r, _, Qc, W = overlap_matrices(xhat[keep], yspin, slice(0, L))
    cosine = lambda M: M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
    np.savez(out, seed=seeds[keep].astype(np.int64), m_r=m_r, Qc=Qc, Cprof=Qc - W,
             W=W, RQc=cosine(Qc), RW=cosine(W))
    print(f"L={L} N={N}: trained {len(block_acc)}, retreat {W.shape[0]}, "
          f"gap(W)/gap(Qc) = {gap(W) / gap(Qc):.3f}   wrote {out.name}", flush=True)


# ---------------------------------------------------------------- Sec. 4.2: controls

def tree_statistics(rw: np.ndarray, tri: np.ndarray):
    """r_coph, R^2_tree, P(delta_um < 0.1) over the triples `tri`, and the K = 4 cluster sizes
    of the average-linkage tree of d = 1 - rw."""
    dc = tree_distance(rw)
    Z = linkage(dc, method="average")
    _, co = cophenet(Z, dc)
    coph = float(np.corrcoef(dc, co)[0, 1])
    return coph, tree_fit(dc, co), near_ultrametric_fraction(rw, tri), cluster_sizes(Z, 4)


def tree_permutation_references():
    """Tree statistics of the (12, 1280) retreat set against two permutation references
    (Sec. 4.2, first control).

    r_coph, R^2_tree and P(delta_um < 0.1) of the 269 retreat endpoints are measured on R^W on
    all positions and on 200 draws of each of two randomized matrices:

      entry permutation   the off-diagonal entries of R^W permuted symmetrically
      row-centered        with mu_r = E_{s != r}[R^W_rs], the entries of each row of
                          R^W_rs - (mu_r + mu_s)/2 permuted independently, the result averaged
                          with its transpose and (mu_r + mu_s)/2 added back; permuting each row
                          on its own keeps each replica's own distribution of similarities

    Both references have unit diagonal.  Neither is guaranteed positive semidefinite, and the
    second does not keep each row mean exactly.  The mean and the maximum over the draws are
    reported.  P(delta_um < 0.1) is measured on one sample of 400000 random triples (those with
    three distinct replicas kept), the same for the observed matrix and every draw.  One
    generator (seed 29) draws the triples and then every permutation, in that order.

    Reads   data/large_data_ensembles/L12_N1280/seeds*.npz
    Writes  results/tree_permutation_references.json: the observed statistics with the K = 4
            cut, and the mean and maximum of each statistic over the draws of each reference.
    """
    label, L, N = "(12,1280)", 12, 1280
    nperm = 200
    statistics = ("r_coph", "r2_tree", "p_um")
    s, a, x, ys = load_large_data(L, N)
    rw = normalized_residual(x[retreat(a)], ys, slice(0, L))
    n = rw.shape[0]
    rng = np.random.default_rng(29)
    tri = rng.integers(0, n, size=(400000, 3))
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 0] != tri[:, 2]) & (tri[:, 1] != tri[:, 2])]
    print(f"{label} retreat, n = {n}, distance = 1 - R^W")
    c, r, p, k4 = tree_statistics(rw, tri)
    print(f"  observed   r_coph {c:.3f}  R2_tree {r:.3f}  P(dum) {p:.3f}   K=4 {k4}")
    out = dict(set=f"retreat|{label}", n=n, draws=nperm, triples=int(len(tri)),
               observed=dict(r_coph=c, r2_tree=r, p_um=p, k4=k4))

    iu = np.triu_indices(n, 1)
    v = rw[iu].copy()
    mu = (rw.sum(1) - np.diag(rw)) / (n - 1)
    base = (mu[:, None] + mu[None, :]) / 2
    E = rw - base
    for tag, key in (("perm(entries) ", "entry_permutation"), ("perm(row resid)", "row_centered")):
        acc = []
        for _ in range(nperm):
            if key == "entry_permutation":
                m = np.zeros((n, n))
                m[iu] = rng.permutation(v)
                m = m + m.T
            else:
                P = np.stack([rng.permutation(E[r]) for r in range(n)])
                m = base + (P + P.T) / 2
            np.fill_diagonal(m, 1.0)
            acc.append(tree_statistics(m, tri)[:3])
        acc = np.array(acc)
        print(f"  {tag}  mean r_coph {acc[:,0].mean():.2f}  R2_tree {acc[:,1].mean():.2f}  "
              f"P(dum) {acc[:,2].mean():.2f}    max {acc[:,0].max():.3f}/{acc[:,1].max():.3f}/"
              f"{acc[:,2].max():.3f}", flush=True)
        out[key] = dict(mean=dict(zip(statistics, acc.mean(axis=0).tolist())),
                        max=dict(zip(statistics, acc.max(axis=0).tolist())))

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "tree_permutation_references.json"
    write_json(path, out)
    print("wrote", path.relative_to(ROOT))


def retreat_definition_controls():
    """Two controls on the definition of the (12, 1280) retreat set (Sec. 4.2, second and third
    control).

    Thresholds.  The retreat set is selected with (0.8, 0.5) and with the tighter pair
    (0.9, 0.6), both at epoch 150.  On each selection, on all positions, the item measures the
    share gap(W)/gap(Q^c) and the cophenetic correlation r_coph of the average-linkage tree on
    d = 1 - R^W, and reports the replica counts and the shift of the two statistics.

    Observation epoch.  The selection with (0.8, 0.5) is applied at epoch 100: reached 0.8
    within the first 100 epochs and below 0.5 at epoch 100.  The item counts how many of these
    replicas are below 0.5 at epoch 150 as well; such a replica belongs to the retreat set of
    Table 3, because it has reached 0.8 by epoch 100.

    Reads   data/large_data_ensembles/L12_N1280/seeds*.npz
    Writes  results/retreat_definition_controls.json
    """
    label, L, N = "(12,1280)", 12, 1280
    thresholds = ((HI, LO), (0.9, 0.6))
    epoch_before = 100

    def residual_share(xhat, yspin):
        _, _, Qc, W = overlap_matrices(xhat, yspin, slice(0, L))
        return gap(W) / gap(Qc)

    def cophenetic_correlation(rw):
        dc = tree_distance(rw)
        _, co = cophenet(linkage(dc, method="average"), dc)
        return float(np.corrcoef(dc, co)[0, 1])

    seeds, block_acc, xhat, yspin = load_large_data(L, N)
    out = dict(ensemble=label, replicas=int(len(seeds)))

    print(f"{label}: {len(seeds)} replicas, observation epoch {EPOCHS}, all output positions")
    rows = []
    for hi, lo in thresholds:
        k = retreat(block_acc, hi, lo)
        row = dict(high=hi, low=lo, n=int(k.sum()),
                   share=residual_share(xhat[k], yspin),
                   r_coph=cophenetic_correlation(normalized_residual(xhat[k], yspin,
                                                                     slice(0, L))))
        rows.append(row)
        print(f"  thresholds ({hi}, {lo}): n = {row['n']:3d}   residual share {row['share']:.3f}"
              f"   r_coph {row['r_coph']:.3f}")
    shift = dict(share=rows[1]["share"] - rows[0]["share"],
                 r_coph=rows[1]["r_coph"] - rows[0]["r_coph"])
    print(f"  shift: residual share {shift['share']:+.3f}   r_coph {shift['r_coph']:+.3f}")
    out["thresholds"] = dict(observation_epoch=EPOCHS, selections=rows, shift=shift)

    before = retreat(block_acc, epoch=epoch_before)
    both = before & (block_acc[:, EPOCHS - 1] < LO)
    assert not (both & ~retreat(block_acc)).any()
    frac = float(both.sum() / before.sum())
    print(f"  retreat at epoch {epoch_before}: {int(before.sum())} replicas, of which "
          f"{int(both.sum())} are in retreat at epoch {EPOCHS} ({frac:.1%})")
    out["observation_epoch"] = dict(high=HI, low=LO, first_epoch=epoch_before,
                                    second_epoch=EPOCHS,
                                    retreat_at_first=int(before.sum()),
                                    retreat_at_both=int(both.sum()), fraction=frac)

    output = RESULTS / "retreat_definition_controls.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(output, out)
    print("wrote", output.relative_to(ROOT))


def k4_cut_equal_count():
    """K = 4 cuts of the retreat and the recovery tree at one equal replica count (Sec. 4.2).

    The cluster sizes of a tree cut depend on the number of replicas, so the retreat and the
    recovery set of one large-data ensemble are compared at the count of the smaller one, the
    recovery set in both ensembles used here.  On R^W on all positions, with the
    average-linkage tree on d = 1 - R^W cut into four clusters:

      recovery   the cut of the whole recovery set, n_rec replicas
      retreat    the cut of 400 random subsets of n_rec retreat replicas (drawn without
                 replacement by one generator with seed 23); the four sizes of each cut are
                 sorted, and the mean and the standard deviation of each rank are reported

    No residual-norm filter is applied.  The mean sizes carry a Monte-Carlo error of
    s.d./sqrt(400), about 0.3 replicas for the two largest clusters.

    Reads   data/large_data_ensembles/{L20_N24576, L16_N3072}/seeds*.npz
    Writes  results/k4_cut_equal_count.json
    """
    ensembles = (("(20,24576)", 20, 24576), ("(16,3072)", 16, 3072))
    K, n_draws, seed = 4, 400, 23

    def cut_sizes(rw):
        sizes = cluster_sizes(linkage(tree_distance(rw), method="average"), K)
        return sizes + [0] * (K - len(sizes))

    out = dict(K=K, draws=n_draws, seed=seed, ensembles={})
    for label, L, N in ensembles:
        seeds, block_acc, xhat, yspin = load_large_data(L, N)
        window = slice(0, L)
        rw_retreat = normalized_residual(xhat[retreat(block_acc)], yspin, window)
        rw_recovery = normalized_residual(xhat[recovery(block_acc)], yspin, window)
        n_ret, n_rec = rw_retreat.shape[0], rw_recovery.shape[0]
        assert n_ret >= n_rec

        recovery_sizes = cut_sizes(rw_recovery)
        rng = np.random.default_rng(seed)
        draws = np.empty((n_draws, K))
        for i in range(n_draws):
            k = rng.choice(n_ret, n_rec, replace=False)
            draws[i] = cut_sizes(rw_retreat[np.ix_(k, k)])
        mean, sd = draws.mean(axis=0), draws.std(axis=0, ddof=1)

        out["ensembles"][label] = dict(
            n_retreat=n_ret, n_recovery=n_rec, matched_count=n_rec,
            recovery_sizes=recovery_sizes, recovery_outside_largest=n_rec - recovery_sizes[0],
            retreat_mean_sizes=mean.tolist(), retreat_sd_sizes=sd.tolist(),
            retreat_se_sizes=(sd / np.sqrt(n_draws)).tolist(),
            retreat_second_cluster_share=float(mean[1] / n_rec),
            retreat_whole_set_sizes=cut_sizes(rw_retreat))
        print(f"{label}: retreat n = {n_ret}, recovery n = {n_rec}, K = {K} cut at "
              f"{n_rec} replicas")
        print("  recovery            " + "/".join(str(v) for v in recovery_sizes)
              + f"   ({n_rec - recovery_sizes[0]} outside the largest cluster)")
        print(f"  retreat, {n_draws} draws  " + "/".join(f"{v:.0f}" for v in mean)
              + "   mean " + " ".join(f"{v:.2f}" for v in mean)
              + "   s.d. " + " ".join(f"{v:.2f}" for v in sd)
              + f"   second cluster {mean[1] / n_rec:.2f} of the replicas")

    output = RESULTS / "k4_cut_equal_count.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(output, out)
    print("wrote", output.relative_to(ROOT))


def truth_scalar_correlation():
    """corr_{r<s}(Q^raw_rs, m_r m_s) of the fourteen replica sets (Sec. 4.2).

    How much of the raw pair pattern the scalar truth alignment explains.  Sec. 4.2 quotes the
    value for the memorization sets and a lower bound for retreat and recovery; the item
    measures it for every set, on all positions ("full", the values quoted) and on the common
    tail ("tail_u").  The correlation is over the binom(R, 2) distinct pairs: the diagonal is
    q_self and does not belong to the pair pattern.

    Reads   the memorization networks and the large-data ensembles
    Writes  results/truth_scalar_correlation.json: "<regime>|(L,N)" -> n, L, N, regime and the
            correlation on the two windows.
    About five minutes, most of them spent evaluating the memorization networks.
    """
    def corr(v, ys, sl):
        m_r, Qraw, _, _ = overlap_matrices(v, ys, sl)
        n = len(m_r)
        iu = np.triu_indices(n, 1)
        return float(np.corrcoef(Qraw[iu], np.outer(m_r, m_r)[iu])[0, 1]), n

    out = {}
    print(f"{'set':26s} {'n':>4s} {'full':>7s} {'tail_u':>7s}")
    for lab, L, N, interpolating_only in MEMORIZATION:
        xh, ys = load_memorization(L, N, interpolating_only)
        cf, n = corr(xh, ys, slice(0, L))
        ct, _ = corr(xh, ys, common_tail(L))
        out[f"memorization|{lab}"] = dict(n=n, L=L, N=N, regime="memorization", full=cf, tail_u=ct)
        print(f"{'memorization ' + lab:26s} {n:4d} {cf:7.3f} {ct:7.3f}", flush=True)
    for lab, L, N in LARGE_DATA:
        s, a, x, ys = load_large_data(L, N)
        for reg, k in endpoint_sets(a):
            cf, n = corr(x[k], ys, slice(0, L))
            ct, _ = corr(x[k], ys, common_tail(L))
            out[f"{reg}|{lab}"] = dict(n=n, L=L, N=N, regime=reg, full=cf, tail_u=ct)
            print(f"{reg + ' ' + lab:26s} {n:4d} {cf:7.3f} {ct:7.3f}", flush=True)
    for w in ("full", "tail_u"):
        for reg in ("memorization", "retreat", "recovered"):
            v = [u[w] for u in out.values() if u["regime"] == reg]
            print(f"[{w:6s}] {reg:13s} span {min(v):.3f} to {max(v):.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "truth_scalar_correlation.json"
    write_json(path, out)
    print("wrote", path.relative_to(ROOT))


def evaluation_outside_training_set():
    """The Sec. 4 endpoint observables on the evaluation draws outside the training set (Sec. 4.2).

    The 2048 evaluation draws of a large-data ensemble are independent of its training sample,
    so some of them are training configurations: at (12, 1280), where the training sample
    covers 963 of the 2048 configurations, 47.1% of the draws; at L = 16 and 20 below 9%.  The
    item drops the draws that occur in the training sample (both regenerated from their seeds,
    TRAIN_SEED and EVAL_SEED) and measures every observable of summarize on the remaining
    draws, on all positions, for the retreat and the recovery set of each large-data
    ensemble.  The replica selection does not change; only the inputs entering the averages
    do.  Sec. 4.2 states that the restriction leaves gap(Q^c) and gap(W) of the (12, 1280)
    retreat set unchanged.

    Reads   the five large-data ensembles
    Writes  results/evaluation_outside_training_set.json: {"worst_shift", "sets"}; sets maps
            "<regime>|(L,N)" to {observable: {"all", "heldout", "shift"}} plus the number of
            evaluation draws and the share of them in the training sample; worst_shift is the
            largest absolute shift over all sets and observables.
    About five minutes.
    """
    keys = ("n", "m", "q_self", "q_cross", "Delta_N", "chi_SG",
            "gap_raw", "gap_Qc", "gap_W", "share", "r2_tree", "p_um")
    out, worst = {}, 0.0
    for lab, L, N in LARGE_DATA:
        s, a, x, ys = load_large_data(L, N)
        te = sample_walls(x.shape[1], L, EVAL_SEED).numpy().astype(np.uint8)
        train = {r.tobytes() for r in sample_walls(N, L, TRAIN_SEED).numpy().astype(np.uint8)}
        inb = np.array([r.tobytes() in train for r in te])
        print(f"\n### {lab}  eval draws={len(te)}  on training inputs={inb.mean():.4f}", flush=True)
        for reg, k in endpoint_sets(a):
            full = summarize(x[k], ys, slice(0, L))
            held = summarize(x[k][:, ~inb, :], ys[~inb], slice(0, L))
            row = {}
            for key in keys:
                d = held[key] - full[key]
                row[key] = dict(all=full[key], heldout=held[key], shift=d)
                if key != "n":
                    worst = max(worst, abs(d))
                print(f"  {reg:<10} {key:<8} all={full[key]:+.4f}  held-out={held[key]:+.4f}"
                      f"  shift={d:+.4f}", flush=True)
            out[f"{reg}|{lab}"] = dict(row, eval_draws=int(len(te)),
                                       train_draw_share=float(inb.mean()))

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "evaluation_outside_training_set.json"
    write_json(path, dict(worst_shift=worst, sets=out))
    print(f"\nlargest shift over all sets and observables: {worst:.4f}")
    print("wrote", path.relative_to(ROOT))


def gf2_reduce(M: np.ndarray, n_var: int):
    """Row-reduce M over GF(2) on its first n_var columns; return the reduced matrix and rank."""
    M = M.copy()
    row = 0
    for col in range(n_var):
        hit = np.flatnonzero(M[row:, col])
        if not len(hit):
            continue
        piv = row + hit[0]
        M[[row, piv]] = M[[piv, row]]
        other = np.flatnonzero(M[:, col])
        other = other[other != row]
        M[other] ^= M[row]
        row += 1
        if row == len(M):
            break
    return M, row


def parity_targets(B: np.ndarray) -> np.ndarray:
    """y_j = parity of the first j walls, y_0 = 0, for wall configurations B (uint8)."""
    Y = np.zeros((len(B), B.shape[1] + 1), dtype=np.uint8)
    Y[:, 1:] = np.cumsum(B, axis=1) % 2
    return Y


def gf2_identifiability():
    """Is the target rule identified by the training set within the parity class? (Sec. 4.2)

    Every target is a GF(2)-linear function of the domain walls, so a training set identifies
    the rule exactly when its input matrix has full column rank D = L-1 over GF(2).  The item
    row-reduces the input matrix of each of the nine training sets of Table 3 (N draws with
    TRAIN_SEED, the training generator; no stored run is read), reports the rank, and at L = 12
    solves for the coefficients and scores the solution on the complete input set.  Full
    column rank already makes the solution unique and equal to the planted rule; the score is
    a check on that argument.  It also tabulates the probability that N random configurations
    span GF(2)^D, prod_{i<D} (1 - 2^(i-N)), and the smallest N where it reaches 0.99, the
    identifiability threshold to compare with the fitted crossover N_c of Sec. 3.1.

    Writes  results/gf2_identifiability.json with the keys "sets" and "threshold"
    """
    sets = [(12, 64), (24, 64), (24, 128), (24, 192),
            (12, 1280), (16, 2048), (16, 2560), (16, 3072), (20, 24576)]
    out = {"sets": {}, "threshold": {}}
    for L, N in sets:
        B = sample_walls(N, L, TRAIN_SEED).numpy().astype(np.uint8)
        red, rank = gf2_reduce(np.concatenate([B, parity_targets(B)], axis=1), L - 1)
        row = dict(L=L, N=N, distinct_inputs=int(len(np.unique(B, axis=0))),
                   n_variables=L - 1, gf2_rank=int(rank), identified=bool(rank == L - 1))
        if L == 12 and rank == L - 1:
            coef = red[:L - 1, L - 1:]
            # the complete input set, rows in the order of common.complete_set
            X = ((np.arange(2 ** (L - 1))[:, None] >> np.arange(L - 1)) & 1).astype(np.uint8)
            pred = (X @ coef) % 2
            row["full_space_block_accuracy"] = float(np.all(pred == parity_targets(X), axis=1).mean())
        out["sets"][f"({L},{N})"] = row
        print(f"L={L:2d} N={N:5d}  distinct={row['distinct_inputs']:5d}  D={L-1:2d}  "
              f"GF(2) rank={rank:2d}  identified={row['identified']}"
              + (f"  full-space block accuracy={row['full_space_block_accuracy']:.4f}"
                 if "full_space_block_accuracy" in row else ""))

    print("\nProbability that N random domain-wall configurations span GF(2)^D:")
    for L in (12, 16, 20):
        D = L - 1
        tab = {}
        for N in range(D, 6 * D):
            tab[N] = float(np.prod([1.0 - 2.0 ** (i - N) for i in range(D)]))
        n99 = min(n for n, p in tab.items() if p >= 0.99)
        nc = CROSSOVER_N[L]
        out["threshold"][f"L={L}"] = dict(D=D, N_for_p99=int(n99), N_c=nc, ratio=float(nc / n99))
        print(f"  L={L:2d}  D={D:2d}  P>=0.99 at N={n99:3d}   fitted N_c={nc:5d}   "
              f"N_c/N={nc/n99:.0f}x")

    output = RESULTS / "gf2_identifiability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, out)
    print("\nwrote", output)


# ---------------------------------------------------------------- Sec. 4.3: obtuse pairs

def retreat_obtuse_pair_fraction():
    """P(R^W < 0) of the five retreat sets on all positions and on the tail (Sec. 4.3).

    A negative R^W_rs is a pair of replicas whose residuals point more than a right angle
    apart.  Sec. 4.3 quotes the range of the fraction over the five retreat sets on all
    positions ("full") and on the tabulated tail ("tail"), which for these ensembles is the
    common tail j >= 2L/3 and where the retreat sets separate from the other regimes more
    sharply.  The fraction uses every replica of a set and needs no equal replica count.

    Reads   the five large-data ensembles
    Writes  results/retreat_obtuse_pair_fraction.json: one row per ensemble with the replica
            count and the fraction on the two windows.
    """
    print(f"{'ensemble':11s} {'n':>4s} {'full':>7s} {'tail':>7s}")
    rows, full, tail = [], [], []
    for lab, L, N in LARGE_DATA:
        s, a, x, ys = load_large_data(L, N)
        k = retreat(a)
        f_ = obtuse_fraction(normalized_residual(x[k], ys, slice(0, L)))
        t_ = obtuse_fraction(normalized_residual(x[k], ys, TAIL[L]))
        full.append(f_)
        tail.append(t_)
        rows.append(dict(set=f"retreat|{lab}", L=L, N=N, n=int(k.sum()), full=f_, tail=t_))
        print(f"{lab:11s} {int(k.sum()):4d} {f_:7.1%} {t_:7.1%}", flush=True)
    print(f"\nfull window: {min(full):.0%} to {max(full):.0%}")
    print(f"tail window: {min(tail):.0%} to {max(tail):.0%}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "retreat_obtuse_pair_fraction.json"
    write_json(path, rows)
    print("wrote", path.relative_to(ROOT))


def split_half_obtuse():
    """Split-half control on the obtuse-pair fraction (Sec. 4.3).

    R^W is estimated from a finite evaluation set, so a set whose pair cosines are all
    non-negative shows a few negative entries once the estimate is noisy.  Halving the
    evaluation set doubles the estimation variance: under that artefact the measured fraction
    rises with less data, while a fraction carried by genuinely opposed pairs stays put.  For
    the retreat and the recovery set of each large-data ensemble, on all positions, the item
    reports P(R^W < 0) on the full set and on each half, and the agreement: of the pairs
    obtuse on the full set, the share obtuse on both halves.  The profile is estimated within
    each half, so a half is a self-contained estimate.  Both a contiguous split and a random
    split (generator seed 0) are reported.

    Reads   the five large-data ensembles
    Writes  results/split_half_obtuse.json: one row per set with the fractions and agreements.
    """
    def rw_on(xhat, yspin, index):
        """R^W on the evaluation inputs `index`, all positions."""
        # numpy raises spurious divide/overflow flags on the BLAS matmul path here; the result
        # is checked below.
        with np.errstate(all="ignore"):
            rw = normalized_residual(xhat[:, index, :], yspin[index], slice(None))
        assert np.isfinite(rw).all()
        return rw

    rows = []
    for lab, L, N in LARGE_DATA:
        s, a, x, ys = load_large_data(L, N)
        for reg, keep in endpoint_sets(a):
            xs = x[keep]
            n, nb = xs.shape[0], xs.shape[1]
            iu = np.triu_indices(n, 1)
            full = rw_on(xs, ys, np.arange(nb))[iu]
            cut = nb // 2
            con = [rw_on(xs, ys, np.arange(cut))[iu],
                   rw_on(xs, ys, np.arange(cut, nb))[iu]]
            perm = np.random.default_rng(0).permutation(nb)
            ran = [rw_on(xs, ys, perm[:cut])[iu], rw_on(xs, ys, perm[cut:])[iu]]
            obtuse = full < 0
            rows.append(dict(
                set=f"{reg}|{lab}", n=n, configurations=nb, pairs=len(full),
                p_full=float(obtuse.mean()),
                p_half=[float((h < 0).mean()) for h in con],
                p_half_random=[float((h < 0).mean()) for h in ran],
                agree=float(((con[0] < 0) & (con[1] < 0))[obtuse].mean()) if obtuse.any() else float("nan"),
                agree_random=float(((ran[0] < 0) & (ran[1] < 0))[obtuse].mean()) if obtuse.any() else float("nan")))
            print(f"{rows[-1]['set']:<22} n={n:<4} b={nb:<6} "
                  f"P_full={rows[-1]['p_full']:.4f}  "
                  f"P_half={rows[-1]['p_half'][0]:.4f}/{rows[-1]['p_half'][1]:.4f}  "
                  f"agree={rows[-1]['agree']:.3f}  "
                  f"[random split P={rows[-1]['p_half_random'][0]:.4f}/"
                  f"{rows[-1]['p_half_random'][1]:.4f} agree={rows[-1]['agree_random']:.3f}]",
                  flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "split_half_obtuse.json"
    write_json(path, rows)
    print("\nwrote", path.relative_to(ROOT))
    for reg in ("retreat", "recovered"):
        q = [r for r in rows if r["set"].startswith(reg)]
        ag = [r["agree"] for r in q]
        rise = [max(r["p_half"]) / r["p_full"] if r["p_full"] else float("nan") for r in q]
        print(f"{reg:<10} agreement {min(ag):.3f}-{max(ag):.3f}   "
              f"half/full fraction ratio {min(rise):.2f}-{max(rise):.2f}")
    print("\nSec. 4.3 quotes the retreat agreement as 88% to 93% and states that halving the "
          "data\nleaves the retreat fractions unchanged while the recovery fractions rise.")


# ---------------------------------------------------------------- App. C: endpoint classes, tails

def endpoint_counts(a: np.ndarray) -> dict:
    """Counts of the five endpoint classes, and of the falling and returning replicas, for the
    block-accuracy curves a [replicas, 150]."""
    n = len(a)
    peak = reached(a)
    falls = [first_fall(curve) for curve in a]
    fell = np.array([fall is not None for fall in falls], dtype=bool)
    # "returning": regains HI at any point after the first fall, held or not
    back = np.array([fall is not None and bool((curve[fall:] >= HI).any())
                     for curve, fall in zip(a, falls)], dtype=bool)
    last = a[:, EPOCHS - 1]
    out = {
        "trained": n,
        "never": int((~peak).sum()),
        "stable": int((peak & ~fell).sum()),
        "retreat": int((fell & (last < LO)).sum()),
        "recovery": int((fell & (last >= HI)).sum()),
        "neither": int((fell & (last >= LO) & (last < HI)).sum()),
        "fell": int(fell.sum()),
        "returned": int(back.sum()),
    }
    assert out["never"] + out["stable"] + out["fell"] == n
    assert out["retreat"] + out["recovery"] + out["neither"] == out["fell"]
    return out


def endpoint_prevalence():
    """The five endpoint classes over every trained large-data replica (Table C1, App. C.1).

    From the 150-epoch held-out A_block curve alone:

        never      peak A_block < HI over the whole window
        stable     reaches HI and never falls below LO afterwards
        retreat    reaches HI, falls below LO afterwards, and is below LO at epoch 150
        recovery   the same, but at or above HI at epoch 150
        neither    the same, but between LO and HI at epoch 150

    The five shares are over all trained runs and add to one.  "Returning" has its own
    denominator: of the runs that fall below LO, the fraction that regain HI at any point
    before epoch 150, held or not.  The retreat and recovery classes are the selections of
    common.retreat and common.recovery, so the counts agree with the replica counts of Sec. 4.

    Reads   the block-accuracy curves of the five large-data ensembles
    Writes  results/endpoint_prevalence.json: the thresholds and, per ensemble, the counts, the
            shares, the returning fraction and N_train/N_c.
    """
    res = {}
    hdr = f"{'ensemble':12s} {'trained':>7s} {'never':>6s} {'stable':>6s} {'retreat':>7s} " \
          f"{'recov':>6s} {'neither':>7s} | {'never':>5s} {'stable':>6s} {'retreat':>7s} " \
          f"{'recov':>6s} {'neither':>7s} {'return':>6s}"
    print(hdr)
    for lab, L, N in LARGE_DATA:
        _, block_acc, _, _ = load_large_data(L, N)
        c = endpoint_counts(block_acc[:, :EPOCHS])
        n = c["trained"]
        c["N_over_Nc"] = round(N / CROSSOVER_N[L], 1)
        for k in ("never", "stable", "retreat", "recovery", "neither"):
            c[f"share_{k}"] = c[k] / n
        c["returning"] = c["returned"] / c["fell"] if c["fell"] else float("nan")
        res[lab] = c
        print(f"{lab:12s} {n:7d} {c['never']:6d} {c['stable']:6d} {c['retreat']:7d} "
              f"{c['recovery']:6d} {c['neither']:7d} | {c['share_never']:5.2f} "
              f"{c['share_stable']:6.2f} {c['share_retreat']:7.2f} {c['share_recovery']:6.2f} "
              f"{c['share_neither']:7.2f} {c['returning']:6.2f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "endpoint_prevalence.json"
    write_json(out, {"hi": HI, "lo": LO, "epochs": EPOCHS, "sets": res})
    print("\nwrote", out.relative_to(ROOT))


def tail_geometry():
    """Tail-window overlap geometry of the four memorization ensembles (Table C3).

    On the tabulated tail of each ensemble (8..11 at L = 12, 18..23 at L = 24), over all its
    replicas:

        q_self, q_cross   mean diagonal and mean off-diagonal entry of Q^raw
        chi               q_self - q_cross
        pair_sd           standard deviation of the off-diagonal entries of Q^raw
        r2_signed         R^2_tree of the average-linkage tree on d = 1 - R^W
        r2_abs            the same on d = 1 - |R^W|, which treats anticorrelated residuals as close
        p_um              P(delta_um < 0.1) over all replica triples, from the signed R^W

    The evaluation sets are those of common.load_memorization.  Every row uses all replicas of
    its ensemble, so tree statistics of rows with different replica counts are not on a
    common footing; Sec. 4.3 compares the regimes on random subsets of one size.

    Reads   the memorization networks
    Writes  results/tail_geometry.json, one entry per ensemble, keyed by its table label
    About seven minutes, most of it evaluating the 411 memorization networks.
    """
    def row(xh, ys, sl):
        _, Q, _, W = overlap_matrices(xh, ys, sl)
        n = Q.shape[0]
        iu = np.triu_indices(n, 1)
        rw = W / np.sqrt(np.outer(np.diag(W), np.diag(W)))

        def r2(signed):
            dc = tree_distance(rw if signed else np.abs(rw))
            _, co = cophenet(linkage(dc, method="average"), dc)
            return tree_fit(dc, co)

        tri = np.array(list(itertools.combinations(range(n), 3)))
        return dict(n=n, q_self=float(np.diag(Q).mean()), q_cross=float(Q[iu].mean()),
                    chi=float(np.diag(Q).mean() - Q[iu].mean()), pair_sd=float(Q[iu].std(ddof=1)),
                    r2_signed=r2(True), r2_abs=r2(False),
                    p_um=near_ultrametric_fraction(rw, tri))

    labels = {(12, 64): "$L=12,\\ N=64$", (24, 64): "$L=24,\\ N=64$",
              (24, 128): "$L=24,\\ N=128$", (24, 192): "$L=24,\\ N=192$"}
    out = {}
    hdr = f"{'ensemble':18s} {'n':>4s} {'q_self':>7s} {'q_cross':>8s} {'chi':>7s} {'pair sd':>8s} {'R2sgn':>7s} {'R2|.|':>7s} {'P(dum)':>7s}"
    print("MEMORIZATION (tail)\n" + hdr, flush=True)
    for _, L, N, interpolating_only in MEMORIZATION:
        lab = labels[(L, N)]
        xh, ys = load_memorization(L, N, interpolating_only)
        r = row(xh, ys, TAIL_MEMORIZATION[(L, N)])
        out[lab] = r
        print(f"{lab:18s} {r['n']:4d} {r['q_self']:7.4f} {r['q_cross']:8.4f} {r['chi']:7.4f} "
              f"{r['pair_sd']:8.4f} {r['r2_signed']:7.3f} {r['r2_abs']:7.3f} {r['p_um']:7.3f}",
              flush=True)
    output = RESULTS / "tail_geometry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, out)
    print("wrote", output)


# ===========================================================================
# Sec. 5.2: the learning-rate intervention
# ===========================================================================

def intervention_pair_outcomes():
    """Paired retreat outcomes of the learning-rate intervention (Sec. 5.2).

    Each pair is one constant-rate run and one intervention run at L = 12 that share the data
    split, the initialization, the minibatch stream and the update budget.  The two runs are
    one run until the held-out block accuracy first reaches 0.8, where the intervention run
    starts to decay its learning rate.  There are 30 pairs at each of N = 512, 768, 1024
    (fifteen data splits with two initializations each).  A run reaches if A_block >= 0.8 at
    some evaluation, retreats if A_block < 0.5 at or after the first such evaluation, and ends
    high if A_block >= 0.8 at the last one.  Fig. 10 counts these events arm by arm; this item
    keeps the pairing: among the pairs in which both runs reach, it counts the pairs in which
    both retreat, only the constant-rate run retreats, only the intervention run retreats, and
    neither, and per arm how many retreat and how many end high.

    Reads   data/learning_rate_intervention/{constant_arm, intervention_arm}/split<d>/
            L12_N<N>_split<d>_init<m>.json
    Writes  results/intervention_pair_outcomes.json: the pooled counts and the counts per N
    """
    arms = ("constant_arm", "intervention_arm")   # constant rate; decay after the first
                                                  # crossing of HI
    outcomes = ("both_retreat", "only_constant_retreats", "only_intervention_retreats",
                "neither_retreats")

    def empty():
        return dict(pairs=0, pairs_reaching=0, **{k: 0 for k in outcomes},
                    constant_retreats=0, intervention_retreats=0,
                    constant_ends_high=0, intervention_ends_high=0)

    runs = {}
    for arm in arms:
        for path in sorted((LEARNING_RATE_INTERVENTION / arm).glob(
                "split*/L12_N*_split*_init*.json")):
            m = re.search(r"_N(\d+)_split(\d+)_init(\d+)\.json$", path.name)
            with open(path) as handle:
                history = json.load(handle)["history"]
            runs[(arm,) + tuple(int(g) for g in m.groups())] = crossing_outcome(
                [float(row["test_block_acc"]) for row in history])

    pairs = sorted({key[1:] for key in runs})
    counts = {"pooled": empty()}
    for key in pairs:
        constant, intervention = runs[("constant_arm",) + key], runs[("intervention_arm",) + key]
        # The two runs of a pair coincide up to the first crossing, so they reach together.
        assert constant[0] == intervention[0], key
        for c in (counts["pooled"], counts.setdefault(f"N={key[0]}", empty())):
            c["pairs"] += 1
            if not constant[0]:
                continue
            c["pairs_reaching"] += 1
            c[outcomes[2 * (not constant[1]) + (not intervention[1])]] += 1
            c["constant_retreats"] += constant[1]
            c["intervention_retreats"] += intervention[1]
            c["constant_ends_high"] += constant[2]
            c["intervention_ends_high"] += intervention[2]

    print(f"{'':8s} {'pairs':>5s} {'reach':>5s} {'both':>5s} {'const':>6s} {'interv':>6s} "
          f"{'neither':>7s}   retreat c/i   ends high c/i")
    for name in [k for k in counts if k != "pooled"] + ["pooled"]:
        c = counts[name]
        print(f"{name:8s} {c['pairs']:5d} {c['pairs_reaching']:5d} {c['both_retreat']:5d} "
              f"{c['only_constant_retreats']:6d} {c['only_intervention_retreats']:6d} "
              f"{c['neither_retreats']:7d}   {c['constant_retreats']:3d} / "
              f"{c['intervention_retreats']:<3d}     {c['constant_ends_high']:3d} / "
              f"{c['intervention_ends_high']:<3d}")
    print("columns: both retreat, only the constant-rate run, only the intervention run, neither")

    output = RESULTS / "intervention_pair_outcomes.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(high=HI, low=LO, **counts))
    print("wrote", output.relative_to(ROOT))


# ===========================================================================
# App. A: gradient and curvature on a fixed diagnostic set
# ===========================================================================

LANCZOS_TOLERANCE = 2e-3
LANCZOS_MAX_ITERATIONS = 250
HELD_OUT_1280 = 1085          # 2048 configurations minus the 963 of the (12, 1280) training set


def _block_forward_twice_differentiable(self, hidden):
    """Pre-norm block forward whose attention supports a double backward pass.

    The default call (need_weights=False) goes through the fused attention kernel, whose
    backward pass has no derivative of its own, so a Hessian-vector product through it
    raises.  need_weights=True selects the attention written in elementary operations, which
    can be differentiated twice.
    """
    normalized = self.norm_attn(hidden)
    attention, _ = self.attn(normalized, normalized, normalized, need_weights=True)
    hidden = hidden + self.dropout(attention)
    return hidden + self.ffn(self.norm_ffn(hidden))


def load_for_curvature(path) -> ChainTransformer:
    """The L = 12 network stored at `path`, in train mode, with every block bound to the
    twice-differentiable forward.  Dropout is zero, so train mode changes no output."""
    model = load_chain_transformer(path, 12)
    for block in model.blocks:
        block.forward = types.MethodType(_block_forward_twice_differentiable, block)
    return model.train()


def held_out_lexicographic():
    """The 1085 held-out inputs of the (12, 1280) training set in lexicographic order.

    The same configurations as common.held_out_set(12, 1280), ordered by
    itertools.product([0, 1], repeat=11), i.e. with the first wall most significant.  The
    order of the rows enters the sums of the loss and the gradient, and the fixed-epoch
    comparison is defined on this order.
    """
    train = sample_walls(1280, 12, TRAIN_SEED)
    seen = {tuple(int(v) for v in row) for row in train}
    rows = [c for c in itertools.product([0, 1], repeat=11) if c not in seen]
    if len(rows) != HELD_OUT_1280:
        raise SystemExit(f"expected {HELD_OUT_1280} held-out configurations, got {len(rows)}")
    return encode(torch.tensor(rows, dtype=torch.float32))


def flatten(tensors) -> np.ndarray:
    """Concatenate parameter-shaped tensors into one float64 vector."""
    return torch.cat([t.detach().reshape(-1).cpu() for t in tensors]).numpy().astype(np.float64)


def unflatten(vector: np.ndarray, parameters):
    """Split a float64 vector into float32 tensors shaped like `parameters`."""
    out, offset = [], 0
    for parameter in parameters:
        end = offset + parameter.numel()
        out.append(torch.from_numpy(vector[offset:end].astype(np.float32, copy=False))
                   .reshape_as(parameter))
        offset = end
    return out


def gradient_metrics(model, inputs, targets) -> dict:
    """Loss, block and bit accuracy and gradient norm |grad L|_2 of `model` on a diagnostic set."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    loss = F.binary_cross_entropy_with_logits(model(inputs), targets)
    grads = torch.autograd.grad(loss, parameters)
    flat = flatten(grads)
    with torch.no_grad():
        logits = model(inputs)
        correct = (logits > 0).float() == targets
    dimension = sum(p.numel() for p in parameters)
    return {
        "diagnostic_loss": float(loss.detach()),
        "diagnostic_block_accuracy": float(correct.all(dim=1).float().mean()),
        "diagnostic_bit_accuracy": float(correct.float().mean()),
        "gradient_norm": float(np.linalg.norm(flat)),
        "parameter_dimension": dimension,
    }


def spectral_extrema(model, inputs, targets, initial_seed: int) -> dict:
    """lambda_min and lambda_max of the Hessian of the diagnostic loss, by Lanczos (ARPACK).

    Both Lanczos runs start from the same unit Gaussian vector drawn with `initial_seed`.  The
    relative residual |H v - lambda v| / |lambda| of each returned pair is reported with it.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    dimension = sum(p.numel() for p in parameters)

    def matvec(vector):
        directions = unflatten(vector, parameters)
        loss = F.binary_cross_entropy_with_logits(model(inputs), targets)
        grads = torch.autograd.grad(loss, parameters, create_graph=True)
        directional = sum((g * d).sum() for g, d in zip(grads, directions))
        return flatten([e.detach() for e in torch.autograd.grad(directional, parameters)])

    operator = LinearOperator((dimension, dimension), matvec=matvec, dtype=np.float64)
    rng = np.random.default_rng(initial_seed)
    initial = rng.standard_normal(dimension)
    initial /= np.linalg.norm(initial)
    smallest, v_min = eigsh(operator, k=1, which="SA", v0=initial, tol=LANCZOS_TOLERANCE,
                            maxiter=LANCZOS_MAX_ITERATIONS, return_eigenvectors=True)
    largest, v_max = eigsh(operator, k=1, which="LA", v0=initial, tol=LANCZOS_TOLERANCE,
                           maxiter=LANCZOS_MAX_ITERATIONS, return_eigenvectors=True)
    lo, hi = float(smallest[0]), float(largest[0])
    return {
        "lambda_min": lo,
        "lambda_max": hi,
        "lambda_min_relative_residual": float(
            np.linalg.norm(matvec(v_min[:, 0]) - lo * v_min[:, 0]) / (abs(lo) + 1e-12)),
        "lambda_max_relative_residual": float(
            np.linalg.norm(matvec(v_max[:, 0]) - hi * v_max[:, 0]) / (abs(hi) + 1e-12)),
        "parameter_dimension": dimension,
    }


def fixed_epoch_class(history: list[dict]) -> str:
    """Retreat, Stable or Other from a run's recorded evaluation_block_acc history."""
    accuracies = [float(row["evaluation_block_acc"]) for row in history]
    final = accuracies[-1]
    reached_high = max(accuracies) >= 0.8
    if reached_high and final < 0.5:
        return "Retreat"
    if reached_high and final > 0.8:
        return "Stable"
    return "Other"


def fixed_epoch_comparison():
    """Fixed-epoch gradient norm and Hessian extrema of sixteen replicas on the held-out set
    (App. A.1, A.2).

    Sixteen (12, 1280) replicas, seeds 0-15, share the training set and differ only in
    initialization and minibatch order.  At epoch 150 each is scored on the 1085 held-out
    configurations (held_out_lexicographic): block and bit accuracy, the mean cross-entropy,
    |grad L|_2 and lambda_min, lambda_max.  The class (Retreat: reached 0.8 and below 0.5 at
    epoch 150; Stable: reached 0.8 and above 0.8; Other) comes from the accuracy history the run
    recorded on its own 2048 evaluation draws.  The class table gives means and medians.  The
    Lanczos start vector of replica s is drawn with seed 20_260_915 + s.

    Reads   data/fixed_epoch_replicas/L12_N1280_epoch150/seed<s>/{model.pt, metrics.json},
            s = 0..15
    Writes  results/fixed_epoch_comparison/seed<ss>.json         one replica each
            results/fixed_epoch_comparison/replica_metrics.csv   the sixteen rows
            results/fixed_epoch_comparison/summary.json          the class table
    One to two minutes per replica.  A replica whose seed<ss>.json is present is read back and
    not evaluated again, so an interrupted run continues where it stopped.
    """
    runs = FIXED_EPOCH_REPLICAS / "L12_N1280_epoch150"
    out_dir = RESULTS / "fixed_epoch_comparison"
    replicas, epoch = 16, 150
    lanczos_seed = 20_260_915          # the start vector of replica s is drawn with this + s

    # two intra-op threads: the last digits of every sum depend on this count
    torch.set_num_threads(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs, targets = held_out_lexicographic()
    rows = []
    for seed in range(replicas):
        cache = out_dir / f"seed{seed:02d}.json"
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        metrics = json.loads((runs / f"seed{seed}" / "metrics.json").read_text())
        model = load_for_curvature(runs / f"seed{seed}" / "model.pt")
        row = {"seed": seed, "trajectory_class": fixed_epoch_class(metrics["history"]),
               "recorded_final_block_acc": float(metrics["history"][-1]["evaluation_block_acc"]),
               **gradient_metrics(model, inputs, targets),
               **spectral_extrema(model, inputs, targets, lanczos_seed + seed)}
        cache.write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(f"seed{seed:02d} {row['trajectory_class']:8s} "
              f"A_block={row['diagnostic_block_accuracy']:.4f} "
              f"L={row['diagnostic_loss']:.4f} |g|={row['gradient_norm']:.4g} "
              f"lmin={row['lambda_min']:.4g} lmax={row['lambda_max']:.4g}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "replica_metrics.csv", index=False)

    def class_row(subset: pd.DataFrame, name: str) -> dict:
        return {
            "class": name, "replicas": int(len(subset)),
            "block_accuracy": float(subset["diagnostic_block_accuracy"].mean()),
            "loss": float(subset["diagnostic_loss"].mean()),
            "grad_mean": float(subset["gradient_norm"].mean()),
            "grad_median": float(subset["gradient_norm"].median()),
            "lambda_min_mean": float(subset["lambda_min"].mean()),
            "lambda_min_median": float(subset["lambda_min"].median()),
            "lambda_max_mean": float(subset["lambda_max"].mean()),
            "lambda_max_median": float(subset["lambda_max"].median()),
        }

    table = [class_row(frame[frame["trajectory_class"] == name], name)
             for name in ("Retreat", "Stable", "Other")]
    table.append(class_row(frame, "All"))
    (out_dir / "summary.json").write_text(json.dumps(
        {"protocol": {"evaluation_set": f"{HELD_OUT_1280} held-out domain-wall configurations",
                      "epoch": epoch, "replicas": replicas, "train_seed": TRAIN_SEED,
                      "classification": "from recorded evaluation_block_acc histories"},
         "quality": {"max_relative_residual": float(frame[
             ["lambda_min_relative_residual", "lambda_max_relative_residual"]].to_numpy().max())},
         "rows": table}, indent=2) + "\n")

    print(f"\nFixed-epoch comparison on the {HELD_OUT_1280} held-out configurations")
    header = f"{'class':8s} {'R':>2s} {'A_block':>8s} {'loss':>7s} " \
             f"{'|g| mean':>9s} {'(med)':>8s} {'lmin mean':>10s} {'(med)':>8s} " \
             f"{'lmax mean':>10s} {'(med)':>8s}"
    print(header)
    for row in table:
        print(f"{row['class']:8s} {row['replicas']:2d} {row['block_accuracy']:8.3f} "
              f"{row['loss']:7.3f} {row['grad_mean']:9.3f} {row['grad_median']:8.3f} "
              f"{row['lambda_min_mean']:10.4g} {row['lambda_min_median']:8.3g} "
              f"{row['lambda_max_mean']:10.4g} {row['lambda_max_median']:8.3g}")
    print(f"\nwrote {out_dir/'summary.json'}")


# ---------------------------------------------------------------- App. A.2: event-paired contrasts

EVENT_SAMPLES = 1024
EVENT_DIAGNOSTIC_SEED = 20_260_820
EVENT_DIAGNOSTIC_SET = f"{EVENT_SAMPLES} uniform draws, seed {EVENT_DIAGNOSTIC_SEED}"
EVENT_BOOTSTRAP_DRAWS = 20000
EVENT_BOOTSTRAP_SEED = 20_260_820
EVENT_PANEL_SEEDS = 12
EVENT_DIR = RESULTS / "retreat_event_contrasts"

# the four checkpoints of an event, and the column of events.csv that names each
EVENT_ROLES = {
    "event_start": "last_high_checkpoint",
    "event_end": "retreat_checkpoint",
    "control_start": "local_control_start_checkpoint",
    "control_end": "local_control_end_checkpoint",
}


# The Lanczos start vector of a checkpoint (replica seed, epoch) is drawn with this seed.
LANCZOS_START_SEEDS = {
    (0, 107): 1957793527, (0, 110): 441362620, (0, 111): 3280170585, (0, 114): 966786723,
    (3, 171): 2628400524, (3, 172): 2858136749, (3, 173): 1629535962, (3, 174): 3436557303,
    (5, 51): 2242192904, (5, 53): 371630179, (5, 159): 1299067211, (5, 161): 4025559527,
    (10, 111): 4191194177, (10, 113): 1564816831, (10, 153): 1104409174, (10, 155): 1887147753,
    (12, 60): 839734986, (12, 63): 2464748599, (12, 72): 3222117973, (12, 75): 1113071471,
    (14, 174): 1252681676, (14, 176): 773047649, (14, 177): 970625248, (14, 179): 653658099,
    (18, 104): 1237281705, (18, 105): 1481302575, (18, 106): 699500389, (18, 107): 461614110,
    (20, 138): 2899627081, (20, 139): 3608213398, (20, 140): 4006190503, (20, 141): 1381568501,
    (22, 93): 3723065761, (22, 94): 2104752433, (22, 95): 2492971165, (22, 96): 2157106967,
    (25, 91): 2178058377, (25, 92): 2112335895, (25, 114): 1576900966, (25, 115): 1488816954,
    (27, 91): 3847139286, (27, 93): 1692454806, (27, 94): 4020841100, (27, 96): 2722920228,
    (29, 145): 3052885829, (29, 146): 2030562538, (29, 147): 3672724747, (29, 148): 1694417509,
}


def checkpoint_key(checkpoint: str) -> tuple[int, int]:
    """(replica seed, epoch) of a checkpoint path .../seed<k>/epoch<e>.pt."""
    path = Path(checkpoint)
    return int(path.parent.name[len("seed"):]), int(path.stem[len("epoch"):])


def event_cache_path(root: Path, checkpoint: str) -> Path:
    """The cache file of one checkpoint: seed<k>_epoch<e>.json."""
    path = Path(checkpoint)
    return root / f"{path.parent.name}_{path.stem}.json"


def checkpoints_of(events: pd.DataFrame) -> list[str]:
    """Every checkpoint the given events refer to, each once, sorted."""
    return sorted({str(events[c].iloc[i]) for c in EVENT_ROLES.values()
                   for i in range(len(events))})


def bootstrap_simple(values: np.ndarray, seed: int) -> dict:
    """Mean, median, central 95% of the bootstrap means and the positive fraction of `values`."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(EVENT_BOOTSTRAP_DRAWS, len(values)))
    draws = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "positive_fraction": float(np.mean(values > 0))}


def bootstrap_cluster(frame: pd.DataFrame, column: str) -> dict:
    """The same for the per-seed means of `column`, resampling seeds (a seed can hold several
    events)."""
    cluster = frame.groupby("seed")[column].mean()
    seeds = cluster.index.to_numpy()
    values = cluster.to_numpy(dtype=float)
    rng = np.random.default_rng(EVENT_BOOTSTRAP_SEED)
    draws = np.empty(EVENT_BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(EVENT_BOOTSTRAP_DRAWS):
        pick = rng.integers(0, len(seeds), size=len(seeds))
        draws[draw] = values[pick].mean()
    return {"seed_cluster_mean": float(values.mean()),
            "seed_cluster_median": float(np.median(values)),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "positive_fraction": float(np.mean(values > 0)),
            "seed_clusters": int(len(seeds))}


def select_panel(events: pd.DataFrame) -> pd.DataFrame:
    """The first event of each of 12 seeds equally spaced in the sorted seed list."""
    seeds = np.asarray(sorted(int(v) for v in events["seed"].unique()))
    picks = seeds[np.linspace(0, len(seeds) - 1, EVENT_PANEL_SEEDS).round().astype(int)]
    panel = (events[events["seed"].isin(picks)]
             .sort_values(["seed", "event_number_within_seed"])
             .groupby("seed", as_index=False).first().sort_values("seed"))
    if len(panel) != EVENT_PANEL_SEEDS:
        raise RuntimeError(f"expected {EVENT_PANEL_SEEDS} panel events, got {len(panel)}")
    return panel


def pair_events(events: pd.DataFrame, lookup: dict, metrics: tuple[str, ...]) -> pd.DataFrame:
    """One row per event: each metric at the four checkpoints, both changes and DD."""
    rows = []
    for event in events.to_dict("records"):
        row = {"event_id": event["event_id"], "seed": int(event["seed"]),
               "lag": int(event["epochs_last_high_to_retreat"])}
        for role, column in EVENT_ROLES.items():
            checkpoint = str(event[column])
            row[f"{role}_checkpoint"] = checkpoint
            for metric in metrics:
                row[f"{role}_{metric}"] = float(lookup[checkpoint][metric])
        for metric in metrics:
            change = row[f"event_end_{metric}"] - row[f"event_start_{metric}"]
            control = row[f"control_end_{metric}"] - row[f"control_start_{metric}"]
            row[f"event_change_{metric}"] = change
            row[f"control_change_{metric}"] = control
            row[f"did_{metric}"] = change - control
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_gradients(events: pd.DataFrame) -> pd.DataFrame:
    """Gradient norm, loss and accuracy on the diagnostic set at every checkpoint of every event."""
    cache = EVENT_DIR / "gradient_cache"
    cache.mkdir(parents=True, exist_ok=True)
    inputs, targets = sample_set(EVENT_SAMPLES, 12, EVENT_DIAGNOSTIC_SEED)
    required = checkpoints_of(events)
    for index, checkpoint in enumerate(required, start=1):
        path = event_cache_path(cache, checkpoint)
        if path.exists():
            continue
        start = time.monotonic()
        row = {"checkpoint": checkpoint, "evaluation_set": EVENT_DIAGNOSTIC_SET,
               **gradient_metrics(load_for_curvature(DATA / checkpoint),
                                  inputs, targets)}
        path.write_text(json.dumps(row, indent=2) + "\n")
        print(f"  grad {index:03d}/{len(required)} {checkpoint}: "
              f"|g|={row['gradient_norm']:.4g} block={row['diagnostic_block_accuracy']:.3f} "
              f"{time.monotonic() - start:.1f}s", flush=True)
    frame = pd.DataFrame([json.loads(event_cache_path(cache, c).read_text()) for c in required])
    frame.to_csv(EVENT_DIR / "checkpoint_gradient.csv", index=False)
    return frame


def checkpoint_curvatures(panel: pd.DataFrame) -> pd.DataFrame:
    """Hessian extrema on the diagnostic set at the four checkpoints of every panel event.

    The Lanczos start vector of a checkpoint is drawn with the seed LANCZOS_START_SEEDS gives it.
    """
    cache = EVENT_DIR / "curvature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    inputs, targets = sample_set(EVENT_SAMPLES, 12, EVENT_DIAGNOSTIC_SEED)
    required = checkpoints_of(panel)
    for index, checkpoint in enumerate(required, start=1):
        path = event_cache_path(cache, checkpoint)
        if path.exists():
            continue
        start = time.monotonic()
        seed = LANCZOS_START_SEEDS[checkpoint_key(checkpoint)]
        row = {"checkpoint": checkpoint, "evaluation_set": EVENT_DIAGNOSTIC_SET,
               **spectral_extrema(load_for_curvature(DATA / checkpoint),
                                  inputs, targets, seed)}
        path.write_text(json.dumps(row, indent=2) + "\n")
        print(f"  curv {index:02d}/{len(required)} {checkpoint}: "
              f"lmin={row['lambda_min']:.4g} lmax={row['lambda_max']:.4g} "
              f"res={max(row['lambda_min_relative_residual'], row['lambda_max_relative_residual']):.2g} "
              f"{time.monotonic() - start:.1f}s", flush=True)
    frame = pd.DataFrame([json.loads(event_cache_path(cache, c).read_text()) for c in required])
    frame.to_csv(EVENT_DIR / "checkpoint_curvature.csv", index=False)
    return frame


def event_contrasts(events, panel, curvature, gradient) -> dict:
    """The DD contrasts with their bootstrap intervals; writes the two paired-event tables."""
    grad_lookup = gradient.set_index("checkpoint").to_dict("index")
    curv_lookup = curvature.set_index("checkpoint").to_dict("index")
    merged = {k: {**v, **grad_lookup.get(k, {})} for k, v in curv_lookup.items()}

    paired = pair_events(panel, merged, ("lambda_min", "lambda_max", "gradient_norm"))
    paired.to_csv(EVENT_DIR / "paired_events_curvature.csv", index=False)

    contrasts = {}
    for index, metric in enumerate(("lambda_min", "lambda_max", "gradient_norm")):
        contrasts[metric] = {kind: bootstrap_simple(
            paired[f"{kind}_{metric}"].to_numpy(dtype=float),
            EVENT_BOOTSTRAP_SEED + 10 * index + len(kind))
            for kind in ("event_change", "control_change", "did")}

    grad_paired = pair_events(events, grad_lookup,
                              ("gradient_norm", "diagnostic_loss", "diagnostic_block_accuracy"))
    grad_paired.to_csv(EVENT_DIR / "paired_events_gradient.csv", index=False)
    grad_contrasts = {m: {k: bootstrap_cluster(grad_paired, f"{k}_{m}")
                          for k in ("event_change", "control_change", "did")}
                      for m in ("gradient_norm", "diagnostic_loss", "diagnostic_block_accuracy")}

    return {
        "protocol": {
            "evaluation_set": EVENT_DIAGNOSTIC_SET,
            "curvature_panel_events": int(len(paired)),
            "curvature_panel_seeds": [int(v) for v in paired["seed"]],
            "gradient_events": int(len(grad_paired)),
            "gradient_seed_clusters": int(grad_paired["seed"].nunique()),
            "tolerance": LANCZOS_TOLERANCE, "max_iterations": LANCZOS_MAX_ITERATIONS,
            "bootstrap_draws": EVENT_BOOTSTRAP_DRAWS,
        },
        "quality": {
            "max_relative_residual": float(curvature[
                ["lambda_min_relative_residual", "lambda_max_relative_residual"]].to_numpy().max()),
        },
        "curvature_panel_contrasts": contrasts,
        "gradient_seed_clustered_contrasts": grad_contrasts,
    }


def retreat_event_contrasts():
    """Event-paired gradient and curvature contrasts at retreat (App. A.2).

    For each retreat of the checkpointed (12, 1280) runs the retreat checkpoint is the first
    with A_block < 0.5 and the last-high checkpoint the last one with A_block >= 0.8 before it.
    The interval is paired with a non-overlapping stable interval of the same seed and the same
    lag in epochs, throughout which A_block stays above 0.8, and a diagnostic X enters through
    DD X = (X_retreat - X_last-high) - (X_stable,end - X_stable,start).

    Every checkpoint is scored on one fixed diagnostic set of 1024 uniform draws from the 2048
    inputs (seed 20_260_820), drawn without reference to the training set (1280 draws with
    TRAIN_SEED): 446 of the 1024 are training configurations.  |grad L|_2 is evaluated at all
    173 checkpoints of the 44 events of 24 seeds; events are averaged within a seed and the
    bootstrap resamples seeds.  lambda_min and lambda_max cost about a minute per checkpoint and
    are evaluated on a panel: the first event of each of 12 seeds taken at equal spacing in the
    sorted seed list; four checkpoints per event give 48 evaluations, and with one event per
    seed resampling events is resampling seeds.  Every interval is the central 95% of 20000
    bootstrap means.

    Reads   data/retreat_events/events.csv (one row per retreat event, with the paths of its
            checkpoints under data/checkpointed_trajectories/ and the lag) and those checkpoints
    Writes  results/retreat_event_contrasts/
                gradient_cache/, curvature_cache/   one JSON per evaluated checkpoint
                checkpoint_gradient.csv             173 rows
                checkpoint_curvature.csv            48 rows
                paired_events_gradient.csv          44 events
                paired_events_curvature.csv         12 panel events
                summary.json                        the contrasts and their intervals
    About half an hour.  A checkpoint whose JSON is present in a cache directory is read back
    and not evaluated again, so an interrupted run continues where it stopped.
    """
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    # two intra-op threads: the last digits of every sum depend on this count
    torch.set_num_threads(2)

    events = pd.read_csv(RETREAT_EVENTS / "events.csv")
    panel = select_panel(events)
    gradient = checkpoint_gradients(events)
    curvature = checkpoint_curvatures(panel)
    summary = event_contrasts(events, panel, curvature, gradient)
    (EVENT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\ncontrasts on {EVENT_DIAGNOSTIC_SET}")
    for metric in ("lambda_min", "lambda_max"):
        d = summary["curvature_panel_contrasts"][metric]["did"]
        print(f"  DD{metric:11s} = {d['mean']:9.4f}  [{d['ci_low']:9.4f}, {d['ci_high']:8.4f}]")
    d = summary["gradient_seed_clustered_contrasts"]["gradient_norm"]["did"]
    print(f"  DD||g||_2     = {d['seed_cluster_mean']:9.4f}  "
          f"[{d['ci_low']:9.4f}, {d['ci_high']:8.4f}]  "
          f"({summary['protocol']['gradient_events']} events, "
          f"{summary['protocol']['gradient_seed_clusters']} seeds)")
    print(f"  wrote {EVENT_DIR/'summary.json'}")


# ---------------------------------------------------------------- App. A.3: the final FFN

FFN_KEY = "blocks.7.ffn.3.weight"     # down-projection of the last block, 16 x 64
FFN_CHECKPOINTS = CHECKPOINTED_TRAJECTORIES / "L12_N1280"
FFN_SEEDS = tuple(range(30))
FFN_DISPLAY_SEEDS = (0, 1, 3, 4, 5)   # the trajectories drawn in Fig. A1
PERMUTATION_DRAWS = 2000
MAX_SUBSPACE_RANK = 16


def ffn_checkpoint_paths(seed: int) -> list[tuple[int, Path]]:
    """(epoch, path) of every stored checkpoint of one training seed, in epoch order."""
    paths = []
    for path in (FFN_CHECKPOINTS / f"seed{seed}").glob("epoch*.pt"):
        paths.append((int(path.stem[len("epoch"):]), path))
    return sorted(paths)


def model_from_state(state: dict) -> ChainTransformer:
    """An L = 12 network of ARCHITECTURE with the given state dict, in eval mode."""
    model = ChainTransformer(L=12, **ARCHITECTURE)
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def logits_and_block(model, inputs: torch.Tensor, target: torch.Tensor):
    """Logits on `inputs` and the block accuracy against `target`."""
    logits = model(inputs)
    block = float((logits > 0).eq(target.bool()).all(dim=1).float().mean())
    return logits.numpy(), block


def top_mode(weight: np.ndarray):
    """sigma_1 and the leading left and right singular vectors of a weight matrix."""
    left, singular, right_t = np.linalg.svd(weight.astype(np.float64), full_matrices=False)
    return float(singular[0]), left[:, 0], right_t[0]


def leading_separation(weight: np.ndarray) -> float:
    """sigma_1 / sigma_2."""
    singular = np.linalg.svd(weight.astype(np.float64), compute_uv=False)
    return float(singular[0] / singular[1])


def subspace_rotation(reference: np.ndarray, retreat_weight: np.ndarray) -> dict:
    """Largest principal angle between the leading rank-k singular subspaces of two matrices,
    k = 1..16, on the left and on the right."""
    ref_left, _, ref_right_t = np.linalg.svd(reference.astype(np.float64), full_matrices=False)
    ret_left, _, ret_right_t = np.linalg.svd(retreat_weight.astype(np.float64), full_matrices=False)
    ref_right = ref_right_t.T
    ret_right = ret_right_t.T
    max_rank = min(MAX_SUBSPACE_RANK, ref_left.shape[1], ret_left.shape[1])
    left_max = []
    right_max = []
    for rank in range(1, max_rank + 1):
        left_cosines = np.linalg.svd(ref_left[:, :rank].T @ ret_left[:, :rank], compute_uv=False)
        right_cosines = np.linalg.svd(ref_right[:, :rank].T @ ret_right[:, :rank], compute_uv=False)
        left_angles = np.degrees(np.arccos(np.clip(left_cosines, 0.0, 1.0)))
        right_angles = np.degrees(np.arccos(np.clip(right_cosines, 0.0, 1.0)))
        left_max.append(float(left_angles.max()))
        right_max.append(float(right_angles.max()))
    return {
        "ranks": list(range(1, max_rank + 1)),
        "left_max_deg": left_max,
        "right_max_deg": right_max,
        "worst_max_deg": [max(left, right) for left, right in zip(left_max, right_max)],
    }


def permutation_ratio(weight: np.ndarray, seed: int) -> float:
    """sigma_1^2 / max(shape) relative to its mean over 2000 entry permutations of the matrix.

    The ratio measures how far the leading mode stands above what the same entries give when
    their arrangement is randomized.  The generator is seeded by the training seed, so the
    peak and the retreat matrix of one pair are shuffled by the same index sequences.
    """
    scale = max(weight.shape)
    observed = np.linalg.svd(weight.astype(np.float64), compute_uv=False)[0] ** 2 / scale
    rng = np.random.default_rng(seed)
    null = []
    entries = weight.ravel()
    for _ in range(PERMUTATION_DRAWS):
        order = rng.permutation(entries.size)
        shuffled = entries[order].reshape(weight.shape)
        null.append(np.linalg.svd(shuffled, compute_uv=False)[0] ** 2 / scale)
    return float(observed / np.mean(null))


def ffn_activations(model, inputs: torch.Tensor):
    """Logits and the input of the last block's down-projection, [inputs, L, 64]."""
    store: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        store["hidden"] = args[0].detach()

    handle = model.blocks[7].ffn[3].register_forward_pre_hook(hook)
    with torch.inference_mode():
        logits = model(inputs)
    handle.remove()
    return logits.numpy(), store["hidden"].numpy()


def walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Normalized Walsh-Hadamard transform of a function on the complete input set, rows in
    binary-index order.

    Coefficient k is E_b[f(b) (-1)^(k . b)], the overlap with the parity character of the
    wall subset k.
    """
    coefficients = np.asarray(values, dtype=np.float64).copy()
    half = 1
    while half < len(coefficients):
        for start in range(0, len(coefficients), 2 * half):
            low = coefficients[start : start + half].copy()
            high = coefficients[start + half : start + 2 * half].copy()
            coefficients[start : start + half] = low + high
            coefficients[start + half : start + 2 * half] = low - high
        half *= 2
    return coefficients / len(coefficients)


def target_parity_fraction(model, inputs: torch.Tensor, target_spin: np.ndarray) -> dict:
    """Bit accuracy and target-parity fraction rho_j of the leading channel at each position.

    The input of the down-projection is projected on its leading right singular vector v_1,
    one scalar per input and position.  Over the complete input set its Walsh-Hadamard transform is
    exact, and rho_j is the share of the non-constant Walsh energy at position j carried by
    the character of the target, the parity of walls 0..j-1.  Position 0 has a constant target
    and no rho.  `target_spin` is 1 - 2 z (+1 where the target bit is 0).
    """
    logits, hidden = ffn_activations(model, inputs)
    _, _, right = top_mode(model.state_dict()[FFN_KEY].numpy())
    channel = hidden.astype(np.float64) @ right
    target_fraction = []
    bit_accuracy = ((logits >= 0) == (target_spin < 0)).mean(axis=0)
    for position in range(12):
        centered = channel[:, position] - channel[:, position].mean()
        energy = walsh_hadamard(centered) ** 2
        energy[0] = 0.0
        # the target at this position is the parity of walls 0..position-1
        target_subset = (1 << position) - 1 if position > 0 else 0
        numerator = float(energy[target_subset])
        target_fraction.append(float(numerator / energy.sum())
                               if position > 0 and energy.sum() > 0 else None)
    return {
        "bit_accuracy": bit_accuracy.tolist(),
        "target_character_fraction": target_fraction,
    }


def leading_mode_ablation(model, inputs: torch.Tensor, target: torch.Tensor,
                          draws: int = 5) -> dict:
    """Per-position loss of bit accuracy on removing the leading mode or a random mode.

    The down-projection W is set to W - sigma_1 u_1 v_1^T with every other parameter fixed
    (Fig. A1(c)); the control removes sigma_1 a b^T for five random unit vectors a, b (generator
    seed 0) and averages the five losses.
    """
    state = model.state_dict()
    weight = state[FFN_KEY].numpy().astype(np.float64)
    singular, left, right = top_mode(weight)
    base_logits, _ = logits_and_block(model, inputs, target)
    base_bit = ((base_logits >= 0) == target.numpy()).mean(axis=0)
    top_state = {key: value.clone() for key, value in state.items()}
    top_state[FFN_KEY] = torch.from_numpy(weight - singular * np.outer(left, right)).float()
    top_logits, _ = logits_and_block(model_from_state(top_state), inputs, target)
    top_drop = base_bit - ((top_logits >= 0) == target.numpy()).mean(axis=0)
    rng = np.random.default_rng(0)
    random_drops = []
    for _ in range(draws):
        left_random = rng.standard_normal(weight.shape[0])
        left_random /= np.linalg.norm(left_random)
        right_random = rng.standard_normal(weight.shape[1])
        right_random /= np.linalg.norm(right_random)
        random_state = {key: value.clone() for key, value in state.items()}
        random_state[FFN_KEY] = torch.from_numpy(
            weight - singular * np.outer(left_random, right_random)).float()
        random_logits, _ = logits_and_block(model_from_state(random_state), inputs, target)
        random_drops.append(base_bit - ((random_logits >= 0) == target.numpy()).mean(axis=0))
    return {"top_drop": top_drop.tolist(), "random_drop_mean": np.mean(random_drops, axis=0).tolist(),
            "singular": singular}


def final_ffn_peak_to_retreat():
    """Peak-to-retreat analysis of the final feed-forward layer (Fig. A1, App. A.3).

    Thirty (12, 1280) networks on the shared training set have their weights stored at every
    epoch from 20 through 220.  Every checkpoint is scored on the complete input set (2^11
    inputs), and per training seed one pair is selected:

        retreat   the first checkpoint with A_block < 0.5 after A_block has reached 0.8
        peak      the checkpoint of highest A_block before the retreat, the earliest on ties

    A seed that never falls below 0.5 after reaching 0.8 gives no pair ("non_retreat_seeds").
    On the down-projection W of the last block (FFN_KEY) the item measures sigma_1 and
    sigma_1/sigma_2 at both checkpoints, the overlaps |u.u'| and |v.v'| of the leading singular
    vectors between peak and retreat, the largest principal angle between the leading rank-k
    singular subspaces (k = 1..16), the permutation ratio against entry permutations (stored
    under "trap_ratio"), the leading-mode ablation of Fig. A1(c) and the target-parity fraction
    rho_j of Fig. A1(b).

    Reads   data/checkpointed_trajectories/L12_N1280/seed<k>/epoch<t>.pt
    Writes  results/final_ffn_peak_to_retreat.json
    All 201 checkpoints of each of the 30 seeds are loaded; about twenty to forty minutes.
    """
    # three intra-op threads: the last digits of the accuracies and of every quantity derived
    # from the captured activations depend on this count
    torch.set_num_threads(3)
    complete_inputs, complete_target = complete_set(12)
    target_spin = 1.0 - 2.0 * complete_target.numpy().astype(np.float64)
    all_results: dict[str, Any] = {}
    for seed in FFN_SEEDS:
        trajectory = []
        models: dict[int, ChainTransformer] = {}
        for epoch, path in ffn_checkpoint_paths(seed):
            model = load_chain_transformer(path, 12).eval()
            _, block = logits_and_block(model, complete_inputs, complete_target)
            trajectory.append({"epoch": epoch, "block_accuracy": block})
            models[epoch] = model
        learned_epochs = [row["epoch"] for row in trajectory if row["block_accuracy"] >= 0.8]
        retreat_epochs = [row["epoch"] for row in trajectory
                          if learned_epochs and row["epoch"] > min(learned_epochs)
                          and row["block_accuracy"] < 0.5]
        if not retreat_epochs:
            continue
        retreat_epoch = retreat_epochs[0]
        preceding = [row for row in trajectory if row["epoch"] < retreat_epoch]
        reference = max(preceding, key=lambda row: row["block_accuracy"])
        reference_epoch = int(reference["epoch"])
        ref_model = models[reference_epoch]
        ret_model = models[retreat_epoch]
        ref_weight = ref_model.state_dict()[FFN_KEY].numpy()
        ret_weight = ret_model.state_dict()[FFN_KEY].numpy()
        ref_s, ref_u, ref_v = top_mode(ref_weight)
        ret_s, ret_u, ret_v = top_mode(ret_weight)
        rotation = subspace_rotation(ref_weight, ret_weight)
        ref_channel = target_parity_fraction(ref_model, complete_inputs, target_spin)
        ret_channel = target_parity_fraction(ret_model, complete_inputs, target_spin)
        ref_ablation = leading_mode_ablation(ref_model, complete_inputs, complete_target)
        ret_ablation = leading_mode_ablation(ret_model, complete_inputs, complete_target)
        all_results[str(seed)] = {
            "reference_epoch": reference_epoch,
            "retreat_epoch": retreat_epoch,
            "reference_block_accuracy": reference["block_accuracy"],
            "retreat_block_accuracy": next(row["block_accuracy"] for row in trajectory
                                           if row["epoch"] == retreat_epoch),
            "trajectory": trajectory,
            "subspace_rotation": rotation,
            "reference": {
                "singular": ref_s,
                "trap_ratio": permutation_ratio(ref_weight, seed=seed),
                "s1_over_s2": leading_separation(ref_weight),
                "u_overlap_to_retreat": float(abs(ref_u @ ret_u)),
                "v_overlap_to_retreat": float(abs(ref_v @ ret_v)),
                "ablation": ref_ablation,
                "channel": ref_channel,
            },
            "retreat": {
                "singular": ret_s,
                "trap_ratio": permutation_ratio(ret_weight, seed=seed),
                "s1_over_s2": leading_separation(ret_weight),
                "ablation": ret_ablation,
                "channel": ret_channel,
            },
        }
    output = RESULTS / "final_ffn_peak_to_retreat.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "L": 12,
            "N_train": 1280,
            "train_seed": TRAIN_SEED,
            "eval_set": "cube",
            "architecture": "ChainTransformer channels=16 depth=8 heads=4 ffn_dim=64",
            "checkpoint_cadence": "every epoch from 20 through 220",
            "learned_selector": "first saved A_block >= 0.8",
            "retreat_selector": "first saved post-learning A_block < 0.5",
            "peak_selector": "earliest pre-retreat checkpoint with maximal A_block",
            "ffn_matrix": FFN_KEY,
            "permutation_null": (
                f"{PERMUTATION_DRAWS} entry permutations; each peak/retreat "
                "pair uses the same permutation indices"
            ),
            "walsh_probe": "complete clean cube of 2**11 inputs; position 0 is constant and omitted from target-character fraction",
            "requested_seeds": list(FFN_SEEDS),
            "trajectory_display_seeds": list(FFN_DISPLAY_SEEDS),
            "retreat_pair_count": len(all_results),
            "non_retreat_seeds": [seed for seed in FFN_SEEDS if str(seed) not in all_results],
        },
        "seeds": all_results,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({seed: {key: value for key, value in data.items()
                             if key.endswith("epoch") or key.endswith("accuracy")}
                      for seed, data in all_results.items()}, indent=2))


# ===========================================================================
# Command line
# ===========================================================================

# (function, what it writes, where the paper uses it), in dependency order:
# retreat_matrices_norm_filter reads what retreat_matrices_l12_first400 writes.
ITEMS = [
    (gf2_identifiability, "results/gf2_identifiability.json", "Sec. 4.2"),
    (endpoint_prevalence, "results/endpoint_prevalence.json", "Table C1"),
    (train_heldout_crossing_epochs, "results/train_heldout_crossing_epochs.csv", "Secs. 1, 3"),
    (block_vs_bit_centered_correlation, "results/block_vs_bit_centered_correlation.json",
     "App. B"),
    (nishimori_gap_small_data, "results/nishimori_gap_small_data.json", "Sec. 3.3"),
    (trajectory_ensemble_retreat, "results/trajectory_ensemble_retreat.json", "Secs. 3.4, 4.2"),
    (intervention_pair_outcomes, "results/intervention_pair_outcomes.json", "Sec. 5.2"),
    (tail_observables, "results/tail_observables.csv", "Fig. 5, Sec. 3.5"),
    (retreat_matrices_l12_first400, "results/retreat_matrices/L12_N1280_first400.{npz,json}",
     "Fig. 7"),
    (retreat_matrices_norm_filter,
     "results/retreat_matrices/L12_N1280_first400_norm_filtered.{npz,json}", "Fig. 7"),
    (retreat_matrices_l16_n2048, "results/retreat_matrices/L16_N2048.npz", "Fig. 7"),
    (tree_permutation_references, "results/tree_permutation_references.json", "Sec. 4.2"),
    (retreat_definition_controls, "results/retreat_definition_controls.json", "Sec. 4.2"),
    (k4_cut_equal_count, "results/k4_cut_equal_count.json", "Sec. 4.2"),
    (retreat_obtuse_pair_fraction, "results/retreat_obtuse_pair_fraction.json", "Sec. 4.3"),
    (split_half_obtuse, "results/split_half_obtuse.json", "Sec. 4.3"),
    (normalized_residuals, "results/normalized_residuals.npz", "Figs. 8, 9, Sec. 4.3"),
    (replica_set_geometry, "results/replica_set_geometry.json",
     "Table 3, Figs. 6, 8, 9, Secs. 4.2, 4.3"),
    (tail_geometry, "results/tail_geometry.json", "Table C3"),
    (truth_scalar_correlation, "results/truth_scalar_correlation.json", "Sec. 4.2"),
    (evaluation_outside_training_set, "results/evaluation_outside_training_set.json", "Sec. 4.2"),
    (fixed_epoch_comparison, "results/fixed_epoch_comparison/", "App. A.1, A.2"),
    (retreat_event_contrasts, "results/retreat_event_contrasts/", "App. A.2"),
    (final_ffn_peak_to_retreat, "results/final_ffn_peak_to_retreat.json", "Fig. A1, App. A.3"),
]


if __name__ == "__main__":
    sys.exit(run_items(ITEMS, str(Path(__file__).resolve()), __doc__.split("\n\n")[0]))
