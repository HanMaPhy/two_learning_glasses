#!/usr/bin/env python3
"""The figures of "Replica Fragmentation and Glassy Dynamics in Parity Learning", and the five
tables that are computed together with them.

One function per figure, named after the file it writes.  Each reads data/ or a table that
analysis.py writes under results/, and writes its PDF under figure/;
figC1_raw_fragmentation_frontier_order draws two figures.  Four functions also write tables under
results/: the logistic fits of Fig. 1, the summary of Fig. B1, the gap decomposition and
bootstrap intervals behind Figs. C1 and C2, and Table C2 with the window
averages of Fig. C3.  Nothing here trains a network.  The task, the network, the loaders, the
endpoint classes and the overlap matrices are defined in common.py, whose docstring gives the
notation used below.

    python3 figures.py --list               every item, what it writes, where the paper uses it
    python3 figures.py NAME [NAME ...]      draw the named figures in this process
    python3 figures.py --all                every figure, each in its own process (a few minutes)

Figs. 5, 6, 7, 8, 9 and A1 are drawn from tables in results/, so the analysis items that write
those tables (named in each docstring) have to have run first; the other nine figures read
data/ directly.  The BLAS thread variables (OMP, MKL, OPENBLAS, VECLIB, NUMEXPR) default to 1
before NumPy is imported, as in analysis.py.  figC3_small_data_diagnosis sets the intra-op thread
count of torch to four, because the last digits of its tables depend on it, and the count is
set back after every item run in one process.  Every figure that changes rcParams does so
inside plt.rc_context, so no setting reaches the next figure drawn in the same process; the
figures drawn with the default rcParams rely on that.  --all starts every item as a separate
process.

===========================================================================
ACCURACIES, EVENTS AND THE CROSSOVER (Figs. 1, 2, 10, B1)
===========================================================================
For one network on one evaluation set, A_bit is the fraction of correct output bits and
A_block the fraction of inputs with all L output bits correct.  A trajectory acquires the rule
at t_0.8, the first epoch with A_block >= 0.8, and retreats when A_block later falls below 0.5
(common.first_crossing, common.first_fall).
If the L outputs failed independently with the mean accuracy, A_block would be (A_bit)^L;
Fig. B1 tests this estimate.

In a training-set-size scan every run draws its own N strings, trains for 120 epochs and
succeeds if its held-out A_block reaches 0.8.  The fraction P(N) of successful runs at
each N is the finite-budget learning probability.  It is fitted by a logistic in log N,

    P(N) = 1 / (1 + exp(-(log N - log N_c) / w)),

by binomial maximum likelihood on the success and failure counts: N_c is the crossover and w
its width in log N.  The error bars are Wilson score intervals at z = 1.

===========================================================================
OVERLAPS ALONG A TRAJECTORY (Figs. 2, 3, 4)
===========================================================================
With the soft spin u = tanh(logit / 2) and the truth spin y of common.py, and E the average
over the evaluation inputs and the output positions of one network,

    m                   = E[u y]          truth overlap (Mattis magnetization)
    q_self              = E[u^2]          self-overlap
    Delta_N             = q_self - m      Nishimori gap
    f_uncertain(zeta)   = P(|u| < zeta)   fraction of uncertain outputs

For the posterior mean of a Bayes-optimal estimator the Nishimori identity gives q_self = m, so
Delta_N > 0 is confidence in excess of accuracy.  Resolved by output position j, the profiles
m(j) and q_self(j) fall along the chain.  A frontier is the linearly interpolated position at
which such a profile crosses one half (frontier_position): j*_m for m(j), j*_q for q_self(j).

===========================================================================
REPLICA ENSEMBLES (Figs. 5 to 9, C1 to C3)
===========================================================================
The fourteen replica sets, their windows and the replica matrices m_r, Q^raw, Q^c, C^prof, W,
R^W and gap(A) are those of common.py.  q_cross is the mean off-diagonal entry of Q^raw and
chi_SG = q_self - q_cross the self-cross gap.  The tree statistics of Figs. 8 and 9 are the
ones analysis.replica_set_geometry measures on the distance d = 1 - R^W on random subsets of
n = 47 replicas: R^2_tree, the fit of the average-linkage tree, and P(delta_um < 0.1), the
fraction of replica triples whose sorted similarities q1 <= q2 <= q3 have
(q2 - q1) / (q3 - q1) < 0.1.  The obtuse-pair fraction P(R^W < 0) uses every replica
(common.obtuse_pair_fraction).

===========================================================================
PAPER LOCATION -> FUNCTION
===========================================================================
    Fig. 1, Sec. 3.1                    fig1_generalization_crossover
    Fig. 2, Sec. 3.2                    fig2_shared_training_set_diagnostics
    Fig. 3, Sec. 3.3                    fig3_nishimori_gap_dynamics
    Fig. 4, Sec. 3.4                    fig4_position_resolved_retreat_recovery
    Fig. 5, Sec. 3.5                    fig5_tail_observables
    Figs. 6, 7, Sec. 4.2                fig6_raw_overlaps_and_gaps, fig7_retreat_residual_tree
    Fig. 8, Sec. 4.3                    fig8_fragmentation_residual_organization
    Fig. 9, Sec. 5.1                    fig9_two_planes_common_tail
    Fig. 10, Sec. 5.2                   fig10_learning_rate_intervention
    Fig. A1, App. A.3                   figA1_final_layer_ffn
    Fig. B1, App. B                     figB1_bit_and_block_accuracy
    Sec. 4.2; Figs. C1, C2, App. C.2    figC1_raw_fragmentation_frontier_order
    Table C2, App. C.3                  figC3_small_data_diagnosis
    Fig. C3, App. C.4                   figC3_small_data_diagnosis
"""
from __future__ import annotations

import os

# One BLAS thread unless the caller sets otherwise; this has to precede the NumPy import.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")

import csv
import glob
import json
import math
import re
import sys
from collections import defaultdict
from itertools import chain, combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from common import (
    ARCHITECTURE, CNN_SIZE_SCAN, EPOCHS, FIGURE, HI, LEARNING_RATE_INTERVENTION, LO,
    REPRESENTATIVE_CNN_RUNS, REPRESENTATIVE_TRANSFORMER_RUNS, RESULTS, ROOT, SMALL_DATA_ENSEMBLES,
    TRAIN_SEED, TRAJECTORY_ENSEMBLES, TRAJECTORY_EPOCHS, TRANSFORMER_SIZE_SCAN,
    crossing_epoch, crossing_outcome, ends_high, first_crossing, first_fall, held_out_set,
    history_crossing, load_chain_transformer, load_history, load_l12_n1280_in_storage_order,
    normalize_by_diagonal, obtuse_pair_fraction, overlap_matrices, retreat, run_items, sample_set,
    wall_codes, write_rows,
)


def output_path(directory: Path, name: str) -> Path:
    """directory / name, creating the directory if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def frontier_position(profile: np.ndarray) -> float:
    """The position at which a profile over the output positions crosses one half.

    The profile is high at small j and low at large j.  The crossing is found by linear
    interpolation between positions, so it is not the last position above one half.  A
    profile at or above one half everywhere gives the number of positions, one below one half
    everywhere gives zero.  Used for the frontiers of Fig. 4 and the frontier order of
    Fig. C1(d).
    """
    length = profile.size
    if np.all(profile >= 0.5):
        return float(length)
    if np.all(profile < 0.5):
        return 0.0
    return float(np.interp(0.5, profile[::-1], np.arange(length)[::-1]))


# ===========================================================================
# Fig. 1: the generalization crossover of the fixed-training-set scans (Sec. 3.1)
# ===========================================================================

CNN_SEED = 3                 # the 600-epoch CNN run of Fig. 1(a), N = 2048
TRANSFORMER_SEED = 0         # the 600-epoch Transformer run of Figs. 1(a) and B1(a), N = 1280
CNN_COLOR = "#167d4a"
TRANSFORMER_COLOR = "#c43c35"
LENGTH_STYLE = [(12, "#c43c35", "o"), (16, "#7b3fa0", "s"), (20, "#1f6fb4", "^")]


def scan_success_fractions(root: Path, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The training-set sizes N of one scan, the fraction of successful runs at each N and the
    number of runs at each N.  A run succeeds if its held-out A_block reaches HI = 0.8."""
    pattern = re.compile(rf"L{length}_N(\d+)_seed(\d+)$")
    by_n: dict[int, list[bool]] = defaultdict(list)
    for path in root.glob(f"L{length}_N*_seed*/metrics.json"):
        parsed = pattern.fullmatch(path.parent.name)
        if parsed is None:
            continue
        history = load_history(path)
        if history:
            by_n[int(parsed.group(1))].append(history_crossing(history) is not None)
    if not by_n:
        raise FileNotFoundError(f"no fixed-training-set runs found under {root}")
    ns = np.asarray(sorted(by_n), dtype=float)
    successes = np.asarray([sum(by_n[int(n)]) for n in ns], dtype=float)
    totals = np.asarray([len(by_n[int(n)]) for n in ns], dtype=float)
    return ns, successes / totals, totals


def logistic_fit(ns: np.ndarray, probabilities: np.ndarray,
                 totals: np.ndarray) -> tuple[float, float]:
    """(N_c, w) of the logistic in log N, by binomial maximum likelihood (Nelder-Mead).

    The parameters are (log N_c, log w), started at the median of log N and log 0.5.  The
    logits are clipped at +-40 and the probabilities shifted by 1e-12 inside the logarithms.
    """
    successes = probabilities * totals
    log_ns = np.log(ns)

    def negative_log_likelihood(params: np.ndarray) -> float:
        center, log_width = params
        width = math.exp(float(log_width))
        logits = np.clip((log_ns - center) / width, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        return float(-np.sum(successes * np.log(p + eps)
                             + (totals - successes) * np.log(1.0 - p + eps)))

    result = minimize(negative_log_likelihood, np.asarray([np.median(log_ns), math.log(0.5)]),
                      method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(result.message)
    return math.exp(float(result.x[0])), math.exp(float(result.x[1]))


def wilson_errors(probability: np.ndarray, totals: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Lower and upper arm of the Wilson score interval at z, the interval clipped to [0, 1]."""
    denominator = 1.0 + z**2 / totals
    center = (probability + z**2 / (2.0 * totals)) / denominator
    half = z / denominator * np.sqrt(probability * (1.0 - probability) / totals
                                     + z**2 / (4.0 * totals**2))
    lower = np.clip(center - half, 0.0, 1.0)
    upper = np.clip(center + half, 0.0, 1.0)
    return np.vstack((np.maximum(probability - lower, 0.0), np.maximum(upper - probability, 0.0)))


def fit_scan(root: Path, length: int) -> dict:
    ns, probabilities, totals = scan_success_fractions(root, length)
    nc, width = logistic_fit(ns, probabilities, totals)
    return {"ns": ns, "p": probabilities, "totals": totals, "nc": nc, "w": width}


def annotate_transformer_events(ax: plt.Axes, history: list[dict], color: str) -> None:
    """Mark the acquisition and the retreat of the 600-epoch Transformer run of Fig. 1(a).

    The acquisition is the first epoch with A_block >= 0.8.  The retreat is the epoch after it
    with the deepest drop below the running maximum of A_block, marked if that drop is at
    least 0.2.
    """
    epochs = np.asarray([int(row["epoch"]) for row in history])
    block = np.asarray([float(row["block_acc"]) for row in history])
    first = first_crossing(block)
    ax.annotate("acquisition", xy=(epochs[first], block[first]),
                xytext=(epochs[first] + 22, 0.62), fontsize=8.3, color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.9})
    peak = np.maximum.accumulate(block)
    drawdown = peak - block
    post = np.arange(len(block)) >= first
    deepest = int(np.argmax(np.where(post, drawdown, -np.inf)))
    if drawdown[deepest] >= 0.2:
        ax.annotate("collapse", xy=(epochs[deepest], block[deepest]),
                    xytext=(epochs[deepest] - 48, 0.22), fontsize=8.3, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                    arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.9})


def draw_representative_runs(ax: plt.Axes) -> None:
    """Fig. 1(a): held-out A_block of the two 600-epoch runs at L = 12."""
    cnn = load_history(REPRESENTATIVE_CNN_RUNS / f"seed{CNN_SEED}" / "metrics.json")
    ax.plot([int(row["epoch"]) for row in cnn], [float(row["block_acc"]) for row in cnn],
            color=CNN_COLOR, linewidth=2.4, label=f"CNN seed {CNN_SEED}, N=2048")
    transformer = load_history(REPRESENTATIVE_TRANSFORMER_RUNS / f"seed{TRANSFORMER_SEED}"
                               / "metrics.json")
    ax.plot([int(row["epoch"]) for row in transformer],
            [float(row["block_acc"]) for row in transformer],
            color=TRANSFORMER_COLOR, linewidth=2.4,
            label=f"Transformer seed {TRANSFORMER_SEED}, N=1280")
    annotate_transformer_events(ax, transformer, TRANSFORMER_COLOR)
    ax.axhline(HI, color="0.35", linestyle=":", linewidth=1.0)
    ax.set(xlabel="epoch", ylabel="held-out block accuracy", ylim=(-0.03, 1.03), xlim=(0, 600),
           title="(a) Held-out block-accuracy trajectories at L=12")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=9.5)


def draw_crossover(ax: plt.Axes, scan: dict, color: str, marker: str, alpha: float,
                   margin: float, label: str) -> None:
    """Measured fractions with Wilson arms, the fitted logistic over the scanned range of N
    widened by the factor `margin` on both sides, and a dotted line at N_c."""
    ns, nc, width = scan["ns"], scan["nc"], scan["w"]
    ax.errorbar(ns, scan["p"], yerr=wilson_errors(scan["p"], scan["totals"]), color=color,
                marker=marker, markersize=4.5, linewidth=0, elinewidth=0.9, capsize=2, alpha=alpha)
    grid = np.geomspace(ns.min() / margin, ns.max() * margin, 300)
    ax.plot(grid, 1.0 / (1.0 + np.exp(-(np.log(grid) - math.log(nc)) / width)), color=color,
            linewidth=2.2, label=label)
    ax.axvline(nc, color=color, linestyle=":", linewidth=1.0, alpha=0.8)


def finish_probability_panel(ax: plt.Axes, title: str) -> None:
    ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set(xlabel="training-set size N", ylabel="finite-budget learning probability",
           ylim=(-0.04, 1.04), title=title)
    ax.grid(alpha=0.18, which="both")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)


def fig1_generalization_crossover():
    """Fig. 1: generalization with independently sampled fixed training sets (Sec. 3.1).

    (a) Held-out A_block against epoch for one 600-epoch CNN run (N = 2048, seed 3) and one
        600-epoch Transformer run (N = 1280, seed 0) at L = 12, with the acquisition and the
        retreat of the Transformer run marked (annotate_transformer_events).
    (b) The finite-budget learning probability P(N) at L = 12 for the CNN and the Transformer.
    (c) The Transformer at L = 12, 16 and 20; the L = 12 curve is the one of (b).

    Points are the fractions of successful runs at each N with Wilson intervals at z = 1, curves
    the logistics in log N fitted by binomial maximum likelihood, and dotted vertical lines the
    fitted N_c (module docstring).  Two logistics with centers mu_1, mu_2 and widths w_1, w_2
    cross at log N* = (mu_1 w_2 - mu_2 w_1) / (w_2 - w_1).  The function prints the three
    pairwise crossings of the curves in (c); none lies where both curves are measured.

    Reads   metrics.json of every run of common.CNN_SIZE_SCAN and common.TRANSFORMER_SIZE_SCAN,
            and of the two 600-epoch runs under common.REPRESENTATIVE_CNN_RUNS and
            common.REPRESENTATIVE_TRANSFORMER_RUNS
    Writes  figure/fig1_generalization_crossover.pdf
            results/generalization_crossover_fits.csv: per curve the architecture, L, the number
            of runs, the range of N and the fitted N_c and w, the N_c quoted in Sec. 3.1
    """
    scans = {("cnn", 12): fit_scan(CNN_SIZE_SCAN, 12)}
    for length, _, _ in LENGTH_STYLE:
        scans[("transformer", length)] = fit_scan(TRANSFORMER_SIZE_SCAN[length], length)

    out_pdf = output_path(FIGURE, "fig1_generalization_crossover.pdf")
    with plt.rc_context({"font.size": 11.5}):
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0), constrained_layout=True)
        draw_representative_runs(axes[0])
        for architecture, color, label in (("cnn", CNN_COLOR, "CNN"),
                                           ("transformer", TRANSFORMER_COLOR, "Transformer")):
            draw_crossover(axes[1], scans[(architecture, 12)], color, marker="o", alpha=0.8,
                           margin=1.0, label=f"{label} logistic guide")
        finish_probability_panel(axes[1], "(b) Finite-budget crossover at L=12")
        for length, color, marker in LENGTH_STYLE:
            scan = scans[("transformer", length)]
            draw_crossover(axes[2], scan, color, marker=marker, alpha=0.85, margin=1.5,
                           label=rf"$L={length}$:  $N_c={scan['nc']:.0f}$,  $w={scan['w']:.2f}$")
        finish_probability_panel(axes[2], "(c) The crossover moves right with string length")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    out_csv = output_path(RESULTS, "generalization_crossover_fits.csv")
    write_rows(out_csv, [{"architecture": architecture, "L": length,
                          "n_runs": int(scan["totals"].sum()),
                          "N_min": int(scan["ns"].min()), "N_max": int(scan["ns"].max()),
                          "logistic_Nc": scan["nc"], "width_log_N": scan["w"]}
                         for (architecture, length), scan in scans.items()])
    print("wrote", out_pdf)
    print("wrote", out_csv)

    print("\npairwise crossings of the fitted curves in (c):")
    for a, b in ((12, 16), (12, 20), (16, 20)):
        m1, w1 = math.log(scans[("transformer", a)]["nc"]), scans[("transformer", a)]["w"]
        m2, w2 = math.log(scans[("transformer", b)]["nc"]), scans[("transformer", b)]["w"]
        x = (m1 * w2 - m2 * w1) / (w2 - w1)
        pr = 1.0 / (1.0 + np.exp(-(x - m1) / w1))
        print(f"  L={a} vs L={b}:  N* = {math.exp(x):.3g}   at probability {pr:.4f}")


# ===========================================================================
# Figs. 2, 3: the shared-training-set trajectories (Secs. 3.2, 3.3)
# ===========================================================================
# Sixteen Transformer replicas at L = 12, N = 1280 train for 600 epochs on one shared training
# set and differ only in initialization and minibatch order.  Each seed<i>/history.csv holds
# one row per epoch with the columns
#
#     epoch, loss, bit_acc, block_acc, independent_block_acc, m, q_self, delta_N,
#     uncertain_frac_0p5, uncertain_frac_0p2, certainty_0p5, certainty_0p2
#
# every column except loss measured on the complete set of 2048 inputs (common.complete_set):
# A_bit, A_block, the product prod_j A_j of the per-position accuracies, m, q_self, Delta_N,
# f_uncertain(zeta) and 1 - f_uncertain(zeta) for zeta = 0.5, 0.2.

OVERLAP_TRAJECTORIES = TRAJECTORY_ENSEMBLES / "L12_N1280_overlaps"
OVERLAP_TRAJECTORIES_N64 = TRAJECTORY_ENSEMBLES / "L12_N64_overlaps" / "per_seed_epoch_metrics.csv"

# Text sizes of Figs. 2 and 3; TrueType fonts are embedded in the PDF.
TRAJECTORY_RC = {"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
                 "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42}


def load_histories(data_dir: Path, seed_indices) -> list[tuple[int, pd.DataFrame]]:
    """[(seed index, history frame), ...] of the requested replicas.

    A history that does not cover epochs 1 to 600 is refused, so a figure is never drawn from a
    run trained for another number of epochs.
    """
    frames = []
    for seed_index in seed_indices:
        history_path = data_dir / f"seed{seed_index}" / "history.csv"
        frame = pd.read_csv(history_path)
        if len(frame) != TRAJECTORY_EPOCHS or int(frame["epoch"].iloc[-1]) != TRAJECTORY_EPOCHS:
            raise RuntimeError(f"incomplete history: {history_path}")
        frames.append((seed_index, frame))
    return frames


def acquisition_epoch(frame: pd.DataFrame):
    """t_0.8 of one history: the first epoch with A_block >= 0.8, or None."""
    return crossing_epoch(frame["epoch"], frame["block_acc"])


def plot_mean_sem(ax, aligned, key, label, color, linestyle="-", linewidth=2.0):
    """The aligned ensemble mean of one observable with its standard-error band."""
    x = aligned["tau"].to_numpy(dtype=float)
    mean = aligned[f"{key}_mean"].to_numpy(dtype=float)
    sem = aligned[f"{key}_sem"].to_numpy(dtype=float)
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.14, linewidth=0)


HIGHLIGHTED_SEEDS = (0, 1, 2)
HIGHLIGHT_COLORS = {0: "#0072b2", 1: "#d55e00", 2: "#009e73"}
REPRESENTATIVE_SEED = 1

# The trajectories of panel (a) leave no room for a legend inside the axes: the rise fills the
# left half and the plateau near 1 the right half.  Its legend sits just above the frame as one
# right-aligned row.  This pad lifts the titles of both top panels by the same amount, so that
# the legend fits between the frame and the title and the two titles stay on one line.
TOP_TITLE_PAD = 26.0

# Upper-right corner of the panel (c) legend, in axes fractions: the lowest-right position at
# which the single-column legend box covers no curve.
PANEL_C_LEGEND_ANCHOR = (0.9900, 0.3589)


def fig2_shared_training_set_diagnostics():
    """Fig. 2: hidden progress before the first block-accuracy crossing (Sec. 3.2).

    (a) A_block of the sixteen replicas against epoch; seeds 0, 1 and 2 are highlighted and
        labelled with their t_0.8.
    (b) Ensemble means of A_block, of prod_j A_j and of A_bit on the aligned time axis
        tau_0.8 = t - t_0.8.
    (c) m, q_self, Delta_N and A_block of seed 1 against epoch, with its t_0.8 marked.
    (d) Ensemble means of m, q_self, 1 - f_uncertain(0.5) and Delta_N on the aligned axis.

    The bands in (b) and (d) are standard errors across the replicas.

    Reads   data/trajectory_ensembles/L12_N1280_overlaps/seed{0..15}/history.csv
            data/trajectory_ensembles/L12_N1280_overlaps/aligned_summary.csv: for every tau_0.8
            from -60 to 70 the number n of replicas and the mean and standard error of each
            observable across them
    Writes  figure/fig2_shared_training_set_diagnostics.pdf
    """
    frames = load_histories(OVERLAP_TRAJECTORIES, range(16))
    aligned = pd.read_csv(OVERLAP_TRAJECTORIES / "aligned_summary.csv")
    print(f"loaded {len(frames)} replicas, aligned rows {len(aligned)}, n at tau=0 "
          f"{int(aligned.loc[aligned['tau'] == 0, 'n'].iloc[0])}")

    output = output_path(FIGURE, "fig2_shared_training_set_diagnostics.pdf")
    with plt.rc_context(TRAJECTORY_RC):
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), constrained_layout=True)
        ax_raw, ax_aligned_acc, ax_rep, ax_aligned_hidden = axes.ravel()

        for seed_index, frame in frames:
            highlighted = seed_index in HIGHLIGHTED_SEEDS
            ax_raw.plot(frame["epoch"], frame["block_acc"],
                        color=HIGHLIGHT_COLORS.get(seed_index, "0.65"),
                        linewidth=1.8 if highlighted else 0.9, alpha=0.9 if highlighted else 0.45,
                        label=(f"seed {seed_index}; t0.8={acquisition_epoch(frame)}"
                               if highlighted else None))
        ax_raw.axhline(HI, color="0.2", linestyle="--", linewidth=0.9)
        ax_raw.set_title("(a) Shared-training-set block accuracy", pad=TOP_TITLE_PAD)
        ax_raw.set_xlabel("epoch")
        ax_raw.set_ylabel("block accuracy")
        ax_raw.set_ylim(-0.03, 1.03)

        plot_mean_sem(ax_aligned_acc, aligned, "block_acc", "block accuracy", "#111111")
        plot_mean_sem(ax_aligned_acc, aligned, "independent_block_acc",
                      r"$\prod_{j=0}^{L-1}A_j$", "#009e73", linestyle=":")
        plot_mean_sem(ax_aligned_acc, aligned, "bit_acc", "bit accuracy", "0.45", linestyle="--")
        ax_aligned_acc.axvline(0, color="#cc0000", linestyle="--", linewidth=1.0)
        ax_aligned_acc.axhline(HI, color="0.2", linestyle="--", linewidth=0.8)
        ax_aligned_acc.set_title("(b) Aligned by first 0.8 crossing", pad=TOP_TITLE_PAD)
        ax_aligned_acc.set_xlabel(r"$\tau_{0.8}=t-t_{0.8}$")
        ax_aligned_acc.set_ylabel("aligned mean")
        ax_aligned_acc.set_ylim(-0.03, 1.03)

        representative = dict(frames)[REPRESENTATIVE_SEED]
        ax_rep.plot(representative["epoch"], representative["m"], label=r"$m$", color="#009e73")
        ax_rep.plot(representative["epoch"], representative["q_self"], label=r"$q_{\rm self}$",
                    color="#e69f00")
        ax_rep.plot(representative["epoch"], representative["delta_N"], label=r"$\Delta_N$",
                    color="#cc79a7")
        ax_rep.plot(representative["epoch"], representative["block_acc"], label="block accuracy",
                    color="black", linestyle=":", linewidth=1.5)
        ax_rep.axvline(acquisition_epoch(representative), color="#cc0000", linestyle="--",
                       linewidth=1.0)
        ax_rep.set_title(f"(c) Representative run (seed {REPRESENTATIVE_SEED})")
        ax_rep.set_xlabel("epoch")
        ax_rep.set_ylabel("observable value")
        ax_rep.set_ylim(-0.08, 1.03)

        plot_mean_sem(ax_aligned_hidden, aligned, "m", r"$m$", "#009e73")
        plot_mean_sem(ax_aligned_hidden, aligned, "q_self", r"$q_{\rm self}$", "#e69f00")
        plot_mean_sem(ax_aligned_hidden, aligned, "certainty_0p5",
                      r"$1-f_{\rm uncertain}(\zeta=0.5)$", "#56b4e9")
        plot_mean_sem(ax_aligned_hidden, aligned, "delta_N", r"$\Delta_N$", "#cc79a7")
        ax_aligned_hidden.axvline(0, color="#cc0000", linestyle="--", linewidth=1.0)
        ax_aligned_hidden.axhline(0, color="0.35", linewidth=0.7)
        ax_aligned_hidden.set_title("(d) Aligned overlap and certainty observables")
        ax_aligned_hidden.set_xlabel(r"$\tau_{0.8}=t-t_{0.8}$")
        ax_aligned_hidden.set_ylabel("aligned mean")
        ax_aligned_hidden.set_ylim(-0.08, 1.03)

        ax_raw.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
                      borderaxespad=0.3, columnspacing=1.6)
        ax_aligned_acc.legend(frameon=False)
        ax_rep.legend(frameon=False, ncol=1, loc="upper right",
                      bbox_to_anchor=PANEL_C_LEGEND_ANCHOR)
        ax_aligned_hidden.legend(frameon=False, ncol=2)

        fig.suptitle("Transformer diagnostics (L=12, N=1280)", fontsize=12.5)
        fig.savefig(output)
        plt.close(fig)
    print(f"wrote {output}")


LARGE_N_SEEDS = (0, 1)        # the two N = 1280 replicas of the left column
SMALL_N_SEEDS = (0, 6)        # the two N = 64 replicas of the right column
DISPLAYED_EPOCH_MAX = 350
PEAK_THRESHOLD = 0.05


def local_peak_indices(values: np.ndarray, threshold: float) -> list[int]:
    """Indices of the interior local maxima of `values` that reach `threshold`.

    A point counts when it is not below either neighbour, so both ends of a flat top count.
    """
    peaks = []
    for index in range(1, len(values) - 1):
        if (values[index] >= threshold and values[index] >= values[index - 1]
                and values[index] >= values[index + 1]):
            peaks.append(index)
    return peaks


def fig3_nishimori_gap_dynamics():
    """Fig. 3: Nishimori-gap dynamics at large and at small training-set size (Sec. 3.3).

    Every curve is measured on the complete set of 2048 inputs at L = 12 and shown over epochs
    1 to 350.  The top row is Delta_N = q_self - m, the bottom row A_block; the columns share
    their axes.

    Left column: seeds 0 and 1 of the N = 1280 shared-training-set ensemble of Fig. 2.  A dot
    marks every local maximum of Delta_N that reaches 0.05, and a dotted vertical line carries
    its epoch into the A_block panel: the gap opens in short peaks at a retreat and closes
    within a few epochs.
    Right column: two N = 64 replicas, seeds 0 and 6, on one shared training set.  Delta_N stays
    positive while A_block stays low.

    Reads   data/trajectory_ensembles/L12_N1280_overlaps/seed{0,1}/history.csv
            data/trajectory_ensembles/L12_N64_overlaps/per_seed_epoch_metrics.csv, one row per
            seed and epoch with the same observables
    Writes  figure/fig3_nishimori_gap_dynamics.pdf
    """
    large_frames = load_histories(OVERLAP_TRAJECTORIES, LARGE_N_SEEDS)
    small_frame = pd.read_csv(OVERLAP_TRAJECTORIES_N64).sort_values(["seed", "epoch"])

    output = output_path(FIGURE, "fig3_nishimori_gap_dynamics.pdf")
    with plt.rc_context(TRAJECTORY_RC):
        large_map = dict(large_frames)
        colors = ("#0072b2", "#d55e00")
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.4), sharex="col", sharey="row",
                                 constrained_layout=True)

        for color, seed_index in zip(colors, LARGE_N_SEEDS):
            frame = large_map[seed_index]
            local = frame.loc[frame["epoch"] <= DISPLAYED_EPOCH_MAX].copy()
            axes[0, 0].plot(local["epoch"], local["delta_N"], color=color, linewidth=1.7,
                            label=f"seed {seed_index}")
            axes[1, 0].plot(local["epoch"], local["block_acc"], color=color, linewidth=1.7,
                            label=f"seed {seed_index}")
            for peak_index in local_peak_indices(local["delta_N"].to_numpy(), PEAK_THRESHOLD):
                peak = local.iloc[peak_index]
                epoch = float(peak["epoch"])
                for axis in axes[:, 0]:
                    axis.axvline(epoch, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
                axes[0, 0].scatter([epoch], [peak["delta_N"]], color=color, s=24, zorder=4)
                axes[1, 0].scatter([epoch], [peak["block_acc"]], color=color, s=24, zorder=4)

        for color, seed_index in zip(colors, SMALL_N_SEEDS):
            local = small_frame.loc[(small_frame["seed"] == seed_index)
                                    & (small_frame["epoch"] <= DISPLAYED_EPOCH_MAX)]
            axes[0, 1].plot(local["epoch"], local["delta_N"], color=color, linewidth=1.7,
                            label=f"seed {seed_index}")
            axes[1, 1].plot(local["epoch"], local["block_acc"], color=color, linewidth=1.7,
                            label=f"seed {seed_index}")

        for axis in axes[0, :]:
            axis.axhline(0, color="0.25", linewidth=0.8, alpha=0.6)
            axis.axhline(PEAK_THRESHOLD, color="0.35", linestyle="--", linewidth=0.8, alpha=0.55)
            axis.set_ylim(-0.08, 0.72)
            axis.legend(frameon=False, loc="upper left")
        for axis in axes[1, :]:
            axis.axhline(HI, color="0.25", linestyle="--", linewidth=0.9, alpha=0.65)
            axis.set_xlim(1, DISPLAYED_EPOCH_MAX)
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlabel("epoch")

        axes[0, 0].set_title(r"(a) Large $N=1280$: shared-set transient peaks")
        axes[0, 1].set_title(r"(b) Small $N=64$: shared-set overconfidence")
        axes[0, 0].set_ylabel(r"$\Delta_N=q_{\rm self}-m$")
        axes[1, 0].set_ylabel("complete-cube block accuracy")
        fig.suptitle(r"Shared-training-set confidence--truth dynamics at $L=12$ (epochs 1--350)",
                     fontsize=13)
        fig.savefig(output)
        plt.close(fig)
    print(f"wrote {output}")


# ===========================================================================
# Fig. 4: the truth and self-overlap frontiers aligned on the first retreat (Sec. 3.4)
# ===========================================================================

# The legend of panel (a) sits above the axes, between the frame and the title.  Both titles of
# the top row get this pad so that they stay on one line.
EVENT_TITLE_PAD = 26


def fig4_position_resolved_retreat_recovery():
    """Fig. 4: the truth and self-overlap frontiers, aligned on the first retreat (Sec. 3.4).

    Sixteen replicas at L = 12, N = 1280 share one training set.  xhat_history.npz holds the
    soft spins u of every replica after every one of its 150 epochs on a fixed set of 4096
    held-out inputs (replica x epoch x input x position) and yspin the truth spins of those
    inputs.  A_block is computed here from the signs of u.

    A replica enters the figure if A_block reaches 0.8 and falls below 0.5 at some later epoch;
    the first such epoch is its retreat event.  Every selected replica is shifted so that its
    event sits at tau = 0, over the window tau = -20, ..., +30.  The profiles
    m(j, tau) = E_b[u_j y_j] and q_self(j, tau) = E_b[u_j^2] are averaged over the held-out
    inputs; the group frontiers j*_m and j*_q are the frontier positions of the
    replica-averaged profiles.

    (a) aligned A_block of each selected replica and their mean;
    (b) the truth frontier of each selected replica (faint), the group truth frontier and the
        group self-overlap frontier;
    (c) the replica-averaged m(j, tau) with the group truth frontier;
    (d) the replica-averaged q_self(j, tau) with the group self-overlap frontier.

    The window is added to the event epoch without a bounds check.  Eight of the sixteen
    replicas retreat, at epochs 63 to 113 of 150 (array indices 62 to 112), so the window fits
    for all of them.

    Reads   data/trajectory_ensembles/L12_N1280_soft_outputs/xhat_history.npz (300 MB, a few
            seconds)
    Writes  figure/fig4_position_resolved_retreat_recovery.pdf
    """
    with np.load(TRAJECTORY_ENSEMBLES / "L12_N1280_soft_outputs" / "xhat_history.npz") as data:
        xhat = data["xhat"]
        yspin = data["yspin"]

    block = (np.sign(xhat) == yspin[None, None]).all(axis=-1).mean(axis=-1)
    first_retreat = [first_fall(trajectory) for trajectory in block]

    selected = np.array([i for i, event in enumerate(first_retreat) if event is not None],
                        dtype=int)
    event_epochs = np.array([first_retreat[i] for i in selected], dtype=int)
    tau = np.arange(-20, 31)
    aligned = np.stack([xhat[selected, event_epochs + offset] for offset in tau], axis=1)
    aligned_block = np.stack([block[selected, event_epochs + offset] for offset in tau], axis=1)

    replicas, times, _, length = aligned.shape
    per_replica_profiles = (aligned * yspin[None, None]).mean(axis=2)
    mean_profile = per_replica_profiles.mean(axis=0)
    per_replica_self_profiles = (aligned**2).mean(axis=2)
    mean_self_profile = per_replica_self_profiles.mean(axis=0)
    individual_frontiers = np.empty((replicas, times), dtype=np.float64)
    mean_frontier = np.empty(times, dtype=np.float64)
    mean_self_frontier = np.empty(times, dtype=np.float64)
    for time_index in range(times):
        mean_frontier[time_index] = frontier_position(mean_profile[time_index])
        mean_self_frontier[time_index] = frontier_position(mean_self_profile[time_index])
        for replica in range(replicas):
            individual_frontiers[replica, time_index] = frontier_position(
                per_replica_profiles[replica, time_index])

    output = output_path(FIGURE, "fig4_position_resolved_retreat_recovery.pdf")
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.9), constrained_layout=True)
        axes = axes.ravel()

        for replica in range(replicas):
            axes[0].plot(tau, aligned_block[replica], color="#8b70a1", alpha=0.27, lw=0.9,
                         label="individual" if replica == 0 else "_nolegend_")
        axes[0].plot(tau, aligned_block.mean(axis=0), color="#5f287f", lw=2.3, label="mean")
        axes[0].axvline(0, color="black", ls="--", lw=1)
        axes[0].axhline(LO, color="0.5", ls=":", lw=0.8)
        axes[0].set(xlabel=r"epoch relative to first retreat $\tau$", ylabel="block accuracy")
        axes[0].set_title("(a) Event-aligned retreat", pad=EVENT_TITLE_PAD)
        axes[0].legend(fontsize=9, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)

        # The individual truth frontiers carry no legend entry: a third entry would cover too
        # much of the panel.  The caption identifies them as the faint curves.
        for replica in range(replicas):
            axes[1].plot(tau, individual_frontiers[replica], color="#8b70a1", alpha=0.25, lw=0.9)
        axes[1].plot(tau, mean_frontier, color="#5f287f", lw=2.3,
                     label=r"group truth frontier $j_m^\star$")
        axes[1].plot(tau, mean_self_frontier, color="#d26a1b", ls="--", lw=2.1,
                     label=r"group self-overlap frontier $j_q^\star$")
        axes[1].axvline(0, color="black", ls="--", lw=1)
        axes[1].set(xlabel=r"relative epoch $\tau$", ylabel="frontier coordinate",
                    ylim=(-0.5, length + 0.5))
        axes[1].set_title("(b) Truth and self-overlap frontiers", pad=EVENT_TITLE_PAD)
        axes[1].legend(fontsize=9)

        truth_image = axes[2].imshow(
            mean_profile.T, origin="lower", aspect="auto",
            extent=(tau[0] - 0.5, tau[-1] + 0.5, -0.5, length - 0.5),
            vmin=0.0, vmax=1.0, cmap="viridis",
        )
        axes[2].plot(tau, mean_frontier, color="white", lw=1.8)
        axes[2].axvline(0, color="white", ls="--", lw=1)
        axes[2].set(xlabel=r"relative epoch $\tau$", ylabel="output position $j$",
                    title=r"(c) Truth alignment $m(j,\tau)$")
        fig.colorbar(truth_image, ax=axes[2], fraction=0.046, pad=0.04, label=r"$m(j,\tau)$")

        self_image = axes[3].imshow(
            mean_self_profile.T, origin="lower", aspect="auto",
            extent=(tau[0] - 0.5, tau[-1] + 0.5, -0.5, length - 0.5),
            vmin=0.0, vmax=1.0, cmap="viridis",
        )
        axes[3].plot(tau, mean_self_frontier, color="white", lw=1.8)
        axes[3].axvline(0, color="white", ls="--", lw=1)
        axes[3].set(xlabel=r"relative epoch $\tau$", ylabel="output position $j$",
                    title=r"(d) Self-overlap $q_{\rm self}(j,\tau)$")
        fig.colorbar(self_image, ax=axes[3], fraction=0.046, pad=0.04,
                     label=r"$q_{\rm self}(j,\tau)$")

        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {output}")


# ===========================================================================
# Fig. 5: the L = 12 tail crossover (Sec. 3.5)
# ===========================================================================

def plot_tail_series(axis: plt.Axes, ns: np.ndarray, rows: list[dict[str, float]], key: str,
                     fmt: str, label: str) -> None:
    """One column of the table against N, with its <key>_se column as error bar."""
    values = np.asarray([row[key] for row in rows])
    errors = np.asarray([row[f"{key}_se"] for row in rows])
    axis.errorbar(ns, values, yerr=errors, fmt=fmt, label=label, linewidth=1.8, markersize=5,
                  capsize=2.5, elinewidth=1)


def shade_tail_regimes(axis: plt.Axes) -> None:
    """The three regime bands N = 28-72, 72-143, 143-290, each labelled at its geometric center."""
    regimes = ((28, 72, "memorization\nglass", "#efb5b5"),
               (72, 143, "crossover", "#f3d28c"),
               (143, 290, "truth-aligned\nretrieval", "#bbe3bb"))
    for lower, upper, label, color in regimes:
        axis.axvspan(lower, upper, color=color, alpha=0.22, zorder=0)
        axis.text(np.sqrt(lower * upper), 0.97, label, transform=axis.get_xaxis_transform(),
                  ha="center", va="top", fontsize=8,
                  bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color,
                        "alpha": 0.88, "linewidth": 0.7})


def fig5_tail_observables():
    """Fig. 5: the L = 12 tail crossover from memorization to truth-aligned retrieval
    (Sec. 3.5).

    One row of the table per training-set size N = 32, ..., 256 of the six-replica
    shared-training-set ensembles, every quantity averaged over the tail positions j = 8, ..., 11
    with a leave-one-replica jackknife error in the column <key>_se.

    (a) q_self, q_cross, m and chi_SG = q_self - q_cross on the held-out inputs;
    (b) training accuracy, held-out accuracy and the training-set q_self, against chance 0.5.

    The regime bands, the axis range 28 to 290 and the ticks 32, 64, 128, 256 are set for this
    scan; a wider scan needs them changed.

    Reads   results/tail_observables.csv (analysis.py tail_observables)
    Writes  figure/fig5_tail_observables.pdf
    """
    with open(RESULTS / "tail_observables.csv", encoding="utf-8", newline="") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    ns = np.asarray([row["N"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)

    plot_tail_series(axes[0], ns, rows, "tail_q_self", "o-", r"$q_{\rm self}$")
    plot_tail_series(axes[0], ns, rows, "tail_q_cross", "s-", r"$q_{\rm cross}$")
    plot_tail_series(axes[0], ns, rows, "tail_m_truth", "^-", r"$m$")
    plot_tail_series(axes[0], ns, rows, "tail_chi_sg", "d-", r"$\chi_{\rm SG}$")
    axes[0].set_title(r"(a) Tail order parameters")
    axes[0].set_ylabel(r"test-set tail average, $j=8,\ldots,11$")
    axes[0].legend(fontsize=9, ncol=2, loc="lower left")

    plot_tail_series(axes[1], ns, rows, "tail_train_acc", "o-", "train accuracy")
    plot_tail_series(axes[1], ns, rows, "tail_test_acc", "s-", "test accuracy")
    plot_tail_series(axes[1], ns, rows, "tail_train_q_self", "^-", r"train $q_{\rm self}$")
    axes[1].axhline(0.5, color="0.45", linestyle=":", linewidth=1.1, label="chance accuracy")
    axes[1].set_title("(b) Memorization and generalization")
    axes[1].set_ylabel(r"tail average, $j=8,\ldots,11$")
    axes[1].legend(fontsize=9, ncol=2, loc="lower right")

    for axis in axes:
        shade_tail_regimes(axis)
        axis.set_xscale("log", base=2)
        axis.set_xticks((32, 64, 128, 256), labels=("32", "64", "128", "256"))
        axis.set_xlim(28, 290)
        axis.set_ylim(-0.08, 1.05)
        axis.set_xlabel(r"fixed training-set size $N$")
        axis.grid(alpha=0.28, which="both")

    output = output_path(FIGURE, "fig5_tail_observables.pdf")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


# ===========================================================================
# Figs. 6 to 9: the fourteen replica sets (Secs. 4.2, 4.3, 5.1)
# ===========================================================================
# results/replica_set_geometry.json holds under "sets" one dict per replica set, keyed
# "<regime>|(L,N)" with the regime names of common.py: the overlaps, gaps and tree
# statistics of analysis.summarize on all output positions, with L, N, regime and the same
# quantities on the tabulated tail ("tail") and on the common tail j >= 2L/3 ("tail_u").
# "matched_n" is the replica count of the random subsets of the tree statistics.
# results/normalized_residuals.npz holds R^W of every set under "<key>|full" and "<key>|tail_u".

REGIME_STYLE = [("memorization", "#8c564b", "s", -0.26),     # regime, color, marker, offset
                ("retreat",      "#1f77b4", "o",  0.00),
                ("recovered",    "#2ca02c", "^",  0.26)]
REGIME_COLOR = {regime: color for regime, color, _, _ in REGIME_STYLE}
REPLICA_SET_RC = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                  "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5}


def load_replica_set_geometry() -> tuple[dict, int]:
    """The per-set dicts of results/replica_set_geometry.json and the size of the random replica
    subsets of the tree statistics."""
    geometry = read_json(RESULTS / "replica_set_geometry.json")
    return geometry["sets"], geometry["matched_n"]


OVERLAPS_AND_GAPS = [("m", r"$m$"), ("q_self", r"$q_{\mathrm{self}}$"),
                    ("q_cross", r"$q_{\mathrm{cross}}$"), ("Delta_N", r"$\Delta_N$"),
                    ("chi_SG", r"$\chi_{\mathrm{SG}}$")]


def fig6_raw_overlaps_and_gaps():
    """Fig. 6: raw overlaps and their two gaps for the fourteen replica sets, on two windows
    (Sec. 4.2).

    For every set, m, q_self, q_cross, Delta_N = q_self - m and chi_SG = q_self - q_cross;
    (a) on all output positions, (b) on the common tail j >= 2L/3.  Drawing every set on both
    windows shows that the memorization signature lives on the tail while the retreat and
    recovery signatures do not depend on the window.  Panel (b) uses the common window rather
    than each set's tabulated tail because the tabulated windows are not one rule: the L = 24
    memorization ensembles use j >= 18 while every other set follows 2L/3.

    Points are grouped by regime within each quantity, sets in order of (L, N) inside a group.
    Error bars are one closed-form standard error: m, q_self and Delta_N are means of
    per-replica quantities, while q_cross and chi_SG are degree-2 U-statistics whose standard
    error uses the Hajek projection on the per-replica row means.  Most are smaller than the
    marker.

    Reads   results/replica_set_geometry.json (analysis.py replica_set_geometry)
    Writes  figure/fig6_raw_overlaps_and_gaps.pdf
    Prints the range of each quantity over the sets of each regime, per window.
    """
    sets, _ = load_replica_set_geometry()
    windows = (("full", "(a) all output positions"), ("tail_u", r"(b) common tail $j \geq 2L/3$"))
    output = output_path(FIGURE, "fig6_raw_overlaps_and_gaps.pdf")
    with plt.rc_context(REPLICA_SET_RC):
        fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
        for p, (window, title) in enumerate(windows):
            a = ax[p]
            for i in range(1, len(OVERLAPS_AND_GAPS)):
                a.axvline(i - 0.5, color="0.85", lw=0.8, zorder=0)
            for regime, color, marker, dx in REGIME_STYLE:
                keys = [k for k in sets if sets[k]["regime"] == regime]
                keys.sort(key=lambda k: (sets[k]["L"], sets[k]["N"]))
                jitter = np.linspace(-0.085, 0.085, len(keys))
                for i, (q, _) in enumerate(OVERLAPS_AND_GAPS):
                    values = [(sets[k] if window == "full" else sets[k][window]) for k in keys]
                    a.errorbar(i + dx + jitter, [u[q] for u in values],
                               yerr=[u[q + "_se"] for u in values],
                               fmt=marker, ms=5.2, mfc=color, mec="0.25", mew=0.5, ecolor="0.35",
                               elinewidth=0.9, capsize=1.8, ls="none", zorder=3,
                               label=regime if i == 0 else None)
            a.set_xticks(range(len(OVERLAPS_AND_GAPS)))
            a.set_xticklabels([t for _, t in OVERLAPS_AND_GAPS], fontsize=12)
            a.set_xlim(-0.55, len(OVERLAPS_AND_GAPS) - 0.45)
            a.set_ylim(-0.08, 1.08)
            a.axhline(0.0, color="0.6", lw=0.8, zorder=1)
            a.set_title(title)
            a.grid(axis="y", alpha=0.25)
            a.set_ylabel("value")
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=10, ncol=3, loc="outside lower center")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    print("wrote", output.relative_to(ROOT))

    for window, _ in windows:
        print("===", window)
        for q, _ in OVERLAPS_AND_GAPS:
            by_regime = {regime: [(sets[k] if window == "full" else sets[k][window])[q]
                                  for k in sets if sets[k]["regime"] == regime]
                         for regime, *_ in REGIME_STYLE}
            print("  %-8s " % q + "  ".join(
                f"{regime[:3]} {min(v):+.3f}..{max(v):+.3f}" for regime, v in by_regime.items()))


RETREAT_TREES = {
    # row label -> (matrix file, whether the residual-norm filter is applied here)
    "L=12": (RESULTS / "retreat_matrices" / "L12_N1280_first400_norm_filtered.npz", False),
    "L=16": (RESULTS / "retreat_matrices" / "L16_N2048.npz", True),
}
TREE_CLUSTERS = 4


def fig7_retreat_residual_tree():
    """Fig. 7: clustering tree of the residual overlap in two retreat ensembles (Sec. 4.2).

    A 2 x 3 panel: the top row is the (12, 1280) retreat ensemble, the bottom row the (16, 2048)
    one.  W is normalized to R^W_rs = W_rs / sqrt(W_rr W_ss) and clustered by average linkage on
    the signed distance d_rs = 1 - R^W_rs.  The columns show the dendrogram with the sizes of
    the four blocks of its K = 4 cut, R^W in tree order with the cluster boundaries drawn in
    black, and the mean R^W within and between the four clusters (within a cluster over its
    distinct pairs; a singleton has none and is marked n=1).

    The distance keeps the sign of R^W: truth alignment is removed from W, so anticorrelated
    residuals are different functions, not one state and its global flip, and the heatmaps use
    a diverging scale centered on zero.  Both rows are shown after the bottom-5% residual-norm
    filter, which keeps the floor(0.95 R) replicas with the largest W_rr.  The L = 12 file is
    stored with the filter applied; the L = 16 file holds every retreat replica, so the filter
    is applied here.

    Reads   results/retreat_matrices/L12_N1280_first400_norm_filtered.npz
            (analysis.py retreat_matrices_norm_filter)
            results/retreat_matrices/L16_N2048.npz (analysis.py retreat_matrices_l16_n2048)
    Writes  figure/fig7_retreat_residual_tree.pdf
    Prints, for each row, the replica count, the K = 4 cluster sizes, the averages of the
    within-cluster and of the between-cluster block means of R^W, and the fraction of negative
    off-diagonal entries of R^W.
    """
    K = TREE_CLUSTERS
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2), constrained_layout=True)
    tags = "abcdef"

    for row, (label, (path, needs_filter)) in enumerate(RETREAT_TREES.items()):
        W = np.load(path, allow_pickle=True)["W"]
        if needs_filter:
            keep = np.sort(np.argsort(np.diag(W))[::-1][:int(np.floor(0.95 * len(W)))])
            W = W[np.ix_(keep, keep)]
        R = W / np.sqrt(np.outer(np.diag(W), np.diag(W)))
        d = 1.0 - R
        np.fill_diagonal(d, 0.0)
        Z = linkage(squareform(d, checks=False), method="average")
        order = leaves_list(Z)
        labels = fcluster(Z, t=K, criterion="maxclust")
        sizes = [int((labels == c).sum()) for c in range(1, K + 1)]

        # dendrogram, colored below the height of the K = 4 cut
        ax = axes[row, 0]
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-(K - 1), 2],
                   above_threshold_color="0.45")
        ax.set_ylabel(rf"$L={label[2:]}$" + "\n" + r"distance $1-R^W_{r,s}$")
        ax.set_title(f"({tags[3*row]}) dendrogram, $K=4$ blocks "
                     + "|".join(str(s) for s in sizes))
        ax.set_xticks([])

        # R^W with the replicas in tree order
        ax = axes[row, 1]
        im = ax.imshow(R[np.ix_(order, order)], vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        edges = np.flatnonzero(np.diff(labels[order])) + 0.5
        for e in edges:
            ax.axhline(e, color="black", lw=0.9)
            ax.axvline(e, color="black", lw=0.9)
        ax.set_title(f"({tags[3*row+1]}) " + r"$R^W_{r,s}$ in tree order")
        ax.set_xlabel("replica in tree order")
        ax.set_ylabel("replica in tree order")
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(r"$R^W_{r,s}$")

        # mean R^W of each pair of clusters
        ax = axes[row, 2]
        block = np.full((K, K), np.nan)
        for a in range(1, K + 1):
            ia = labels == a
            for b in range(1, K + 1):
                ib = labels == b
                sub = R[np.ix_(ia, ib)]
                if a == b:
                    if ia.sum() < 2:
                        continue
                    iu = np.triu_indices(ia.sum(), 1)
                    block[a-1, b-1] = sub[iu].mean()
                else:
                    block[a-1, b-1] = sub.mean()
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(block, vmin=-0.75, vmax=0.75, cmap=cmap)
        for a in range(K):
            for b in range(K):
                txt = "n=1" if np.isnan(block[a, b]) else f"{block[a, b]:.2f}"
                ax.text(b, a, txt, ha="center", va="center", fontsize=9)
        ax.set_xticks(range(K))
        ax.set_xticklabels(range(1, K+1))
        ax.set_yticks(range(K))
        ax.set_yticklabels(range(1, K+1))
        ax.set_title(f"({tags[3*row+2]}) mean " + r"$R^W_{r,s}$ by cluster")
        ax.set_xlabel("cluster")
        ax.set_ylabel("cluster")
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(r"mean $R^W_{r,s}$")
        within = [block[a, a] for a in range(K) if not np.isnan(block[a, a])]
        between = [block[a, b] for a in range(K) for b in range(K)
                   if a != b and not np.isnan(block[a, b])]
        iu = np.triu_indices(len(W), 1)
        print(label, "n =", len(W), "K=4 sizes:", sizes,
              f"within {np.mean(within):.3f} between {np.mean(between):.3f}",
              f"negative off-diagonal {100*(R[iu] < 0).mean():.1f}%")

    output = output_path(FIGURE, "fig7_retreat_residual_tree.pdf")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print("wrote", output)


RESIDUAL_STATISTICS = [("r2_tree", r"$R^2_{\mathrm{tree}}$"),
                       ("p_um", r"$P(\delta_{\mathrm{um}}<0.1)$"),
                       ("p_neg", r"$P(R^W<0)$")]


def fig8_fragmentation_residual_organization():
    """Fig. 8: fragmentation amplitude, its decomposition and the residual organization of the
    fourteen replica sets, on all output positions (Sec. 4.3).

    (a) gap(Q^c) against L.  The three regimes fall into three bands that do not drift with
        string length; capped bars give the ratios of the band means.
    (b) The residual share gap(W) / gap(Q^c) against gap(Q^c).  It separates the regimes again
        in another order: retreat drops below recovery although its gap is about ten times
        larger.  Each capped bar spans the empty interval between two adjacent bands and is
        labelled by its width.
    (c) The pair distribution of R^W for each set: mean, one standard deviation and the 1st to
        99th percentile.  Memorization is narrow and never approaches zero, retreat is broad and
        runs through it, and recovery is as broad as retreat and yet stops there.
    (d) R^2_tree, P(delta_um < 0.1) and P(R^W < 0).  None of the three separates all three
        regimes on its own.
    (e) P(delta_um < 0.1) against P(R^W < 0).  The obtuse-pair fraction alone selects the five
        retreat sets and the near-ultrametric fraction alone the four memorization sets, so the
        plane is a two-by-two of cells with one cell empty; the grey bands are the two intervals
        that contain no set.

    Every panel uses all output positions, the window of every comparison across regimes
    (Sec. 4.1); the common-tail values are drawn in Fig. 9.  The tree statistics are means over
    random subsets of n = 47 replicas because they depend on the replica count; the share
    and P(R^W < 0) do not and use every replica.  All error bars are leave-one-replica jackknife
    standard errors.  P(R^W < 0) and its error are computed here from the stored R^W
    (common.obtuse_pair_fraction).

    Reads   results/replica_set_geometry.json (analysis.py replica_set_geometry)
            results/normalized_residuals.npz (analysis.py normalized_residuals)
    Writes  figure/fig8_fragmentation_residual_organization.pdf
    Prints the margin by which P(delta_um < 0.1) isolates the memorization sets, and the range
    of each plotted quantity per regime.
    """
    S, M = load_replica_set_geometry()
    RW = dict(np.load(RESULTS / "normalized_residuals.npz"))
    for k in S:
        S[k]["p_neg"], S[k]["p_neg_se"] = obtuse_pair_fraction(RW[f"{k}|full"])

    def sets(regime):
        return sorted((S[k] for k in S if S[k]["regime"] == regime), key=lambda u: (u["L"], u["N"]))

    def value_range(regime, q):
        return min(u[q] for u in sets(regime)), max(u[q] for u in sets(regime))

    output = output_path(FIGURE, "fig8_fragmentation_residual_organization.pdf")
    with plt.rc_context(REPLICA_SET_RC):
        fig = plt.figure(figsize=(9.8, 9.9), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.72, 1.0])
        a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        a4 = fig.add_subplot(gs[1, :])
        a2, a3 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

        # (a) the amplitude ladder
        for regime, c, mk, _ in REGIME_STYLE:
            v = sets(regime)
            x = np.array([q["L"] for q in v], float) + np.linspace(-0.35, 0.35, len(v))
            y = np.array([q["gap_Qc"] for q in v])
            # the same jackknife errors as on the horizontal axis of panel (b)
            a0.errorbar(x, y, yerr=np.array([q["gap_Qc_se"] for q in v]), fmt="none", ecolor=c,
                        elinewidth=1.0, capsize=1.8, alpha=0.9, zorder=2)
            a0.scatter(x, y, c=c, marker=mk, s=58,
                       edgecolor="0.25", linewidth=0.5, label=regime, zorder=3)
            a0.axhspan(y.min(), y.max(), color=c, alpha=0.12, zorder=0)
        levels = {regime: np.array([q["gap_Qc"] for q in sets(regime)])
                  for regime, *_ in REGIME_STYLE}
        for p, q in (("memorization", "retreat"), ("retreat", "recovered")):
            f = levels[p].mean() / levels[q].mean()
            # The capped bar spans the two band means the ratio is taken between.
            a0.plot([25.2, 25.2], [levels[q].mean(), levels[p].mean()], color="0.45", lw=0.9,
                    marker="_", ms=5, mew=0.9, zorder=2)
            a0.annotate(rf"$\times{f:.0f}$",
                        xy=(25.5, np.sqrt(levels[p].mean() * levels[q].mean())), fontsize=10,
                        color="0.25", va="center", ha="left", annotation_clip=False)
        a0.set_yscale("log")
        a0.set_xticks([12, 16, 20, 24])
        a0.set_xlim(10.8, 26.8)
        a0.set_xlabel("string length $L$")
        a0.set_ylabel(r"connected gap $\mathrm{gap}(Q^c)$")
        a0.set_title("(a) connected-gap separation across the scanned sizes")
        a0.legend(frameon=False, loc="upper center", fontsize=9)
        a0.grid(axis="y", alpha=0.25, which="both")

        # (b) the share against the gap: both separate the three regimes with no overlap, and
        # they disagree on the order.
        for regime, c, mk, _ in REGIME_STYLE:
            v = sets(regime)
            x = np.array([q["gap_Qc"] for q in v])
            y = np.array([q["share"] for q in v])
            # jackknife on both coordinates
            a1.errorbar(x, y, xerr=np.array([q["gap_Qc_se"] for q in v]),
                        yerr=np.array([q["share_se"] for q in v]), fmt="none", ecolor=c,
                        elinewidth=1.0, capsize=1.8, alpha=0.9, zorder=2)
            a1.scatter(x, y, c=c, marker=mk, s=58, edgecolor="0.25", linewidth=0.5, zorder=3)
            lo, hi = value_range(regime, "share")
            a1.axhspan(lo, hi, color=c, alpha=0.12, zorder=0)
        tr = a1.get_yaxis_transform()
        for below, above in (("retreat", "recovered"), ("recovered", "memorization")):
            edge_lo = max(u["share"] for u in sets(below))
            edge_hi = min(u["share"] for u in sets(above))
            a1.plot([0.045, 0.045], [edge_lo, edge_hi], transform=tr, color="0.45", lw=0.9,
                    marker="_", ms=5, mew=0.9, zorder=2)
            a1.annotate(f"${edge_hi - edge_lo:.2f}$", xy=(0.075, 0.5 * (edge_lo + edge_hi)),
                        xycoords=tr, fontsize=10, color="0.25", va="center", ha="left")
        a1.set_xscale("log")
        a1.set_ylim(0.25, 1.06)
        a1.set_xlim(1.3e-3, 1.0)
        a1.set_xlabel(r"connected gap $\mathrm{gap}(Q^c)$")
        a1.set_ylabel(r"residual share $\mathrm{gap}(W)/\mathrm{gap}(Q^c)$")
        a1.set_title("(b) residual share, retreat and recovery inverted")
        a1.annotate("retreat drops below recovered\nat ten times the gap",
                    xy=(0.077, 0.45), xytext=(0.0042, 0.30), fontsize=8.5, color="0.15",
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
        a1.grid(alpha=0.25, which="both")

        # (d) the three residual statistics.  The spread over the subset draws is not used as
        # an error bar: it measures how much one draw of M replicas moves, and it collapses to
        # zero on a set whose replica count equals M.
        for i in range(1, len(RESIDUAL_STATISTICS)):
            a2.axvline(i - 0.5, color="0.85", lw=0.8, zorder=0)
        for regime, c, mk, dx in REGIME_STYLE:
            v = sets(regime)
            jitter = np.linspace(-0.08, 0.08, len(v))
            for i, (q, _) in enumerate(RESIDUAL_STATISTICS):
                x = i + dx + jitter
                y = np.array([u[q] for u in v])
                e = np.array([u[q + "_se"] for u in v])
                a2.errorbar(x, y, yerr=e, fmt="none", ecolor=c, elinewidth=1.1, capsize=2.0,
                            alpha=0.9, zorder=2)
                a2.scatter(x, y, c=c, marker=mk, s=42, edgecolor="0.25", linewidth=0.5, zorder=3)
        a2.set_xticks(range(len(RESIDUAL_STATISTICS)))
        a2.set_xticklabels([t for _, t in RESIDUAL_STATISTICS], fontsize=11)
        a2.set_xlim(-0.55, len(RESIDUAL_STATISTICS) - 0.45)
        a2.set_ylim(-0.045, 1.0)
        a2.set_ylabel("value")
        a2.set_title("(d) the three residual statistics")
        a2.grid(axis="y", alpha=0.25)

        # (e) two of them together separate all three
        ret_nlo = min(q["p_neg"] for q in sets("retreat"))
        oth_nhi = max(q["p_neg"] for r in ("memorization", "recovered") for q in sets(r))
        mem_phi = max(q["p_um"] for q in sets("memorization"))
        oth_plo = min(q["p_um"] for r in ("retreat", "recovered") for q in sets(r))
        a3.axvspan(oth_nhi, ret_nlo, color="0.45", alpha=0.25, zorder=1)
        a3.axhspan(mem_phi, oth_plo, color="0.45", alpha=0.25, zorder=1)
        for regime, c, mk, _ in REGIME_STYLE:
            v = sets(regime)
            x = np.array([q["p_neg"] for q in v])
            y = np.array([q["p_um"] for q in v])
            ex = np.array([q["p_neg_se"] for q in v])
            ey = np.array([q["p_um_se"] for q in v])
            a3.errorbar(x, y, xerr=ex, yerr=ey, fmt="none", ecolor=c, elinewidth=1.0,
                        capsize=1.8, alpha=0.9, zorder=3)
            a3.scatter(x, y, c=c, marker=mk, s=58, edgecolor="0.25", linewidth=0.5, zorder=4)
        a3.set_xlim(-0.02, 0.30)
        a3.set_ylim(0.04, 0.44)
        a3.annotate("retreat", xy=(0.19, 0.415), color=REGIME_COLOR["retreat"], fontsize=10,
                    ha="center")
        a3.annotate("recovered", xy=(0.03, 0.415), color=REGIME_COLOR["recovered"], fontsize=10,
                    ha="center")
        a3.annotate("memorization", xy=(0.04, 0.062), color=REGIME_COLOR["memorization"],
                    fontsize=10, ha="center")
        a3.set_xlabel(r"obtuse-pair fraction $P(R^W<0)$")
        a3.set_ylabel(rf"$P(\delta_{{\mathrm{{um}}}}<0.1)$ at $n={M}$")
        a3.set_title("(e) two intervals that contain no set")
        a3.grid(alpha=0.2)

        # (c) the distribution behind the obtuse-pair fraction
        pos, labs = 0, []
        for regime, c, mk, _ in REGIME_STYLE:
            for q in sets(regime):
                R = RW[f"{regime}|({q['L']},{q['N']})|full"]
                pairs = R[np.triu_indices(R.shape[0], 1)]
                mean, sd = pairs.mean(), pairs.std()
                # The whisker is a percentile rather than the extreme pair: the number of pairs
                # runs from 1081 to 86320 across the fourteen sets, and an extreme value would
                # track that.
                p01, p99 = np.quantile(pairs, [0.01, 0.99])
                a4.plot([pos, pos], [p01, p99], color=c, lw=1.0, alpha=0.65, zorder=2)
                a4.plot([pos, pos], [mean - sd, mean + sd], color=c, lw=6.5, alpha=0.9,
                        solid_capstyle="butt", zorder=2)
                a4.plot([pos], [mean], marker=mk, ms=5.5, mfc="white", mec=c, mew=1.4, zorder=4)
                labs.append(f"({q['L']},{q['N']})")
                pos += 1
        a4.axhline(0, color="0.25", lw=1.1, zorder=1)
        for boundary in (3.5, 8.5):
            a4.axvline(boundary, color="0.85", lw=0.8, ls=":", zorder=0)
        a4.set_xticks(range(len(labs)))
        a4.set_xticklabels(labs, rotation=45, ha="right", fontsize=9)
        a4.set_xlim(-0.7, len(labs) - 0.3)
        for c0, c1, regime in ((0, 3, "memorization"), (4, 8, "retreat"), (9, 13, "recovered")):
            a4.annotate(regime, xy=((c0 + c1) / 2, 1.02), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=10, color=REGIME_COLOR[regime])
        a4.set_ylabel(r"$R^W_{r,s}$")
        a4.set_title(r"(c) mean, one s.d. and the 1st--99th percentile of the pair distribution",
                     pad=20)
        a4.grid(axis="y", alpha=0.25)

        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    print("wrote", output.relative_to(ROOT))

    q, isolated = "p_um", "memorization"
    inside = max((S[k][q], k) for k in S if S[k]["regime"] == isolated)
    outside = min((S[k][q], k) for k in S if S[k]["regime"] != isolated)
    spread = (S[inside[1]][q + "_sd"] ** 2 + S[outside[1]][q + "_sd"] ** 2) ** 0.5
    print("%-8s isolates %-13s %s %.3f vs %s %.3f | margin %+.3f  spread %.3f"
          % (q, isolated, inside[1], inside[0], outside[1], outside[0], outside[0] - inside[0],
             spread))
    for q in ("gap_Qc", "share", "r2_tree", "p_um"):
        sd = [S[k].get(q + "_sd") for k in S]
        tail = ("  sd %.3f-%.3f" % (min(sd), max(sd))) if None not in sd else ""
        print("  %-8s " % q + "  ".join("%s %.3f-%.3f" % (r[:3], *value_range(r, q))
                                        for r, *_ in REGIME_STYLE) + tail)


def fig9_two_planes_common_tail():
    """Fig. 9: two planes that separate the three regimes, both on the common tail (Sec. 5.1).

    This figure repeats the separation of Fig. 8 on the common tail j >= 2L/3, where the
    memorization signature is sharpest and the retreat signal weakest.

    (a) Delta_N against chi_SG, the two gaps of Sec. 1.  The memorization sets sit far from the
        line q_self = m while the retreat and recovery sets stay on it, and chi_SG orders the
        three.  Error bars are the closed-form standard errors of Fig. 6.
    (b) P(delta_um < 0.1) against P(R^W < 0).  On the tail P(delta_um < 0.1) alone puts the three
        regimes in three ranges that do not overlap (the two horizontal grey bands), and
        P(R^W < 0) separates the retreat sets from the other nine by an order of magnitude (the
        vertical band).  Vertical bars are the spread of P(delta_um < 0.1) over the subset draws,
        horizontal bars the jackknife error of P(R^W < 0).

    Inside a regime the sets are drawn in order of their replica count.

    Reads   results/replica_set_geometry.json (analysis.py replica_set_geometry): gaps and
            tree statistics on "tail_u"
            results/normalized_residuals.npz (analysis.py normalized_residuals): R^W on
            "tail_u", from which P(R^W < 0) is computed here
    Writes  figure/fig9_two_planes_common_tail.pdf
    Prints the bands of panel (b) and the range of each plotted quantity per regime.
    """
    S, M = load_replica_set_geometry()
    window = "tail_u"
    RW = dict(np.load(RESULTS / "normalized_residuals.npz"))
    for k in S:
        S[k][window]["p_neg"], S[k][window]["p_neg_se"] = obtuse_pair_fraction(RW[f"{k}|{window}"])

    def sets(regime):
        return sorted((S[k][window] for k in S if S[k]["regime"] == regime), key=lambda u: u["n"])

    output = output_path(FIGURE, "fig9_two_planes_common_tail.pdf")
    with plt.rc_context(REPLICA_SET_RC):
        fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)

        a0 = ax[0]
        a0.axhline(0.0, color="0.45", lw=1.0, ls="--", zorder=1)
        for regime, c, mk, _ in REGIME_STYLE:
            v = sets(regime)
            x = np.array([q["chi_SG"] for q in v])
            y = np.array([q["Delta_N"] for q in v])
            # The vertical errors are smaller than the markers for every set; the horizontal
            # ones are visible where chi_SG is small, because the axis is logarithmic there.
            a0.errorbar(x, y, xerr=np.array([q["chi_SG_se"] for q in v]),
                        yerr=np.array([q["Delta_N_se"] for q in v]), fmt="none", ecolor=c,
                        elinewidth=1.0, capsize=1.8, alpha=0.9, zorder=2)
            a0.scatter(x, y, c=c, marker=mk, s=58,
                       edgecolor="0.25", linewidth=0.5, label=regime, zorder=3)
        a0.set_xscale("log")
        a0.set_ylim(-0.12, 1.14)
        a0.annotate(r"$q_{\mathrm{self}}=m$", xy=(0.021, 0.055), fontsize=9, color="0.35")
        a0.annotate("retreat and recovered separate\nin $\\chi_{\\mathrm{SG}}$ alone",
                    xy=(0.021, 0.22), fontsize=8.5, color="0.30")
        a0.set_xlabel(r"self--cross gap $\chi_{\mathrm{SG}}$")
        a0.set_ylabel(r"Nishimori gap $\Delta_N$")
        a0.set_title("(a) the two gaps")
        a0.legend(frameon=False, loc="upper left", fontsize=9.5)
        a0.grid(alpha=0.25, which="both")

        # (b) the geometry plane; the grey bands are the intervals that contain no set
        a1 = ax[1]
        mem_hi = max(q["p_um"] for q in sets("memorization"))
        ret_lo = min(q["p_um"] for q in sets("retreat"))
        ret_hi = max(q["p_um"] for q in sets("retreat"))
        rec_lo = min(q["p_um"] for q in sets("recovered"))
        ret_nlo = min(q["p_neg"] for q in sets("retreat"))
        oth_nhi = max(q["p_neg"] for r in ("memorization", "recovered") for q in sets(r))
        a1.axhspan(mem_hi, ret_lo, color="0.45", alpha=0.25, zorder=1)
        a1.axhspan(ret_hi, rec_lo, color="0.45", alpha=0.25, zorder=1)
        a1.axvspan(oth_nhi, ret_nlo, color="0.45", alpha=0.25, zorder=1)
        for regime, c, mk, _ in REGIME_STYLE:
            v = sets(regime)
            x = np.array([q["p_neg"] for q in v])
            y = np.array([q["p_um"] for q in v])
            ex = np.array([q["p_neg_se"] for q in v])
            ey = np.array([q["p_um_sd"] for q in v])
            a1.errorbar(x, y, xerr=ex, yerr=ey, fmt="none", ecolor=c, elinewidth=1.0,
                        capsize=1.8, alpha=0.9, zorder=3)
            a1.scatter(x, y, c=c, marker=mk, s=58, edgecolor="0.25", linewidth=0.5, zorder=4)
        a1.set_xlim(-0.02, 0.46)
        a1.set_ylim(0.06, 0.40)
        a1.annotate("retreat", xy=(0.30, 0.235), color=REGIME_COLOR["retreat"], fontsize=10,
                    ha="center")
        a1.annotate("recovered", xy=(0.10, 0.345), color=REGIME_COLOR["recovered"], fontsize=10,
                    ha="center")
        a1.annotate("memorization", xy=(0.12, 0.095), color=REGIME_COLOR["memorization"],
                    fontsize=10, ha="center")
        a1.set_xlabel(r"obtuse-pair fraction $P(R^W<0)$")
        a1.set_ylabel(rf"$P(\delta_{{\mathrm{{um}}}}<0.1)$ at $n={M}$")
        a1.set_title("(b) residual geometry on the tail")
        a1.grid(alpha=0.25)

        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    print("wrote", output.relative_to(ROOT))
    print(f"  P(dum) bands  mem<{mem_hi:.3f} | retreat {ret_lo:.3f}-{ret_hi:.3f} "
          f"| recovered>{rec_lo:.3f}")
    print(f"  P(R^W<0) band others<{oth_nhi:.4f} | retreat>{ret_nlo:.4f}  "
          f"(factor {ret_nlo/oth_nhi:.1f})")
    for regime, *_ in REGIME_STYLE:
        v = sets(regime)
        print("  %-13s P(neg) %.4f-%.4f  P %.3f-%.3f  chi %.3f-%.3f  D_N %+.3f..%+.3f" % (
            regime, min(q["p_neg"] for q in v), max(q["p_neg"] for q in v),
            min(q["p_um"] for q in v), max(q["p_um"] for q in v),
            min(q["chi_SG"] for q in v), max(q["chi_SG"] for q in v),
            min(q["Delta_N"] for q in v), max(q["Delta_N"] for q in v)))


# ===========================================================================
# Fig. 10: the learning-rate intervention after acquisition (Sec. 5.2)
# ===========================================================================

ARMS = ("constant_arm", "intervention_arm")
ARM_LABEL = {"constant_arm": "constant rate", "intervention_arm": "decay after acquisition"}
ARM_COLOR = {"constant_arm": "#d55e00", "intervention_arm": "#009e73"}
INTERVENTION_NS = (512, 768, 1024)
SHOWN_N = 512                  # panel (a) shows the first pair of the smallest size,
SHOWN_SPLIT, SHOWN_INIT = 0, 0     # data split 0 and initialization 0


def read_intervention_run(arm: str, N: int, split: int, init: int) -> dict:
    return read_json(LEARNING_RATE_INTERVENTION / arm / f"split{split}"
                     / f"L12_N{N}_split{split}_init{init}.json")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(k/n, lower, upper) of the Wilson score interval at z, the interval clipped to [0, 1]."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), min(c + h, 1.0)


def arm_counts() -> dict[tuple[str, int], tuple[int, int, int]]:
    """(runs reaching 0.8, of those retreating, of those ending high) per arm and N."""
    out = {}
    for arm in ARMS:
        for f in sorted(glob.glob(
                f"{LEARNING_RATE_INTERVENTION}/{arm}/split*/L12_N*_split*_init*.json")):
            N = int(re.search(r"_N(\d+)_split", f).group(1))
            reached_, retreated, high = crossing_outcome(
                [float(e["test_block_acc"]) for e in read_json(Path(f))["history"]])
            r, t, h = out.get((arm, N), (0, 0, 0))
            out[(arm, N)] = (r + int(reached_), t + int(reached_ and retreated),
                             h + int(reached_ and high))
    return out


def draw_pair(ax: plt.Axes) -> None:
    """Fig. 10(a): A_block against optimizer update for the shown pair of both arms."""
    # The constant run is drawn wider and underneath: where only it shows, the two coincide.
    for arm, lw in zip(ARMS, (3.6, 1.8)):
        r = read_intervention_run(arm, SHOWN_N, SHOWN_SPLIT, SHOWN_INIT)
        s = [e["step"] for e in r["history"]]
        a = [e["test_block_acc"] for e in r["history"]]
        ax.plot(s, a, color=ARM_COLOR[arm], lw=lw, label=ARM_LABEL[arm],
                zorder=2 if arm == "constant_arm" else 3)
        i0 = first_crossing(a)
        if i0 is not None:
            ax.scatter([s[i0]], [a[i0]], s=34, facecolor="white",
                       edgecolor=ARM_COLOR[arm], lw=1.4, zorder=4)
            j = first_fall(a)
            if j is not None:
                ax.scatter([s[j]], [a[j]], marker="v", s=55, color=ARM_COLOR[arm],
                           edgecolor="white", lw=0.7, zorder=5)
    t0 = read_intervention_run("intervention_arm", SHOWN_N, SHOWN_SPLIT,
                               SHOWN_INIT)["summary"]["decay_onset_step"]
    ax.axvline(t0, color=ARM_COLOR["intervention_arm"], ls="--", lw=1.2, zorder=1)
    ax.annotate(r"$t_\ast$", xy=(t0, 1.0), xytext=(4, -2), textcoords="offset points",
                color=ARM_COLOR["intervention_arm"], fontsize=10, va="top")
    ax.axhline(HI, color="0.3", ls="--", lw=1.0)
    ax.axhline(LO, color="0.55", ls=":", lw=1.0)
    ax.set_xlim(0, 2400)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("held-out block accuracy")
    ax.set_title(rf"(a) one matched pair, $N_{{\rm train}}={SHOWN_N}$", loc="left", fontsize=10.5)
    ax.grid(color="0.90", lw=0.7)
    ax.legend(frameon=False, loc="upper right", fontsize=9)


def draw_arm_fraction(ax, counts, idx, ylab, letter, short):
    """Fig. 10(b) (idx 0, retreat) or (c) (idx 1, ending high): the rate per N and arm with
    Wilson 95% intervals and the counts k/n written above each point."""
    x = np.arange(len(INTERVENTION_NS))
    for arm, dx in zip(ARMS, (-0.10, 0.10)):
        p, lo, hi = [], [], []
        for N in INTERVENTION_NS:
            r, t, h = counts[(arm, N)]
            a, b, c = wilson_interval((t, h)[idx], r)
            p.append(a)
            lo.append(a - b)
            hi.append(c - a)
        ax.errorbar(x + dx, p, yerr=[lo, hi], fmt="o", color=ARM_COLOR[arm], capsize=3,
                    markersize=7, lw=1.6, label=ARM_LABEL[arm])
        dy = 6 if arm == "constant_arm" else 16
        for xi, N, pi, up in zip(x + dx, INTERVENTION_NS, p, hi):
            r, t, h = counts[(arm, N)]
            ax.annotate(f"{(t, h)[idx]}/{r}", xy=(xi, pi + up), xytext=(0, dy),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=ARM_COLOR[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in INTERVENTION_NS])
    ax.set_xlim(-0.45, len(INTERVENTION_NS) - 0.55)
    ax.set_ylim(-0.04, 1.22)
    ax.set_xlabel(r"$N_{\rm train}$")
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.set_title(f"({letter}) {short}", loc="left", fontsize=10.5)
    ax.grid(color="0.90", lw=0.7, axis="y")


def fig10_learning_rate_intervention():
    """Fig. 10: the learning-rate intervention, applied only after the rule has been learned
    (Sec. 5.2).

    Both arms are the same run until held-out A_block first reaches 0.8 at update t*: same data
    split, initialization, minibatch stream, optimizer, budget and learning rate.  The
    intervention arm then decays the learning rate over the remaining budget.  Panel (a) is one
    pair (N = 512, data split 0, initialization 0); panels (b) and (c) are two rates over
    the thirty pairs at each of three sizes.  Retreat is scored on the minimum after acquisition
    (below 0.5), while the endpoint rate asks whether A_block is at or above 0.8 when training
    stops.  The denominator of both rates is the number of runs that reach 0.8, the same runs
    in both arms by construction; error bars are Wilson 95% intervals.

    Reads   data/learning_rate_intervention/<arm>/split<d>/L12_N<N>_split<d>_init<m>.json
            for the constant arm and the intervention arm (decay after acquisition),
            N = 512, 768, 1024, fifteen data splits and two initializations
    Writes  figure/fig10_learning_rate_intervention.pdf
    Prints the counts per size and pooled; the paired analysis of the same runs is
    analysis.py intervention_pair_outcomes.
    """
    counts = arm_counts()
    output = output_path(FIGURE, "fig10_learning_rate_intervention.pdf")
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5}):
        # the 9.8-inch width of the other multi-panel figures, so the text scales the same way
        fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.35), constrained_layout=True)
        draw_pair(axes[0])
        draw_arm_fraction(axes[1], counts, 0, "fraction retreating", "b",
                         "retreat after acquisition")
        draw_arm_fraction(axes[2], counts, 1, "fraction still generalizing at the end", "c",
                         "high accuracy at the end")
        axes[2].legend(frameon=False, loc="lower right", fontsize=9)
        fig.savefig(output, facecolor="white")
        plt.close(fig)
    print("wrote", output)

    print("\ndenominator = runs reaching 0.8; the two arms reach on the same runs by construction")
    for N in INTERVENTION_NS:
        row = f"  N={N:<5}"
        for arm in ARMS:
            r, t, h = counts[(arm, N)]
            row += f"  {arm:9s} reach {r}/30  retreat {t}/{r}  ends high {h}/{r}"
        print(row)
    for arm in ARMS:
        r = sum(counts[(arm, N)][0] for N in INTERVENTION_NS)
        t = sum(counts[(arm, N)][1] for N in INTERVENTION_NS)
        h = sum(counts[(arm, N)][2] for N in INTERVENTION_NS)
        print(f"  pooled {arm:9s} reach {r}/90  retreat {t}/{r}  ends high {h}/{r}")


# ===========================================================================
# Fig. A1: the last feed-forward layer at the peak and at the retreat (App. A.3)
# ===========================================================================

def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.12) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=13, va="top")


def figA1_final_layer_ffn():
    """Fig. A1: the down-projection of the last block's feed-forward layer at the peak and at
    the retreat (App. A.3).

    Drawn from the peak-to-retreat checkpoint pairs that analysis.final_ffn_peak_to_retreat
    measures on the complete input set at L = 12, N = 1280.

    (a) A_block against epoch for the display trajectories; the first one is highlighted and the
        others are grey.  A circle marks the peak checkpoint and a square the retreat checkpoint
        of each trajectory; the dashed and dotted lines are 0.8 and 0.5.
    (b) The target-parity fraction rho_j of the leading feed-forward channel at the output
        positions j = 1, ..., 11, averaged over the pairs, at the peak and at the retreat.
    (c) The loss of bit accuracy at each output position when the leading rank-one component of
        the down-projection is removed at evaluation time, against the removal of a random
        rank-one component of the same norm, averaged over the pairs.

    Reads   results/final_ffn_peak_to_retreat.json (analysis.py final_ffn_peak_to_retreat)
    Writes  figure/figA1_final_layer_ffn.pdf
    """
    source = RESULTS / "final_ffn_peak_to_retreat.json"
    if not source.exists():
        raise SystemExit(f"missing {source}\n"
                         "Run python3 analysis.py final_ffn_peak_to_retreat first.")
    payload = read_json(source)
    results = payload["seeds"]
    display_seeds = [seed for seed in payload["protocol"]["trajectory_display_seeds"]
                     if str(seed) in results]
    seeds = sorted(int(seed) for seed in results)
    positions = np.arange(1, 12)

    fig = plt.figure(figsize=(11.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.0))
    ax_traj = fig.add_subplot(grid[0, 0])
    ax_walsh = fig.add_subplot(grid[0, 1])
    ax_ablate = fig.add_subplot(grid[1, :])

    highlight_seed = display_seeds[0]
    for seed in display_seeds:
        data = results[str(seed)]
        highlighted = seed == highlight_seed
        color = "tab:blue" if highlighted else "0.72"
        ax_traj.plot([row["epoch"] for row in data["trajectory"]],
                     [row["block_accuracy"] for row in data["trajectory"]],
                     color=color, alpha=1.0 if highlighted else 0.32,
                     lw=1.8 if highlighted else 0.8,
                     label=f"seed {seed} (highlighted)" if highlighted else None,
                     zorder=3 if highlighted else 1)
        ax_traj.scatter(data["reference_epoch"], data["reference_block_accuracy"],
                        color=color, alpha=1.0 if highlighted else 0.4, marker="o",
                        s=34 if highlighted else 18, zorder=4 if highlighted else 2)
        ax_traj.scatter(data["retreat_epoch"], data["retreat_block_accuracy"],
                        color=color, alpha=1.0 if highlighted else 0.4, marker="s",
                        s=34 if highlighted else 18, zorder=4 if highlighted else 2)
    ax_traj.axhline(HI, color="0.4", ls="--", lw=0.8)
    ax_traj.axhline(LO, color="0.4", ls=":", lw=0.8)
    ax_traj.set(xlabel="epoch", ylabel="block accuracy",
                title="selected peak-to-retreat trajectories")
    ax_traj.legend(fontsize=8, frameon=False, loc="lower right")
    panel_label(ax_traj, "(a)")

    ref_fracs = np.array([[results[str(s)]["reference"]["channel"]["target_character_fraction"][j]
                           for j in positions] for s in seeds], dtype=float)
    ret_fracs = np.array([[results[str(s)]["retreat"]["channel"]["target_character_fraction"][j]
                           for j in positions] for s in seeds], dtype=float)
    ax_walsh.plot(positions, np.nanmean(ref_fracs, axis=0), "o-", ms=7, label="peak")
    ax_walsh.plot(positions, np.nanmean(ret_fracs, axis=0), "s-", ms=7, label="retreat")
    ax_walsh.set(xlabel="output position j", ylabel=r"target-parity fraction $\rho_j$",
                 title="leading-mode parity content")
    ax_walsh.legend(frameon=False)
    panel_label(ax_walsh, "(b)")

    def ablation(checkpoint, key):
        return np.array([results[str(s)][checkpoint]["ablation"][key] for s in seeds])

    top_ref, rand_ref = ablation("reference", "top_drop"), ablation("reference", "random_drop_mean")
    top_ret, rand_ret = ablation("retreat", "top_drop"), ablation("retreat", "random_drop_mean")
    ax_ablate.plot(positions, top_ref[:, 1:].mean(axis=0), "o-", ms=7, color="tab:blue",
                   label="peak leading mode")
    ax_ablate.plot(positions, top_ret[:, 1:].mean(axis=0), "s--", ms=7, color="tab:orange",
                   label="retreat leading mode")
    ax_ablate.plot(positions, rand_ref[:, 1:].mean(axis=0), "^-", ms=7, color="0.35",
                   label="peak random")
    ax_ablate.plot(positions, rand_ret[:, 1:].mean(axis=0), "v--", ms=7, color="0.6",
                   label="retreat random")
    ax_ablate.axhline(0.0, color="0.75", lw=0.8)
    ax_ablate.set(xlabel="output position j", ylabel="bit-accuracy drop",
                  title="evaluation-time leading-mode ablation")
    ax_ablate.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    panel_label(ax_ablate, "(c)", x=-0.055, y=1.10)

    output = output_path(FIGURE, "figA1_final_layer_ffn.pdf")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output} ({len(seeds)} peak-to-retreat pairs)")


# ===========================================================================
# Fig. B1: bit accuracy, block accuracy and the independent-position estimate (App. B)
# ===========================================================================

TAU_WINDOW = (-50, 50)


def history_column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def scan_checkpoints(length: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A_bit and A_block of every logged epoch of every run of the Transformer scan at L = 12,
    the runs in sorted order of their directory names (the order enters the pooled sums)."""
    root = TRANSFORMER_SIZE_SCAN[length]
    pattern = re.compile(rf"L{length}_N(\d+)_seed(\d+)$")
    bit_values: list[float] = []
    block_values: list[float] = []
    for path in sorted(root.glob(f"L{length}_N*_seed*/metrics.json")):
        if pattern.fullmatch(path.parent.name) is None:
            continue
        for row in load_history(path):
            bit_values.append(float(row["bit_acc"]))
            block_values.append(float(row["block_acc"]))
    if not bit_values:
        raise FileNotFoundError(f"no scan checkpoints under {root}")
    return np.asarray(bit_values, dtype=float), np.asarray(block_values, dtype=float)


def binned_curve(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    """Mean of x and of y in the 30 equal bins of x on [0, 1] that hold at least 20 points."""
    edges = np.linspace(0.0, 1.0, 31)
    index = np.digitize(x, edges) - 1
    mean_x, mean_y = [], []
    for i in range(len(edges) - 1):
        mask = index == i
        if int(mask.sum()) < 20:
            continue
        mean_x.append(float(x[mask].mean()))
        mean_y.append(float(y[mask].mean()))
    return mean_x, mean_y


def figB1_bit_and_block_accuracy():
    """Fig. B1: bit accuracy, block accuracy and the independent-position estimate (App. B).

    If every output position had the mean accuracy A_bit and failed independently, the block
    accuracy of a string of length L would be (A_bit)^L.  The figure tests that estimate for the
    L = 12 Transformer.

    (a) Held-out A_bit and A_block of the 600-epoch run at N = 1280, seed 0, against
        tau = t - t_0.8, over the 50 epochs on either side of t_0.8.
    (b) A_block against (A_bit)^12 for every logged epoch of every run of the L = 12
        training-set-size scan (310 runs of 120 epochs, 37,200 checkpoints).  The black curve is
        the mean in 30 equal bins of the estimate that hold at least 20 checkpoints; the dashed
        line y = x is the independent prediction.

    The checkpoints pool runs and epochs, so they are not independent: the correlation
    and the mean deviations summarize the pooled cloud and carry no error bar.  The centered
    correlation at fixed N and epoch is analysis.py block_vs_bit_centered_correlation.

    Reads   metrics.json of common.REPRESENTATIVE_TRANSFORMER_RUNS/seed0 and of every run of
            common.TRANSFORMER_SIZE_SCAN[12]
    Writes  figure/figB1_bit_and_block_accuracy.pdf
            results/block_vs_bit_summary.csv: the number of checkpoints, the Pearson
            correlation of A_block with (A_bit)^12, and the mean absolute and mean signed
            deviation of A_block from it
    """
    length = 12
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    rows = load_history(REPRESENTATIVE_TRANSFORMER_RUNS / f"seed{TRANSFORMER_SEED}"
                        / "metrics.json")
    transition_epoch = history_crossing(rows)
    window = [row for row in rows
              if TAU_WINDOW[0] <= int(row["epoch"]) - transition_epoch <= TAU_WINDOW[1]]
    tau = np.asarray([int(row["epoch"]) - transition_epoch for row in window], dtype=float)
    ax.plot(tau, history_column(window, "block_acc"), color="#111111", linewidth=2.8,
            label=r"block accuracy $A_{\rm block}$")
    ax.plot(tau, history_column(window, "bit_acc"), color="#006d9c", linestyle=(0, (4.0, 2.0)),
            linewidth=2.7, label=r"bit accuracy $A_{\rm bit}$")
    ax.axvline(0, color="#b00020", linestyle="--", linewidth=1.1)
    ax.axhline(HI, color="0.35", linestyle=":", linewidth=1.0)
    ax.set(title=rf"(a) Single run, $t_{{0.8}}={transition_epoch}$", xlabel=r"$\tau=t-t_{0.8}$",
           ylabel="held-out accuracy", xlim=TAU_WINDOW, ylim=(-0.03, 1.03))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")

    ax = axes[1]
    bit_accuracy, block_accuracy = scan_checkpoints(length)
    independent_proxy = bit_accuracy**length
    ax.scatter(independent_proxy, block_accuracy, s=6, color="#777777", alpha=0.045,
               linewidths=0, rasterized=True, label="checkpoints")
    mean_proxy, mean_block = binned_curve(independent_proxy, block_accuracy)
    ax.plot(mean_proxy, mean_block, color="#111111", linewidth=2.1, label="binned mean")
    grid = np.linspace(0.0, 1.0, 400)
    ax.plot(grid, grid, color="#1b9e77", linestyle="--", linewidth=2.1,
            label=r"independent prediction $y=x$")
    ax.set(title="(b) Test of position independence",
           xlabel=r"independent proxy $(A_{\rm bit})^{12}$",
           ylabel=r"held-out block accuracy $A_{\rm block}$", xlim=(-0.01, 1.01),
           ylim=(-0.03, 1.03))
    ax.grid(alpha=0.18)
    # The scatter is nearly transparent; its legend marker is drawn larger and more opaque.
    legend = ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    legend.legend_handles[0].set_alpha(0.85)
    legend.legend_handles[0].set_sizes([20])

    out_summary = output_path(RESULTS, "block_vs_bit_summary.csv")
    write_rows(out_summary, [{
        "n_snapshots": int(len(block_accuracy)),
        "corr_proxy_block": float(np.corrcoef(independent_proxy, block_accuracy)[0, 1]),
        "mean_abs_block_minus_proxy": float(np.mean(np.abs(block_accuracy - independent_proxy))),
        "mean_block_minus_proxy": float(np.mean(block_accuracy - independent_proxy)),
    }])
    out_pdf = output_path(FIGURE, "figB1_bit_and_block_accuracy.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(out_pdf)
    print(out_summary)


# ===========================================================================
# Figs. C1, C2 and the retreat numbers of Sec. 4.2: the (12, 1280) retreat endpoints (App. C.2)
# ===========================================================================
# The functions below take the soft spins xhat [replicas, inputs, L] of one replica set and the
# truth spins yspin [inputs, L] of its evaluation inputs.  u_rj(b) = xhat_rj(b) yspin_j(b) is the
# truth-aligned soft spin and mu_rj = E_b[u_rj(b)] the truth-alignment profile (common.py).
# Several self-cross gaps here are written as the mean diagonal entry minus the mean entry
# above the diagonal; for a symmetric matrix this is common.gap, summed in another order, and
# the tables of this section keep that order.

OVERLAP_TABLES = RESULTS / "retreat_overlap_structure"
BOOTSTRAP_REPLICATES = 4000        # replica resamples of the overlaps and of the gaps
POSITION_BOOTSTRAP_REPLICATES = 1600   # replica resamples of the position profiles
BOOTSTRAP_SEED = 20_260_716        # the generators use this seed, +2 and + the replica count


def pair_values(matrix: np.ndarray) -> np.ndarray:
    """The entries above the diagonal, one per distinct replica pair."""
    return matrix[np.triu_indices(matrix.shape[0], 1)]


def spectrum_summary(matrix: np.ndarray) -> dict[str, float]:
    """Participation ratio and leading fraction of the spectrum of a replica matrix.

    With the eigenvalues lambda clipped at zero, the participation ratio
    (sum lambda)^2 / sum lambda^2 counts the directions that carry the trace, and the leading
    fraction is lambda_max / sum lambda.
    """
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix)[::-1], 0.0, None)
    total = float(eigenvalues.sum())
    return {
        "participation_ratio": float(total**2 / np.square(eigenvalues).sum()),
        "leading_fraction": float(eigenvalues[0] / total),
    }


def ultrametric_summary(matrix: np.ndarray) -> dict[str, float]:
    """Triple statistics of a replica matrix (float64).

    For every replica triple the three pair entries are sorted, q1 <= q2 <= q3.  On an
    ultrametric tree the two smallest coincide, so the absolute gap q2 - q1 and the relative gap
    (q2 - q1) / (q3 - q1) vanish.  The relative gap is taken over the triples with
    q3 - q1 > 1e-12.  All C(R, 3) triples enter (3.2 million for R = 269), in lexicographic
    order r < s < t, the order of the sums behind the means.
    """
    replicas = matrix.shape[0]
    triples = np.fromiter(chain.from_iterable(combinations(range(replicas), 3)), dtype=np.int64,
                          count=3 * math.comb(replicas, 3)).reshape(-1, 3)
    r, s, t = triples[:, 0], triples[:, 1], triples[:, 2]
    q = np.sort(np.stack([matrix[r, s], matrix[r, t], matrix[s, t]], axis=1), axis=1)
    absolute = q[:, 1] - q[:, 0]
    spread_arr = q[:, 2] - q[:, 0]
    resolved = spread_arr > 1e-12
    relative = absolute[resolved] / spread_arr[resolved]
    return {
        "triples": int(len(absolute)),
        "absolute_gap_mean": float(absolute.mean()),
        "absolute_gap_median": float(np.median(absolute)),
        "absolute_gap_q90": float(np.quantile(absolute, 0.90)),
        "relative_gap_mean": float(relative.mean()),
        "relative_gap_median": float(np.median(relative)),
        "relative_gap_q90": float(np.quantile(relative, 0.90)),
        "fraction_relative_gap_below_0p1": float((relative < 0.1).mean()),
        "spread_mean": float(spread_arr.mean()),
    }


def plateau_summary(matrix: np.ndarray) -> dict[str, float]:
    """The largest jump between adjacent sorted pair entries against their range.

    A hierarchy with a few resolved levels puts the sorted pair overlaps on plateaus separated
    by large jumps.
    """
    pairs = np.sort(pair_values(matrix).astype(np.float64))
    gaps = np.diff(pairs)
    width = float(pairs[-1] - pairs[0])
    max_gap = float(gaps.max())
    return {
        "pairs": int(len(pairs)),
        "min": float(pairs[0]),
        "max": float(pairs[-1]),
        "range": width,
        "max_adjacent_gap": max_gap,
        "max_adjacent_gap_over_range": float(max_gap / width),
        "median_adjacent_gap": float(np.median(gaps)),
        "q99_adjacent_gap": float(np.quantile(gaps, 0.99)),
    }


def tree_projection_summary(matrix: np.ndarray) -> dict[str, float]:
    """Fit of an average-linkage tree built on the distance max(q) - q.

    The cophenetic correlation and the R^2 of the tree-implied pair entries measure how much
    of the matrix a tree represents; both are in-sample.
    """
    pairs = pair_values(matrix).astype(np.float64)
    # a monotone overlap-to-distance map; the zero of the distance is arbitrary
    distances = float(pairs.max()) - matrix.astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    condensed = squareform(distances, checks=False)
    tree = linkage(condensed, method="average")
    corr, cophenetic_distances = cophenet(tree, condensed)
    projected_pairs = float(pairs.max()) - cophenetic_distances
    residual = pairs - projected_pairs
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(pairs - pairs.mean()).sum())
    return {
        "average_linkage_cophenetic_correlation": float(corr),
        "average_linkage_tree_r2": float(1.0 - ss_res / ss_tot),
        "average_linkage_tree_rmse": float(np.sqrt(ss_res / len(pairs))),
    }


def linear_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """R^2 and slope of the least-squares line of y on x."""
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    return float(1.0 - ss_res / ss_tot), float(coef[1])


def hierarchy_tests(matrix: np.ndarray) -> dict[str, dict]:
    """The three finite-replica tests of a resolved hierarchy on one replica matrix."""
    return {
        "ultrametric": ultrametric_summary(matrix),
        "plateaus": plateau_summary(matrix),
        "tree_projection": tree_projection_summary(matrix),
    }


def connected_hierarchy(q_connected: np.ndarray, m_r: np.ndarray) -> dict[str, object]:
    """hierarchy_tests of a connected-type matrix and its dependence on truth alignment alone.

    The off-diagonal entries are regressed on two functions of the scalar truth alignment: the
    kernel min(m_r, m_s) - m_r m_s, which is the connected overlap of two hard-step profiles of
    depths m_r and m_s, and the depth mismatch |m_r - m_s|.  m_quartile_connected_means is the
    mean of the matrix inside and between the four quartile groups of m_r (within a group over
    its distinct pairs).
    """
    q_connected = np.asarray(q_connected, dtype=np.float64)
    m_r = np.asarray(m_r, dtype=np.float64)
    upper = np.triu_indices(len(m_r), 1)
    frontier_kernel = np.minimum.outer(m_r, m_r) - np.outer(m_r, m_r)
    depth_distance = np.abs(m_r[:, None] - m_r[None, :])
    connected_pairs = q_connected[upper]
    frontier_pairs = frontier_kernel[upper]
    depth_pairs = depth_distance[upper]
    frontier_r2, frontier_slope = linear_r2(frontier_pairs, connected_pairs)
    depth_r2, depth_slope = linear_r2(depth_pairs, connected_pairs)
    quartiles = np.array_split(np.argsort(m_r), 4)
    quartile_means: list[list[float]] = []
    for first in quartiles:
        row = []
        for second in quartiles:
            values = q_connected[np.ix_(first, second)]
            if np.array_equal(first, second):
                values = values[~np.eye(len(first), dtype=bool)]
            row.append(float(values.mean()))
        quartile_means.append(row)
    return {
        **hierarchy_tests(q_connected),
        "frontier_kernel_corr": float(np.corrcoef(connected_pairs, frontier_pairs)[0, 1]),
        "frontier_kernel_r2": frontier_r2,
        "frontier_kernel_slope": frontier_slope,
        "depth_distance_corr": float(np.corrcoef(connected_pairs, depth_pairs)[0, 1]),
        "depth_distance_r2": depth_r2,
        "depth_distance_slope": depth_slope,
        "m_quartile_connected_means": quartile_means,
    }


def hierarchy_diagnostics(q_raw: np.ndarray, q_connected: np.ndarray, m_r: np.ndarray) -> dict:
    """Finite-replica tests of a Parisi-like hierarchy on Q^raw and on Q^c."""
    return {
        "criterion": (
            "For each triple, sorted overlaps q1<=q2<=q3 should have q1~=q2 "
            "in an ultrametric Parisi hierarchy."
        ),
        "raw": hierarchy_tests(np.asarray(q_raw, dtype=np.float64)),
        "connected": connected_hierarchy(q_connected, m_r),
    }


def overlap_statistics(xhat: np.ndarray, yspin: np.ndarray) -> dict[str, object]:
    """Q^raw, Q^c, W and the overlaps of the replicas in xhat, on all positions.

    m_r, Q^raw, Q^c and W are common.overlap_matrices in float64.  q_self_r = E_bj[xhat_rj^2] is
    the mean square soft spin, the diagonal of Q^raw summed in another order, and
    c_self_r = q_self_r - m_r^2.  The ensemble averages are

        m, q_self, q_cross     means over replicas and over distinct pairs
        raw_gap                q_self - q_cross = chi_SG
        connected_gap          mean of c_self_r minus the mean off-diagonal entry of Q^c
        corr_raw_vs_m_product  corr over pairs r < s of Q^raw_rs and m_r m_s

    with the spread of the pair entries and the spectra of Q^raw and Q^c.
    """
    m_r, q_raw, q_connected, w = overlap_matrices(xhat, yspin, slice(None))
    q_self_r = np.square(xhat.astype(np.float64)).mean(axis=(1, 2))
    raw_cross = pair_values(q_raw)
    connected_cross = pair_values(q_connected)
    ferro_cross = pair_values(np.outer(m_r, m_r))
    c_self_r = q_self_r - np.square(m_r)
    return {
        "m_r": m_r,
        "q_self_r": q_self_r,
        "c_self_r": c_self_r,
        "q_raw": q_raw,
        "q_connected": q_connected,
        "w": w,
        "raw_cross": raw_cross,
        "connected_cross": connected_cross,
        "ferro_cross": ferro_cross,
        "m": float(m_r.mean()),
        "q_self": float(q_self_r.mean()),
        "q_cross": float(raw_cross.mean()),
        "raw_gap": float(q_self_r.mean() - raw_cross.mean()),
        "connected_gap": float(c_self_r.mean() - connected_cross.mean()),
        "corr_raw_vs_m_product": float(np.corrcoef(raw_cross, ferro_cross)[0, 1]),
        "raw_cross_std": float(raw_cross.std()),
        "raw_cross_max": float(raw_cross.max()),
        "connected_cross_std": float(connected_cross.std()),
        "connected_cross_max": float(connected_cross.max()),
        "pairs_raw_above_0p7": int((raw_cross > 0.7).sum()),
        "total_pairs": int(raw_cross.size),
        "raw_spectrum": spectrum_summary(q_raw),
        "connected_spectrum": spectrum_summary(q_connected),
    }


def bootstrap_overlap(stats: dict[str, object]) -> dict[str, dict[str, float]]:
    """Replica bootstrap of m, q_self, q_cross, both gaps and corr(Q^raw_rs, m_r m_s).

    Each of the 4000 draws takes as many replicas as the set holds, with replacement, and
    evaluates the six quantities on the resampled rows and columns of Q^raw; a replica drawn
    twice contributes its diagonal entry to the off-diagonal average.  The interval is the
    central 95% of the draws.  It is conditional on the training set, the evaluation set and
    the endpoint selection, which are not resampled.
    """
    m_r = np.asarray(stats["m_r"])
    q_self_r = np.asarray(stats["q_self_r"])
    q_raw = np.asarray(stats["q_raw"])
    replicas = len(m_r)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = ["m", "q_self", "q_cross", "raw_gap", "connected_gap", "corr"]
    draws = {name: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for name in names}
    upper = np.triu_indices(replicas, 1)
    for b in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, replicas, size=replicas)
        m = m_r[index]
        q_self = q_self_r[index]
        raw = q_raw[np.ix_(index, index)]
        ferro = np.outer(m, m)
        raw_cross = raw[upper]
        ferro_cross = ferro[upper]
        connected = raw - ferro
        c_self = q_self - np.square(m)
        draws["m"][b] = m.mean()
        draws["q_self"][b] = q_self.mean()
        draws["q_cross"][b] = raw_cross.mean()
        draws["raw_gap"][b] = q_self.mean() - raw_cross.mean()
        draws["connected_gap"][b] = c_self.mean() - connected[upper].mean()
        draws["corr"][b] = np.corrcoef(raw_cross, ferro_cross)[0, 1]

    estimate_names = {"m": "m", "q_self": "q_self", "q_cross": "q_cross", "raw_gap": "raw_gap",
                      "connected_gap": "connected_gap", "corr": "corr_raw_vs_m_product"}
    out: dict[str, dict[str, float]] = {}
    for name in names:
        lo, hi = np.quantile(draws[name], [0.025, 0.975])
        out[name] = {
            "estimate": float(stats[estimate_names[name]]),
            "bootstrap_se": float(draws[name].std(ddof=1)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }
    return out


def jsonable_overlap(stats: dict[str, object]) -> dict[str, object]:
    """The scalars of overlap_statistics under the names of the summary table."""
    return {
        "m": stats["m"],
        "q_self": stats["q_self"],
        "q_cross": stats["q_cross"],
        "q_self_minus_q_cross": stats["raw_gap"],
        "chi_SG_connected": stats["connected_gap"],
        "corr_q_vs_m_product": stats["corr_raw_vs_m_product"],
        "raw_cross_std": stats["raw_cross_std"],
        "raw_cross_max": stats["raw_cross_max"],
        "connected_cross_std": stats["connected_cross_std"],
        "connected_cross_max": stats["connected_cross_max"],
        "pairs_raw_above_0p7": stats["pairs_raw_above_0p7"],
        "total_pairs": stats["total_pairs"],
        "raw_participation_ratio": stats["raw_spectrum"]["participation_ratio"],
        "raw_leading_eigenvalue_fraction": stats["raw_spectrum"]["leading_fraction"],
        "connected_participation_ratio": stats["connected_spectrum"]["participation_ratio"],
        "connected_leading_eigenvalue_fraction": stats["connected_spectrum"]["leading_fraction"],
    }


def decreasing_isotonic(values: np.ndarray) -> np.ndarray:
    """The least-squares non-increasing fit of a sequence (pool-adjacent-violators)."""
    blocks: list[list[float | int]] = []
    for value in np.asarray(values, dtype=np.float64):
        blocks.append([float(value), 1])
        while (
            len(blocks) >= 2
            and float(blocks[-2][0]) / int(blocks[-2][1])
            < float(blocks[-1][0]) / int(blocks[-1][1])
        ):
            right_block = blocks.pop()
            left_block = blocks.pop()
            blocks.append([float(left_block[0]) + float(right_block[0]),
                           int(left_block[1]) + int(right_block[1])])
    return np.concatenate([np.full(int(count), float(total) / int(count), dtype=np.float64)
                           for total, count in blocks])


def offdiag_stats(matrix: np.ndarray, upper) -> dict[str, float]:
    values = matrix[upper]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def profile_residual_decomposition(stats: dict[str, object], xhat: np.ndarray,
                                   yspin: np.ndarray) -> dict:
    """Split Q^c into its mean-profile part and its residual, Q^c = C^prof + W.

        C^prof_rs = E_j[mu_rj mu_sj] - m_r m_s
        W_rs      = E_bj[(u_rj(b) - mu_rj) (u_sj(b) - mu_sj)]

    C^prof is what two replicas share through the shape of their mean profiles, in particular
    through the depth of their truth frontier; W is the input-dependent covariance left once
    each replica's profile is removed.  The identity holds with no fitted coefficient, and
    exact_decomposition_max_abs_error records how closely it closes numerically.  W is the one
    of overlap_statistics (stats["w"]).  `xhat` is passed in its stored float32, from which the
    positionwise gaps g_j are computed directly.

    Returns m_profile (mu_rj), the matrices W (exact_profile_residual) and R^W
    (exact_profile_correlation), the positionwise gaps position_residual_gap, and a summary:

    * gap(C^prof) and gap(W) with replica-bootstrap intervals (4000 draws of the rows and
      columns of both matrices, central 95%), and the share gap(W) / gap(Q^c);
    * least-squares fits of the off-diagonal of Q^c on the hard-step kernel
      min(m_r, m_s) - m_r m_s and on C^prof, and what is left of Q^c after removing its one or
      two leading eigenvectors;
    * the principal components of the profiles mu_rj across replicas, and the center and width
      of each replica's frontier.  The width is measured on the decreasing isotonic fit of the
      profile: the drops of the fitted profile between 1 before the first position and 0 after
      the last form a distribution over the boundaries between positions, and center and width
      are its mean and standard deviation;
    * the spectrum of W, of the doubly centered R^W, and connected_hierarchy of W;
    * how R^W depends on the depth mismatch |m_r - m_s|: its correlation with it, its mean over
      the pairs with mismatch <= 0.05 and >= 0.40, its mean inside and between the quartile
      groups of m_r;
    * g_j = gap(K^(j)) with K^(j)_rs = Cov_b(u_rj, u_sj) = E_b[xhat_rj xhat_sj] - mu_rj mu_sj
      (y_j^2 = 1), whose average over j is gap(W), and the share of sum_j g_j carried by
      positions 6 to 11;
    * a split-half check: W and R^W computed separately on the first and on the second half of
      the evaluation inputs, the profile taken within each half, and their off-diagonal entries
      correlated.
    """
    q_connected = np.asarray(stats["q_connected"], dtype=np.float64)
    m_r = np.asarray(stats["m_r"], dtype=np.float64)
    exact_profile_residual = stats["w"]
    replicas = len(m_r)
    upper = np.triu_indices(replicas, 1)
    aligned = np.asarray(xhat, dtype=np.float64) * np.asarray(yspin, dtype=np.float64)[None]
    m_profile = aligned.mean(axis=1)
    profile_kernel = (m_profile @ m_profile.T) / m_profile.shape[1] - np.outer(m_r, m_r)
    depth_kernel = np.minimum.outer(m_r, m_r) - np.outer(m_r, m_r)

    def fit_offdiag(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Coefficients, residual matrix and off-diagonal R^2 of Q^c ~ a + b kernel."""
        x = kernel[upper]
        y = q_connected[upper]
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = q_connected - (coef[0] + coef[1] * kernel)
        r2 = 1.0 - residual[upper].var() / y.var()
        return coef, residual, float(r2)

    depth_coef, depth_residual, depth_r2 = fit_offdiag(depth_kernel)
    profile_coef, profile_residual, profile_r2 = fit_offdiag(profile_kernel)

    # Q^c minus its one or two leading eigenmodes (eigenvalues clipped at zero), symmetrized
    eigenvalues, eigenvectors = np.linalg.eigh(q_connected)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    psd_residuals: dict[int, np.ndarray] = {}
    for components in (1, 2):
        residual = q_connected.copy()
        for component in range(components):
            residual -= eigenvalues[component] * np.outer(
                eigenvectors[:, component], eigenvectors[:, component])
        psd_residuals[components] = 0.5 * (residual + residual.T)

    # Principal components of the profiles; PC1 is oriented to correlate positively with m_r,
    # and PC2 is regressed on a cubic in m_r.
    centered_profile = m_profile - m_profile.mean(axis=0, keepdims=True)
    left, singular_values, right = np.linalg.svd(centered_profile, full_matrices=False)
    profile_variance = np.square(singular_values)
    profile_variance /= profile_variance.sum()
    profile_scores = left[:, :2] * singular_values[:2]
    profile_loadings = right[:2]
    if np.corrcoef(profile_scores[:, 0], m_r)[0, 1] < 0:
        profile_scores[:, 0] *= -1
        profile_loadings[0] *= -1
    cubic_depth_design = np.column_stack([np.ones_like(m_r), m_r, np.square(m_r), np.power(m_r, 3)])
    cubic_depth_coef, *_ = np.linalg.lstsq(cubic_depth_design, profile_scores[:, 1], rcond=None)
    cubic_depth_residual = profile_scores[:, 1] - cubic_depth_design @ cubic_depth_coef
    profile_pc2_depth_cubic_r2 = float(
        1.0
        - np.square(cubic_depth_residual).sum()
        / np.square(profile_scores[:, 1] - profile_scores[:, 1].mean()).sum()
    )

    profile_width = np.empty(replicas, dtype=np.float64)
    profile_center = np.empty(replicas, dtype=np.float64)
    boundary_positions = np.arange(m_profile.shape[1] + 1, dtype=np.float64) - 0.5
    for replica, profile in enumerate(m_profile):
        monotone = decreasing_isotonic(profile)
        padded = np.concatenate([[1.0], monotone, [0.0]])
        drops = np.maximum(padded[:-1] - padded[1:], 0.0)
        drops /= drops.sum()
        center = float(np.sum(drops * boundary_positions))
        profile_center[replica] = center
        profile_width[replica] = float(
            np.sqrt(np.sum(drops * np.square(boundary_positions - center))))

    exact_diag = np.diag(exact_profile_residual)
    exact_pairs = exact_profile_residual[upper]
    exact_gap = float(exact_diag.mean() - exact_pairs.mean())
    profile_gap = float(np.diag(profile_kernel).mean() - profile_kernel[upper].mean())
    gap_draws_profile = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    gap_draws_residual = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    gap_rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    for draw in range(BOOTSTRAP_REPLICATES):
        index = gap_rng.integers(0, replicas, size=replicas)
        sampled_profile = profile_kernel[np.ix_(index, index)]
        sampled_residual = exact_profile_residual[np.ix_(index, index)]
        gap_draws_profile[draw] = np.diag(sampled_profile).mean() - sampled_profile[upper].mean()
        gap_draws_residual[draw] = np.diag(sampled_residual).mean() - sampled_residual[upper].mean()

    def gap_bootstrap_summary(estimate: float, draws: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(draws, [0.025, 0.975])
        return {"estimate": estimate, "bootstrap_se": float(draws.std(ddof=1)),
                "ci95_low": float(low), "ci95_high": float(high)}

    exact_spectrum = spectrum_summary(exact_profile_residual)
    scale = np.sqrt(np.outer(exact_diag, exact_diag))
    exact_correlation = exact_profile_residual / scale
    # R^W with the replica mean removed from rows and columns
    projector = np.eye(replicas) - np.ones((replicas, replicas)) / replicas
    centered_correlation = projector @ exact_correlation @ projector
    centered_eigenvalues = np.clip(np.linalg.eigvalsh(centered_correlation)[::-1], 0.0, None)
    centered_total = float(centered_eigenvalues.sum())
    centered_pr = float(centered_total**2 / np.square(centered_eigenvalues).sum())
    centered_leading_fraction = float(centered_eigenvalues[0] / centered_total)
    exact_residual_hierarchy = connected_hierarchy(exact_profile_residual, m_r)
    depth_mismatch = np.abs(m_r[:, None] - m_r[None, :])
    normalized_exact_pairs = exact_correlation[upper]
    exact_residual_depth_distance_corr = float(
        np.corrcoef(exact_pairs, depth_mismatch[upper])[0, 1])
    normalized_residual_depth_distance_corr = float(
        np.corrcoef(normalized_exact_pairs, depth_mismatch[upper])[0, 1])
    exact_residual_vs_correlation_pattern_corr = float(
        np.corrcoef(exact_pairs, normalized_exact_pairs)[0, 1])
    # each replica's partner of nearest m_r
    depth_for_nearest = depth_mismatch.copy()
    np.fill_diagonal(depth_for_nearest, np.inf)
    nearest_depth_replica = np.argmin(depth_for_nearest, axis=1)
    nearest_depth_residual_mean = float(
        exact_profile_residual[np.arange(replicas), nearest_depth_replica].mean())
    nearest_depth_correlation_mean = float(
        exact_correlation[np.arange(replicas), nearest_depth_replica].mean())
    residual_row_mean = (exact_profile_residual.sum(axis=1) - exact_diag) / (replicas - 1)
    normalized_row_mean = ((exact_correlation.sum(axis=1) - np.diag(exact_correlation))
                           / (replicas - 1))
    residual_row_mean_m_corr = float(np.corrcoef(residual_row_mean, m_r)[0, 1])
    normalized_row_mean_m_corr = float(np.corrcoef(normalized_row_mean, m_r)[0, 1])

    def quartile_block_means(matrix: np.ndarray) -> list[list[float]]:
        groups = np.array_split(np.argsort(m_r), 4)
        block_means: list[list[float]] = []
        for first_index, first in enumerate(groups):
            row: list[float] = []
            for second_index, second in enumerate(groups):
                block = matrix[np.ix_(first, second)]
                if first_index == second_index:
                    block = block[~np.eye(len(first), dtype=bool)]
                row.append(float(block.mean()))
            block_means.append(row)
        return block_means

    residual_quartile_means = quartile_block_means(exact_profile_residual)
    normalized_residual_quartile_means = quartile_block_means(exact_correlation)
    same_depth = depth_mismatch[upper] <= 0.05
    far_depth = depth_mismatch[upper] >= 0.40
    pair_gap = 0.5 * (exact_diag[upper[0]] + exact_diag[upper[1]]) - exact_pairs

    # g_j: the product E_b[xhat_rj xhat_sj] is taken in the stored float32, the profile term
    # mu_rj mu_sj in float64
    position_gap = np.empty(xhat.shape[2], dtype=np.float64)
    for position in range(xhat.shape[2]):
        values = xhat[:, :, position]
        raw_position = values @ values.T / values.shape[1]
        connected_position = raw_position - np.outer(m_profile[:, position], m_profile[:, position])
        position_gap[position] = (
            np.diag(connected_position).mean() - connected_position[upper].mean()
        )

    # W on each half of the evaluation inputs, the profile taken within the half
    split = xhat.shape[1] // 2
    first_half_residual, second_half_residual = (
        overlap_matrices(xhat[:, half, :], yspin[half], slice(None))[3]
        for half in (np.arange(split), np.arange(split, xhat.shape[1])))
    first_half_correlation = normalize_by_diagonal(first_half_residual)
    second_half_correlation = normalize_by_diagonal(second_half_residual)
    residual_split_half_corr = float(
        np.corrcoef(first_half_residual[upper], second_half_residual[upper])[0, 1])
    correlation_split_half_corr = float(
        np.corrcoef(first_half_correlation[upper], second_half_correlation[upper])[0, 1])

    return {
        "exact_profile_residual": exact_profile_residual,
        "exact_profile_correlation": exact_correlation,
        "m_profile": m_profile,
        "position_residual_gap": position_gap,
        "summary": {
            "depth_kernel_r2": depth_r2,
            "depth_kernel_intercept": float(depth_coef[0]),
            "depth_kernel_slope": float(depth_coef[1]),
            "profile_kernel_r2": profile_r2,
            "profile_kernel_intercept": float(profile_coef[0]),
            "profile_kernel_slope": float(profile_coef[1]),
            "exact_decomposition_max_abs_error": float(
                np.max(np.abs(q_connected - profile_kernel - exact_profile_residual))),
            "profile_gap": profile_gap,
            "profile_gap_replica_bootstrap": gap_bootstrap_summary(profile_gap, gap_draws_profile),
            "exact_profile_residual_gap": exact_gap,
            "exact_profile_residual_gap_replica_bootstrap": gap_bootstrap_summary(
                exact_gap, gap_draws_residual),
            "exact_profile_residual_fraction_of_connected_gap": float(
                exact_gap / (np.diag(q_connected).mean() - q_connected[upper].mean())),
            "exact_profile_residual_diag_mean": float(exact_diag.mean()),
            "exact_profile_residual_offdiag": offdiag_stats(exact_profile_residual, upper),
            "exact_profile_residual_leading_fraction": float(exact_spectrum["leading_fraction"]),
            "exact_profile_residual_participation_ratio": float(
                exact_spectrum["participation_ratio"]),
            "exact_profile_residual_centered_correlation_leading_fraction": (
                centered_leading_fraction),
            "exact_profile_residual_centered_correlation_participation_ratio": centered_pr,
            "exact_profile_residual_hierarchy": exact_residual_hierarchy,
            "exact_profile_residual_depth_distance_corr": exact_residual_depth_distance_corr,
            "exact_profile_residual_correlation_depth_distance_corr": (
                normalized_residual_depth_distance_corr),
            "exact_profile_residual_vs_correlation_pattern_corr": (
                exact_residual_vs_correlation_pattern_corr),
            "exact_profile_residual_nearest_depth_mean": nearest_depth_residual_mean,
            "exact_profile_correlation_nearest_depth_mean": nearest_depth_correlation_mean,
            "exact_profile_correlation_offdiag_mean": float(normalized_exact_pairs.mean()),
            "exact_profile_residual_same_depth_0p05_mean": float(exact_pairs[same_depth].mean()),
            "exact_profile_correlation_same_depth_0p05_mean": float(
                normalized_exact_pairs[same_depth].mean()),
            "exact_profile_residual_far_depth_0p40_mean": float(exact_pairs[far_depth].mean()),
            "exact_profile_correlation_far_depth_0p40_mean": float(
                normalized_exact_pairs[far_depth].mean()),
            "far_depth_0p40_pairs": int(far_depth.sum()),
            "exact_profile_residual_row_mean_m_corr": residual_row_mean_m_corr,
            "exact_profile_correlation_row_mean_m_corr": normalized_row_mean_m_corr,
            "exact_profile_residual_diag_width_corr": float(
                np.corrcoef(exact_diag, profile_width)[0, 1]),
            "exact_profile_residual_diag_m_corr": float(np.corrcoef(exact_diag, m_r)[0, 1]),
            "exact_profile_residual_split_half_offdiag_corr": residual_split_half_corr,
            "exact_profile_correlation_split_half_offdiag_corr": correlation_split_half_corr,
            "exact_profile_residual_m_quartile_means": residual_quartile_means,
            "exact_profile_correlation_m_quartile_means": normalized_residual_quartile_means,
            "same_depth_0p05_pairs": int(same_depth.sum()),
            "same_depth_0p05_profile_residual_gap": float(pair_gap[same_depth].mean()),
            "profile_pc1_variance_fraction": float(profile_variance[0]),
            "profile_pc2_variance_fraction": float(profile_variance[1]),
            "profile_pc1_pc2_variance_fraction": float(profile_variance[:2].sum()),
            "profile_pc1_m_correlation": float(np.corrcoef(profile_scores[:, 0], m_r)[0, 1]),
            "profile_pc1_center_correlation": float(
                np.corrcoef(profile_scores[:, 0], profile_center)[0, 1]),
            "profile_pc2_m_correlation": float(np.corrcoef(profile_scores[:, 1], m_r)[0, 1]),
            "profile_pc2_width_correlation": float(
                np.corrcoef(profile_scores[:, 1], profile_width)[0, 1]),
            "profile_pc2_depth_cubic_r2": profile_pc2_depth_cubic_r2,
            "profile_width_mean": float(profile_width.mean()),
            "profile_width_std": float(profile_width.std()),
            "profile_width_min": float(profile_width.min()),
            "profile_width_max": float(profile_width.max()),
            "profile_width_m_correlation": float(np.corrcoef(profile_width, m_r)[0, 1]),
            "position_residual_gap": position_gap.tolist(),
            "late_positions_6_to_11_fraction_of_residual_gap": float(
                position_gap[6:12].sum() / position_gap.sum()),
            "connected_offdiag": offdiag_stats(q_connected, upper),
            "depth_residual_offdiag": offdiag_stats(depth_residual, upper),
            "profile_residual_offdiag": offdiag_stats(profile_residual, upper),
            "psd_residual_1_offdiag": offdiag_stats(psd_residuals[1], upper),
            "psd_residual_2_offdiag": offdiag_stats(psd_residuals[2], upper),
            "psd_residual_1_diag_mean": float(np.diag(psd_residuals[1]).mean()),
            "psd_residual_2_diag_mean": float(np.diag(psd_residuals[2]).mean()),
        },
    }


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """A_rs / sqrt(A_rr A_ss), symmetrized, with unit diagonal, clipped to [-1, 1]."""
    normalized = normalize_by_diagonal(matrix)
    normalized = 0.5 * (normalized + normalized.T)
    np.fill_diagonal(normalized, 1.0)
    return np.clip(normalized, -1.0, 1.0)


def position_components(xhat: np.ndarray, yspin: np.ndarray) -> dict[str, np.ndarray]:
    """Per-replica, per-position ingredients of the position-resolved overlaps.

    For every replica r and position j: mu_rj, E_b[xhat_rj^2], the fraction of inputs with
    |xhat_rj| < 0.2 (uncertain) and with u_rj < -0.5 (confidently wrong), and the replica matrix
    q_matrices[j] = E_b[xhat_rj xhat_sj] of position j, so that a bootstrap can resample
    replicas without going back to the inputs.  The arithmetic is in the dtype of xhat, which
    the caller passes as stored (float32).
    """
    replicas, n_inputs, length = xhat.shape
    m_r = (xhat * yspin[None]).mean(axis=1)
    q_self_r = np.square(xhat).mean(axis=1)
    uncertain_r = (np.abs(xhat) < 0.2).mean(axis=1)
    confident_wrong_r = (xhat * yspin[None] < -0.5).mean(axis=1)
    q_matrices = np.empty((length, replicas, replicas), dtype=np.float64)
    for position in range(length):
        values = xhat[:, :, position]
        q_matrices[position] = (values @ values.T) / n_inputs
    return {
        "m_r": m_r,
        "q_self_r": q_self_r,
        "uncertain_r": uncertain_r,
        "confident_wrong_r": confident_wrong_r,
        "q_matrices": q_matrices,
    }


POSITION_PROFILE_KEYS = ["m", "q_self", "q_cross", "chi_connected", "delta_N",
                         "uncertain_fraction", "confident_wrong_fraction"]


def position_profiles(components: dict[str, np.ndarray],
                      index: np.ndarray) -> dict[str, np.ndarray]:
    """Position profiles of the overlaps of the replicas listed in `index`.

    As functions of j: m(j), q_self(j), q_cross(j), the connected gap chi^c_SG(j) (mean diagonal
    minus mean off-diagonal entry of the connected overlap at position j), Delta_N(j) =
    q_self(j) - m(j), and the uncertain and confidently-wrong fractions.  `index` may repeat
    replicas, as a bootstrap draw does.
    """
    m_r = components["m_r"][index]
    q_self_r = components["q_self_r"][index]
    uncertain_r = components["uncertain_r"][index]
    confident_wrong_r = components["confident_wrong_r"][index]
    q_matrices = components["q_matrices"][:, index][:, :, index]
    upper = np.triu_indices(len(index), 1)
    q_cross = np.array([matrix[upper].mean() for matrix in q_matrices])
    q_connected_cross = np.array(
        [(matrix - np.outer(m_r[:, j], m_r[:, j]))[upper].mean()
         for j, matrix in enumerate(q_matrices)]
    )
    c_self = (q_self_r - np.square(m_r)).mean(axis=0)
    q_self = q_self_r.mean(axis=0)
    m = m_r.mean(axis=0)
    return {
        "m": m,
        "q_self": q_self,
        "q_cross": q_cross,
        "chi_connected": c_self - q_connected_cross,
        "delta_N": q_self - m,
        "uncertain_fraction": uncertain_r.mean(axis=0),
        "confident_wrong_fraction": confident_wrong_r.mean(axis=0),
    }


def bootstrap_position_profiles(components: dict[str, np.ndarray]) -> dict[str, dict[str, list]]:
    """Position profiles with pointwise 95% replica-bootstrap bands, as nested lists.

    The estimate uses every replica once; the band is the 2.5th to 97.5th percentile over 1600
    resamples of the replicas with replacement, taken separately at each position.  The seed of
    the generator is BOOTSTRAP_SEED plus the replica count, so two groups of different size
    draw independent resamples.
    """
    replicas = components["m_r"].shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED + replicas)
    estimate = position_profiles(components, np.arange(replicas))
    draws = {key: np.empty((POSITION_BOOTSTRAP_REPLICATES, estimate[key].size), dtype=np.float64)
             for key in POSITION_PROFILE_KEYS}
    for b in range(POSITION_BOOTSTRAP_REPLICATES):
        index = rng.integers(0, replicas, size=replicas)
        profile = position_profiles(components, index)
        for key in POSITION_PROFILE_KEYS:
            draws[key][b] = profile[key]
    return {
        key: {
            "estimate": estimate[key].tolist(),
            "ci95_low": np.quantile(draws[key], 0.025, axis=0).tolist(),
            "ci95_high": np.quantile(draws[key], 0.975, axis=0).tolist(),
        }
        for key in estimate
    }


def draw_overlap_structure(stats: dict[str, object], m_profile: np.ndarray,
                           frontier: np.ndarray) -> Path:
    """Fig. C1, written to figure/figC1_raw_fragmentation_frontier_order.pdf.

    (a) Distribution of the raw pair overlaps Q^raw_rs with q_cross and q_self.  (b) Distribution
    of the connected pair overlaps Q^c_rs with the mean c_self_r.  (c) Q^raw_rs against m_r m_s
    for every distinct pair.  (d) The profiles mu_rj, replicas sorted by frontier position
    (white curve).
    """
    raw_cross = np.asarray(stats["raw_cross"])
    connected_cross = np.asarray(stats["connected_cross"])
    c_self = np.asarray(stats["c_self_r"])
    ferro_cross = np.asarray(stats["ferro_cross"])
    order = np.argsort(frontier)
    replicas = len(frontier)

    fig, axes = plt.subplot_mosaic([["a", "b"], ["c", "d"]], figsize=(13.6, 9.8),
                                   constrained_layout=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.15]})
    axes["a"].hist(raw_cross, bins=24, density=True, color="#b64a4a", alpha=0.78)
    axes["a"].axvline(stats["q_cross"], color="#8c2d2d", lw=1.4,
                      label=rf"$\langle q_{{\rm cross}}\rangle={stats['q_cross']:.3f}$")
    axes["a"].axvline(stats["q_self"], color="black", ls="--", lw=1.2,
                      label=rf"$\langle q_{{\rm self}}\rangle={stats['q_self']:.3f}$")
    axes["a"].set(
        xlabel=r"raw pair overlap $Q^{\mathrm{raw}}_{r,s}$", ylabel="density",
        title=rf"(a) Fragmentation: $q_{{\rm self}}-q_{{\rm cross}}={stats['raw_gap']:.3f}$",
    )
    axes["a"].legend(fontsize=8)

    axes["b"].hist(connected_cross, bins=26, density=True, color="#4c78a8", alpha=0.78)
    axes["b"].axvline(connected_cross.mean(), color="#2f5f8f", lw=1.4,
                      label=rf"$\langle Q^c_{{r,s}}\rangle={connected_cross.mean():.3f}$")
    axes["b"].axvline(c_self.mean(), color="black", ls="--", lw=1.2,
                      label=rf"$\langle Q^c_{{r,r}}\rangle={c_self.mean():.3f}$")
    axes["b"].set(xlabel=r"connected pair overlap $Q^c_{r,s}$", ylabel="density",
                  title=rf"(b) Connected gap: ${stats['connected_gap']:.3f}$")
    axes["b"].legend(fontsize=8)

    axes["c"].scatter(ferro_cross, raw_cross, s=10, alpha=0.30, color="#b64a4a")
    limits = [min(ferro_cross.min(), raw_cross.min()), max(ferro_cross.max(), raw_cross.max())]
    axes["c"].plot(limits, limits, "k--", lw=1)
    axes["c"].set(
        xlabel=r"truth component $m_r m_s$", ylabel=r"$Q^{\mathrm{raw}}_{r,s}$",
        title=rf"(c) Raw overlap vs $m_r m_s$: corr. $={stats['corr_raw_vs_m_product']:.3f}$",
    )

    profile_image = axes["d"].imshow(m_profile[order], vmin=0.0, vmax=1.0, cmap="viridis",
                                     aspect="auto", interpolation="nearest")
    axes["d"].plot(frontier[order], np.arange(replicas), color="white", lw=1.2)
    axes["d"].set(xlabel="output position $j$", ylabel="replicas sorted by half-height frontier",
                  title="(d) Position-resolved truth frontier")
    axes["d"].set_xticks(np.arange(m_profile.shape[1]))
    fig.colorbar(profile_image, ax=axes["d"], fraction=0.025, pad=0.02, label=r"$\mu_{r,j}$")

    output = output_path(FIGURE, "figC1_raw_fragmentation_frontier_order.pdf")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def draw_position_profile(stats: dict[str, object], decomposition: dict[str, object]) -> Path:
    """Fig. C2, written to figure/figC2_normalized_residual_similarity.pdf.

    (a) R^W_rs of every distinct pair against |m_r - m_s|.  The range of the mismatch is cut
    into eight equal intervals; the heavy curve is the mean of R^W in each interval, placed at
    the mean mismatch of its pairs, and the band spans the 25th to 75th percentile.  The means
    describe the cloud; the pairs share replicas and are not independent.  (b) The positionwise
    residual gaps g_j, whose average is gap(W); positions 6 to 11 are set off by color and the
    two largest, j = 8 and 9, by a darker one.
    """
    m_r = np.asarray(stats["m_r"], dtype=np.float64)
    upper = np.triu_indices(len(m_r), 1)
    exact_correlation = np.asarray(decomposition["exact_profile_correlation"])
    position_gap = np.asarray(decomposition["position_residual_gap"])
    summary = decomposition["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    depth_pairs = np.abs(m_r[:, None] - m_r[None, :])[upper]
    correlation_pairs = exact_correlation[upper]
    axes[0].scatter(depth_pairs, correlation_pairs, s=12, alpha=0.16, color="#4c78a8",
                    edgecolors="none", label="replica pairs")
    edges = np.linspace(0.0, float(depth_pairs.max()) + 1e-12, 9)
    bin_index = np.digitize(depth_pairs, edges[1:-1])
    bin_x: list[float] = []
    bin_mean: list[float] = []
    bin_q25: list[float] = []
    bin_q75: list[float] = []
    for index in range(len(edges) - 1):
        mask = bin_index == index
        if not np.any(mask):
            continue
        bin_x.append(float(depth_pairs[mask].mean()))
        bin_mean.append(float(correlation_pairs[mask].mean()))
        q25, q75 = np.quantile(correlation_pairs[mask], [0.25, 0.75])
        bin_q25.append(float(q25))
        bin_q75.append(float(q75))
    axes[0].fill_between(bin_x, bin_q25, bin_q75, color="#e45756", alpha=0.16,
                         label="pairwise IQR")
    axes[0].plot(bin_x, bin_mean, "o-", color="#e45756", lw=2.0, ms=5,
                 label="descriptive bin mean")
    axes[0].axhline(0.0, color="0.35", lw=0.8, ls="--")
    axes[0].set(
        xlabel=r"truth-alignment mismatch $|m_r-m_s|$",
        ylabel=r"normalized residual similarity $R^W_{r,s}$",
        title=(
            r"(a) Residual similarity vs truth-alignment mismatch"
            + "\n"
            + rf"corr. $={summary['exact_profile_residual_correlation_depth_distance_corr']:.3f}$"
        ),
    )
    axes[0].text(
        0.97, 0.96,
        r"$|m_r-m_s|\leq0.05$: "
        f"{summary['exact_profile_correlation_same_depth_0p05_mean']:.3f}\n"
        r"$|m_r-m_s|\geq0.4$: "
        f"{summary['exact_profile_correlation_far_depth_0p40_mean']:.3f}",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    axes[0].legend(fontsize=7, loc="lower left")

    positions = np.arange(len(position_gap))
    bar_colors = np.array(["#9ecae1"] * len(position_gap), dtype=object)
    bar_colors[6:] = "#f28e2b"
    bar_colors[8:10] = "#d94f3d"
    axes[1].bar(positions, position_gap, color=bar_colors, width=0.78)
    axes[1].set(
        xlabel="output position $j$", ylabel="positionwise residual gap",
        title=(
            r"(b) Residual support is late/frontier-local"
            + "\n"
            + rf"$j=6$--$11$: {summary['late_positions_6_to_11_fraction_of_residual_gap']:.1%}"
        ),
    )
    axes[1].set_xticks(positions)
    axes[1].text(0.97, 0.96, "largest at $j=8,9$", transform=axes[1].transAxes, ha="right",
                 va="top", fontsize=8)
    axes[1].grid(axis="y", alpha=0.18)

    output = output_path(FIGURE, "figC2_normalized_residual_similarity.pdf")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def figC1_raw_fragmentation_frontier_order():
    """Figs. C1 and C2 and the retreat numbers of Sec. 4.2: the overlap structure of the
    (12, 1280) retreat endpoints and its position profile (App. C.2).

    The (12, 1280) ensemble holds 1200 replicas on one shared training set, read in storage
    order (common.load_l12_n1280_in_storage_order).  With A_block on the 2048 evaluation draws
    and the observation epoch 150,

        retreat     reached HI = 0.8, below LO = 0.5 at epoch 150 (common.retreat)   269 replicas
        ends high   reached HI, at or above HI at epoch 150                          556 replicas

    The second group holds the stable and the recovery class of App. C.1 together.  On the
    retreat endpoints, on all output positions, the function measures

    1. m, q_self, q_cross, gap(Q^raw), gap(Q^c) and corr(Q^raw_rs, m_r m_s), with 95% replica
       bootstrap intervals (bootstrap_overlap);
    2. the tests of an ultrametric hierarchy on Q^raw and Q^c (hierarchy_diagnostics);
    3. the exact decomposition Q^c = C^prof + W, the gaps of its two terms with bootstrap
       intervals, the dependence of R^W on |m_r - m_s| and the positionwise gaps g_j
       (profile_residual_decomposition): the numbers of the retreat paragraph of Sec. 4.2 and
       of App. C.2;
    4. the position profiles of m, q_self, q_cross, chi^c_SG, Delta_N and the uncertain and
       confidently-wrong fractions of the retreat and of the ends-high replicas, with 95%
       bootstrap bands (bootstrap_position_profiles);
    5. the hierarchy tests of R^Qc and R^W (correlation_matrix of Q^c and W, the replicas
       sorted by m_r).  Those of R^W are the observed r_coph, R^2_tree and P(delta_um < 0.1) of
       Sec. 4.2, the last one over all replica triples.

    Fig. C1 shows the distributions of Q^raw_rs and Q^c_rs, Q^raw_rs against m_r m_s and the
    profiles mu_rj sorted by frontier position (draw_overlap_structure).  Fig. C2 shows R^W_rs
    against |m_r - m_s| and g_j (draw_position_profile).

    Reads   data/large_data_ensembles/L12_N1280/seeds*.npz
    Writes  figure/figC1_raw_fragmentation_frontier_order.pdf         (Fig. C1)
            figure/figC2_normalized_residual_similarity.pdf           (Fig. C2)
            results/retreat_overlap_structure/summary.json            every number above
            results/retreat_overlap_structure/overlap_arrays.npz      seeds, m_r, frontier
                                                                      positions, q_self_r,
                                                                      Q^raw, Q^c
    About two minutes on one core; the loops over the 3.2 million replica triples of each
    tested matrix take most of it.
    """
    xhat, block, seeds, yspin = load_l12_n1280_in_storage_order()
    retreat_mask = retreat(block)
    ends_high_mask = ends_high(block)
    retreat_xhat = xhat[retreat_mask]
    retreat_seeds = seeds[retreat_mask]

    stats = overlap_statistics(retreat_xhat, yspin)
    bootstrap = bootstrap_overlap(stats)
    hierarchy = hierarchy_diagnostics(stats["q_raw"], stats["q_connected"], stats["m_r"])
    decomposition = profile_residual_decomposition(stats, retreat_xhat, yspin)
    position_summary = {
        "bootstrap_unit": "initialization replica",
        "bootstrap_replicates": POSITION_BOOTSTRAP_REPLICATES,
        "retreat": bootstrap_position_profiles(position_components(retreat_xhat, yspin)),
        "ends_high": bootstrap_position_profiles(position_components(xhat[ends_high_mask], yspin)),
    }

    m_profile = np.asarray(decomposition["m_profile"])
    frontier = np.array([frontier_position(profile) for profile in m_profile])
    fig_c1 = draw_overlap_structure(stats, m_profile, frontier)
    np.savez_compressed(
        output_path(OVERLAP_TABLES, "overlap_arrays.npz"),
        seeds=retreat_seeds,
        m_r=np.asarray(stats["m_r"]),
        frontier_position=frontier,
        q_self_r=np.asarray(stats["q_self_r"]),
        q_raw=np.asarray(stats["q_raw"]),
        q_connected=np.asarray(stats["q_connected"]),
    )
    fig_c2 = draw_position_profile(stats, decomposition)

    # The hierarchy tests of the diagonal-normalized Q^c and W, the replicas sorted by m_r.
    m_r = np.asarray(stats["m_r"], dtype=np.float64)
    order = np.argsort(m_r)
    clustering_metrics = {}
    for name, matrix in (("Qc", stats["q_connected"]),
                         ("W", decomposition["exact_profile_residual"])):
        sorted_matrix = np.asarray(matrix, dtype=np.float64)[np.ix_(order, order)]
        tests = connected_hierarchy(correlation_matrix(sorted_matrix), m_r[order])
        clustering_metrics[name] = {
            "near_ultrametric_fraction_delta_lt_0p1":
                tests["ultrametric"]["fraction_relative_gap_below_0p1"],
            "ultrametric_delta_median": tests["ultrametric"]["relative_gap_median"],
            "plateau_max_gap_over_range": tests["plateaus"]["max_adjacent_gap_over_range"],
            "tree_r2": tests["tree_projection"]["average_linkage_tree_r2"],
            "cophenetic_correlation":
                tests["tree_projection"]["average_linkage_cophenetic_correlation"],
            "depth_distance_r2": tests["depth_distance_r2"],
            "depth_distance_corr": tests["depth_distance_corr"],
        }

    summary = {
        "protocol": {
            "replicas": int(len(seeds)),
            "seed_min": int(seeds.min()),
            "seed_max": int(seeds.max()),
            "retreat_selector": "max_t A_block(t) >= 0.8 and A_block(t_obs) < 0.5",
            "retreat_replicas": int(retreat_mask.sum()),
            "ends_high_selector": "max_t A_block(t) >= 0.8 and A_block(t_obs) >= 0.8",
            "ends_high_replicas": int(ends_high_mask.sum()),
            "other_replicas": int(len(seeds) - retreat_mask.sum() - ends_high_mask.sum()),
            "shared_training_seed": TRAIN_SEED,
            "observation_epoch": EPOCHS,
            "t_obs_scope": ("end of the saved 150-epoch analysis window, not an "
                            "asymptotic endpoint"),
        },
        "retreat_overlap": jsonable_overlap(stats),
        "retreat_overlap_bootstrap": bootstrap,
        "retreat_hierarchy": hierarchy,
        "frontier_deflation": decomposition["summary"],
        "endpoint_spatial_profiles": position_summary,
        "clustering_diagnostics": {
            "metric_scope": ("All metrics use diagonal-normalized connected/residual overlap "
                             "matrices R^Qc and R^W."),
            "ultrametric_residual": ("For each replica triple, sort the three pair similarities "
                                     "q1<=q2<=q3 and record (q2-q1)/(q3-q1). Exact ultrametric "
                                     "triangles have value 0."),
            "metrics": clustering_metrics,
        },
    }
    out_summary = output_path(OVERLAP_TABLES, "summary.json")
    with out_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    printed_decomposition = ("profile_gap", "exact_profile_residual_gap",
                             "exact_profile_residual_fraction_of_connected_gap",
                             "exact_profile_residual_correlation_depth_distance_corr",
                             "exact_profile_correlation_same_depth_0p05_mean",
                             "exact_profile_correlation_far_depth_0p40_mean",
                             "late_positions_6_to_11_fraction_of_residual_gap",
                             "exact_decomposition_max_abs_error")
    print(json.dumps({
        "protocol": summary["protocol"],
        "retreat_overlap": {key: summary["retreat_overlap"][key]
                            for key in ("m", "q_self", "q_cross", "q_self_minus_q_cross",
                                        "chi_SG_connected", "corr_q_vs_m_product")},
        "frontier_deflation": {key: summary["frontier_deflation"][key]
                               for key in printed_decomposition},
    }, indent=2))
    for path in (fig_c1, fig_c2, out_summary, OVERLAP_TABLES / "overlap_arrays.npz"):
        print("wrote", path)


# ===========================================================================
# Fig. C3 and Table C2: the (12, 64) memorization ensemble on the complete held-out set
# (App. C.3, C.4)
# ===========================================================================

MEMORIZATION_L, MEMORIZATION_N = 12, 64
MEMORIZATION_TAIL = np.arange(8, 12)       # the tabulated tail j = 8, ..., 11 of Table 3
MEMORIZATION_SEEDS = tuple(range(100))


def first_walls_codes(inputs: torch.Tensor, length: int) -> torch.Tensor:
    """The integer code sum_{i < length} s_i 2^i of the first `length` domain walls of each
    input, which fix output `length`."""
    if length == 0:
        return torch.zeros(len(inputs), dtype=torch.long)
    walls = inputs[:, 0, 1:length + 1].to(torch.long)
    weights = 2 ** torch.arange(length, dtype=torch.long)
    return walls @ weights


def mean_and_replica_se(values: np.ndarray) -> tuple[float, float]:
    """Mean over replicas and its standard error sd / sqrt(R)."""
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def load_memorization_replica(run_dir: Path):
    """The network of one run directory in eval mode (fused attention kernel).

    Its metrics.json must record L = 12, the training set of TRAIN_SEED and the widths of
    common.ARCHITECTURE.  A different head count would load without a shape error, so it is
    checked here.
    """
    payload = read_json(run_dir / "metrics.json")
    config = payload["config"]
    stored = (int(config["L"]), int(payload["train_seed"]), int(config["channels"]),
              int(config["depth"]), int(config["heads"]), int(config["ffn"]))
    expected = (MEMORIZATION_L, TRAIN_SEED, ARCHITECTURE["channels"], ARCHITECTURE["depth"],
                ARCHITECTURE["num_heads"], ARCHITECTURE["ffn_dim"])
    if stored != expected:
        raise ValueError(f"incompatible replica: {run_dir}")
    return load_chain_transformer(run_dir / "model.pt", MEMORIZATION_L).eval()


@torch.no_grad()
def collect_predictions(run_dirs: list[Path], test_inputs: torch.Tensor,
                        train_inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft spins on the held-out and the training inputs and the hard bits on the training
    inputs, each of shape [replicas, inputs, L]."""
    test_soft = []
    train_soft = []
    train_hard = []
    for run_dir in run_dirs:
        model = load_memorization_replica(run_dir)
        test_logits = model(test_inputs)
        train_logits = model(train_inputs)
        test_soft.append(torch.tanh(test_logits / 2.0).numpy())
        train_soft.append(torch.tanh(train_logits / 2.0).numpy())
        train_hard.append((train_logits > 0.0).numpy())
    return np.stack(test_soft), np.stack(train_soft), np.stack(train_hard)


def cross_profile(predictions: np.ndarray) -> np.ndarray:
    """q_cross(j): mean of u_rj u_sj over ordered replica pairs r != s and over inputs.

    The pair sum is (sum_r u_r)^2 - sum_r u_r^2, taken input by input.
    """
    replicas, n_inputs, _ = predictions.shape
    replica_sum = predictions.sum(axis=0)
    numerator = (replica_sum**2 - (predictions**2).sum(axis=0)).sum(axis=0)
    return numerator / (replicas * (replicas - 1) * n_inputs)


PER_POSITION_COLUMNS = [
    "j", "m", "q_self", "q_cross",
    "configurations_covered", "configurations_total",
    "heldout_records_covered_configuration", "heldout_records_uncovered_configuration",
    "accuracy", "accuracy_replica_se",
    "covered_configuration_accuracy", "covered_configuration_accuracy_replica_se",
    "uncovered_configuration_accuracy", "uncovered_configuration_accuracy_replica_se",
]


def figC3_small_data_diagnosis():
    """Fig. C3 and Table C2: 100 memorization replicas on the complete held-out set
    (App. C.3, C.4).

    The (12, 64) memorization ensemble is 100 networks trained from different initializations
    on the shared training set of N = 64 draws with TRAIN_SEED, which are 64 distinct inputs of
    the 2^11 = 2048.  Every replica is evaluated on all the other 1984 (common.held_out_set), so
    the held-out averages carry no sampling error over inputs and the only error bar is the
    standard error over replicas.  At every output position j the function measures m(j),
    q_self(j), q_cross(j) (over ordered replica pairs, cross_profile) and the bit accuracy,
    averaged over held-out inputs and replicas.

    Output j is the parity of the first j domain walls and of nothing else.  A held-out input is
    covered at j when some training string has the same first j walls, so that the training set
    contains the right answer for that output, and uncovered otherwise; the accuracy is reported
    separately for the two groups (Table C2, App. C.3).

    On the tail j = 8, ..., 11 two replica matrices are formed: Q^raw over held-out inputs and
    tail positions, whose off-diagonal distribution is panel (b), and R^W of the tail
    (common.py), symmetrized, clipped to [-1, 1], with unit diagonal.  Panel (c) shows R^W with the
    replicas in the leaf order of the average-linkage tree on the distance max(1 - R^W, 0); the
    signed R^W is used throughout, and every off-diagonal entry of this ensemble is positive.

    (a) q_self(j), m(j) and q_cross(j), the tail shaded;
    (b) the histogram of the 4950 off-diagonal tail entries Q^raw_rs (bins from 0.05 to 0.5,
        which hold every pair of this ensemble), with their mean;
    (c) the tail R^W in tree order.

    The network evaluations run with four intra-op threads of torch; the last digits of the
    tables depend on it.

    Reads   data/small_data_ensembles/L12_N64/seed{0..99}/model.pt and metrics.json; stops if a
            replica is missing or records another training seed or architecture
    Writes  figure/figC3_small_data_diagnosis.pdf              (Fig. C3)
            results/memorization_heldout_per_position.csv      (Table C2: per position j the
                                                                profiles, the coverage counts
                                                                and the accuracies)
            results/memorization_heldout_summary.json          (window averages of App. C.4)
    """
    L, N_train, tail = MEMORIZATION_L, MEMORIZATION_N, MEMORIZATION_TAIL
    torch.set_num_threads(4)
    run_dirs = [SMALL_DATA_ENSEMBLES / "L12_N64" / f"seed{seed}" for seed in MEMORIZATION_SEEDS]
    missing = [str(path) for path in run_dirs if not (path / "model.pt").exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} replicas; first missing: {missing[0]}")

    test_inputs, test_targets = held_out_set(L, N_train)
    train_inputs, train_targets = sample_set(N_train, L, TRAIN_SEED)
    distinct_training_inputs = len(torch.unique(wall_codes(train_inputs)))

    test_soft, train_soft, train_hard = collect_predictions(run_dirs, test_inputs, train_inputs)
    test_targets_np = test_targets.numpy()
    truth_spin = 2.0 * test_targets_np - 1.0
    test_correct = (test_soft > 0.0) == test_targets_np[None, :, :]

    m_profile = np.mean(test_soft * truth_spin[None, :, :], axis=(0, 1))
    q_self_profile = np.mean(test_soft**2, axis=(0, 1))
    q_cross_profile = cross_profile(test_soft)

    bit_accuracy, bit_accuracy_se = mean_and_replica_se(test_correct.mean(axis=(1, 2)))
    block_accuracy, block_accuracy_se = mean_and_replica_se(test_correct.all(axis=2).mean(axis=1))
    tail_accuracy, tail_accuracy_se = mean_and_replica_se(
        test_correct[:, :, tail].mean(axis=(1, 2)))

    # Accuracy at position j, split by whether the first j walls of the held-out input occur as
    # the first j walls of a training string.
    coverage_rows = []
    for j in range(L):
        training_configurations = torch.unique(first_walls_codes(train_inputs, j))
        evaluation_configurations = first_walls_codes(test_inputs, j)
        covered_mask = torch.isin(evaluation_configurations, training_configurations).numpy()
        uncovered_mask = ~covered_mask
        position_accuracy, position_accuracy_se = mean_and_replica_se(
            test_correct[:, :, j].mean(axis=1))
        if covered_mask.any():
            covered_accuracy, covered_accuracy_se = mean_and_replica_se(
                test_correct[:, covered_mask, j].mean(axis=1))
        else:
            covered_accuracy, covered_accuracy_se = float("nan"), float("nan")
        if uncovered_mask.any():
            uncovered_accuracy, uncovered_accuracy_se = mean_and_replica_se(
                test_correct[:, uncovered_mask, j].mean(axis=1))
        else:
            uncovered_accuracy, uncovered_accuracy_se = float("nan"), float("nan")
        coverage_rows.append([
            int(len(training_configurations)), 2**j,
            int(covered_mask.sum()), int(uncovered_mask.sum()),
            position_accuracy, position_accuracy_se,
            covered_accuracy, covered_accuracy_se,
            uncovered_accuracy, uncovered_accuracy_se,
        ])

    # Raw replica overlap on the tail; its upper triangle is the P(q) of panel (b).
    tail_predictions = test_soft[:, :, tail]
    flat_tail = tail_predictions.reshape(len(run_dirs), -1)
    overlap_matrix = flat_tail @ flat_tail.T / flat_tail.shape[1]
    pair_i, pair_j = np.triu_indices(len(run_dirs), k=1)
    pair_overlaps = overlap_matrix[pair_i, pair_j]

    # R^W of the truth-gauged tail outputs, and the average-linkage tree on 1 - R^W that orders
    # panel (c).
    gauged_tail = tail_predictions * truth_spin[None, :, tail]
    centered_tail = gauged_tail - gauged_tail.mean(axis=1, keepdims=True)
    flat_centered_tail = centered_tail.reshape(len(run_dirs), -1)
    w_tail = flat_centered_tail @ flat_centered_tail.T / flat_centered_tail.shape[1]
    r_w_tail = w_tail / np.sqrt(np.outer(np.diag(w_tail), np.diag(w_tail)))
    r_w_tail = np.clip(0.5 * (r_w_tail + r_w_tail.T), -1.0, 1.0)
    np.fill_diagonal(r_w_tail, 1.0)
    tree_distance = np.maximum(1.0 - r_w_tail, 0.0)
    np.fill_diagonal(tree_distance, 0.0)
    tree = linkage(squareform(tree_distance, checks=False), method="average")
    tree_order = leaves_list(tree)
    tree_ordered_r_w = r_w_tail[np.ix_(tree_order, tree_order)]

    train_tail_soft = train_soft[:, :, tail]
    train_tail_hard = train_hard[:, :, tail]
    train_tail_targets = train_targets.numpy()[None, :, tail]

    summary = {
        "L": L,
        "N_train": N_train,
        "replicas": len(run_dirs),
        "training_unique_records": int(distinct_training_inputs),
        "all_possible_records": 2 ** (L - 1),
        "heldout_records": int(len(test_inputs)),
        "tail_positions": tail.tolist(),
        "test_bit_accuracy": bit_accuracy,
        "test_bit_accuracy_replica_se": bit_accuracy_se,
        "test_block_accuracy": block_accuracy,
        "test_block_accuracy_replica_se": block_accuracy_se,
        "m_full": float(np.mean(test_soft * truth_spin[None, :, :])),
        "m_tail": float(np.mean(m_profile[tail])),
        "q_self_tail": float(np.mean(q_self_profile[tail])),
        "q_cross_tail": float(pair_overlaps.mean()),
        "delta_N_tail": float(np.mean(q_self_profile[tail] - m_profile[tail])),
        "chi_SG_tail": float(np.mean(q_self_profile[tail]) - pair_overlaps.mean()),
        "train_tail_q_self": float(np.mean(train_tail_soft**2)),
        "train_tail_accuracy": float(np.mean(train_tail_hard == train_tail_targets)),
        "test_tail_accuracy": tail_accuracy,
        "test_tail_accuracy_replica_se": tail_accuracy_se,
        "q_pair_mean": float(pair_overlaps.mean()),
        "q_pair_std": float(pair_overlaps.std()),
        "q_pair_min": float(pair_overlaps.min()),
        "q_pair_max": float(pair_overlaps.max()),
        "q_pair_above_0_7": int(np.sum(pair_overlaps > 0.7)),
    }
    summary_path = output_path(RESULTS, "memorization_heldout_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    per_position_path = output_path(RESULTS, "memorization_heldout_per_position.csv")
    with per_position_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PER_POSITION_COLUMNS)
        for j, coverage in enumerate(coverage_rows):
            writer.writerow([j, m_profile[j], q_self_profile[j], q_cross_profile[j], *coverage])

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)

    ax = axes[0]
    positions = np.arange(L)
    ax.axvspan(7.5, 11.5, color="0.88", zorder=0, label="measured tail")
    ax.plot(positions, q_self_profile, "o-", linewidth=1.7, markersize=4,
            label=r"$q_{\rm self}(j)$")
    ax.plot(positions, m_profile, "^-", linewidth=1.7, markersize=4, label=r"$m(j)$")
    ax.plot(positions, q_cross_profile, "s-", linewidth=1.7, markersize=4,
            label=r"$q_{\rm cross}(j)$")
    ax.axhline(0.0, color="0.45", linewidth=0.8)
    ax.set(xlabel="output position $j$", ylabel="overlap on held-out samples",
           title="(a) Position-resolved overlaps")
    ax.set_xticks(positions)
    ax.set_ylim(-0.08, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    bins = np.linspace(0.05, 0.5, 24)
    ax.hist(pair_overlaps, bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white")
    ax.axvline(pair_overlaps.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"mean = {pair_overlaps.mean():.3f}")
    ax.set(xlabel=r"tail pair overlap $Q^{\mathrm{raw}}_{r,s}$", ylabel="replica-pair count",
           title=r"(b) Off-diagonal $P(q)$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    image = ax.imshow(tree_ordered_r_w, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
    ax.set(xlabel="replica leaf", ylabel="replica leaf",
           title="(c) Profile-centered similarity, tree order")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("similarity")

    figure_file = output_path(FIGURE, "figC3_small_data_diagnosis.pdf")
    fig.savefig(figure_file, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(figure_file)
    print(per_position_path)
    print(summary_path)


# ===========================================================================
# Command line
# ===========================================================================

# (function, what it writes, where the paper uses it), in the order of the paper.  Each figure
# sets its rcParams in an rc_context, so they do not carry over from one figure to the next.
ITEMS = [
    (fig1_generalization_crossover,
     "figure/fig1_generalization_crossover.pdf, results/generalization_crossover_fits.csv",
     "Fig. 1, Sec. 3.1"),
    (fig2_shared_training_set_diagnostics, "figure/fig2_shared_training_set_diagnostics.pdf",
     "Fig. 2, Sec. 3.2"),
    (fig3_nishimori_gap_dynamics, "figure/fig3_nishimori_gap_dynamics.pdf", "Fig. 3, Sec. 3.3"),
    (fig4_position_resolved_retreat_recovery, "figure/fig4_position_resolved_retreat_recovery.pdf",
     "Fig. 4, Sec. 3.4"),
    (fig5_tail_observables, "figure/fig5_tail_observables.pdf", "Fig. 5, Sec. 3.5"),
    (fig6_raw_overlaps_and_gaps, "figure/fig6_raw_overlaps_and_gaps.pdf", "Fig. 6, Sec. 4.2"),
    (fig7_retreat_residual_tree, "figure/fig7_retreat_residual_tree.pdf", "Fig. 7, Sec. 4.2"),
    (fig8_fragmentation_residual_organization,
     "figure/fig8_fragmentation_residual_organization.pdf", "Fig. 8, Sec. 4.3"),
    (fig9_two_planes_common_tail, "figure/fig9_two_planes_common_tail.pdf", "Fig. 9, Sec. 5.1"),
    (fig10_learning_rate_intervention, "figure/fig10_learning_rate_intervention.pdf",
     "Fig. 10, Sec. 5.2"),
    (figA1_final_layer_ffn, "figure/figA1_final_layer_ffn.pdf", "Fig. A1, App. A.3"),
    (figB1_bit_and_block_accuracy,
     "figure/figB1_bit_and_block_accuracy.pdf, results/block_vs_bit_summary.csv",
     "Fig. B1, App. B"),
    (figC1_raw_fragmentation_frontier_order,
     "figure/figC1_raw_fragmentation_frontier_order.pdf, "
     "figure/figC2_normalized_residual_similarity.pdf, results/retreat_overlap_structure/",
     "Figs. C1, C2, Sec. 4.2, App. C.2"),
    (figC3_small_data_diagnosis,
     "figure/figC3_small_data_diagnosis.pdf, results/memorization_heldout_per_position.csv, "
     "results/memorization_heldout_summary.json", "Fig. C3, Table C2, App. C.3, C.4"),
]


if __name__ == "__main__":
    sys.exit(run_items(ITEMS, str(Path(__file__).resolve()), __doc__.split("\n\n")[0]))
