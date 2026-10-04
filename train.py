#!/usr/bin/env python3
"""Every training protocol behind data/ of "Replica Fragmentation and Glassy Dynamics in Parity
Learning", one subcommand per protocol.

    python3 train.py --list                  every subcommand, the data/ directory it writes and
                                             where the paper uses it
    python3 train.py SUBCOMMAND --help       the options of one subcommand
    python3 train.py SUBCOMMAND [options]    run it

Every subcommand skips a run whose output is present unless --overwrite is given, so a call
against complete data trains nothing and writes nothing.  The task (the domain-wall encoding,
the input sets), the Transformer and the directories are defined in common.py, whose docstring
gives the notation used here.  Trained networks are stored as state dicts of torch.

===========================================================================
TRAINING AS RELAXATION AT QUENCHED DISORDER
===========================================================================
For a training set D of inputs b with target bits t_j(b), the energy is the empirical loss

    E_D(theta) = mean over b in D and positions j of
                 BCE(z_j, t_j) = log(1 + e^(z_j)) - t_j z_j,     z_j = z_j(b; theta) the logit,

and training is AdamW descent on it: learning rate 2e-3 held constant (except in one arm of
Fig. 10), decoupled weight decay 1e-4, minibatches of 128.  A run is fixed by three random
elements: the training set D, the initialization theta_0 and the noise history (the order in
which minibatches are drawn).  The protocols differ in which of the three are shared:

    shared training set   D is drawn once (N draws with TRAIN_SEED) and is common to every
                          replica; replica r has its own theta_0 and noise history.  These are
                          replicas at fixed disorder, and the overlaps between them are the
                          observables of Secs. 3 and 4.
    independent sets      every run draws its own D from its seed, so disorder, initialization
                          and noise all change with the seed; averages over runs are disorder
                          averages (the scans over N of Fig. 1 and App. B).
    fresh minibatches     every minibatch is a fresh uniform draw and there is no fixed D: the
                          annealed counterpart, the runs from which the L = 12 scans of Fig. 1
                          take their configurations (data/fresh_minibatch_runs/).
    nested split          the complete input set is split once per data seed into 512 held-out
                          inputs and an ordered pool of 1536; D_N is the first N of the pool and
                          minibatches are drawn with replacement (Fig. 10).

With the logit z_j, the soft spin u_j = tanh(z_j / 2) = 2 sigmoid(z_j) - 1 and the truth spin
y_j = 2 t_j - 1, the recorded observables are

    bit accuracy       fraction of positions whose predicted bit is right
    A_block            fraction of inputs with every position right (block accuracy)
    m = E[u y]         truth overlap, a Mattis magnetization
    q_self = E[u^2]    self-overlap;  Delta_N = q_self - m is the Nishimori gap
    wall residual      fraction of domain walls that the predicted spins violate, the
                       frustration of the output configuration
    uncertain fraction fraction of outputs with |u| below a threshold (0.2 or 0.5)

The predicted bit is sigmoid(z_j) > 0.5 in independent_set_cnn and independent_set_transformer,
sigmoid(z_j) >= 0.5 in independent_set_transformer_overlaps and overlap_trajectories_l12_n1280,
and z_j > 0 in the other protocols.

===========================================================================
SEEDS
===========================================================================
    TRAIN_SEED = 12345     the shared training set (common.sample_set)
    EVAL_SEED = 777        the evaluation draws of the replica ensembles
    replica r              torch.manual_seed(r) before the network is built; the minibatch
                           order comes from its own generator seeded with r + 1
    trajectories, Fig. 10  initialization seed 100000 + 1003 d + r, minibatch seed
                           200000 + 1009 d + r, with d the data seed (12345 for the
                           trajectories)
    independent sets       the global generator, seeded with the run's seed, draws D, the
                           test inputs and every permutation.  independent_set_cnn and
                           independent_set_transformer_overlaps build the network after
                           seeding; independent_set_transformer builds it before, so its
                           initialization comes from the generator torch seeds afresh in every
                           process and differs from one execution to the next.

Per-network results depend on the machine, the thread count and the version of torch.

===========================================================================
THREADS AND PROCESSES
===========================================================================
Every trainer sets the intra-op thread count of torch.  A driver starts each run as its own
process with the thread variables of its protocol, and the log of each run goes to
results/logs/<driver>/<run>.log.  The trainers of the chain task (independent_set_cnn,
independent_set_transformer, independent_set_transformer_overlaps,
overlap_trajectories_l12_n1280, overlap_trajectories_l12_n64, learning_rate_intervention) first
default the thread variables to 4 and take one inter-op thread (chain_task_threads), then set
their own intra-op count.

===========================================================================
DATA DIRECTORY -> SUBCOMMAND
===========================================================================
    independent_training_sets/size_scan/       size_scan_transformer_l12,
                                               size_scan_transformer_l16_l20, and
                                               independent_set_cnn once per CNN run
    independent_training_sets/representative_runs/
                                               representative_runs_transformer,
                                               representative_runs_cnn
    fresh_minibatch_runs/                      independent_set_transformer and
                                               independent_set_cnn with --fixed-train-set 0
    trajectory_ensembles/                      overlap_trajectories_l12_n1280,
                                               soft_output_trajectories_l12_n1280,
                                               overlap_trajectories_l12_n64
    small_data_ensembles/                      small_data_ensembles, one small_data_replica
                                               child per replica
    large_data_ensembles/                      large_data_ensembles, one large_data_replicas
                                               child per part file
    checkpointed_trajectories/                 checkpointed_trajectories
    fixed_epoch_replicas/                      fixed_epoch_replicas
    learning_rate_intervention/                learning_rate_intervention
    retreat_events/events.csv names checkpoints of checkpointed_trajectories/; no subcommand
    writes it.

===========================================================================
PAPER LOCATION -> SUBCOMMAND
===========================================================================
    Fig. 1, Secs. 1, 3.1           size_scan_transformer_l12, size_scan_transformer_l16_l20,
                                   independent_set_transformer, independent_set_cnn,
                                   representative_runs_transformer,
                                   independent_set_transformer_overlaps, representative_runs_cnn
    App. B, Fig. B1                size_scan_transformer_l12, representative_runs_transformer,
                                   independent_set_cnn
    Figs. 2, 3, Secs. 3.2-3.4      overlap_trajectories_l12_n1280
    Fig. 3, Sec. 3.3               overlap_trajectories_l12_n64
    Fig. 4, Secs. 3.4, 4.2         soft_output_trajectories_l12_n1280
    Fig. 5, Sec. 3.5, Fig. C3      small_data_ensembles, small_data_replica
    Table 3, Sec. 4, Tables C1, C3 small_data_ensembles, small_data_replica,
                                   large_data_ensembles, large_data_replicas
    Fig. 10, Sec. 5.2              learning_rate_intervention
    App. A.1, A.2                  fixed_epoch_replicas
    App. A.2, Fig. A1, App. A.3    checkpointed_trajectories
"""
from __future__ import annotations

import os
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def default_thread_variables() -> None:
    """Set every unset thread variable to 4 and allow a second OpenMP runtime in the process."""
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "4")
    # Intel OpenMP aborts when it is loaded twice in one process (through MKL and through
    # torch) unless this is set.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# The convolutional trainer sets the thread variables before NumPy and torch are loaded, so the
# BLAS library reads them when it loads.
if sys.argv[1:2] == ["independent_set_cnn"]:
    default_thread_variables()

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ARCHITECTURE, CHECKPOINTED_TRAJECTORIES, EVAL_SEED, FIXED_EPOCH_REPLICAS, FRESH_MINIBATCH_RUNS,
    HI, LARGE_DATA_ENSEMBLES, LO, REPRESENTATIVE_CNN_RUNS, REPRESENTATIVE_TRANSFORMER_RUNS,
    RESULTS, ROOT, SMALL_DATA_ENSEMBLES, TRAIN_SEED, TRAJECTORY_ENSEMBLES, TRAJECTORY_EPOCHS,
    TRANSFORMER_SIZE_SCAN,
    ChainTransformer, complete_set, crossing_epoch, encode, sample_set, write_json, write_rows,
)

TRAIN_PY = Path(__file__).resolve()
LOGS = RESULTS / "logs"

# The optimizer of every protocol: AdamW at this learning rate and weight decay, minibatch 128.
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
BATCH = 128


# ===========================================================================
# Helpers: paths, threads, the two descent loops, child processes
# ===========================================================================

def resolve(path) -> Path:
    """A path argument; a relative one is taken from the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def shown(path) -> str:
    """A path for messages, relative to the repository root when it lies inside it."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def present(*paths) -> bool:
    """True when every path exists."""
    return all(Path(path).exists() for path in paths)


def int_list(value: str) -> list[int]:
    """A comma-separated list of integers."""
    return [int(item) for item in value.split(",") if item.strip()]


def chain_task_threads() -> None:
    """The thread settings every trainer of the chain task starts with.

    The thread variables default to 4, torch takes its intra-op count from OMP_NUM_THREADS and
    runs one inter-op thread.  The protocol then sets its own intra-op count before it computes
    anything; the inter-op setting and the variables stay.
    """
    default_thread_variables()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # torch accepts the inter-op count only before its first parallel region
        pass


def draw_fresh(num: int, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """num uniform domain-wall configurations from the global generator, encoded.

    The independent-set protocols draw their training set, their test inputs and their fresh
    minibatches with this call, so the seed of the global generator names all of them.
    """
    return encode(torch.randint(0, 2, (num, L - 1)).float())


def replica_epochs(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
                   order_seed: int, epochs: int, batch: int = BATCH):
    """AdamW descent of one replica on the shared training set; yields after every epoch.

    Each epoch is one pass over a fresh permutation of the training set, drawn from a generator
    of its own seeded with order_seed, in minibatches of `batch`.  The generator yields
    (epoch, mean minibatch loss) with the network in training mode; the caller evaluates it.
    The global generator is not touched, so the noise history is fixed by order_seed alone.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order = torch.Generator().manual_seed(order_seed)
    n = len(inputs)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n, generator=order)
        total = 0.0
        for start in range(0, n, batch):
            index = permutation[start:start + batch]
            loss = F.binary_cross_entropy_with_logits(model(inputs[index]), targets[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        yield epoch, total / math.ceil(n / batch)


def independent_set_epochs(model: nn.Module, lr: float, weight_decay: float, batch_size: int,
                           epochs: int, num_train: int, L: int, train_set=None):
    """AdamW descent with every random choice from the global generator; yields per epoch.

    With a fixed training set (train_set = (inputs, targets) of num_train draws) each epoch is
    one pass over torch.randperm(num_train); without one, every minibatch is a fresh draw of
    the same size.  Yields (epoch, mean minibatch loss) with the network in training mode.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(num_train / batch_size)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        if train_set is not None:
            permutation = torch.randperm(num_train)
            batches = [permutation[start:start + batch_size]
                       for start in range(0, num_train, batch_size)]
        else:
            batches = [min(batch_size, num_train - start)
                       for start in range(0, num_train, batch_size)]
        for batch in batches:
            if train_set is not None:
                x, y = train_set[0][batch], train_set[1][batch]
            else:
                x, y = draw_fresh(batch, L)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        yield epoch, total / steps_per_epoch


def child(subcommand: str, *arguments) -> list[str]:
    """The command line that runs `python3 train.py subcommand arguments` in its own process."""
    return [sys.executable, str(TRAIN_PY), subcommand, *[str(item) for item in arguments]]


def pinned_environment(threads: int) -> dict:
    """The environment of a child with every thread variable set to `threads`."""
    environment = os.environ.copy()
    for name in THREAD_VARIABLES:
        environment[name] = str(threads)
    environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return environment


def run_jobs(jobs: list[tuple[str, list[str], dict]], workers: int, log_dir: Path) -> int:
    """Run (name, command, environment) jobs, at most `workers` at a time; 1 if one failed.

    Each child runs from the repository root and writes its output to log_dir/<name>.log.
    With no job nothing is started and nothing is written.
    """
    if not jobs:
        print("every run is present; nothing to train", flush=True)
        return 0
    print(f"{len(jobs)} runs to train, {workers} at a time", flush=True)
    pending, running, failures = list(jobs), [], []
    try:
        while pending or running:
            while pending and len(running) < workers:
                name, command, environment = pending.pop(0)
                log_dir.mkdir(parents=True, exist_ok=True)
                log = open(log_dir / f"{name}.log", "w", encoding="utf-8")
                print(f"start {name}", flush=True)
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
                running.append((name, process, log))
            time.sleep(2.0)
            still = []
            for name, process, log in running:
                if process.poll() is None:
                    still.append((name, process, log))
                    continue
                log.close()
                if process.returncode:
                    failures.append(name)
                    print(f"FAIL  {name} (status {process.returncode}), see "
                          f"{shown(log_dir / (name + '.log'))}", flush=True)
                else:
                    print(f"done  {name}", flush=True)
            running = still
    finally:
        for _, process, log in running:
            if process.poll() is None:
                process.terminate()
            log.close()
    if failures:
        print(f"failed runs: {', '.join(failures)}", flush=True)
        return 1
    return 0


# ===========================================================================
# Fig. 1, App. B: single runs on independently drawn training sets
# ===========================================================================
# independent_set_transformer and independent_set_cnn train one network on the parity task and
# store the configuration of the run with every field of the stored runs.  The fields of
# TRANSFORMER_FIXED and CNN_FIXED have no option; they take one value in every stored run and
# are written with it.

TRANSFORMER_FIELDS = ("L", "p", "train_samples", "test_samples", "batch_size", "epochs",
                      "channels", "depth", "kernel_size", "lr", "weight_decay", "seed",
                      "out_dir", "device", "plot_format", "task", "interval_length", "anchor",
                      "fixed_train_set", "test_all_syndromes", "num_threads", "num_heads",
                      "ffn_dim", "dropout")
TRANSFORMER_FIXED = {"p": 0.08, "kernel_size": 3, "device": "cpu", "plot_format": "pdf",
                     "task": "parity", "interval_length": 4, "anchor": -1,
                     "test_all_syndromes": 0}

CNN_FIELDS = ("L", "p", "train_samples", "test_samples", "batch_size", "epochs", "channels",
              "depth", "kernel_size", "lr", "weight_decay", "probe_epochs", "bp_iters",
              "max_probe_samples", "probe_all_interval_lengths", "seed", "out_dir", "device",
              "plot_format", "task", "interval_length", "anchor", "fixed_train_set",
              "test_all_syndromes", "run_diagnostics", "num_threads", "readability_threshold")
# kernel_size is the width of every convolution of ChainCNN; the others are inert.
CNN_FIXED = {"p": 0.08, "kernel_size": 3, "probe_epochs": 1, "bp_iters": 4,
             "max_probe_samples": 128, "probe_all_interval_lengths": 1, "device": "cpu",
             "plot_format": "pdf", "task": "parity", "interval_length": 4, "anchor": -1,
             "test_all_syndromes": 0, "run_diagnostics": 0, "readability_threshold": 0.9}


def run_configuration(fields: tuple[str, ...], fixed: dict, args) -> dict:
    """The stored configuration of one run: the fixed fields and the options of the call."""
    return {key: fixed[key] if key in fixed else getattr(args, key) for key in fields}


def wall_residual(predicted_bits: torch.Tensor, inputs: torch.Tensor) -> float:
    """Fraction of the L-1 domain walls that the predicted spin configuration violates.

    Predicted bits t^ are consistent with the input when t^_i XOR t^_(i+1) = s_i at every wall;
    the residual counts the walls where this fails, the frustration of the output.
    """
    walls = inputs[:, 1, :-1]
    predicted_walls = (predicted_bits[:, :-1] + predicted_bits[:, 1:]).remainder(2.0)
    return (predicted_walls != walls).float().mean().item()


class ChainCNN(nn.Module):
    """Residual stack of local convolutions on the L positions: [B, 2, L] -> [B, L] logits.

    in_proj (a convolution of width 3) and GELU, then `depth` blocks h <- h + GELU(GroupNorm(
    Conv(h))), then a width-1 convolution to one logit per position.  Each block widens the
    receptive field by one position per side, so at depth 16 every position sees the whole
    L = 12 chain.
    The parameters are created in the order in_proj, blocks, norms, out_proj, which fixes the
    initialization a seed gives.
    """

    def __init__(self, channels: int, depth: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.in_proj = nn.Conv1d(2, channels, kernel_size, padding=pad)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=pad) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, channels) for _ in range(depth)])
        self.out_proj = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_proj(x))
        for conv, norm in zip(self.blocks, self.norms):
            h = h + F.gelu(norm(conv(h)))
        return self.out_proj(h).squeeze(1)


def train_independent_set(model: nn.Module, args) -> list[dict]:
    """Train one network on its own training set (or on fresh minibatches); the per-epoch rows.

    With --fixed-train-set 1 the global generator draws the train_samples training inputs and
    then test_samples test inputs; with 0 only the test inputs, and every minibatch is a fresh
    draw.  After every epoch, in evaluation mode: loss (mean minibatch loss of the epoch), bit
    and block accuracy on the training set (NaN without one) and on the test inputs, and the
    wall residual of the test predictions.  A bit is predicted 1 when sigmoid(z) > 0.5.
    """
    fixed = bool(args.fixed_train_set)
    train_set = draw_fresh(args.train_samples, args.L) if fixed else None
    x_test, y_test = draw_fresh(args.test_samples, args.L)
    history = []
    for epoch, loss in independent_set_epochs(model, args.lr, args.weight_decay, args.batch_size,
                                              args.epochs, args.train_samples, args.L,
                                              train_set):
        model.eval()
        with torch.no_grad():
            if fixed:
                train_pred = (model(train_set[0]).sigmoid() > 0.5).float()
                train_bit_acc = (train_pred == train_set[1]).float().mean().item()
                train_block_acc = (train_pred == train_set[1]).all(dim=1).float().mean().item()
            else:
                train_bit_acc = float("nan")
                train_block_acc = float("nan")
            pred = (model(x_test).sigmoid() > 0.5).float()
            bit_acc = (pred == y_test).float().mean().item()
            block_acc = (pred == y_test).all(dim=1).float().mean().item()
            residual = wall_residual(pred, x_test)
        history.append({"epoch": epoch, "loss": loss, "train_bit_acc": train_bit_acc,
                        "train_block_acc": train_block_acc, "bit_acc": bit_acc,
                        "block_acc": block_acc, "syndrome_residual": residual})
        print(f"epoch {epoch:03d} loss={loss:.4f} bit_acc={bit_acc:.4f} "
              f"block_acc={block_acc:.4f} residual={residual:.4f}", flush=True)
    return history


def option(*flags, **kwargs):
    """One argparse option of a subcommand."""
    return flags, kwargs


COMMANDS: dict = {}


def subcommand(writes: str, paper: str, *options):
    """Register the decorated function as the subcommand of its name, with its options."""
    def register(function):
        COMMANDS[function.__name__] = (function, writes, paper, options)
        return function
    return register


INDEPENDENT_SET_OPTIONS = (
    option("--out-dir", required=True, help="run directory (relative: from the repository root)"),
    option("--train-samples", type=int, required=True, help="N, the size of the training set"),
    option("--seed", type=int, required=True, help="seed of the global generator"),
    option("--test-samples", type=int, default=1000),
    option("--batch-size", type=int, default=BATCH),
    option("--epochs", type=int, default=120),
    option("--channels", type=int, default=16),
    option("--lr", type=float, default=LEARNING_RATE),
    option("--weight-decay", type=float, default=WEIGHT_DECAY),
    option("--fixed-train-set", type=int, default=0, choices=(0, 1),
           help="1: one training set kept for the run; 0: a fresh draw for every minibatch"),
    option("--num-threads", type=int, default=4, help="intra-op threads of torch"),
)


@subcommand("data/independent_training_sets/size_scan/cnn_L12/, one run", "Fig. 1, App. B",
            *INDEPENDENT_SET_OPTIONS,
            option("--L", type=int, default=12),
            option("--depth", type=int, default=16))
def independent_set_cnn(args) -> int:
    """Train one ChainCNN on its own training set: chain_cnn.pt and metrics.json in --out-dir.

    The seed of the global generator is set first, so it fixes the initialization, the
    training set, the test inputs and the minibatch order together.  metrics.json holds
    "train_history" (the rows of train_independent_set) and "config" (the stored configuration);
    a run whose metrics.json is present is skipped.
    The scan over N of the convolutional network in Fig. 1 is one call per run with the
    configuration stored in that run's metrics.json; the 600-epoch runs of Fig. 1 come from
    representative_runs_cnn.
    """
    out = resolve(args.out_dir)
    if not args.overwrite and present(out / "metrics.json"):
        print(f"present {shown(out)}")
        return 0
    chain_task_threads()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    out.mkdir(parents=True, exist_ok=True)
    model = ChainCNN(args.channels, args.depth, CNN_FIXED["kernel_size"])
    history = train_independent_set(model, args)
    torch.save(model.state_dict(), out / "chain_cnn.pt")
    write_json(out / "metrics.json",
               {"train_history": history, "config": run_configuration(CNN_FIELDS, CNN_FIXED, args)})
    print(f"wrote {shown(out)}", flush=True)
    return 0


@subcommand("data/independent_training_sets/size_scan/transformer_L*/, one run", "Fig. 1, App. B",
            *INDEPENDENT_SET_OPTIONS,
            option("--L", type=int, required=True),
            option("--depth", type=int, default=8),
            option("--num-heads", type=int, default=4),
            option("--ffn-dim", type=int, default=64),
            option("--dropout", type=float, default=0.0))
def independent_set_transformer(args) -> int:
    """Train one ChainTransformer on its own training set: model.pt and metrics.json in --out-dir.

    The network is built before the global generator is seeded: the seed fixes the training
    set, the test inputs and the minibatch order, and the initialization comes from the
    generator torch seeds afresh in each process.  metrics.json holds "train_history" (the
    rows of train_independent_set) and "config"; a run whose metrics.json is present is
    skipped.  The scans over N of Fig. 1 are calls of this subcommand (size_scan_transformer_l12,
    size_scan_transformer_l16_l20).
    """
    out = resolve(args.out_dir)
    if not args.overwrite and present(out / "metrics.json"):
        print(f"present {shown(out)}")
        return 0
    chain_task_threads()
    model = ChainTransformer(L=args.L, channels=args.channels, depth=args.depth,
                             num_heads=args.num_heads, ffn_dim=args.ffn_dim, dropout=args.dropout)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    out.mkdir(parents=True, exist_ok=True)
    history = train_independent_set(model, args)
    torch.save(model.state_dict(), out / "model.pt")
    write_json(out / "metrics.json", {"train_history": history,
                                      "config": run_configuration(TRANSFORMER_FIELDS,
                                                                  TRANSFORMER_FIXED, args)})
    print(f"wrote {shown(out)}", flush=True)
    return 0


def fixed_set_counterpart(subcommand_name: str, config: dict, out_dir: Path, overwrite: bool):
    """The job that trains the fixed-training-set counterpart of a stored configuration, or None.

    The counterpart is the run of `config` with fixed_train_set = 1 and nothing else changed.  A
    counterpart whose metrics.json is present is kept after checking that its stored
    configuration equals `config` in every field except fixed_train_set and out_dir; a mismatch
    raises.  The child runs with every thread variable set to the num_threads of the
    configuration.
    """
    fields, fixed = ((CNN_FIELDS, CNN_FIXED) if subcommand_name == "independent_set_cnn"
                     else (TRANSFORMER_FIELDS, TRANSFORMER_FIXED))
    metrics = out_dir / "metrics.json"
    if not overwrite and metrics.exists():
        stored = json.loads(metrics.read_text(encoding="utf-8"))["config"]
        for key, value in config.items():
            expected = 1 if key == "fixed_train_set" else value
            if key != "out_dir" and stored.get(key) != expected:
                raise ValueError(f"{metrics} has {key}={stored.get(key)!r}, expected {expected!r}")
        print(f"present {shown(out_dir)}")
        return None
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError(f"{key}={config[key]!r} is not a setting {subcommand_name} runs")
    arguments = ["--out-dir", out_dir]
    for key in fields:
        if key not in fixed and key not in ("out_dir", "fixed_train_set"):
            arguments += [f"--{key.replace('_', '-')}", config[key]]
    arguments += ["--fixed-train-set", 1, "--overwrite"]
    return (out_dir.name, child(subcommand_name, *arguments),
            pinned_environment(int(config["num_threads"])))


@subcommand("data/independent_training_sets/size_scan/transformer_L12/", "Fig. 1, Fig. B1")
def size_scan_transformer_l12(args) -> int:
    """The fixed-training-set Transformer runs of the L = 12 axis of Fig. 1 (also Fig. B1).

    Every run is the fixed-training-set counterpart (fixed_set_counterpart) of a fresh-minibatch
    run whose configuration is stored in
    data/fresh_minibatch_runs/transformer_L12/L12_N<N>_seed<s>/metrics.json: depth 8, 120 epochs,
    1000 test draws, four threads.  The counterparts are written to
    data/independent_training_sets/size_scan/transformer_L12/L12_N<N>_seed<s>/, one after the other,
    in the order (N, seed).
    """
    pattern = re.compile(r"L12_N(\d+)_seed(\d+)$")
    sources = sorted((path for path in (FRESH_MINIBATCH_RUNS / "transformer_L12").glob(
                          "L12_N*_seed*/metrics.json")
                      if pattern.fullmatch(path.parent.name)),
                     key=lambda path: tuple(int(v) for v in
                                            pattern.fullmatch(path.parent.name).groups()))
    jobs = []
    for source in sources:
        config = json.loads(source.read_text(encoding="utf-8"))["config"]
        job = fixed_set_counterpart("independent_set_transformer", config,
                                    TRANSFORMER_SIZE_SCAN[12] / source.parent.name, args.overwrite)
        if job:
            jobs.append(job)
    return run_jobs(jobs, 1, LOGS / "size_scan_transformer_l12")


SIZE_SCAN_GRID = {16: (256, 512, 768, 1024, 1536, 2048, 3072, 4096),
                20: (2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576)}
SIZE_SCAN_SEEDS = 20
SIZE_SCAN_WORKERS = 6


@subcommand("data/independent_training_sets/size_scan/transformer_L16/, transformer_L20/", "Fig. 1")
def size_scan_transformer_l16_l20(args) -> int:
    """The fixed-training-set scans over N at L = 16 and L = 20 (Fig. 1).

    The protocol of the L = 12 axis with L and the grid of N changed: for each N of
    SIZE_SCAN_GRID and seeds 0-19, one independent_set_transformer with --fixed-train-set 1,
    depth 8, 120 epochs, 1000 test draws and one thread, into
    data/independent_training_sets/size_scan/transformer_L<L>/L<L>_N<N>_seed<s>/.  Six runs train
    at a time, each with OMP, MKL and VECLIB threads set to one.  A run whose metrics.json is
    present is skipped.
    """
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    jobs = []
    for L, grid in SIZE_SCAN_GRID.items():
        for N in grid:
            for seed in range(SIZE_SCAN_SEEDS):
                out = TRANSFORMER_SIZE_SCAN[L] / f"L{L}_N{N}_seed{seed}"
                if not args.overwrite and (out / "metrics.json").exists():
                    continue
                jobs.append((out.name, child(
                    "independent_set_transformer", "--out-dir", out, "--L", L, "--train-samples", N,
                    "--test-samples", 1000, "--batch-size", BATCH, "--epochs", 120,
                    "--channels", 16, "--depth", 8, "--lr", "0.002", "--weight-decay", "0.0001",
                    "--seed", seed, "--fixed-train-set", 1, "--num-threads", 1,
                    "--num-heads", 4, "--ffn-dim", 64, "--dropout", "0.0", "--overwrite"),
                    environment))
    return run_jobs(jobs, SIZE_SCAN_WORKERS, LOGS / "size_scan_transformer_l16_l20")


# The 600-epoch convolutional runs: the configuration of the fresh-minibatch run below with
# 600 epochs, a fixed training set and seeds 0, 3, 7.  The paper uses seed 3.
CNN_CONFIGURATION_SOURCE = FRESH_MINIBATCH_RUNS / "cnn_L12" / "L12_N2048_seed0"
REPRESENTATIVE_EPOCHS = 600
REPRESENTATIVE_CNN_SEEDS = (0, 3, 7)


@subcommand("data/independent_training_sets/representative_runs/cnn_L12_N2048/", "Fig. 1")
def representative_runs_cnn(args) -> int:
    """The 600-epoch fixed-training-set convolutional runs at L = 12, N = 2048 (Fig. 1).

    The configuration is that of data/fresh_minibatch_runs/cnn_L12/L12_N2048_seed0 (depth 16,
    1000 test draws, four threads) with 600 epochs, the seed set to 0, 3 or 7, and
    fixed_train_set = 1; each run is an independent_set_cnn child (fixed_set_counterpart) writing
    data/independent_training_sets/representative_runs/cnn_L12_N2048/seed<s>/.  A seed whose
    metrics.json is present is kept.
    """
    base = json.loads((CNN_CONFIGURATION_SOURCE / "metrics.json").read_text(
        encoding="utf-8"))["config"]
    jobs = []
    for seed in REPRESENTATIVE_CNN_SEEDS:
        config = dict(base)
        config["epochs"] = REPRESENTATIVE_EPOCHS
        config["seed"] = seed
        job = fixed_set_counterpart("independent_set_cnn", config,
                                    REPRESENTATIVE_CNN_RUNS / f"seed{seed}", args.overwrite)
        if job:
            jobs.append(job)
    return run_jobs(jobs, 1, LOGS / "representative_runs_cnn")


# ---------------------------------------------------------------------------
# The 600-epoch Transformer runs with per-epoch overlaps and confidence (Fig. 1, Fig. B1)
# ---------------------------------------------------------------------------

def confidence_configuration(**values) -> dict:
    """The stored configuration of a 600-epoch Transformer run, in its field order, with `values`
    set."""
    configuration = {"L": 12, "p": 0.08, "train_samples": 1280, "test_samples": 4096,
              "batch_size": BATCH, "epochs": 600, "channels": 16, "depth": 8, "lr": 0.002,
              "weight_decay": 0.0001, "seed": 0, "out_dir": "", "device": "cpu",
              "plot_format": "pdf", "task": "parity", "interval_length": 4, "anchor": -1,
              "fixed_train_set": 1, "max_exact_test": 65536, "num_threads": 2,
              "num_heads": 4, "ffn_dim": 64, "dropout": 0.0}
    configuration.update(values)
    return configuration


def confidence_observables(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor) -> dict:
    """Accuracies, m, q and the confidence of right and wrong outputs on the test inputs.

    u = 2 sigmoid(z) - 1, a bit is predicted 1 when sigmoid(z) >= 0.5.  wrong_conf and
    correct_conf are the mean |u| over wrong and over right outputs; negative_strong_frac is
    the fraction with u y < -0.5 (confidently wrong), uncertain_frac the fraction with
    |u| < 0.2.  Also returns the position profiles m_j, q_j and the signed margins u y.
    """
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_test))
        pred = (prob >= 0.5).float()
        soft = 2.0 * prob - 1.0
        signed_margin = soft * (2.0 * y_test - 1.0)
        correct = pred == y_test
        wrong = ~correct
        m = signed_margin.mean().item()
        q = (soft.square()).mean().item()
        return {
            "bit_acc": correct.float().mean().item(),
            "block_acc": correct.all(dim=1).float().mean().item(),
            "m": m,
            "q": q,
            "q_minus_m": q - m,
            "mean_abs_soft": soft.abs().mean().item(),
            "wrong_frac": wrong.float().mean().item(),
            "wrong_conf": soft.abs()[wrong].mean().item() if wrong.any() else float("nan"),
            "correct_conf": soft.abs()[correct].mean().item() if correct.any() else float("nan"),
            "negative_strong_frac": (signed_margin < -0.5).float().mean().item(),
            "uncertain_frac": (soft.abs() < 0.2).float().mean().item(),
            "per_pos_m": signed_margin.mean(dim=0).numpy(),
            "per_pos_q": soft.square().mean(dim=0).numpy(),
            "signed_margin": signed_margin.numpy().astype(np.float32),
        }


@subcommand("data/independent_training_sets/representative_runs/transformer_L12_N1280/seed<s>/",
            "Fig. 1, Fig. B1",
            option("--seed", type=int, required=True, help="seed of the global generator"),
            option("--out-dir", required=True),
            option("--epochs", type=int, default=600),
            option("--num-threads", type=int, default=2))
def independent_set_transformer_overlaps(args) -> int:
    """One L = 12, N = 1280 fixed-training-set Transformer with its overlaps after every epoch.

    A retreat of A_block can go two ways: paramagnetic (m -> 0 and q -> 0, the outputs become
    undecided) or confidently wrong (m -> 0 with q large).  The seed of the global generator is
    set before the network is built, so it fixes the initialization, the training set (1280
    draws) and the minibatch order.  After every epoch the network is evaluated on all 2048
    inputs, so m and q carry no sampling noise (confidence_observables).

    Writes into --out-dir: model.pt; metrics.json ("config", "test_mode", "train_history");
    confidence.csv, the same per-epoch rows; confidence_arrays.npz with the signed margins
    u y [epoch, input, position] and the profiles per_pos_m, per_pos_q [epoch, position].
    """
    out = resolve(args.out_dir)
    outputs = [out / name for name in ("model.pt", "metrics.json", "confidence.csv",
                                       "confidence_arrays.npz")]
    if not args.overwrite and present(*outputs):
        print(f"present {shown(out)}")
        return 0
    chain_task_threads()
    configuration = confidence_configuration(epochs=args.epochs, seed=args.seed,
                                             out_dir=args.out_dir, num_threads=args.num_threads)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    out.mkdir(parents=True, exist_ok=True)
    model = ChainTransformer(L=12, **ARCHITECTURE)
    train_set = draw_fresh(configuration["train_samples"], 12)
    x_test, y_test = complete_set(12)
    rows, margins, profile_m, profile_q = [], [], [], []
    for epoch, loss in independent_set_epochs(model, configuration["lr"],
                                              configuration["weight_decay"], BATCH,
                                              args.epochs, configuration["train_samples"], 12,
                                              train_set):
        diag = confidence_observables(model, x_test, y_test)
        margins.append(diag.pop("signed_margin"))
        profile_m.append(diag.pop("per_pos_m"))
        profile_q.append(diag.pop("per_pos_q"))
        rows.append({"epoch": epoch, "loss": loss, **diag})
        print(f"epoch {epoch:03d} loss={loss:.4f} block={diag['block_acc']:.3f} "
              f"m={diag['m']:.3f} q={diag['q']:.3f}", flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    write_json(out / "metrics.json",
               {"config": configuration, "test_mode": "exact_all_2048", "train_history": rows})
    write_rows(out / "confidence.csv", rows)
    np.savez_compressed(out / "confidence_arrays.npz", signed_margin=np.stack(margins, axis=0),
                        per_pos_m=np.stack(profile_m, axis=0), per_pos_q=np.stack(profile_q, axis=0))
    print(f"wrote {shown(out)}", flush=True)
    return 0


REPRESENTATIVE_TRANSFORMER_SEEDS = (0, 4, 11)


@subcommand("data/independent_training_sets/representative_runs/transformer_L12_N1280/",
            "Fig. 1, Fig. B1")
def representative_runs_transformer(args) -> int:
    """The three 600-epoch runs at L = 12, N = 1280, seeds 0, 4, 11 (paper: seed 0).

    One independent_set_transformer_overlaps child per seed, one after the other, on two threads,
    writing data/independent_training_sets/representative_runs/transformer_L12_N1280/seed<s>/.
    The seed sets both the initialization and the training set, so the three runs are three
    independent training sets, not three replicas of one.  A seed whose confidence.csv is
    present is skipped.
    """
    environment = os.environ.copy()
    environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    environment.setdefault("OMP_NUM_THREADS", "2")
    jobs = []
    for seed in REPRESENTATIVE_TRANSFORMER_SEEDS:
        out = REPRESENTATIVE_TRANSFORMER_RUNS / f"seed{seed}"
        if not args.overwrite and (out / "confidence.csv").exists():
            print(f"present {shown(out)}")
            continue
        jobs.append((f"seed{seed}", child("independent_set_transformer_overlaps", "--seed", seed,
                                          "--out-dir", out, "--epochs", 600, "--num-threads", 2,
                                          "--overwrite"),
                     environment))
    return run_jobs(jobs, 1, LOGS / "representative_runs_transformer")


# ===========================================================================
# Small-data replicas: the memorization ensembles (Fig. 5, Table 3, Tables C1, C3, Fig. C3)
# ===========================================================================
# At small N a network with more capacity than the training set needs memorizes the positions it
# cannot learn.  The memorized values depend on the replica, so on unseen inputs the outputs
# at those positions are confident (q_self > 0), uncorrelated with the truth (m -> 0) and different
# between replicas (q_cross < q_self): the memorization regime.  Every replica gets the same
# budget of UPDATE_BUDGET updates, whatever N is.

UPDATE_BUDGET = 4000
SMALL_DATA_EVALUATION = 4000      # evaluation draws with EVAL_SEED


def train_replica_for_steps(L: int, N: int, seed: int, inputs: torch.Tensor,
                            targets: torch.Tensor) -> tuple[nn.Module, int, int]:
    """One replica on the shared training set for about UPDATE_BUDGET updates.

    torch.manual_seed(seed) initializes the network, seed + 1 orders the minibatches.  The
    number of epochs is ceil(UPDATE_BUDGET / steps per epoch); for every N used the steps per
    epoch (1 or 2) divide UPDATE_BUDGET, so the budget is exactly 4000 updates.  Returns the
    network in evaluation mode, the number of epochs and the number of updates.
    """
    torch.manual_seed(seed)
    model = ChainTransformer(L=L, **ARCHITECTURE)
    steps_per_epoch = math.ceil(N / BATCH)
    epochs = math.ceil(UPDATE_BUDGET / steps_per_epoch)
    for epoch, loss in replica_epochs(model, inputs, targets, seed + 1, epochs):
        if epoch % max(1, epochs // 6) == 0:
            print(f"[N={N} seed={seed}] epoch {epoch}/{epochs} loss={loss:.4f}", flush=True)
    model.eval()
    return model, epochs, epochs * steps_per_epoch


def small_data_dir(L: int, N: int) -> Path:
    """data/small_data_ensembles/L<L>_N<N>, the directory of the (L, N) small-data ensemble."""
    return SMALL_DATA_ENSEMBLES / f"L{L}_N{N}"


def small_data_replica_complete(run_dir: Path, L: int, N: int, seed: int) -> bool:
    """A small-data replica is complete when its weights are stored and its metrics.json gives
    L, N, the seed, TRAIN_SEED and at least UPDATE_BUDGET updates."""
    if not present(run_dir / "metrics.json", run_dir / "model.pt"):
        return False
    try:
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = payload["config"]
        return (int(config["L"]) == L and int(config["N"]) == N and int(config["seed"]) == seed
                and int(payload["train_seed"]) == TRAIN_SEED
                and int(payload["steps"]) >= UPDATE_BUDGET)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


@subcommand("data/small_data_ensembles/L<L>_N<N>/seed<s>/",
            "Fig. 5, Table 3, Tables C1, C3, Fig. C3",
            option("--L", type=int, required=True),
            option("--N", type=int, required=True, help="size of the shared training set"),
            option("--seed", type=int, required=True, help="replica seed"),
            option("--threads", type=int, default=1, help="intra-op threads of torch"),
            option("--out", default=None,
                   help="ensemble directory (default data/small_data_ensembles/L<L>_N<N>)"))
def small_data_replica(args) -> int:
    """One small-data replica on the shared training set for UPDATE_BUDGET updates: <out>/seed<s>/.

    The training set is N draws with TRAIN_SEED, the same for every replica (the quenched
    disorder); --seed changes only the initialization and the minibatch order
    (train_replica_for_steps).  Writes model.pt and metrics.json with the settings, the training
    and evaluation seeds, the update and epoch counts, and the final bit and block accuracy on
    the training set and on SMALL_DATA_EVALUATION draws with EVAL_SEED (final_train_bit_acc,
    final_train_block_acc, final_test_bit_acc, final_test_block_acc).  A complete replica
    (small_data_replica_complete) is skipped.
    """
    root = resolve(args.out) if args.out else small_data_dir(args.L, args.N)
    run_dir = root / f"seed{args.seed}"
    if not args.overwrite and small_data_replica_complete(run_dir, args.L, args.N, args.seed):
        print(f"present {shown(run_dir)}")
        return 0
    torch.set_num_threads(args.threads)
    x_tr, z_tr = sample_set(args.N, args.L, TRAIN_SEED)
    x_te, z_te = sample_set(SMALL_DATA_EVALUATION, args.L, EVAL_SEED)
    start = time.time()
    model, epochs, steps = train_replica_for_steps(args.L, args.N, args.seed, x_tr, z_tr)
    with torch.no_grad():
        right_train = (model(x_tr) > 0).float() == z_tr
        right_test = (model(x_te) > 0).float() == z_te
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_json(run_dir / "metrics.json", {
        "config": {"L": args.L, "N": args.N, "seed": args.seed, "batch": BATCH, "channels": 16,
                   "depth": 8, "heads": 4, "ffn": 64, "lr": LEARNING_RATE, "wd": WEIGHT_DECAY},
        "train_seed": TRAIN_SEED, "evaluation_seed": EVAL_SEED, "target_steps": UPDATE_BUDGET,
        "steps": steps, "epochs": epochs,
        "final_train_bit_acc": float(right_train.float().mean()),
        "final_train_block_acc": float(right_train.all(1).float().mean()),
        "final_test_bit_acc": float(right_test.float().mean()),
        "final_test_block_acc": float(right_test.all(1).float().mean())})
    print(f"wrote {shown(run_dir)} in {time.time() - start:.0f} s", flush=True)
    return 0


# (L, N) -> number of replicas, seeds 0 to n - 1, of the small-data ensembles: the L = 12 scan
# over the training-set size of Fig. 5 (six replicas at each N) and the memorization ensembles
# (12, 64), (24, 64), (24, 128) and (24, 192) of Table 3.
SMALL_DATA_REPLICAS = {(12, 32): 6, (12, 64): 100, (12, 80): 6, (12, 96): 6, (12, 112): 6,
                       (12, 128): 6, (12, 160): 6, (12, 256): 6,
                       (24, 64): 100, (24, 128): 100, (24, 192): 120}


def ensembles_to_train(table: dict, args) -> list[tuple[int, int]]:
    """The (L, N) keys of `table`, or the one ensemble that --L and --N name."""
    if (args.L is None) != (args.N is None):
        raise SystemExit("give both --L and --N, or neither")
    if args.L is None:
        return list(table)
    if (args.L, args.N) not in table:
        raise SystemExit(f"no ensemble at (L, N) = ({args.L}, {args.N}); the ensembles are "
                         f"{sorted(table)}")
    return [(args.L, args.N)]


@subcommand("data/small_data_ensembles/L<L>_N<N>/seed<s>/",
            "Fig. 5, Table 3, Tables C1, C3, Fig. C3",
            option("--L", type=int, default=None, help="train one ensemble only, with --N"),
            option("--N", type=int, default=None),
            option("--threads", type=int, default=1, help="intra-op threads of each replica"),
            option("--workers", type=int, default=6, help="replicas trained at a time"))
def small_data_ensembles(args) -> int:
    """Every replica of the small-data ensembles (SMALL_DATA_REPLICAS) that is not complete.

    One small_data_replica child per replica, --workers at a time, each with every thread
    variable set to --threads.  A complete replica (small_data_replica_complete) is skipped.
    """
    jobs = []
    for L, N in ensembles_to_train(SMALL_DATA_REPLICAS, args):
        for seed in range(SMALL_DATA_REPLICAS[(L, N)]):
            run_dir = small_data_dir(L, N) / f"seed{seed}"
            if not args.overwrite and small_data_replica_complete(run_dir, L, N, seed):
                continue
            jobs.append((f"L{L}_N{N}_seed{seed}",
                         child("small_data_replica", "--L", L, "--N", N, "--seed", seed,
                               "--threads", args.threads, "--overwrite"),
                         pinned_environment(args.threads)))
    return run_jobs(jobs, args.workers, LOGS / "small_data_ensembles")


# ===========================================================================
# Large-data replica ensembles: endpoints (Table 3, Figs. 6-9, Sec. 4, Table C1)
# ===========================================================================

def large_data_part(L: int, N: int, first: int, last: int) -> Path:
    """data/large_data_ensembles/L<L>_N<N>/seeds<first>-<last>.npz, the part of those seeds."""
    return LARGE_DATA_ENSEMBLES / f"L{L}_N{N}" / f"seeds{first:04d}-{last:04d}.npz"


@subcommand("data/large_data_ensembles/L<L>_N<N>/seeds<first>-<last>.npz",
            "Table 3, Figs. 6-9, C1, C2, Sec. 4, Table C1",
            option("--seed-lo", type=int, required=True),
            option("--seed-hi", type=int, required=True, help="one past the last seed"),
            option("--out", default=None,
                   help="part file (default data/large_data_ensembles/L<L>_N<N>/"
                        "seeds<lo>-<hi-1>.npz)"),
            option("--L", type=int, default=12),
            option("--N", type=int, default=1280),
            option("--epochs", type=int, default=150),
            option("--batch", type=int, default=BATCH),
            option("--test-samples", type=int, default=2048),
            option("--threads", type=int, default=1))
def large_data_replicas(args) -> int:
    """Replicas --seed-lo ... --seed-hi - 1 on one shared training set; their endpoints.

    Every replica trains on the N draws of TRAIN_SEED and is evaluated on --test-samples
    draws of EVAL_SEED; replica s is initialized from seed s and ordered by seed s + 1,
    --epochs epochs of AdamW.  Writes one part of the (L, N) large-data ensemble, the file
    <out> with

        final_xhat   [replica, input, position]   u = tanh(z/2) after the last epoch (float32)
        block_acc    [replica, epoch]         A_block on the evaluation draws after each epoch
        seeds        [replica]
        yspin        [input, position]            truth spins of the evaluation draws (float32)

    The A_block curve gives the endpoint class of Sec. 4.2 (common.endpoint_sets) and
    final_xhat the function the replica ends at.
    """
    out = (resolve(args.out) if args.out
           else large_data_part(args.L, args.N, args.seed_lo, args.seed_hi - 1))
    if not args.overwrite and present(out):
        print(f"present {shown(out)}")
        return 0
    torch.set_num_threads(args.threads)
    out.parent.mkdir(parents=True, exist_ok=True)
    x_tr, z_tr = sample_set(args.N, args.L, TRAIN_SEED)
    x_te, z_te = sample_set(args.test_samples, args.L, EVAL_SEED)
    yspin = (2 * z_te - 1).numpy().astype(np.float32)
    seeds = list(range(args.seed_lo, args.seed_hi))
    finals, curves = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = ChainTransformer(L=args.L, **ARCHITECTURE)
        curve = []
        for _ in replica_epochs(model, x_tr, z_tr, seed + 1, args.epochs, args.batch):
            model.eval()
            with torch.no_grad():
                curve.append(float(((model(x_te) > 0).float() == z_te).all(1).float().mean()))
        with torch.no_grad():
            finals.append(torch.tanh(model(x_te) / 2.0).numpy().astype(np.float32))
        curves.append(curve)
        print(f"seed {seed}: peak={max(curve):.2f} final={curve[-1]:.2f}", flush=True)
    np.savez_compressed(out, final_xhat=np.stack(finals, 0),
                        block_acc=np.array(curves), seeds=np.array(seeds), yspin=yspin)
    print(f"wrote {shown(out)}", flush=True)
    return 0


# (L, N) -> number of replicas, seeds 0 to n - 1, of the five large-data ensembles of Table 3.
LARGE_DATA_REPLICAS = {(12, 1280): 1200, (16, 2048): 330, (16, 2560): 290, (16, 3072): 260,
                       (20, 24576): 156}
LARGE_DATA_PART = 10      # at most this many consecutive seeds in one part file


def stored_large_data_seeds(L: int, N: int) -> set[int]:
    """The seeds that the part files of the (L, N) large-data ensemble hold."""
    return {int(seed) for path in (LARGE_DATA_ENSEMBLES / f"L{L}_N{N}").glob("seeds*.npz")
            for seed in np.load(path)["seeds"]}


@subcommand("data/large_data_ensembles/L<L>_N<N>/seeds<first>-<last>.npz",
            "Table 3, Figs. 6-9, C1, C2, Sec. 4, Table C1",
            option("--L", type=int, default=None, help="train one ensemble only, with --N"),
            option("--N", type=int, default=None),
            option("--threads", type=int, default=1, help="intra-op threads of each part"),
            option("--workers", type=int, default=6, help="parts trained at a time"))
def large_data_ensembles(args) -> int:
    """Every replica of the large-data ensembles (LARGE_DATA_REPLICAS) that no part file holds.

    The missing seeds of an ensemble are cut into runs of consecutive seeds and each run into
    parts of at most LARGE_DATA_PART seeds.  A part is one large_data_replicas child (150 epochs,
    batch 128, 2048 evaluation draws) writing
    data/large_data_ensembles/L<L>_N<N>/seeds<first>-<last>.npz, with every thread variable set
    to --threads; --workers parts train at a time.  With --overwrite every seed is trained again.
    """
    jobs = []
    for L, N in ensembles_to_train(LARGE_DATA_REPLICAS, args):
        stored = set() if args.overwrite else stored_large_data_seeds(L, N)
        parts: list[list[int]] = []
        for seed in range(LARGE_DATA_REPLICAS[(L, N)]):
            if seed in stored:
                continue
            if parts and parts[-1][-1] == seed - 1 and len(parts[-1]) < LARGE_DATA_PART:
                parts[-1].append(seed)
            else:
                parts.append([seed])
        for part in parts:
            out = large_data_part(L, N, part[0], part[-1])
            jobs.append((f"{out.parent.name}_{out.stem}",
                         child("large_data_replicas", "--L", L, "--N", N, "--seed-lo", part[0],
                               "--seed-hi", part[-1] + 1, "--epochs", 150, "--batch", BATCH,
                               "--test-samples", 2048, "--threads", args.threads, "--out", out,
                               "--overwrite"),
                         pinned_environment(args.threads)))
    return run_jobs(jobs, args.workers, LOGS / "large_data_ensembles")


# ===========================================================================
# Replica trajectories at (12, 1280): per-epoch outputs, weights and observables
# ===========================================================================

SOFT_OUTPUT_EPOCHS = 150
SOFT_OUTPUT_EVALUATION = 4096


@subcommand("data/trajectory_ensembles/L12_N1280_soft_outputs/", "Fig. 4, Secs. 3.4, 4.2",
            option("--seeds", type=int_list, default=list(range(16))),
            option("--out-dir", default=str(TRAJECTORY_ENSEMBLES / "L12_N1280_soft_outputs")),
            option("--threads", type=int, default=2))
def soft_output_trajectories_l12_n1280(args) -> int:
    """Replicas at (12, 1280) with their soft spins after every epoch (Fig. 4).

    The replicas share the training set of TRAIN_SEED and differ in initialization (seed s)
    and minibatch order (seed s + 1).  After every epoch u = tanh(z/2) is stored on
    SOFT_OUTPUT_EVALUATION draws with EVAL_SEED, so the overlaps can be followed through
    acquisition and retreat.  Writes into --out-dir

        xhat_history.npz            xhat [replica, epoch, input, position] (float32), seeds, yspin
        per_seed_epoch_metrics.csv  seed, epoch, A_block on the evaluation draws
    """
    out = resolve(args.out_dir)
    if not args.overwrite and present(out / "xhat_history.npz", out / "per_seed_epoch_metrics.csv"):
        print(f"present {shown(out)}")
        return 0
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)
    x_tr, z_tr = sample_set(1280, 12, TRAIN_SEED)
    x_te, z_te = sample_set(SOFT_OUTPUT_EVALUATION, 12, EVAL_SEED)
    yspin = (2 * z_te - 1).numpy().astype(np.float32)
    rows, history = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = ChainTransformer(L=12, **ARCHITECTURE)
        per_epoch = []
        for epoch, _ in replica_epochs(model, x_tr, z_tr, seed + 1, SOFT_OUTPUT_EPOCHS):
            model.eval()
            with torch.no_grad():
                logits = model(x_te)
                block = float(((logits > 0).float() == z_te).all(dim=1).float().mean())
            rows.append({"seed": seed, "epoch": epoch, "block_acc": block})
            per_epoch.append(torch.tanh(logits / 2.0).numpy().astype(np.float32))
            if epoch % 20 == 0 or epoch == 1:
                print(f"seed {seed} epoch {epoch:03d}: block={block:.3f}", flush=True)
        history.append(np.stack(per_epoch, 0))
    np.savez_compressed(out / "xhat_history.npz", xhat=np.stack(history, 0),
                        seeds=np.array(args.seeds), yspin=yspin)
    write_rows(out / "per_seed_epoch_metrics.csv", rows, ["seed", "epoch", "block_acc"])
    print(f"wrote {shown(out)}", flush=True)
    return 0


CHECKPOINT_EPOCHS = 220
CHECKPOINT_START = 20
CHECKPOINT_EVALUATION = 4096


def call_files(seeds) -> tuple[str, str]:
    """The log and the manifest of one checkpointed_trajectories call, named by its seeds."""
    tag = f"seeds{min(seeds):02d}-{max(seeds):02d}"
    return f"trajectory_{tag}.csv", f"manifest_{tag}.json"


@subcommand("data/checkpointed_trajectories/L12_N1280/", "Fig. A1, Apps. A.2, A.3",
            option("--seeds", type=int_list, default=list(range(6))),
            option("--out-dir", default=str(CHECKPOINTED_TRAJECTORIES / "L12_N1280")),
            option("--threads", type=int, default=6))
def checkpointed_trajectories(args) -> int:
    """Replicas at (12, 1280) with their weights after every epoch from 20 to 220.

    The protocol of soft_output_trajectories_l12_n1280 run for CHECKPOINT_EPOCHS epochs; the
    complete state dict is saved after every epoch from CHECKPOINT_START on.  With the weights
    on disk the curvature and the final FFN can be followed through acquisition, retreat and
    re-acquisition.  Writes into --out-dir

        seed<s>/epoch<e>.pt            state dict after epoch e (201 per seed)
        trajectory_seeds<a>-<b>.csv    seed, epoch, A_block and loss on CHECKPOINT_EVALUATION
                                       draws with EVAL_SEED, every epoch, for the seeds a..b
                                       of this call
        manifest_seeds<a>-<b>.json     every checkpoint of the call with its A_block and loss

    A call is present when every checkpoint of its seeds is.  The paper's calls are

        (defaults)                    seeds 0-5
        --seeds 6,7,8
        --seeds 9,10,11,12,13,14,15
        --seeds 16,...,22
        --seeds 23,...,29

    A seed with at least five epochs above A_block 0.8 and three below 0.2 is printed as
    one that acquires, retreats and acquires again.
    """
    out = resolve(args.out_dir)
    trajectory_name, manifest_name = call_files(args.seeds)
    checkpoints = [out / f"seed{seed}" / f"epoch{epoch}.pt" for seed in args.seeds
                   for epoch in range(CHECKPOINT_START, CHECKPOINT_EPOCHS + 1)]
    if not args.overwrite and present(*checkpoints):
        print(f"present {shown(out)}, seeds {args.seeds}")
        return 0
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)
    x_tr, z_tr = sample_set(1280, 12, TRAIN_SEED)
    x_te, z_te = sample_set(CHECKPOINT_EVALUATION, 12, EVAL_SEED)
    rows, manifest = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = ChainTransformer(L=12, **ARCHITECTURE)
        (out / f"seed{seed}").mkdir(parents=True, exist_ok=True)
        for epoch, _ in replica_epochs(model, x_tr, z_tr, seed + 1, CHECKPOINT_EPOCHS):
            model.eval()
            with torch.no_grad():
                logits = model(x_te)
                block = float(((logits > 0).float() == z_te).all(dim=1).float().mean())
                loss = float(F.binary_cross_entropy_with_logits(logits, z_te))
            rows.append({"seed": seed, "epoch": epoch, "block_acc": block, "loss": loss})
            if epoch >= CHECKPOINT_START:
                name = Path(f"seed{seed}") / f"epoch{epoch}.pt"
                torch.save(model.state_dict(), out / name)
                # the manifest names the file by the --out-dir as given
                manifest.append({"seed": seed, "epoch": epoch, "file": str(Path(args.out_dir) / name),
                                 "block_acc": block, "loss": loss})
            if epoch % 40 == 0 or epoch == 1:
                print(f"seed {seed} epoch {epoch:03d}: block={block:.3f}", flush=True)
    write_rows(out / trajectory_name, rows, ["seed", "epoch", "block_acc", "loss"])
    write_json(out / manifest_name, manifest)
    for seed in args.seeds:
        blocks = [row["block_acc"] for row in rows if row["seed"] == seed]
        high, low = sum(b > 0.8 for b in blocks), sum(b < 0.2 for b in blocks)
        print(f"seed {seed}: {high} epochs above 0.8, {low} below 0.2"
              f"{'  acquires again' if high >= 5 and low >= 3 else ''}", flush=True)
    return 0


# The ensemble of App. A.1 is defined at L = 12, N = 1280 and observed at epoch 150.
FIXED_EPOCH = 150
FIXED_EPOCH_EVALUATION = 2048


@subcommand("data/fixed_epoch_replicas/L12_N1280_epoch150/", "App. A.1, A.2",
            option("--seed-lo", type=int, default=0),
            option("--seed-hi", type=int, default=16, help="one past the last seed"),
            option("--threads", type=int, default=2),
            option("--out", default=str(FIXED_EPOCH_REPLICAS / "L12_N1280_epoch150")))
def fixed_epoch_replicas(args) -> int:
    """Sixteen replicas at (12, 1280) with their weights at epoch 150, for App. A.1.

    The protocol of large_data_replicas (shared training set of TRAIN_SEED, 2048
    evaluation draws of EVAL_SEED, seeds 0-15), keeping the weights that the curvature
    (Hessian) and stationarity measurements need.  Writes into --out

        seed<k>/model.pt       weights after epoch 150
        seed<k>/metrics.json   the protocol, the settings and, per epoch, loss, bit and block
                               accuracy on the training set and on the evaluation draws
        manifest.json          the seeds whose weights are present

    A seed whose two files are present is skipped; manifest.json is written only when a seed
    was trained or it is missing.
    """
    out = resolve(args.out)
    seeds = range(args.seed_lo, args.seed_hi)
    todo = [seed for seed in seeds if args.overwrite
            or not present(out / f"seed{seed}" / "model.pt", out / f"seed{seed}" / "metrics.json")]
    if not todo and present(out / "manifest.json"):
        print(f"present {shown(out)}")
        return 0
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)
    x_tr, z_tr = sample_set(1280, 12, TRAIN_SEED)
    x_ev, z_ev = sample_set(FIXED_EPOCH_EVALUATION, 12, EVAL_SEED)

    def block_accuracy(logits, targets) -> float:
        return float(((logits > 0).float() == targets).all(dim=1).float().mean())

    for seed in todo:
        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        model = ChainTransformer(L=12, **ARCHITECTURE)
        history = []
        for epoch, _ in replica_epochs(model, x_tr, z_tr, seed + 1, FIXED_EPOCH):
            model.eval()
            with torch.no_grad():
                train_logits, eval_logits = model(x_tr), model(x_ev)
                history.append({
                    "epoch": epoch,
                    "train_loss": float(F.binary_cross_entropy_with_logits(train_logits, z_tr)),
                    "train_bit_acc": float(((train_logits > 0).float() == z_tr).float().mean()),
                    "train_block_acc": block_accuracy(train_logits, z_tr),
                    "evaluation_loss": float(F.binary_cross_entropy_with_logits(eval_logits, z_ev)),
                    "evaluation_bit_acc": float(((eval_logits > 0).float() == z_ev).float().mean()),
                    "evaluation_block_acc": block_accuracy(eval_logits, z_ev)})
            if epoch == 1 or epoch % 25 == 0 or epoch == FIXED_EPOCH:
                print(f"seed {seed:02d}, epoch {epoch:03d}: evaluation block="
                      f"{history[-1]['evaluation_block_acc']:.3f}", flush=True)
        torch.save(model.state_dict(), seed_dir / "model.pt")
        write_json(seed_dir / "metrics.json", {
            "protocol": "l12_shared_fixed_disorder", "train_seed": TRAIN_SEED,
            "evaluation_seed": EVAL_SEED,
            "config": {"L": 12, "N_train": 1280, "evaluation_samples": FIXED_EPOCH_EVALUATION,
                       "epochs": FIXED_EPOCH, "batch_size": BATCH, "channels": 16,
                       "depth": 8, "num_heads": 4, "ffn_dim": 64, "dropout": 0.0,
                       "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
                       "weight_decay": WEIGHT_DECAY, "initialization_and_sgd_seed": seed,
                       "batches_per_epoch": math.ceil(1280 / BATCH)},
            "history": history})
    write_json(out / "manifest.json", {
        "protocol": "l12_shared_fixed_disorder", "train_seed": TRAIN_SEED,
        "evaluation_seed": EVAL_SEED, "L": 12, "N_train": 1280, "epochs": FIXED_EPOCH,
        "available_seeds": [seed for seed in seeds if (out / f"seed{seed}" / "model.pt").exists()]})
    return 0


# ---------------------------------------------------------------------------
# The sixteen trajectories of Secs. 3.2-3.4 (Figs. 2, 3)
# ---------------------------------------------------------------------------

TRAJECTORY_SEEDS = tuple(range(16))
TRAJECTORY_THREADS = 2
UNCERTAIN_PRIMARY, UNCERTAIN_SECONDARY = 0.5, 0.2
TAU_MIN, TAU_MAX = -60, 70                   # window of the crossing-aligned average
DISPLAYED_SEEDS, DISPLAYED_EPOCH_MAX = (0, 1), 350   # the two trajectories drawn in Fig. 3


def initialization_seed(data_seed: int, index: int) -> int:
    """Initialization seed of replica `index` at data seed d: 100000 + 1003 d + index."""
    return 100_000 + 1_003 * data_seed + index


def minibatch_seed(data_seed: int, index: int) -> int:
    """Minibatch-order seed of replica `index` at data seed d: 200000 + 1009 d + index."""
    return 200_000 + 1_009 * data_seed + index


def tensor_sha256(*tensors: torch.Tensor) -> str:
    """SHA-256 of the bytes of the tensors, which identifies a data set."""
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def trajectory_observables(model: nn.Module, x_eval: torch.Tensor, y_eval: torch.Tensor) -> dict:
    """Accuracies, m, q_self, Delta_N and the uncertain fractions on the complete input set.

    u = 2 sigmoid(z) - 1, a bit is predicted 1 when sigmoid(z) >= 0.5.
    independent_block_acc is the product over positions of the position accuracies, the block
    accuracy the positions would have if their errors were independent.
    """
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(x_eval))
        correct = (probability >= 0.5) == y_eval.bool()
        soft = 2.0 * probability - 1.0
        margin = soft * (2.0 * y_eval - 1.0)
        return {
            "bit_acc": float(correct.float().mean().item()),
            "block_acc": float(correct.all(dim=1).float().mean().item()),
            "independent_block_acc": float(correct.float().mean(dim=0).prod().item()),
            "m": float(margin.mean().item()),
            "q_self": float(soft.square().mean().item()),
            "delta_N": float((soft.square().mean() - margin.mean()).item()),
            "uncertain_frac_0p5": float((soft.abs() < UNCERTAIN_PRIMARY).float().mean().item()),
            "uncertain_frac_0p2": float((soft.abs() < UNCERTAIN_SECONDARY).float().mean().item()),
        }


def history_is_complete(path: Path, epochs: int) -> bool:
    """True when a history.csv holds exactly `epochs` rows, the last at epoch `epochs`."""
    import pandas as pd
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, usecols=["epoch"])
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return len(frame) == epochs and int(frame["epoch"].iloc[-1]) == epochs


def aligned_summary(frames: list):
    """Mean and standard error over replicas at each tau = epoch - first crossing.

    Only the replicas that cross enter, and only TAU_MIN <= tau <= TAU_MAX; n is the number of
    replicas at that tau.  The standard error is std(ddof=1) / sqrt(n), zero for n = 1.
    """
    import pandas as pd
    pieces = []
    for index, frame in frames:
        crossing = crossing_epoch(frame["epoch"], frame["block_acc"])
        if crossing is None:
            continue
        local = frame.copy()
        local["seed_index"] = index
        local["tau"] = local["epoch"] - crossing
        pieces.append(local.loc[local["tau"].between(TAU_MIN, TAU_MAX)])
    if not pieces:
        raise RuntimeError("no replica crosses the block-accuracy threshold")
    aligned = pd.concat(pieces, ignore_index=True)
    keys = ["block_acc", "bit_acc", "independent_block_acc", "m", "q_self", "delta_N",
            "certainty_0p5", "certainty_0p2"]
    rows = []
    for tau, group in aligned.groupby("tau", sort=True):
        row = {"tau": int(tau), "n": int(group["seed_index"].nunique())}
        for key in keys:
            values = group[key].to_numpy(dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_sem"] = (float(values.std(ddof=1) / math.sqrt(len(values)))
                                 if len(values) > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_summary(frames: list, aligned, data_metadata: dict) -> dict:
    """The protocol, per-replica crossings and retreats, and counts of Delta_N after crossing.

    Upcrossings count the rises of A_block to HI or above (a curve starting above counts
    one); downcrossings count the falls below LO from the first crossing on.  The
    post-crossing counts pool every epoch from the first crossing of every crossing replica:
    |Delta_N| <= 0.05, Delta_N > 0.05, > 0.1, and the same at A_block < LO; once for
    all replicas and once for seeds 0 and 1 up to epoch 350, the context of Fig. 3.
    """
    import pandas as pd

    def upcrossings(values: np.ndarray) -> int:
        above = values >= HI
        return int(above[0]) + int(np.sum(above[1:] & ~above[:-1]))

    def downcrossings(frame, crossing) -> int:
        if crossing is None:
            return 0
        below = frame.loc[frame["epoch"] >= crossing, "block_acc"].to_numpy() < LO
        return int(np.sum(below[1:] & ~below[:-1]))

    def post_crossing(selected: list, epoch_max: int | None = None) -> dict:
        pieces = []
        for index, frame in selected:
            crossing = crossing_epoch(frame["epoch"], frame["block_acc"])
            if crossing is None:
                continue
            local = frame.loc[frame["epoch"] >= crossing].copy()
            if epoch_max is not None:
                local = local.loc[local["epoch"] <= epoch_max]
            local["seed_index"] = index
            pieces.append(local)
        pooled = pd.concat(pieces, ignore_index=True)
        low = pooled.loc[pooled["block_acc"] < LO]
        return {"observations": int(len(pooled)),
                "abs_delta_le_0p05": int((pooled["delta_N"].abs() <= 0.05).sum()),
                "delta_gt_0p05": int((pooled["delta_N"] > 0.05).sum()),
                "delta_gt_0p1": int((pooled["delta_N"] > 0.1).sum()),
                "low_block_observations": int(len(low)),
                "low_block_abs_delta_le_0p05": int((low["delta_N"].abs() <= 0.05).sum())}

    seed_rows = []
    for index, frame in frames:
        crossing = crossing_epoch(frame["epoch"], frame["block_acc"])
        block = frame["block_acc"].to_numpy(dtype=float)
        seed_rows.append({"seed_index": index, "first_block_0p8_crossing": crossing,
                          "block_0p8_upcrossings": upcrossings(block),
                          "post_crossing_block_0p5_downcrossings": downcrossings(frame, crossing),
                          "max_block_acc": float(block.max()), "final_block_acc": float(block[-1])})
    tau_minus_20 = aligned.loc[aligned["tau"] == -20]
    return {
        "protocol": {**data_metadata, "replicas": len(frames), "shared_training_set": True,
                     "crossing_alignment_threshold": HI,
                     "retreat_threshold": LO,
                     "primary_uncertainty_threshold": UNCERTAIN_PRIMARY,
                     "secondary_uncertainty_threshold": UNCERTAIN_SECONDARY},
        "successful_replicas": sum(row["first_block_0p8_crossing"] is not None for row in seed_rows),
        "replicas_with_multiple_0p8_upcrossings": sum(row["block_0p8_upcrossings"] >= 2
                                                      for row in seed_rows),
        "replicas_with_post_crossing_retreat_below_0p5": sum(
            row["post_crossing_block_0p5_downcrossings"] >= 1 for row in seed_rows),
        "all_replica_post_crossing": post_crossing(frames),
        "displayed_seed_0_1_post_crossing_through_epoch_350": post_crossing(
            [item for item in frames if item[0] in DISPLAYED_SEEDS], DISPLAYED_EPOCH_MAX),
        "tau_minus_20": None if tau_minus_20.empty else tau_minus_20.iloc[0].to_dict(),
        "seeds": seed_rows,
    }


@subcommand("data/trajectory_ensembles/L12_N1280_overlaps/", "Figs. 2, 3, Secs. 3.2-3.4")
def overlap_trajectories_l12_n1280(args) -> int:
    """Sixteen replicas at (12, 1280) with their observables after every epoch (Figs. 2, 3).

    One training set of 1280 independent draws with replacement (963 distinct inputs) is drawn
    from the global generator seeded with the data seed d = TRAIN_SEED and shared by every
    replica; replica i is initialized from seed 100000 + 1003 d + i and ordered by
    200000 + 1009 d + i.  Every replica trains TRAJECTORY_EPOCHS epochs on two threads and is
    evaluated after every epoch on the complete set of 2048 inputs (trajectory_observables).
    Writes into data/trajectory_ensembles/L12_N1280_overlaps/

        protocol.json           the configuration, the seed rules, the hashes of the training
                                set and of the evaluation set
        seed<i>/history.csv     one row per epoch: loss, the observables, certainty = 1 -
                                uncertain fraction
        seed<i>/metadata.json   the seeds of the replica, its first epoch with A_block >= 0.8
                                and the training time
        aligned_summary.csv     mean and standard error at each tau = epoch - first crossing
        ensemble_summary.json   per-replica crossings and retreats, counts of Delta_N

    A replica whose history.csv is complete is kept; with every output present nothing is
    written.  Fig. 2 reads the sixteen replicas, Fig. 3 seeds 0 and 1.
    """
    import pandas as pd
    out = TRAJECTORY_ENSEMBLES / "L12_N1280_overlaps"
    histories = [out / f"seed{index}" / "history.csv" for index in TRAJECTORY_SEEDS]
    if (not args.overwrite
            and present(out / "protocol.json", out / "aligned_summary.csv",
                        out / "ensemble_summary.json",
                        *[out / f"seed{index}" / "metadata.json" for index in TRAJECTORY_SEEDS])
            and all(history_is_complete(path, TRAJECTORY_EPOCHS) for path in histories)):
        print(f"present {shown(out)}")
        return 0
    chain_task_threads()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(TRAJECTORY_THREADS)
    config = confidence_configuration(test_samples=2048, epochs=TRAJECTORY_EPOCHS)

    # The shared training set and the complete evaluation set, from the data seed alone.
    torch.manual_seed(TRAIN_SEED)
    x_train, y_train = draw_fresh(1280, 12)
    x_eval, y_eval = complete_set(12)
    data_metadata = {
        "data_seed": TRAIN_SEED,
        "training_set_sha256": tensor_sha256(x_train, y_train),
        "evaluation_cube_sha256": tensor_sha256(x_eval, y_eval),
        "training_draws": 1280,
        "unique_training_inputs": int(torch.unique(x_train[:, 0, 1:], dim=0).shape[0]),
        "evaluation_inputs": int(x_eval.shape[0]),
        "training_sampling": "fixed IID draws with replacement",
        "evaluation_sampling": "complete 2^(L-1) input cube",
    }
    thresholds = {"primary": UNCERTAIN_PRIMARY, "secondary": UNCERTAIN_SECONDARY}
    (out / "protocol.json").write_text(json.dumps({
        "config": config, "seed_indices": list(TRAJECTORY_SEEDS),
        "model_seed_rule": "100000 + 1003 * data_seed + seed_index",
        "minibatch_seed_rule": "200000 + 1009 * data_seed + seed_index",
        "shared_training_set": True,
        "uncertainty_thresholds": [UNCERTAIN_PRIMARY, UNCERTAIN_SECONDARY],
        **data_metadata}, indent=2), encoding="utf-8")

    fields = ["epoch", "loss", "bit_acc", "block_acc", "independent_block_acc", "m", "q_self",
              "delta_N", "uncertain_frac_0p5", "uncertain_frac_0p2", "certainty_0p5",
              "certainty_0p2"]
    for index, history in zip(TRAJECTORY_SEEDS, histories):
        if history_is_complete(history, TRAJECTORY_EPOCHS) and not args.overwrite:
            print(f"present {shown(history)}")
            continue
        history.parent.mkdir(parents=True, exist_ok=True)
        partial = history.with_name("history.partial.csv")
        torch.manual_seed(initialization_seed(TRAIN_SEED, index))
        model = ChainTransformer(L=12, **ARCHITECTURE)
        started, crossing = time.time(), None
        with open(partial, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch, loss in replica_epochs(model, x_train, y_train,
                                              minibatch_seed(TRAIN_SEED, index), TRAJECTORY_EPOCHS):
                observed = trajectory_observables(model, x_eval, y_eval)
                row = {"epoch": epoch, "loss": loss, **observed,
                       "certainty_0p5": 1.0 - observed["uncertain_frac_0p5"],
                       "certainty_0p2": 1.0 - observed["uncertain_frac_0p2"]}
                writer.writerow(row)
                handle.flush()
                if crossing is None and row["block_acc"] >= HI:
                    crossing = epoch
                if epoch == 1 or epoch % 25 == 0 or epoch == crossing:
                    print(f"seed={index:02d} epoch={epoch:03d} block={row['block_acc']:.3f} "
                          f"m={row['m']:.3f} q={row['q_self']:.3f}", flush=True)
        partial.replace(history)
        (history.parent / "metadata.json").write_text(json.dumps({
            "complete": True, "seed_index": index,
            "initialization_seed": initialization_seed(TRAIN_SEED, index),
            "minibatch_order_seed": minibatch_seed(TRAIN_SEED, index),
            "first_block_0p8_crossing": crossing, "elapsed_seconds": time.time() - started,
            "config": config, "uncertainty_thresholds": thresholds, **data_metadata},
            indent=2), encoding="utf-8")

    frames = [(index, pd.read_csv(path)) for index, path in zip(TRAJECTORY_SEEDS, histories)]
    aligned = aligned_summary(frames)
    aligned.to_csv(out / "aligned_summary.csv", index=False)
    summary = ensemble_summary(frames, aligned, data_metadata)
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"crossing replicas {summary['successful_replicas']}/{len(frames)}, retreating "
          f"{summary['replicas_with_post_crossing_retreat_below_0p5']}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Two memorization replicas at (12, 64) scored on the complete input set (Fig. 3, Sec. 3.3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverlapTrajectorySettings:
    """Settings of the two (12, 64) runs, stored in summary.json under "args"."""
    L: int = 12
    N: int = 64
    seeds: tuple = (0, 6)
    epochs: int = 1000
    batch_size: int = BATCH
    channels: int = 16
    depth: int = 8
    heads: int = 4
    ffn: int = 64
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_threads: int = 4
    out_dir: str = str(TRAJECTORY_ENSEMBLES / "L12_N64_overlaps")
    task: str = "parity"
    interval_length: int = 4
    anchor: int = -1


def complete_set_observables(model, x_test, y_test, x_train, y_train) -> dict:
    """Accuracies, m, q_self, Delta_N and the uncertain and strongly-wrong fractions.

    On the complete set of 2048 inputs, with u = tanh(z/2) and a bit predicted 1 when z > 0;
    strong_wrong_frac is the fraction with u y < -0.5, uncertain_frac with |u| < 0.2.  The
    training-set accuracies are on the 64 training draws.
    """
    with torch.no_grad():
        logits = model(x_test)
        xhat = torch.tanh(logits / 2.0)
        signed = xhat * (2.0 * y_test - 1.0)
        correct = (logits > 0).float() == y_test
        train_correct = (model(x_train) > 0).float() == y_train
        return {
            "bit_acc": float(correct.float().mean().item()),
            "block_acc": float(correct.all(dim=1).float().mean().item()),
            "m_truth": float(signed.mean().item()),
            "q_self": float((xhat * xhat).mean().item()),
            "delta_N": float(((xhat * xhat) - signed).mean().item()),
            "uncertain_frac": float((torch.abs(xhat) < 0.2).float().mean().item()),
            "strong_wrong_frac": float((signed < -0.5).float().mean().item()),
            "train_bit_acc": float(train_correct.float().mean().item()),
            "train_block_acc": float(train_correct.all(dim=1).float().mean().item()),
        }


@subcommand("data/trajectory_ensembles/L12_N64_overlaps/", "Fig. 3, Sec. 3.3",
            option("--num-threads", type=int, default=4))
def overlap_trajectories_l12_n64(args) -> int:
    """Two (12, 64) replicas for 1000 epochs, scored after every epoch on all 2048 inputs.

    Seeds 0 and 6 train on the shared training set of 64 draws with TRAIN_SEED, the set of
    small_data_replica.  N = 64 is below the batch of 128, so an epoch is one
    full-batch update.
    The complete enumeration leaves no sampling noise in the observables
    (complete_set_observables).  A replica that memorizes keeps q_self large while m stays small:
    it is confident and wrong.  Writes per_seed_epoch_metrics.csv and summary.json.
    """
    settings = OverlapTrajectorySettings(num_threads=args.num_threads)
    out = Path(settings.out_dir)
    metrics_path = out / "per_seed_epoch_metrics.csv"
    if not args.overwrite and present(metrics_path, out / "summary.json"):
        print(f"present {shown(out)}")
        return 0
    chain_task_threads()
    torch.set_num_threads(settings.num_threads)
    out.mkdir(parents=True, exist_ok=True)
    x_train, y_train = sample_set(settings.N, settings.L, TRAIN_SEED)
    x_test, y_test = complete_set(settings.L)
    rows = []
    for seed in settings.seeds:
        torch.manual_seed(seed)
        model = ChainTransformer(L=settings.L, **ARCHITECTURE)
        for epoch, loss in replica_epochs(model, x_train, y_train, seed + 1, settings.epochs):
            model.eval()
            row = complete_set_observables(model, x_test, y_test, x_train, y_train)
            row.update({"seed": int(seed), "epoch": int(epoch), "loss": loss})
            rows.append(row)
            if epoch == 1 or epoch % 100 == 0:
                print(f"seed {seed} epoch {epoch:04d}: block={row['block_acc']:.3f} "
                      f"m={row['m_truth']:.3f} q={row['q_self']:.3f}", flush=True)
    write_rows(metrics_path, rows)
    write_json(out / "summary.json", {"args": asdict(settings), "train_seed": TRAIN_SEED,
                                      "test_mode": f"exact_all_{2 ** (settings.L - 1)}",
                                      "metrics_path": str(metrics_path)})
    print(f"wrote {shown(out)}", flush=True)
    return 0


# ===========================================================================
# Fig. 10, Sec. 5.2: the learning-rate intervention on nested training sets
# ===========================================================================

@dataclass(frozen=True)
class InterventionConfig:
    """The settings of one call, stored in every result file and in manifest.json."""
    L: int = 12
    test_size: int = 512
    batch_size: int = BATCH
    steps: int = 2400
    eval_every: int = 100
    eval_batch_size: int = 512
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    lr_schedule: str = "constant"
    min_lr_factor: float = 0.02
    decay_trigger_acc: float = 0.80
    device: str = "cpu"
    threads: int = 1
    output_dir: str = ""
    save_models: bool = False


def nested_split(L: int, test_size: int, data_seed: int):
    """The held-out set and the ordered training pool of one data seed, and their metadata.

    The complete set of 2^(L-1) inputs is permuted once with a generator seeded by data_seed;
    the first test_size inputs are held out and the rest, in permutation order, is the pool
    whose first N inputs are D_N, so the training sets are nested.
    """
    x_all, y_all = complete_set(L)
    domain_size = x_all.shape[0]
    permutation = torch.randperm(domain_size, generator=torch.Generator(device="cpu").manual_seed(data_seed))
    test_indices, train_indices = permutation[:test_size], permutation[test_size:]
    metadata = {
        "domain_size": int(domain_size),
        "test_size": int(test_size),
        "train_pool_size": int(train_indices.numel()),
        "data_seed": int(data_seed),
        "permutation_sha256": hashlib.sha256(permutation.numpy().astype(np.int32).tobytes()).hexdigest(),
        "train_test_overlap": int(torch.isin(train_indices, test_indices).sum().item()),
    }
    return (x_all[train_indices], y_all[train_indices], x_all[test_indices], y_all[test_indices],
            metadata)


def batched_scores(model, inputs, targets, batch_size: int) -> dict:
    """Loss per bit, bit and block accuracy (z > 0 predicts 1), in batches of batch_size."""
    model.eval()
    loss_sum, correct_bits, correct_blocks = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, targets.shape[0], batch_size):
            x, y = inputs[start:start + batch_size], targets[start:start + batch_size]
            logits = model(x)
            loss_sum += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            equality = (logits > 0.0) == (y > 0.5)
            correct_bits += int(equality.sum().item())
            correct_blocks += int(equality.all(dim=1).sum().item())
    return {"loss": loss_sum / targets.numel(), "bit_acc": correct_bits / targets.numel(),
            "block_acc": correct_blocks / targets.shape[0], "correct_blocks": correct_blocks,
            "num_blocks": targets.shape[0]}


def stable_crossing(history: list[dict], key: str, threshold: float) -> int | None:
    """First evaluation step after which every recorded value stays at or above threshold."""
    for index, row in enumerate(history):
        if all(float(later[key]) >= threshold for later in history[index:]):
            return int(row["step"])
    return None


def intervention_run(config: InterventionConfig, N: int, data_seed: int, model_index: int, split) -> dict:
    """Train one Transformer on D_N for config.steps updates under the configured schedule.

    Each update takes 128 indices drawn uniformly with replacement from D_N by a generator
    seeded with 200000 + 1009 d + model_index, the same stream for every N; the network is
    initialized from 100000 + 1003 d + model_index.  Training-set and held-out scores are
    recorded at step 0 and every eval_every updates.

    postgen-cosine: the learning rate eta_0 is held until held-out A_block first reaches
    decay_trigger_acc at a recorded step s_0, then follows
        eta(s) = eta_min + (eta_0 - eta_min) (1 + cos(pi (s - s_0) / (S - s_0))) / 2,
    eta_min = 0.02 eta_0, from the update after s_0 to the last update S.  Up to s_0 this arm
    and the constant arm are the same run.  The summary gives the step after which the
    training A_block stays >= 0.99 and the held-out A_block stays >= 0.80, peaks, final and
    tail (last three evaluations) accuracies, and the step at which the decay started.
    """
    x_pool, y_pool, x_test, y_test, split_metadata = split
    x_train, y_train = x_pool[:N], y_pool[:N]
    init_seed, order_seed = initialization_seed(data_seed, model_index), minibatch_seed(data_seed, model_index)
    torch.manual_seed(init_seed)
    model = ChainTransformer(L=config.L, **ARCHITECTURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    eta_min = config.lr * config.min_lr_factor
    onset_step = None
    index_generator = torch.Generator(device="cpu").manual_seed(order_seed)
    history = []
    running_loss, running_steps = 0.0, 0
    started = time.time()

    def evaluate(step: int) -> None:
        nonlocal running_loss, running_steps, onset_step
        train = batched_scores(model, x_train, y_train, config.eval_batch_size)
        test = batched_scores(model, x_test, y_test, config.eval_batch_size)
        row = {"step": step, "lr": optimizer.param_groups[0]["lr"],
               "minibatch_loss": running_loss / max(1, running_steps),
               "train_loss": train["loss"], "train_bit_acc": train["bit_acc"],
               "train_block_acc": train["block_acc"], "train_correct_blocks": train["correct_blocks"],
               "train_num_blocks": train["num_blocks"], "test_loss": test["loss"],
               "test_bit_acc": test["bit_acc"], "test_block_acc": test["block_acc"],
               "test_correct_blocks": test["correct_blocks"], "test_num_blocks": test["num_blocks"]}
        history.append(row)
        # the decay starts at the update after the evaluation that first sees the trigger
        if (config.lr_schedule == "postgen-cosine" and onset_step is None and step > 0
                and row["test_block_acc"] >= config.decay_trigger_acc):
            onset_step = step
        running_loss, running_steps = 0.0, 0
        print(f"  step={step:>5} train_block={row['train_block_acc']:.3f} "
              f"test_block={row['test_block_acc']:.3f}", flush=True)

    evaluate(0)
    model.train()
    for step in range(1, config.steps + 1):
        indices = torch.randint(N, (config.batch_size,), generator=index_generator, device="cpu")
        loss = F.binary_cross_entropy_with_logits(model(x_train[indices]), y_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if onset_step is not None and config.steps > onset_step:
            progress = (step - onset_step) / (config.steps - onset_step)
            eta = eta_min + (config.lr - eta_min) * (1 + math.cos(math.pi * min(progress, 1.0))) / 2
            for group in optimizer.param_groups:
                group["lr"] = eta
        running_loss += float(loss.item())
        running_steps += 1
        if step % config.eval_every == 0 or step == config.steps:
            evaluate(step)
            model.train()

    final = history[-1]
    tail = history[-min(3, len(history)):]
    tail_train = sum(row["train_block_acc"] for row in tail) / len(tail)
    tail_test = sum(row["test_block_acc"] for row in tail) / len(tail)
    return {
        "protocol": {**asdict(config), "architecture": "transformer",
                     "architecture_spec": dict(ARCHITECTURE), "N": N, "data_seed": data_seed,
                     "model_seed_index": model_index, "initialization_seed": init_seed,
                     "minibatch_seed": order_seed,
                     "sampling": "uniform minibatches with replacement from fixed D_N",
                     "nested_training_sets": True},
        "split": split_metadata,
        "summary": {
            "stable_train_block_0p99_step": stable_crossing(history, "train_block_acc", 0.99),
            "stable_test_block_0p80_step": stable_crossing(history, "test_block_acc", HI),
            "max_train_block_acc": max(row["train_block_acc"] for row in history),
            "max_test_block_acc": max(row["test_block_acc"] for row in history),
            "final_train_bit_acc": final["train_bit_acc"],
            "final_train_block_acc": final["train_block_acc"],
            "final_test_bit_acc": final["test_bit_acc"],
            "final_test_block_acc": final["test_block_acc"],
            "tail_train_block_acc": tail_train,
            "tail_test_block_acc": tail_test,
            "decay_onset_step": onset_step,
            "fit_success": tail_train >= 0.99,
            "generalization_success": tail_test >= 0.80,
            "fit_conditioned_generalization_success": tail_train >= 0.99 and tail_test >= 0.80,
            "elapsed_s": time.time() - started,
        },
        "history": history,
    }


@subcommand("data/learning_rate_intervention/<arm>_arm/split<d>/", "Fig. 10, Sec. 5.2",
            option("--output-dir", required=True),
            option("--lr-schedule", choices=("constant", "postgen-cosine"), default="constant"),
            option("--Ns", type=int_list, default=[512, 768, 1024]),
            option("--data-seeds", type=int_list, default=[0]),
            option("--model-seeds", type=int_list, default=[0, 1]),
            option("--steps", type=int, default=2400),
            option("--threads", type=int, default=1))
def learning_rate_intervention(args) -> int:
    """The learning-rate intervention of Fig. 10: nested D_N at L = 12, constant or decaying rate.

    For each data seed the split of nested_split (512 held out, a pool of 1536), and for each
    model seed and N one intervention_run, in process and one after the other.  The number of
    updates is the same at every N.  Writes into --output-dir

        L12_N<N>_split<d>_init<m>.json   protocol, split, history, summary (data split d,
                                         initialization m)
        summary.csv                      one row per run, written after every run
        manifest.json                    the settings of the call

    The runs of Fig. 10 are, for every data split d = 0, ..., 14 and (schedule, arm) =
    (constant, constant), (postgen-cosine, intervention),

        --Ns 512,768,1024 --data-seeds d --model-seeds 0,1 --steps 2400 --threads 1
        --lr-schedule <schedule> --output-dir data/learning_rate_intervention/<arm>_arm/split<d>

    A result file that is present is read instead of trained; with every file present nothing
    is written.
    """
    out = resolve(args.output_dir)
    names = [f"L12_N{N}_split{d}_init{m}.json"
             for d in args.data_seeds for m in args.model_seeds for N in args.Ns]
    if not args.overwrite and present(out / "summary.csv", out / "manifest.json",
                                      *[out / name for name in names]):
        print(f"present {shown(out)}")
        return 0
    chain_task_threads()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    config = InterventionConfig(steps=args.steps, lr_schedule=args.lr_schedule, threads=args.threads,
                        output_dir=args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for data_seed in args.data_seeds:
        split = nested_split(config.L, config.test_size, data_seed)
        if split[4]["train_test_overlap"] != 0:
            raise RuntimeError("training and test splits overlap")
        for model_index in args.model_seeds:
            for N in args.Ns:
                name = f"L{config.L}_N{N}_split{data_seed}_init{model_index}.json"
                print(f"N={N} data_seed={data_seed} model_seed={model_index}", flush=True)
                if (out / name).exists() and not args.overwrite:
                    result = json.loads((out / name).read_text(encoding="utf-8"))
                else:
                    result = intervention_run(config, N, data_seed, model_index, split)
                    write_json(out / name, result)
                protocol = result["protocol"]
                rows.append({"architecture": protocol["architecture"], "N": protocol["N"],
                             "data_seed": protocol["data_seed"],
                             "model_seed_index": protocol["model_seed_index"],
                             "result_path": str(Path(args.output_dir) / name),
                             **result["summary"]})
                write_rows(out / "summary.csv", rows)
    write_json(out / "manifest.json", {"scan_config": asdict(config), "Ns": args.Ns,
                                       "data_seeds": args.data_seeds,
                                       "model_seeds": args.model_seeds,
                                       "architectures": ["transformer"], "num_runs": len(rows)})
    print(f"wrote {len(rows)} runs to {shown(out)}", flush=True)
    return 0


# ===========================================================================
# Command line
# ===========================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="list the subcommands with the data/ directory each writes")
    commands = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    for name, (function, writes, paper, options) in COMMANDS.items():
        summary = function.__doc__.strip().splitlines()[0]
        sub = commands.add_parser(name, help=summary, description=function.__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
        for flags, kwargs in options:
            sub.add_argument(*flags, **kwargs)
        sub.add_argument("--overwrite", action="store_true",
                         help="train again the runs whose output is present")
    args = parser.parse_args(argv)
    if args.list:
        for name, (_, writes, paper, _) in COMMANDS.items():
            print(f"{name:40s} {paper}\n{'':40s} writes {writes}")
        return 0
    if not args.command:
        parser.error("name a subcommand, or use --list")
    return COMMANDS[args.command][0](args) or 0


if __name__ == "__main__":
    sys.exit(main())
