# Replica Fragmentation and Glassy Dynamics in Parity Learning: source code

The training protocols, the analyses and the figures of the paper.

The repository holds code only. The analyses read trained networks and training
records from `data/`, which is distributed as `data.tar.gz` (1.2 GB), attached to
the release [`data`](https://github.com/HanMaPhy/two_learning_glasses/releases/tag/data) of this repository. Unpack it at the root of the
repository:

```sh
shasum -a 256 data.tar.gz     # prints eeb951279be4e28a332af39dbaddcd642956b2b46397a06f420c0fe98a616eee
tar -xzf data.tar.gz          # creates data/
```

Python 3.9 or later, with `requirements.txt` installed.

## Files

```
common.py             the task, the network, the replica sets and the overlap matrices shared by the files below
analysis.py           the analyses: data/ -> results/
figures.py            the fifteen figures: data/ and results/ -> figure/
train.py              the training protocols: -> data/
replica_overlaps.py   the training procedure and the replica observables in one self-contained file
```

Each file opens with a docstring that states the physics of what it computes and maps
paper locations to functions; `common.py` defines the task, the endpoint classes and the
overlaps (m, Q^raw, Q^c, W, R^W) once for all of them. `data/`, `results/` and `figure/`
are not tracked. Every file finds the repository root from its own location, so it can
be started from any directory.

To see what is measured, start with `replica_overlaps.py`. It depends on nothing else
here and reads no data:

```sh
python3 replica_overlaps.py --check   # self-checks only, about a second
python3 replica_overlaps.py           # self-checks, then a 16-replica ensemble at (L, N) = (12, 1280), a few minutes
```

## Running

```sh
python3 analysis.py --list                   # every analysis: name, paper location, what it writes
python3 analysis.py --all                    # all tables in results/, from data/   (about 80 minutes)
python3 analysis.py replica_set_geometry     # one analysis
python3 figures.py --all                     # the fifteen figures                   (about 2 minutes)
python3 figures.py fig7_retreat_residual_tree
```

`--all` runs every item in dependency order, each in its own process. Nothing in
`analysis.py` or `figures.py` trains. Figures 5, 6, 7, 8, 9 and A1 are drawn from
tables in `results/`, so they need the analyses to have run; the other nine figures
read `data/` directly.

Both files set the BLAS thread variables to 1 unless they are set already. Four items
set the thread count of torch themselves: `fixed_epoch_comparison` and
`retreat_event_contrasts` (two threads), `final_ffn_peak_to_retreat` (three) and the
figure `figC3_small_data_diagnosis` (four).

## Figures

Each figure is written to `figure/<name>.pdf` by the function `<name>` of `figures.py`.

| Figure | Name | Reads |
|---|---|---|
| 1 | `fig1_generalization_crossover` | `data/independent_training_sets/` |
| 2 | `fig2_shared_training_set_diagnostics` | `data/trajectory_ensembles/L12_N1280_overlaps/` |
| 3 | `fig3_nishimori_gap_dynamics` | `data/trajectory_ensembles/L12_N1280_overlaps/`, `data/trajectory_ensembles/L12_N64_overlaps/` |
| 4 | `fig4_position_resolved_retreat_recovery` | `data/trajectory_ensembles/L12_N1280_soft_outputs/` |
| 5 | `fig5_tail_observables` | `results/tail_observables.csv` |
| 6 | `fig6_raw_overlaps_and_gaps` | `results/replica_set_geometry.json` |
| 7 | `fig7_retreat_residual_tree` | `results/retreat_matrices/` |
| 8 | `fig8_fragmentation_residual_organization` | `results/replica_set_geometry.json`, `results/normalized_residuals.npz` |
| 9 | `fig9_two_planes_common_tail` | the same two |
| 10 | `fig10_learning_rate_intervention` | `data/learning_rate_intervention/` |
| A1 | `figA1_final_layer_ffn` | `results/final_ffn_peak_to_retreat.json` |
| B1 | `figB1_bit_and_block_accuracy` | `data/independent_training_sets/` |
| C1, C2 | `figC1_raw_fragmentation_frontier_order` (writes C1 and `figC2_normalized_residual_similarity.pdf`) | `data/large_data_ensembles/L12_N1280/` |
| C3 | `figC3_small_data_diagnosis` | `data/small_data_ensembles/L12_N64/` |

Four figure functions also write tables: `fig1_generalization_crossover` the logistic
fits of Sec. 3.1 (`results/generalization_crossover_fits.csv`),
`figB1_bit_and_block_accuracy` the summary of App. B (`results/block_vs_bit_summary.csv`),
`figC1_raw_fragmentation_frontier_order` the gap decomposition, bootstrap intervals and
position profiles of Sec. 4.2 and App. C (`results/retreat_overlap_structure/`), and
`figC3_small_data_diagnosis` Table C2 and the window averages of App. C.4
(`results/memorization_heldout_per_position.csv`, `memorization_heldout_summary.json`).

## Analyses

Each function of `analysis.py` writes the file or directory of the same name under
`results/`.

| Function | Computes | Paper |
|---|---|---|
| `replica_set_geometry` | the fourteen replica sets: gaps of Q^c and W, overlaps, tree statistics at equal replica count | Table 3, Figs. 6, 8, 9, Secs. 4.2, 4.3 |
| `normalized_residuals` | normalized residual matrices R^W of the fourteen sets, all positions and common tail | Figs. 8, 9, Sec. 4.3 |
| `retreat_matrices_l12_first400`, `retreat_matrices_norm_filter`, `retreat_matrices_l16_n2048` | Q^c, C^prof and W of the two retreat ensembles drawn as trees | Fig. 7 |
| `tree_permutation_references` | tree statistics of the (12,1280) retreat ensemble against two permutation references | Sec. 4.2, first control |
| `retreat_definition_controls` | the retreat set under thresholds (0.9, 0.6) and at observation epoch 100 | Sec. 4.2, second and third control |
| `trajectory_ensemble_retreat` | reach and retreat counts of the 16-replica trajectory ensemble; two-time and cross-replica tail overlap in retreat | Sec. 3.4; Sec. 4.2, fourth control |
| `evaluation_outside_training_set` | the Sec. 4 observables on the evaluation inputs that are not in the training set | Sec. 4.2 |
| `truth_scalar_correlation` | how much of the raw pair pattern the scalar truth alignment explains | Sec. 4.2 |
| `k4_cut_equal_count` | four-cluster cut of the recovery tree and of the retreat tree sampled to the recovery count | Sec. 4.2 |
| `gf2_identifiability` | GF(2) rank of the nine training sets and the block accuracy of the eliminated rule | Sec. 4.2 |
| `retreat_obtuse_pair_fraction`, `split_half_obtuse` | fraction of replica pairs with R^W < 0, and its split-half control | Sec. 4.3 |
| `endpoint_prevalence` | the five endpoint classes over every trained replica | Table C1 |
| `tail_geometry` | tail-window overlap geometry of the memorization ensembles | Table C3 |
| `tail_observables` | tail observables of the L=12 scan over training-set size | Fig. 5, Sec. 3.5 |
| `nishimori_gap_small_data` | persistence of the Nishimori gap in the two N=64 runs of Fig. 3 | Sec. 3.3 |
| `train_heldout_crossing_epochs` | lag between the training and the held-out crossing of 0.8 in the four scans of Fig. 1 | Secs. 1, 3 |
| `intervention_pair_outcomes` | paired retreat outcomes of the learning-rate intervention | Sec. 5.2 |
| `block_vs_bit_centered_correlation` | block accuracy against the equal-position estimate, pooled and centered at fixed N and epoch | App. B |
| `fixed_epoch_comparison` | gradient norm and Hessian extrema of sixteen replicas at a fixed epoch, on the held-out set | App. A.1, A.2 |
| `retreat_event_contrasts` | event-paired gradient and curvature contrasts at retreat, on a fixed set of 1024 configurations | App. A.2 |
| `final_ffn_peak_to_retreat` | peak-to-retreat analysis of the last feed-forward layer on the complete input space | Fig. A1, App. A.3 |

The three longest are `retreat_event_contrasts` (about 25 minutes),
`final_ffn_peak_to_retreat` (about 16) and `replica_set_geometry` (about 11).

## Data and the training protocols that produce it

`data.tar.gz` holds the files the analyses read, and the configurations under
`fresh_minibatch_runs/` that two training protocols take:

```
data/independent_training_sets/size_scan/<network>_L<L>/L<L>_N<N>_seed<k>/   Fig. 1, App. B: every run draws its own fixed training set (metrics.json)
data/independent_training_sets/representative_runs/<network>_L12_N<N>/seed<k>/   Fig. 1: 600 epochs, the run the paper shows (metrics.json)
data/fresh_minibatch_runs/<network>_L12/                    configurations of runs on fresh minibatches (metrics.json only), read by train.py
data/trajectory_ensembles/{L12_N1280_overlaps, L12_N1280_soft_outputs, L12_N64_overlaps}/   Table 2, Figs. 2-4
data/small_data_ensembles/L<L>_N<N>/seed<k>/               Table 3 (small data), Fig. 5, Tables C2, C3, Fig. C3
data/large_data_ensembles/L<L>_N<N>/seeds<aaaa>-<bbbb>.npz Table 3 (large data), Figs. 6-9, C1, C2, Table C1
data/learning_rate_intervention/{constant_arm, intervention_arm}/split<d>/   Sec. 5.2, Fig. 10
data/checkpointed_trajectories/L12_N1280/seed<k>/epoch<e>.pt   App. A.2, A.3, Fig. A1: weights at epochs 20 to 220
data/fixed_epoch_replicas/L12_N1280_epoch150/seed<k>/      App. A.1, A.2
data/retreat_events/events.csv                             App. A.2: the retreat events of the checkpointed trajectories
```

`python3 train.py --list` shows every protocol with what it writes, and
`python3 train.py PROTOCOL --help` its options. The defaults are the settings that
produced `data/`; the replicas of `small_data_ensembles/` and `large_data_ensembles/`
were trained with one to four intra-op threads, and `--threads` sets that count. All
commands below are `python3 train.py ...`.

| Under `data/` | Command |
|---|---|
| `independent_training_sets/size_scan/transformer_L12/` | `size_scan_transformer_l12`, which takes the configuration of each run from `fresh_minibatch_runs/transformer_L12/L12_N*_seed*/metrics.json` |
| `independent_training_sets/size_scan/transformer_L{16,20}/` | `size_scan_transformer_l16_l20` |
| `independent_training_sets/size_scan/cnn_L12/` | `independent_set_cnn --out-dir <run> --train-samples <N> --seed <s> --fixed-train-set 1`, once per run |
| `independent_training_sets/representative_runs/transformer_L12_N1280/` | `representative_runs_transformer` (the paper uses seed 0) |
| `independent_training_sets/representative_runs/cnn_L12_N2048/` | `representative_runs_cnn` (the paper uses seed 3), with the configuration of `fresh_minibatch_runs/cnn_L12/L12_N2048_seed0/` |
| `fresh_minibatch_runs/transformer_L12/`, `fresh_minibatch_runs/cnn_L12/` | `independent_set_transformer --out-dir <run> --L 12 --train-samples <N> --seed <s> --fixed-train-set 0` and `independent_set_cnn --out-dir <run> --train-samples 2048 --seed 0` |
| `trajectory_ensembles/L12_N1280_overlaps/` | `overlap_trajectories_l12_n1280` |
| `trajectory_ensembles/L12_N1280_soft_outputs/` | `soft_output_trajectories_l12_n1280` |
| `trajectory_ensembles/L12_N64_overlaps/` | `overlap_trajectories_l12_n64` |
| `small_data_ensembles/` | `small_data_ensembles`, one `small_data_replica` per network |
| `large_data_ensembles/` | `large_data_ensembles`, one `large_data_replicas` call per part file `seeds<a>-<b>.npz` |
| `checkpointed_trajectories/L12_N1280/` | `checkpointed_trajectories` for seeds 0 to 5, then `--seeds 6,7,8`, `--seeds 9,...,15`, `--seeds 16,...,22` and `--seeds 23,...,29` |
| `fixed_epoch_replicas/L12_N1280_epoch150/` | `fixed_epoch_replicas` |
| `learning_rate_intervention/<arm>_arm/split<d>/` | `learning_rate_intervention --Ns 512,768,1024 --data-seeds <d> --model-seeds 0,1 --steps 2400 --threads 1 --lr-schedule <schedule> --output-dir data/learning_rate_intervention/<arm>_arm/split<d>`, for d = 0 to 14, with `constant` for the constant arm and `postgen-cosine` for the intervention arm |

Every protocol skips a run whose output is present and trains it again only with
`--overwrite`. The package leaves out the outputs no analysis reads, among them the
weights of the size scan and of the representative runs, the confidence tables of the
representative runs, nine weight files of `small_data_ensembles/L24_N192/`, and the
training logs, manifests and summaries. Train into a checkout that holds only
`data/fresh_minibatch_runs/` from the package; over the unpacked package a protocol would
train the runs with left-out files again and write over files the analyses read.
`retreat_events/events.csv` lists the retreat events of the checkpointed trajectories;
no command in this repository writes it.

A call of `checkpointed_trajectories` writes the per-epoch log and the manifest of the
seeds it trains, named after them (`trajectory_seeds09-15.csv`, `manifest_seeds09-15.json`);
no analysis reads them.

## What reproduces exactly

Every analysis and every figure. Run on the same data, `analysis.py --all` and
`figures.py --all` give the same tables byte for byte and the same figures.

The training values of an individual network depend on the machine, the thread count and
the version of torch.

In `independent_set_transformer` the model is built before the random generator is
seeded. For the runs it trains (the fixed-training-set scans of Fig. 1), the seed
therefore fixes the training set and the minibatch order, and the initial weights
differ from one execution to the next.
