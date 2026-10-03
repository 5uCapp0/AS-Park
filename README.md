# AS-Park

AS-Park is a reproduction package for **closed-form symmetry-injection (lambda-star) scheduling
for multi-agent cooperative parking**: it derives a closed-form strength lambda* for injecting
permutation equivariance into parking-dispatch policies and validates it across offline lot
benchmarks, strictly-online dispatchers, and MAPPO baselines.

This repository contains the reproducible subset: experiment code under `code/` and the paper's
result tables/figures data under `data/`.

## Repository layout

```
AS-Park-repro/
├── code/                 # all experiment / analysis / plotting scripts (Python 3)
│   ├── parking_sim.py    # core grid parking simulation + lambda* derivation demos (E1/E2/E3)
│   ├── common.py         # shared torch utilities
│   ├── l1_robustness.py  # L1: robustness of lambda* under non-Gaussian noise
│   ├── l2_common.py / l2_calibrate.py / l2_stats.py / l2_visualize.py
│   ├── l2_benchmark.py   # L2 offline benchmark
│   ├── l2_a1.py ... l2_a5.py   # L2 ablations (lambda / capacity / estimator / dual-perm / dispatcher)
│   ├── l2_fig4.py        # Fig.4 alignment experiment
│   ├── l3_online_dispatcher.py / l3_online_2d.py   # L3 strictly-online dispatchers
│   ├── l3_mappo_baseline.py / l3_mappo_typeaware.py / l3_mappo_scale.py
│   ├── l3_sample_efficiency.py / l3_generalization.py / l3_multiinit.py
│   ├── l3_scale.py / l3_scale_stable.py / l3_sigma_sensitivity.py / l3_typeaware_arm.py
│   ├── sched_benchmark_multiseed.py
│   ├── run_all_l2.py / run_l3_main_10seed.py / run_seed_job.py / run_sens_cell.py / launch_jobs.py
│   ├── merge_jobs.py / merge_main10.py / merge_scale.py / merge_l3_dispatcher.py
│   ├── rebuild_dispatcher_result.py / plot_dispatcher_result.py / plot_l3_5seed.py
│   └── ...               # recovery / probe / phase-diagram helpers
├── data/                 # result JSONs (and a few CSVs) produced by the experiments
├── README.md
├── LICENSE               # MIT
└── .gitignore
```

## Dependencies and hardware

- Python 3.9+ (developed against CPython; uses only the standard library plus scientific stack).
- Required packages: `torch`, `numpy`, `matplotlib`.
  Install with: `pip install torch numpy matplotlib`
- The offline (L1/L2) and online-dispatcher experiments run comfortably on a CPU for small
  fleet sizes; the MAPPO baselines (`l3_mappo_*.py`) and large-fleet scans benefit from a CUDA
  GPU. A single-GPU workstation is sufficient to reproduce every figure.

## How to run

All scripts are run from the repository root. Each script writes its output JSON into `data/`
(or the current directory) and prints a short progress summary.

- Core derivation demos (E1 lambda sweep, E2 heterogeneity sweep, E3 data efficiency):
  ```
  python code/parking_sim.py
  ```
- L1 robustness:
  ```
  python code/l1_robustness.py
  ```
- L2 offline benchmark and the five ablations (A1-A5), end to end:
  ```
  python code/run_all_l2.py
  ```
  Individual experiments: `python code/l2_benchmark.py`, `python code/l2_a1.py`, ...,
  `python code/l2_a5.py`, `python code/l2_fig4.py`.
- L3 strictly-online dispatcher (multi-seed):
  ```
  python code/run_l3_main_10seed.py
  ```
  Single seed: `python code/run_seed_job.py`; the 2D-grid variant:
  `python code/l3_online_2d.py`.
- MAPPO baselines:
  ```
  python code/l3_mappo_baseline.py
  python code/l3_mappo_typeaware.py
  python code/l3_mappo_scale.py
  ```
- Sensitivity / generalization / scale studies:
  ```
  python code/l3_sigma_sensitivity.py        # sigma^2 scan, one cell: run_sens_cell.py
  python code/l3_generalization.py           # fleet-size N generalization
  python code/l3_scale.py / l3_scale_stable.py   # large-fleet scale
  python code/l3_sample_efficiency.py         # epoch vs gap learning curves
  ```
- Scheduler multi-seed benchmark:
  ```
  python code/sched_benchmark_multiseed.py
  ```
- Merging per-seed outputs and producing paper figures:
  ```
  python code/merge_jobs.py
  python code/merge_main10.py
  python code/merge_scale.py
  python code/merge_l3_dispatcher.py
  python code/plot_dispatcher_result.py
  python code/plot_l3_5seed.py
  ```

## Result data format

Files in `data/` are the artifacts produced by the runs above.

- `*.json` files are structured result tables. Each JSON is a dict (or list of dicts) with keys
  such as the fleet size `N`, the heterogeneity level `p` / defect `epsilon^2`, the closed-form
  strength `lambda*`, the measured optimization `gap`, the random `seed`, and per-seed metric
  arrays. Numeric strings that look like 11-digit numbers are random seeds, not identifiers;
  they are part of the data and must not be edited.
- `l2_a*.csv` and `l2_fig4.csv` are long-format tables backing the L2 ablation figures.
- `l3_mappo_*` files hold the MAPPO baseline learning curves; `scale_*` files hold the
  large-fleet sweep; `sens_*.json` holds the sigma^2 sensitivity grid.

## Notes

- This is a cleaned, English-only reproduction snapshot. Docstrings are short English summaries;
  experiment logic is unchanged from the paper's code.
- If you reproduce the numbers, expect small run-to-run differences in the last digits due to
  nondeterministic CUDA kernels; the reported trends and significance tests are stable across
  seeds.
