# Learning to Target with Network Interference

This repository contains the code for reproducing the numerical experiments in:

> **Learning to target with network interference**
> Xiaomeng Wang, Hamsa Bastani, Osbert Bastani, and Zhimei Ren.
> 2026.

## Overview

We study adaptive targeting under network interference in a bandit setting where treatments applied to one individual may affect others through spillover effects. We develop algorithms for three regimes with different levels of network information. Algorithm numbering and regret bounds below follow Table 1 of the paper; `ρ_j` denotes the column-support size of the network treatment effect matrix, and under row sparsity `Σ_j ρ_j ≤ ds`.

| Algorithm | Class | Network Info | Regret (upper) |
|---|---|---|---|
| Baseline (Section 3)  | `BaselineBandit` | None (aggregated reward)    | Ω(d^{3/2} sqrt(T) ∧ dT) lower bound |
| NSE-FS (Alg. 1)       | `NSEFSBandit`    | Full support known          | Õ(sqrt(T) Σ_j sqrt(ρ_j)) ≤ Õ(d sqrt(sT)) |
| NSE (Alg. 2)          | `NSEBandit`      | Column support sizes known  | Õ(sqrt(T) Σ_j ρ_j) ≤ Õ(ds sqrt(T)) |
| NETC (Alg. 3)         | `NETCBandit`     | No support information      | Õ(d (sT)^{2/3}) |

NSE-FS is minimax-optimal (matches the lower bound up to log factors); NSE is near-optimal up to a √ρ_j factor; NETC remains linear in `d` despite having no support information.

## Repository Structure

```
.
├── algorithms.py           # All algorithm implementations and data generation
├── run_main.py             # Main experiment (Figure 1): NETC / NSE / NSE-FS, d=100
├── run_scaling_d.py        # Scaling experiment (Figure 2): regret vs. d (100..900)
├── run_sweep_signal.py     # Signal strength sweep (Appendix D)
├── run_sweep_sparsity.py   # Sparsity sweep (Appendix D)
├── run_village.py          # Semi-synthetic village experiment (Figure 3, Table 2)
└── README.md
```

Each `run_*.py` script writes JSON results to a dedicated `output_*/` directory (created on first use). Result files are keyed by seed / task ID, and existing files are skipped on re-run, so jobs are safe to restart.

## Algorithm Configurations

The publication runs use the following pinned hyperparameters across all experiments:

- **NSE**: one-hot estimator, `tau_constant=0.2`
- **NSE-FS**: ridge estimator, `alpha=0.05`, `estimation_alpha=1.0`
- **NETC**: `lambda_explore=0.035`, `T1=200` exploration rounds (in `run_village.py`, `T1 = ceil(2 s log(2dT))`)
- **Baseline**: `lambda_confidence=1`, `reg_param=0`

In every `run_*.py` script except `run_scaling_d.py`, the `BaselineBandit` call is intentionally commented out: Baseline results are re-used from earlier simulation outputs and merged at plotting time. To regenerate Baseline from scratch, uncomment the `baseline = BaselineBandit(...)` block and the corresponding entry in the returned `results` dict. Baseline is excluded entirely from `run_scaling_d.py` because of its computational cost at large `d`.

## Requirements

- Python >= 3.10
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib
- [Gurobi](https://www.gurobi.com/) (with a valid license)
  - `gurobipy` Python package

Install dependencies:
```bash
pip install numpy scipy scikit-learn pandas matplotlib gurobipy
```

**Note:** Gurobi requires a license. Free academic licenses are available at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Usage

All scripts support two execution modes:
- **Cluster (SGE array job):** set `SGE_TASK_ID` in the environment — the script runs exactly one task and exits.
- **Local batch:** pass `--task-start` / `--task-end` (or `--seed-start` / `--seed-end` for `run_main.py`) to loop over a range sequentially in one process.

If neither is provided, the script defaults to task ID 1.

### Main experiment (Figure 1)

Compares NETC, NSE, and NSE-FS on a simulated network with `d=100`, `s=20`, `T=20000`, `signal_strength=0.1`.

```bash
python run_main.py                                # seed=1
python run_main.py --seed-start 1 --seed-end 100  # local batch over seeds 1..100
SGE_TASK_ID=42 python run_main.py                 # cluster: seed=42
```

Results are saved to `output_main/seed_<seed>.json`.

### Scaling experiment (Figure 2)

Evaluates per-individual regret as the network size `d` varies over `[100, 200, ..., 900]` (9 values × 100 seeds = **900 tasks**). Task IDs are decoded as `d_index = (task_id-1) // 100`, `seed_within_d = ((task_id-1) % 100) + 1`.

```bash
SGE_TASK_ID=1 python run_scaling_d.py     # d=100, seed_within_d=1
SGE_TASK_ID=150 python run_scaling_d.py   # d=200, seed_within_d=50
python run_scaling_d.py --task-start 1 --task-end 900   # full local sweep
```

Results are saved to `output_scaling_d/d_<d>_seed_<s>.json`.

### Signal strength and sparsity sweeps (Appendix D)

Each task ID maps to a (parameter value, within-bucket seed) pair via 100-seed buckets.

```bash
# Signal strength sweep: 6 values [0.01, 0.05, 0.1, 0.15, 0.2, 0.5] x 100 seeds = 600 tasks
SGE_TASK_ID=1 python run_sweep_signal.py        # signal=0.01, seed=1
SGE_TASK_ID=101 python run_sweep_signal.py      # signal=0.05, seed=1

# Sparsity sweep: 6 values [5, 10, 15, 20, 25, 50] x 100 seeds = 600 tasks
SGE_TASK_ID=1 python run_sweep_sparsity.py      # s=5, seed=1
SGE_TASK_ID=101 python run_sweep_sparsity.py    # s=10, seed=1
```

Results are saved to `output_sweep_signal/` and `output_sweep_sparsity/` respectively.

### Semi-synthetic village experiment (Figure 3, Table 2)

Uses adjacency matrices from the Indian village dataset (Banerjee et al., 2013). Each task ID corresponds to one of **74 villages** (numbers 1–76 with 13 and 22 excluded), and each village is run `NUM_RUNS=5` times with different seeds.

```bash
SGE_TASK_ID=1 python run_village.py                       # village 1, runs 1..5
SGE_TASK_ID=50 python run_village.py                      # village 52, runs 1..5
python run_village.py --task-start 1 --task-end 74        # full local sweep
```

Results are saved to `output_village/village_<v>_run_<r>_seed_<s>.json`.

**Data location.** `algorithms.generate_village_network` resolves the dataset path relative to `algorithms.py` as `<parent_of_repo>/datav4.0/Data/1. Network Data/Adjacency Matrices/`, i.e. **one level above the repository directory**. Either place the unzipped `datav4.0/` folder next to (not inside) this repo, or symlink it accordingly.

The village dataset is available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/21538

## Citation

```bibtex
@article{wang2026learning,
  title={Learning to target with network interference},
  author={Wang, Xiaomeng and Bastani, Hamsa and Bastani, Osbert and Ren, Zhimei},
  year={2026}
}
```

## License

This project is for academic and research purposes.
