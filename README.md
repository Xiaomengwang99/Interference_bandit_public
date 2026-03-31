# Learning to Target with Network Interference

This repository contains the code for reproducing the numerical experiments in:

> **Learning to target with network interference**
> Xiaomeng Wang, Hamsa Bastani, Osbert Bastani, and Zhimei Ren.
> 2026.

## Overview

We study adaptive targeting under network interference in a bandit setting where treatments applied to one individual may affect others through spillover effects. We develop algorithms for three regimes with different levels of network information:

| Algorithm | Class | Network Info | Regret |
|---|---|---|---|
| Baseline (Alg. 3) | `BaselineBandit` | None (aggregated reward) | O(d sqrt(dT)) |
| NETC (Alg. 2) | `NETCBandit` | No support information | O(d (sT)^{2/3}) |
| NSE (Alg. 1) | `NSEBandit` | Column support sizes known | O(ds sqrt(T)) |
| NSE-FS (Alg. 4) | `NSEFSBandit` | Full support known | O(ds sqrt(T)) |

## Repository Structure

```
.
├── algorithms.py           # All algorithm implementations and data generation
├── run_main.py             # Main experiment (Figure 1): all algorithms, d=100
├── run_scaling_d.py        # Scaling experiment (Figure 2): regret vs. d
├── run_sweep_signal.py     # Signal strength sweep (Appendix D)
├── run_sweep_sparsity.py   # Sparsity sweep (Appendix D)
├── run_village.py          # Semi-synthetic village experiment (Figure 3)
└── README.md
```

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

### Main experiment (Figure 1)

Compares all four algorithms on a simulated network with d=100, s=20, T=20000.

```bash
python run_main.py
```

Results are saved to `output_main/`.

### Scaling experiment (Figure 2)

Evaluates per-individual regret as network size d varies from 100 to 1000.

```bash
python run_scaling_d.py
```

### Signal strength and sparsity sweeps (Appendix D)

These are designed for cluster array jobs (SGE). Each task ID maps to a parameter bucket:

```bash
# Signal strength sweep: 6 values x 100 seeds = 600 tasks
SGE_TASK_ID=1 python run_sweep_signal.py

# Sparsity sweep: 6 values x 100 seeds = 600 tasks
SGE_TASK_ID=1 python run_sweep_sparsity.py
```

### Semi-synthetic village experiment (Figure 3)

Uses adjacency matrices from the Indian village dataset (Banerjee et al., 2013). Requires the village network data in `datav4.0/Data/1. Network Data/Adjacency Matrices/`.

```bash
# Each task ID corresponds to one of 75 villages
SGE_TASK_ID=1 python run_village.py
```

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
