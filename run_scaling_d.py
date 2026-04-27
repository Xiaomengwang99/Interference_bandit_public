"""
Scaling experiment: per-individual regret as a function of network size d.

Corresponds to Figure 2 in the paper. Sweeps d from 100 to 900 and runs
NETC, NSE, and NSE-FS. Baseline is excluded due to computational cost at
large d.

Each SGE array task ID maps to one (d, seed) pair, indexed as
    task_id = (d_index * SEEDS_PER_D) + seed_within_d
where d_index enumerates D_VALUES (0-based).

For cluster runs, this means:
    task 1..100   -> d=100, seeds 1..100
    task 101..200 -> d=200, seeds 1..100
    ...
    task 801..900 -> d=900, seeds 1..100

NOTE ON REUSED BASELINE RESULTS:
    Baseline is not part of this sweep (never was -- excluded at large d).
    Reuse old NETC/NSE/NSE-FS results is NOT possible: algorithms diverged
    between old and new code; all three must be re-run here. The final
    algorithm configurations are:
      - NSE:    one-hot estimator, tau_constant=0.2
      - NSE-FS: ridge, alpha=0.05, estimation_alpha=1.0
      - NETC:   lambda_explore=0.035

Usage
-----
    SGE_TASK_ID=1 python run_scaling_d.py    # d=100, seed=1
    SGE_TASK_ID=150 python run_scaling_d.py  # d=200, seed=50
"""

import json
import os
from pathlib import Path

import numpy as np

from algorithms import (
    generate_network,
    NETCBandit,
    NSEBandit,
    NSEFSBandit,
    BaselineBandit,  # imported for documentation; not called in this experiment
)

# --- Default parameters (fixed across the sweep) ---
DEFAULT_S = 20
DEFAULT_TAU = 20000
DEFAULT_B = 100
DEFAULT_SIGNAL_STRENGTH = 0.1
DEFAULT_T1 = 200
NETC_LAMBDA = 0.035
NSE_TAU_CONSTANT = 0.2
NSEFS_ALPHA = 0.05
NSEFS_EST_ALPHA = 1.0

# --- Sweep configuration ---
D_VALUES = [100, 200, 300, 400, 500, 600, 700, 800, 900]
SEEDS_PER_D = 100
OUTPUT_DIR = Path(__file__).resolve().parent / "output_scaling_d"


def _get_job_id():
    job_id = os.getenv("SGE_TASK_ID")
    if job_id is None:
        return 1
    try:
        return int(job_id)
    except ValueError:
        return 1


def _decode_task_id(task_id):
    """Map 1-indexed task id -> (d, seed_within_d)."""
    task_id = max(1, task_id)
    d_index = (task_id - 1) // SEEDS_PER_D
    seed_within_d = ((task_id - 1) % SEEDS_PER_D) + 1
    d_index = min(d_index, len(D_VALUES) - 1)
    return D_VALUES[d_index], seed_within_d


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _run_for_d(seed, d):
    """Run NETC, NSE, and NSE-FS for a given seed and population size d."""
    s = DEFAULT_S
    tau = DEFAULT_TAU
    signal_strength = DEFAULT_SIGNAL_STRENGTH
    combined_seed = seed + d * 1000  # matches old Simulation_new_per_individual_vs_d.py

    X = generate_network(combined_seed, d, s, signal_strength)

    # --- Baseline (Algorithm 3) ---
    # Not part of this experiment due to computational cost at large d.
    # baseline = BaselineBandit(
    #     X, tau=tau, b=DEFAULT_B, lambda_confidence=1, reg_param=0,
    #     sparsity=s, R_max=s*signal_strength,
    # )
    # baseline_results = baseline.run()

    netc = NETCBandit(X, tau=tau, b=DEFAULT_B, exploration_rounds=DEFAULT_T1,
                      lambda_explore=NETC_LAMBDA)
    netc_results = netc.run()

    nse = NSEBandit(X, tau=tau, tau_constant=NSE_TAU_CONSTANT,
                    estimation_method="onehot")
    nse_results = nse.run()

    nsefs = NSEFSBandit(X, tau=tau, alpha=NSEFS_ALPHA,
                        estimation_alpha=NSEFS_EST_ALPHA,
                        estimation_method="ridge")
    nsefs_results = nsefs.run()

    return {
        "combined_seed": combined_seed,
        "parameters": {
            "d": d, "s": s, "tau": tau, "b": DEFAULT_B,
            "signal_strength": signal_strength, "T1": DEFAULT_T1,
        },
        "results": {
            # "baseline": _json_ready(baseline_results),  # excluded
            "netc": _json_ready(netc_results),
            "nse": _json_ready(nse_results),
            "nsefs": _json_ready(nsefs_results),
        },
    }


def _run_one_task(task_id):
    d, seed_within_d = _decode_task_id(task_id)
    out_path = OUTPUT_DIR / f"d_{d}_seed_{seed_within_d}.json"
    if out_path.exists():
        print(f"[task {task_id}] d={d} seed={seed_within_d} already exists, skipping.")
        return
    print(f"[task {task_id}] d={d}, seed_within_d={seed_within_d}")
    payload = _run_for_d(seed_within_d, d)
    payload["task_id"] = task_id
    payload["d"] = d
    payload["seed_within_d"] = seed_within_d
    with out_path.open("w") as f:
        json.dump(payload, f)
    print(f"Saved to {out_path}.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-start", type=int, default=None)
    parser.add_argument("--task-end", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.task_start is not None and args.task_end is not None:
        task_ids = range(args.task_start, args.task_end + 1)
    else:
        task_ids = [_get_job_id()]

    for task_id in task_ids:
        _run_one_task(task_id)


if __name__ == "__main__":
    main()
