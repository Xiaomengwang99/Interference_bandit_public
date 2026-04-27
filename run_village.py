"""
Semi-synthetic experiment using Indian village social network data.

Corresponds to Figure 3 and Table 2 in the paper. Uses household-level
adjacency matrices from Banerjee et al. (2013) as the network structure,
with randomly generated heterogeneous treatment effects.

Each SGE array task ID maps to one of 75 villages. For each village, the
experiment is repeated NUM_RUNS times with different random seeds.

NOTE ON REUSED BASELINE RESULTS:
    The Baseline algorithm call below is intentionally left in source but
    commented out. For publication runs, Baseline is reused from previous
    simulations (see output_village_new_algos2/). Only NETC, NSE, and
    NSE-FS are re-run with the final algorithm configurations:
      - NSE:    one-hot estimator, tau_constant=0.2
      - NSE-FS: ridge, alpha=0.05, estimation_alpha=1.0
      - NETC:   lambda_explore=0.035

Usage
-----
    SGE_TASK_ID=1 python run_village.py    # village 1
    SGE_TASK_ID=50 python run_village.py   # village 52
"""

import json
import math
import os
from pathlib import Path

import numpy as np

from algorithms import (
    generate_village_network,
    VILLAGE_NUMBERS,
    NETCBandit,
    NSEBandit,
    NSEFSBandit,
    BaselineBandit,
)

# --- Experiment parameters ---
SIGNAL_STRENGTH = 0.1
BASE_SPARSITY = 10
TAU = 20000
BASELINE_LAMBDA = 1
NETC_LAMBDA = 0.035
NSE_TAU_CONSTANT = 0.2
NSEFS_ALPHA = 0.05
NSEFS_EST_ALPHA = 1.0
NUM_RUNS = 5
RUN_SEED_OFFSET = 10000

OUTPUT_DIR = Path(__file__).resolve().parent / "output_village"


def _get_job_id():
    job_id = os.getenv("SGE_TASK_ID")
    if job_id is None:
        return 1
    try:
        return int(job_id)
    except ValueError:
        return 1


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


def _get_village_number(job_id):
    """Map job ID (1-indexed) to a valid village number."""
    if job_id < 1 or job_id > len(VILLAGE_NUMBERS):
        raise ValueError(f"job_id {job_id} out of range 1-{len(VILLAGE_NUMBERS)}")
    return int(VILLAGE_NUMBERS[job_id - 1])


def _run_single(X, village_number, run_index, run_seed):
    """Run all algorithms on one village with one random seed."""
    d = X.shape[0]
    s = min(BASE_SPARSITY, d)
    b = d
    tau = TAU
    R_max = s * SIGNAL_STRENGTH
    T1 = math.ceil(2 * s * np.log(2 * d * max(tau, 1)))

    print(f"[village {village_number}, run {run_index + 1}] d={d}, s={s}, T1={T1}")

    # --- Baseline (Algorithm 3) ---
    # Reused from previous simulation outputs; do not re-run here.
    # baseline = BaselineBandit(
    #     X, tau=tau, b=b, lambda_confidence=BASELINE_LAMBDA,
    #     reg_param=0, sparsity=s, R_max=R_max, random_state=run_seed + 1,
    # )
    # baseline_results = baseline.run()

    netc = NETCBandit(
        X, tau=tau, b=b, exploration_rounds=T1,
        lambda_explore=NETC_LAMBDA, random_state=run_seed + 2,
    )
    netc_results = netc.run()

    nse = NSEBandit(
        X, tau=tau, tau_constant=NSE_TAU_CONSTANT,
        estimation_method="onehot", random_state=run_seed + 4,
    )
    nse_results = nse.run()

    nsefs = NSEFSBandit(
        X, tau=tau, alpha=NSEFS_ALPHA,
        estimation_alpha=NSEFS_EST_ALPHA,
        estimation_method="ridge", random_state=run_seed + 5,
    )
    nsefs_results = nsefs.run()

    return {
        "run_index": run_index,
        "run_seed": run_seed,
        "parameters": {
            "d": d, "s": s, "tau": tau, "b": b,
            "signal_strength": SIGNAL_STRENGTH, "R_max": R_max, "T1": T1,
            "village_number": village_number,
        },
        "results": {
            # "baseline": _json_ready(baseline_results),  # reused from old outputs
            "netc": _json_ready(netc_results),
            "nse": _json_ready(nse_results),
            "nsefs": _json_ready(nsefs_results),
        },
    }


def _run_one_job(job_id):
    village_number = _get_village_number(job_id)
    x_seed = job_id + 200

    X = generate_village_network(x_seed, village_number, SIGNAL_STRENGTH)
    print(f"Village {village_number}: d={X.shape[0]}")

    for run_idx in range(NUM_RUNS):
        run_seed = job_id * RUN_SEED_OFFSET + run_idx + 1
        path = OUTPUT_DIR / f"village_{village_number}_run_{run_idx + 1}_seed_{run_seed}.json"
        if path.exists():
            print(f"[village {village_number} run {run_idx + 1}] already exists, skipping.")
            continue
        run_results = _run_single(X, village_number, run_idx, run_seed)

        payload = {
            "job_id": job_id,
            "village_number": village_number,
            "x_seed": x_seed,
            "run": run_results,
        }
        with path.open("w") as f:
            json.dump(payload, f)
        print(f"Saved run {run_idx + 1}/{NUM_RUNS} to {path}.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-start", type=int, default=None)
    parser.add_argument("--task-end", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.task_start is not None and args.task_end is not None:
        job_ids = range(args.task_start, args.task_end + 1)
    else:
        job_ids = [_get_job_id()]

    for job_id in job_ids:
        _run_one_job(job_id)


if __name__ == "__main__":
    main()
