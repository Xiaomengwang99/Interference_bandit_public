"""
Main simulation: compare algorithms on a simulated random network.

Corresponds to Figure 1 in the paper. Runs NETC, NSE, and NSE-FS on a
single randomly generated network with d=100, s=20, T=20000.

NOTE ON REUSED BASELINE RESULTS:
    The Baseline algorithm call below is intentionally left in source but
    commented out. For publication runs, Baseline is reused from previous
    simulations (see output_simulations_main_03022026/). Only NETC, NSE,
    and NSE-FS are re-run with the final algorithm configurations:
      - NSE:    one-hot estimator, tau_constant=0.2
      - NSE-FS: ridge, alpha=0.05, estimation_alpha=1.0
      - NETC:   lambda_explore=0.035

Usage
-----
    python run_main.py                         # seed=1 (or SGE_TASK_ID)
    python run_main.py --seed-start 1 --seed-end 10   # local sequential loop
    SGE_TASK_ID=42 python run_main.py          # cluster array job, seed=42
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from algorithms import (
    generate_network,
    NETCBandit,
    NSEBandit,
    NSEFSBandit,
    BaselineBandit,  # still imported for documentation; call site commented below
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_main"


def _get_job_id():
    """Read the SGE array task ID, defaulting to 1."""
    job_id = os.getenv("SGE_TASK_ID")
    if job_id is None:
        return 1
    try:
        return int(job_id)
    except ValueError:
        return 1


def _json_ready(value):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _run_single_seed(seed):
    """Run the non-baseline algorithms for one random seed."""
    d = 100
    s = 20
    tau = 20000
    b = 100
    signal_strength = 0.1
    R_max = s * signal_strength
    reg_param = 0
    T1 = 200  # exploration rounds for NETC

    print(f"[seed {seed}] d={d}, s={s}, T={tau}, signal={signal_strength}")
    X = generate_network(seed, d, s, signal_strength)

    # --- Baseline (Algorithm 3) ---
    # Reused from previous simulation outputs; do not re-run here.
    # baseline = BaselineBandit(
    #     X, tau=tau, b=b,
    #     lambda_confidence=1, reg_param=reg_param, sparsity=s, R_max=R_max,
    # )
    # baseline_results = baseline.run()

    # --- NETC (Algorithm 2), lambda=0.035 ---
    netc = NETCBandit(X, tau=tau, b=b, exploration_rounds=T1, lambda_explore=0.035)
    netc_results = netc.run()

    # --- NSE (Algorithm 1) with paper's one-hot estimator ---
    nse = NSEBandit(X, tau=tau, tau_constant=0.2, estimation_method="onehot")
    nse_results = nse.run()

    # --- NSE-FS (Algorithm 4) with ridge ---
    nsefs = NSEFSBandit(X, tau=tau, alpha=0.05, estimation_alpha=1.0,
                        estimation_method="ridge")
    nsefs_results = nsefs.run()

    return {
        "seed": seed,
        "parameters": {
            "d": d, "s": s, "tau": tau, "b": b,
            "signal_strength": signal_strength, "R_max": R_max,
            "T1": T1,
        },
        "results": {
            # "baseline": _json_ready(baseline_results),  # reused from old outputs
            "netc": _json_ready(netc_results),
            "nse": _json_ready(nse_results),
            "nsefs": _json_ready(nsefs_results),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=None,
                        help="Start seed (inclusive) for local batch runs.")
    parser.add_argument("--seed-end", type=int, default=None,
                        help="End seed (inclusive). If set with --seed-start, loops locally.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.seed_start is not None and args.seed_end is not None:
        seeds = range(args.seed_start, args.seed_end + 1)
    else:
        seeds = [_get_job_id()]

    for seed in seeds:
        out_path = OUTPUT_DIR / f"seed_{seed}.json"
        if out_path.exists():
            print(f"[seed {seed}] already exists, skipping.")
            continue
        results = _run_single_seed(seed)
        with out_path.open("w") as f:
            json.dump(results, f)
        print(f"Saved results to {out_path}.")


if __name__ == "__main__":
    main()
