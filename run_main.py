"""
Main simulation: compare all four algorithms on a simulated random network.

Corresponds to Figure 1 in the paper. Runs Baseline, NETC (two lambda values),
NSE, and NSE-FS on a single randomly generated network with d=100, s=20,
T=20000.

Usage
-----
    python run_main.py                # uses SGE_TASK_ID or defaults to seed=1
    SGE_TASK_ID=42 python run_main.py # for cluster array jobs
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
    BaselineBandit,
)

OUTPUT_DIR = Path("output_main")


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
    """Run all algorithms for one random seed and return collected results."""
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
    baseline = BaselineBandit(
        X, tau=tau, b=b,
        lambda_confidence=1, reg_param=reg_param, sparsity=s, R_max=R_max,
    )
    baseline_results = baseline.run()

    # --- NETC (Algorithm 2), two lambda values ---
    netc1 = NETCBandit(X, tau=tau, b=b, exploration_rounds=T1, lambda_explore=0.035)
    netc1_results = netc1.run()

    netc2 = NETCBandit(X, tau=tau, b=b, exploration_rounds=T1, lambda_explore=1.0)
    netc2_results = netc2.run()

    # --- NSE (Algorithm 1) ---
    nse = NSEBandit(X, tau=tau, tau_constant=0.1, estimation_alpha=1.0)
    nse_results = nse.run()

    # --- NSE-FS (Algorithm 4) ---
    nsefs = NSEFSBandit(X, tau=tau, alpha=0.05, estimation_alpha=1.0)
    nsefs_results = nsefs.run()

    return {
        "seed": seed,
        "parameters": {
            "d": d, "s": s, "tau": tau, "b": b,
            "signal_strength": signal_strength, "R_max": R_max,
            "T1": T1,
        },
        "results": {
            "baseline": _json_ready(baseline_results),
            "netc_lambda1": _json_ready(netc1_results),
            "netc_lambda2": _json_ready(netc2_results),
            "nse": _json_ready(nse_results),
            "nsefs": _json_ready(nsefs_results),
        },
    }


def main():
    seed = _get_job_id()
    results = _run_single_seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"seed_{seed}.json"
    with path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}.")


if __name__ == "__main__":
    main()
