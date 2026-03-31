"""
Sparsity sweep: evaluate all algorithms under varying sparsity levels.

Part of the robustness experiments in Appendix D. Each SGE array task maps
to a (sparsity, within-bucket seed) pair.

Usage
-----
    SGE_TASK_ID=1 python run_sweep_sparsity.py   # s=5,  seed 1
    SGE_TASK_ID=101 python run_sweep_sparsity.py # s=10, seed 1
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

# --- Default parameters ---
DEFAULT_D = 100
DEFAULT_TAU = 20000
DEFAULT_B = 100
DEFAULT_SIGNAL_STRENGTH = 0.1
DEFAULT_T1 = 200
NETC_LAMBDA1 = 0.035
NETC_LAMBDA2 = 1.0
NSE_TAU_CONSTANT = 0.1
NSE_ALPHA = 1.0
NSEFS_ALPHA = 0.05
NSEFS_EST_ALPHA = 1.0

# --- Sweep configuration ---
SPARSITY_VALUES = [5, 10, 15, 20, 25, 50]
BUCKET_SIZE = 100  # seeds per sparsity value
OUTPUT_DIR = Path("output_sweep_sparsity")


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


def _sparsity_from_seed(seed):
    """Map the SGE task ID to a sparsity bucket."""
    bucket = min((max(seed, 1) - 1) // BUCKET_SIZE, len(SPARSITY_VALUES) - 1)
    return SPARSITY_VALUES[bucket]


def _run_single_seed(seed, sparsity):
    d = DEFAULT_D
    s = sparsity
    tau = DEFAULT_TAU
    b = DEFAULT_B
    signal_strength = DEFAULT_SIGNAL_STRENGTH
    R_max = s * signal_strength
    T1 = DEFAULT_T1

    print(f"[seed {seed}] s={s}, d={d}, T={tau}, signal={signal_strength}")
    X = generate_network(seed, d, s, signal_strength)

    baseline = BaselineBandit(
        X, tau=tau, b=b, lambda_confidence=1, reg_param=0, sparsity=s, R_max=R_max,
    )
    baseline_results = baseline.run()

    netc1 = NETCBandit(X, tau=tau, b=b, exploration_rounds=T1, lambda_explore=NETC_LAMBDA1)
    netc1_results = netc1.run()

    netc2 = NETCBandit(X, tau=tau, b=b, exploration_rounds=T1, lambda_explore=NETC_LAMBDA2)
    netc2_results = netc2.run()

    nse = NSEBandit(X, tau=tau, tau_constant=NSE_TAU_CONSTANT, estimation_alpha=NSE_ALPHA)
    nse_results = nse.run()

    nsefs = NSEFSBandit(X, tau=tau, alpha=NSEFS_ALPHA, estimation_alpha=NSEFS_EST_ALPHA)
    nsefs_results = nsefs.run()

    return {
        "seed": seed,
        "parameters": {
            "d": d, "s": s, "tau": tau, "b": b,
            "signal_strength": signal_strength, "R_max": R_max, "T1": T1,
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
    sparsity = _sparsity_from_seed(seed)
    results = _run_single_seed(seed, sparsity)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"seed_{seed}_sparsity_{sparsity}.json"
    with path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {path}.")


if __name__ == "__main__":
    main()
