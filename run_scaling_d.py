"""
Scaling experiment: per-individual regret as a function of network size d.

Corresponds to Figure 2 in the paper. Sweeps d from 100 to 1000 and runs
NETC, NSE, and NSE-FS (excluding the Baseline due to computational cost
at large d). For each d, multiple seeds are used and the per-individual
cumulative regret (total regret / d) is reported.

Usage
-----
    python run_scaling_d.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms import (
    generate_network,
    NETCBandit,
    NSEBandit,
    NSEFSBandit,
    BaselineBandit,
)

# --- Default parameters (fixed across the sweep) ---
DEFAULT_S = 20
DEFAULT_TAU = 20000
DEFAULT_B = 100
DEFAULT_SIGNAL_STRENGTH = 0.1
DEFAULT_T1 = 200
NETC_LAMBDA = 0.05
NSE_TAU_CONSTANT = 0.5
NSE_ALPHA = 1.0
NSEFS_ALPHA = 0.05
NSEFS_EST_ALPHA = 1.0

# --- Sweep configuration ---
D_VALUES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
SEEDS_PER_D = 100
OUTPUT_DIR = Path("output_scaling_d")
ALGO_LABELS = {"netc": "No info (NETC)", "nse": "Partial info (NSE)", "nsefs": "Full info (NSE)"}


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

    X = generate_network(seed, d, s, signal_strength)

    netc = NETCBandit(X, tau=tau, b=DEFAULT_B, exploration_rounds=DEFAULT_T1,
                      lambda_explore=NETC_LAMBDA)
    netc_results = netc.run()

    nse = NSEBandit(X, tau=tau, tau_constant=NSE_TAU_CONSTANT,
                    estimation_alpha=NSE_ALPHA)
    nse_results = nse.run()

    nsefs = NSEFSBandit(X, tau=tau, alpha=NSEFS_ALPHA,
                        estimation_alpha=NSEFS_EST_ALPHA)
    nsefs_results = nsefs.run()

    return {
        "netc": netc_results,
        "nse": nse_results,
        "nsefs": nsefs_results,
    }


def _plot(d_values, stats, output_path):
    """Plot per-individual cumulative regret vs. d."""
    d_axis = np.asarray(d_values, dtype=float)
    plt.figure(figsize=(8, 5))
    for key, label in ALGO_LABELS.items():
        means = np.asarray(stats[key]["means"])
        stds = np.asarray(stats[key]["stds"])
        plt.plot(d_axis, means, "o-", label=label)
        plt.fill_between(d_axis, means - stds, means + stds, alpha=0.2)
    plt.xlabel("Network size (d)")
    plt.ylabel("Per-individual cumulative regret")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    algo_keys = list(ALGO_LABELS.keys())
    stats = {k: {"means": [], "stds": []} for k in algo_keys}

    for d in D_VALUES:
        print(f"\n{'='*60}\n[d={d}] Running {SEEDS_PER_D} seeds...\n{'='*60}")
        scaled = {k: [] for k in algo_keys}

        for seed_idx in range(SEEDS_PER_D):
            combined_seed = (seed_idx + 1) + d * 1000
            results = _run_for_d(combined_seed, d)
            for k in algo_keys:
                total = sum(results[k]["regret"])
                scaled[k].append(total / d)

        for k in algo_keys:
            vals = np.array(scaled[k])
            stats[k]["means"].append(float(vals.mean()))
            stats[k]["stds"].append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
        print(f"[d={d}] means: { {k: stats[k]['means'][-1]:.2f for k in algo_keys} }")

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(_json_ready({"d_values": D_VALUES, "stats": stats}), f, indent=2)

    _plot(D_VALUES, stats, OUTPUT_DIR / "regret_vs_d.png")


if __name__ == "__main__":
    main()
