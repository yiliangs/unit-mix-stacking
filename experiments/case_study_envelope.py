"""Two-dimensional (K, plate_tol) feasibility envelope on the case-study
instance. Reproduces Figure 4 (Section 7.3).

Sweeps K x main-plate tolerance on a grid and records status per cell.
The other plates stay at the baseline 0.5% tolerance; only the main
plate is varied.

Output: experiments/results/case_study_envelope.csv
"""

import argparse
import csv
from typing import Dict, List

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance, PLATE_TOL_BASELINE


def sweep(K_values: List[int], tau_values: List[float],
          time_limit: float) -> List[Dict]:
    rows = []
    for K in K_values:
        for tau in tau_values:
            print(f"  K={K}, tau_M={tau:.3f} ... ", end="", flush=True)
            cfg = build_instance(
                K,
                plate_tols=(tau, PLATE_TOL_BASELINE, PLATE_TOL_BASELINE),
                time_limit=time_limit,
            )
            sol = v2b.solve(cfg)
            print(f"{sol.status} ({sol.solve_time_sec:.1f}s)")
            rows.append({
                "K": K,
                "tau_M": tau,
                "status": sol.status,
                "feasible": sol.status.startswith("Optimal"),
                "solve_time_sec": round(sol.solve_time_sec, 3),
                "objective": sol.objective,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args()

    if args.quick:
        K_values = [3, 5, 7]
        tau_values = [0.0, 0.005, 0.02]
    else:
        # 6 K x 4 tau = 24 cells; trim to focus on the interesting band.
        # K=3 is the structurally infeasible Diophantine cell (one row
        # for context). K=4..7 is the tooling-boundary band. K=8 is the
        # safely feasible top. tau_M includes the locked stress test
        # (0) and the baseline (0.005) plus two relaxation steps.
        K_values = [3, 4, 5, 6, 7, 8]
        tau_values = [0.0, 0.005, 0.01, 0.02]

    write_machine_json("case_study_envelope.py")

    rows = sweep(K_values, tau_values, args.time_limit)
    with open(out_path("case_study_envelope.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path('case_study_envelope.csv')} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
