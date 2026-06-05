"""Case-study extras (envelope + headroom) under Gurobi.

Bundles three §7 follow-ups that were slow or numerically delicate on
HiGHS:

  1. envelope: (K, tau_M) sweep mirroring case_study_envelope.py
  2. headroom: tau_M sweep at fixed K in bounded-count mode, mirroring
     case_study_headroom.py
  3. ksweep:   full K=3..11 sweep on Gurobi for direct comparison with
     case_study_ksweep.csv

Three separate CSV outputs; existing HiGHS results are untouched.

Outputs:
    experiments/results/gurobi_case_study_envelope.csv
    experiments/results/gurobi_case_study_headroom.csv
    experiments/results/gurobi_case_study_ksweep.csv
"""

import argparse
import csv
from typing import Dict, List

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance as build_ksweep_instance
from case_study_ksweep import PLATE_TOL_BASELINE
from case_study_headroom import build_instance as build_headroom_instance
from case_study_headroom import CONSULTANT


GUROBI_OPTIONS = {
    "Threads": 4,
    "Symmetry": 2,
    "Presolve": 2,
    # NumericFocus left at default; raising it slowed easy cells dramatically
    # in an earlier run for no real numeric benefit.
}


def _gurobify(cfg: v2b.StackingConfig) -> v2b.StackingConfig:
    cfg.solver_backend = "gurobi"
    cfg.solver_options = dict(GUROBI_OPTIONS)
    return cfg


def run_envelope(time_limit: float) -> List[Dict]:
    rows = []
    K_values = [3, 4, 5, 6, 7, 8]
    tau_values = [0.0, PLATE_TOL_BASELINE, 0.01, 0.02]
    for K in K_values:
        for tau in tau_values:
            cfg = build_ksweep_instance(
                K,
                plate_tols=(tau, PLATE_TOL_BASELINE, PLATE_TOL_BASELINE),
                time_limit=time_limit,
            )
            _gurobify(cfg)
            print(f"  envelope K={K}, tau_M={tau:.3f} ... ", end="", flush=True)
            sol = v2b.solve(cfg)
            print(f"{sol.status} ({sol.solve_time_sec:.1f}s)")
            rows.append({
                "K": K,
                "tau_M": tau,
                "solver": "Gurobi",
                "status": sol.status,
                "feasible": sol.status.startswith("Optimal"),
                "solve_time_sec": round(sol.solve_time_sec, 3),
                "objective": sol.objective,
                "mip_gap_rel": sol.mip_gap_rel,
            })
    return rows


def run_headroom(K: int, time_limit: float) -> List[Dict]:
    tau_values = [PLATE_TOL_BASELINE, 0.01, 0.02, 0.03, 0.04, 0.05]
    base = sum(CONSULTANT.values())
    rows = []
    for tau in tau_values:
        cfg = build_headroom_instance(K, tau, time_limit=time_limit)
        _gurobify(cfg)
        print(f"  headroom K={K}, tau_M={tau:.3f} ... ", end="", flush=True)
        sol = v2b.solve(cfg)
        units = sol.total_units or 0
        headroom = units - base if units else 0
        print(f"{sol.status} ({sol.solve_time_sec:.1f}s) units={units}")
        rows.append({
            "K": K,
            "tau_M": tau,
            "solver": "Gurobi",
            "status": sol.status,
            "feasible": sol.status.startswith("Optimal"),
            "total_units": units,
            "consultant_floor": base,
            "headroom_units": headroom,
            "headroom_pct": (headroom / base * 100.0) if base else 0.0,
            "solve_time_sec": round(sol.solve_time_sec, 3),
            "objective": sol.objective,
            "mip_gap_rel": sol.mip_gap_rel,
        })
    return rows


def run_ksweep(time_limit: float) -> List[Dict]:
    rows = []
    for K in range(3, 12):
        cfg = build_ksweep_instance(K, time_limit=time_limit)
        _gurobify(cfg)
        print(f"  ksweep K={K} ... ", end="", flush=True)
        sol = v2b.solve(cfg)
        print(f"{sol.status} ({sol.solve_time_sec:.1f}s) units={sol.total_units}")
        rows.append({
            "K": K,
            "solver": "Gurobi",
            "status": sol.status,
            "zones_used": sol.n_zones_used,
            "solve_time_sec": round(sol.solve_time_sec, 3),
            "total_units": sol.total_units,
            "total_area": round(sol.total_area, 1),
            "objective": sol.objective,
            "mip_dual_bound": sol.mip_dual_bound,
            "mip_gap_rel": sol.mip_gap_rel,
            "mip_node_count": sol.mip_node_count,
        })
    return rows


def _write(rows: List[Dict], name: str) -> None:
    path = out_path(name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--headroom-K", type=int, default=8)
    parser.add_argument("--skip-envelope", action="store_true")
    parser.add_argument("--skip-headroom", action="store_true")
    parser.add_argument("--skip-ksweep", action="store_true")
    args = parser.parse_args()

    write_machine_json("gurobi_case_study_extras.py")

    if not args.skip_ksweep:
        print("Gurobi K-sweep (tau_i=0.5% baseline):")
        _write(run_ksweep(args.time_limit), "gurobi_case_study_ksweep.csv")
    if not args.skip_envelope:
        print("\nGurobi envelope sweep:")
        _write(run_envelope(args.time_limit), "gurobi_case_study_envelope.csv")
    if not args.skip_headroom:
        print(f"\nGurobi headroom sweep (K={args.headroom_K}):")
        _write(run_headroom(args.headroom_K, args.time_limit),
               "gurobi_case_study_headroom.csv")


if __name__ == "__main__":
    main()
