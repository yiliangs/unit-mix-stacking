"""Solver-driven program-headroom analysis on the case-study instance.

Reframes the §7 case study in BOUNDED_COUNT mode: the consultant counts
become per-type *minimum* requirements, max is unbounded, and the
objective shifts to maximize the total unit count. Sweeps main-plate
tolerance tau_M at a fixed K to characterize how much additional
density the solver discovers above the consultant program as the
plate envelope opens.

Reproduces §7.4 of the paper.

Output: experiments/results/case_study_headroom.csv
"""

import argparse
import csv
from typing import Dict, List

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import PLATE_TOL_BASELINE, HIGHS_OPTIONS


# Consultant program (matches fixed_count values used in
# case_study_ksweep.build_instance). Becomes min_count in bounded mode.
CONSULTANT = {
    "studio": 58,
    "1B":     228,
    "2B":     264,
    "2B+D":   120,
    "3B":     144,
}


def build_instance(K: int, tau_M: float, time_limit: float = 600.0) -> v2b.StackingConfig:
    types = [
        v2b.UnitType("studio", target_area=40.0,  min_count=CONSULTANT["studio"], area_tol=0.05),
        v2b.UnitType("1B",     target_area=55.0,  min_count=CONSULTANT["1B"],     area_tol=0.05),
        v2b.UnitType("2B",     target_area=78.0,  min_count=CONSULTANT["2B"],     area_tol=0.05),
        v2b.UnitType("2B+D",   target_area=92.0,  min_count=CONSULTANT["2B+D"],   area_tol=0.05),
        v2b.UnitType("3B",     target_area=118.0, min_count=CONSULTANT["3B"],     area_tol=0.05),
    ]
    plates = [
        v2b.PlateClass("M", 1380.0, 37, area_tol=tau_M),
        v2b.PlateClass("T",  700.0, 12, area_tol=PLATE_TOL_BASELINE),
        v2b.PlateClass("C",  580.0,  6, area_tol=PLATE_TOL_BASELINE),
    ]
    return v2b.StackingConfig(
        types=types, plates=plates, n_zones=K,
        min_floors_per_zone=2,
        max_count_per_type_per_floor=8,
        enumeration_max_count_total=18,
        pattern_simplicity_weight=1e-6,
        solver_objective="maximize_total_units",
        time_limit=time_limit, solver_msg=False,
        solver_options=HIGHS_OPTIONS,
    )


def run_one(K: int, tau_M: float, time_limit: float) -> Dict:
    cfg = build_instance(K, tau_M, time_limit=time_limit)
    print(f"  K={K}, tau_M={tau_M:.3f} ... ", end="", flush=True)
    sol = v2b.solve(cfg)
    print(f"{sol.status} ({sol.solve_time_sec:.1f}s)  units={sol.total_units}")

    base = sum(CONSULTANT.values())
    headroom = sol.total_units - base if sol.total_units else 0
    return {
        "K": K,
        "tau_M": tau_M,
        "status": sol.status,
        "feasible": sol.status.startswith("Optimal"),
        "total_units": sol.total_units,
        "consultant_floor": base,
        "headroom_units": headroom,
        "headroom_pct": (headroom / base * 100.0) if base else 0.0,
        "solve_time_sec": round(sol.solve_time_sec, 3),
        "objective": sol.objective,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=8,
                        help="Compactness cap (default 8).")
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args()

    write_machine_json("case_study_headroom.py")

    # Sweep main-plate tolerance starting at the baseline (0.5%).
    tau_values = [PLATE_TOL_BASELINE, 0.01, 0.02, 0.03, 0.04, 0.05]
    rows: List[Dict] = []
    for tau in tau_values:
        rows.append(run_one(args.K, tau, args.time_limit))

    csv_path = out_path("case_study_headroom.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
