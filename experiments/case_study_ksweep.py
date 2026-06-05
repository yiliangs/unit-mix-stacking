"""Case-study K-sweep on the synthetic 55-floor tower instance.

Reproduces Section 7.2: K-sweep at tight (0.5%) plate tolerance, plus a
single-knob diagnostic run on the smallest infeasible K.

Backend: HiGHS via PuLP, with mip_heuristic_effort tuned upward to help
find primal feasible points in the tight K=4..7 middle band.

Outputs:
    experiments/results/case_study_ksweep.csv
    experiments/results/case_study_diag.csv
"""

import argparse
import csv
from typing import Dict, List

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b


# ---------------------------------------------------------------------------
# Synthetic case-study instance (see Appendix F of the paper)
# ---------------------------------------------------------------------------
#
# Plate net areas (1380 / 700 / 580 m^2) and floor distribution (37/12/6)
# reflect a typical mid-rise setback profile (street-frontage main plate,
# transition, crown). The 0.5% baseline plate tolerance is realistic
# for construction tolerance on a poured-in-place concrete plate; the
# locked case (tau_i = 0) is recovered in the diagnostic and used as a
# stress test in the envelope sweep.
#
# Five unit types (studio through 3B); 4B luxury units omitted to keep
# the count-vector pool tractable.
#
# Program counts (total 814) are designed so that:
#   - the program target area sits ~0.9% above the nominal plate envelope
#     (63,484 m^2 vs 62,940 m^2), absorbed by the per-type +/-5% slack;
#   - the studio count of 58 is the single Diophantine blocker at K=3:
#     58 mod 6 = 4, so 37 a + 6 m = 58 has no non-negative integer
#     solution against the 37-floor main plate and the 6-floor crown
#     (12 is also divisible by 6), forcing infeasibility at K=3 by
#     pure LP-detectable reasoning.

PLATE_TOL_BASELINE = 0.005   # 0.5% baseline; locked case (0.0) swept in envelope
TIME_LIMIT_PER_CELL = 3600.0 # 1 hour per cell, per user direction

# HiGHS tuning to help find primals in the tight middle band. Default
# mip_heuristic_effort is 0.05; we push it to 0.5 to spend more time in
# primal heuristics relative to dual bounds. presolve forced on to
# exploit constraint structure (the floor-partition equation is a
# strong reduction). mip_detect_symmetry=True because templates within
# a plate are exchangeable, and HiGHS's symmetry exploitation helps.
HIGHS_OPTIONS = {
    "mip_heuristic_effort": 0.5,
    "presolve": "on",
    "mip_detect_symmetry": True,
    "parallel": "on",
    "threads": 4,
}


def build_instance(
    K: int,
    plate_tols=(PLATE_TOL_BASELINE, PLATE_TOL_BASELINE, PLATE_TOL_BASELINE),
    time_limit: float = TIME_LIMIT_PER_CELL,
) -> v2b.StackingConfig:
    # Program counts are designed so that:
    #   (a) the studio count is the Diophantine blocker at K=3:
    #       37 s_M + 12 s_T + 6 s_C = 58 has no non-negative integer
    #       solution since 58 mod 6 = 4;
    #   (b) the other four type counts are all multiples of 12, so
    #       each one decomposes cleanly across the (12, 6) divisors
    #       of the setback plates at K=4 (with the main plate split
    #       into two templates, the joint Diophantine system has many
    #       solutions HiGHS can locate by primal heuristic);
    #   (c) the total target area sums to ~63{,}484 m^2 against a
    #       plate envelope of 62{,}940 m^2, leaving the per-type
    #       +/-5% area bands as the absorber for the ~0.9% overshoot.
    # This keeps the constraint-envelope narrative (tight program
    # against the envelope, structural infeasibility at K=3) while
    # giving HiGHS a tractable search at K>=4.
    types = [
        v2b.UnitType("studio", target_area=40.0,  fixed_count=58,  area_tol=0.05),
        v2b.UnitType("1B",     target_area=55.0,  fixed_count=228, area_tol=0.05),
        v2b.UnitType("2B",     target_area=78.0,  fixed_count=264, area_tol=0.05),
        v2b.UnitType("2B+D",   target_area=92.0,  fixed_count=120, area_tol=0.05),
        v2b.UnitType("3B",     target_area=118.0, fixed_count=144, area_tol=0.05),
    ]
    plates = [
        v2b.PlateClass("M", 1380.0, 37, area_tol=plate_tols[0]),
        v2b.PlateClass("T",  700.0, 12, area_tol=plate_tols[1]),
        v2b.PlateClass("C",  580.0,  6, area_tol=plate_tols[2]),
    ]
    return v2b.StackingConfig(
        types=types, plates=plates, n_zones=K,
        min_floors_per_zone=2,
        max_count_per_type_per_floor=8,
        enumeration_max_count_total=18,
        deviation_weight=1000.0,
        pattern_simplicity_weight=1e-6,
        time_limit=time_limit, solver_msg=False,
        solver_options=HIGHS_OPTIONS,
    )


def ksweep(time_limit: float = TIME_LIMIT_PER_CELL) -> List[Dict]:
    rows = []
    for K in range(3, 12):
        cfg = build_instance(K, time_limit=time_limit)
        print(f"  K={K} ... ", end="", flush=True)
        sol = v2b.solve(cfg)
        print(f"{sol.status} ({sol.solve_time_sec:.1f}s)")
        rows.append({
            "K": K,
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


def first_infeasible_K(rows: List[Dict]) -> int:
    """Identify the smallest K whose status is not Optimal/Optimal_TimeLimit."""
    for r in rows:
        if not r["status"].startswith("Optimal"):
            return r["K"]
    return None


def run_diagnostic(K_star: int, time_limit_per_probe: float = 600.0) -> List[Dict]:
    print(f"\nDiagnostic at K={K_star}:")
    cfg = build_instance(K_star, time_limit=time_limit_per_probe)
    probes = v2b.diagnose_infeasibility(
        cfg,
        plate_tol_step=0.005,
        unit_tol_step=0.02,
        zone_step=1,
        count_window_step=0,
        time_limit_per_probe=time_limit_per_probe,
        verbose=True,
    )
    return [{
        "knob": p.knob,
        "description": p.description,
        "delta": p.delta,
        "status": p.status,
        "solve_time_sec": round(p.solve_time_sec, 3),
    } for p in probes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-diagnostic", action="store_true")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_PER_CELL,
                        help=f"Per-cell time limit (default {TIME_LIMIT_PER_CELL}s).")
    parser.add_argument("--diag-time", type=float, default=600.0,
                        help="Per-probe time limit for the diagnostic (default 600s).")
    args = parser.parse_args()

    write_machine_json("case_study_ksweep.py")

    print("Case-study K-sweep (tau_i=0.5% baseline, HiGHS, "
          f"time={args.time_limit:.0f}s/cell):")
    ksweep_rows = ksweep(time_limit=args.time_limit)
    with open(out_path("case_study_ksweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ksweep_rows[0].keys()))
        w.writeheader()
        w.writerows(ksweep_rows)
    print(f"Wrote {out_path('case_study_ksweep.csv')}.")

    if args.no_diagnostic:
        return

    K_star = first_infeasible_K(ksweep_rows)
    if K_star is None:
        print("\nNo infeasible K observed; skipping diagnostic.")
        return
    diag_rows = run_diagnostic(K_star, time_limit_per_probe=args.diag_time)
    with open(out_path("case_study_diag.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)
    print(f"Wrote {out_path('case_study_diag.csv')}.")


if __name__ == "__main__":
    main()
