"""Real-case constraint envelope + infeasibility diagnostic: De Piek.

Companion to case_study_ksweep.py (synthetic Sec. 6/7). Takes the
157-unit posed program (the SAME program fig14 renders, read from the
canonical stacking CSV so the two never drift), pins it as fixed_count, and
maps the feasibility frontier as:

  1. the plate (construction) tolerance is tightened from the +/-2% baseline
     toward locked (0%), at baseline unit tolerance;
  2. if still feasible at locked plates, the unit-area tolerance is tightened
     at locked plates.

At the first infeasible corner it runs the single-knob diagnostic: which lone
relaxation (widen one plate's tol, widen one unit's tol, or add a zone)
restores a feasible stacking. That is the honest-infeasibility / design-
intelligence claim, on a real building.

Honest envelope: if the realistic program stays feasible to locked plates,
that is reported as headroom, not massaged into a failure.

Backend: HiGHS via PuLP (headline). K=3 (one template per architectural plate,
the compact target). Feasibility objective (not vertical_sort, which is for the
figure only). Outputs:
    experiments/results/case_study_real_envelope.csv
    experiments/results/case_study_real_diag.csv
"""
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import out_path, write_machine_json
import unit_mix_solver_v2b as v2b

UNIT_TYPES = ["studio", "1B", "2B", "3B", "PH"]
UNIT_AREA = {"studio": 50.0, "1B": 68.0, "2B": 92.0, "3B": 120.0, "PH": 155.0}
PLATES = [("lower", 940.0, 3), ("mid", 850.0, 8), ("upper", 640.0, 8)]
K = 3

HIGHS = {"mip_heuristic_effort": 0.5, "presolve": "on",
         "mip_detect_symmetry": True, "parallel": "on"}


def delivered_counts() -> dict:
    """The 157-unit posed program fig14 renders (summed from its stacking CSV)."""
    df = pd.read_csv(out_path("case_study_real_stacking.csv"))
    return {ut: int(df[f"{ut}_count"].sum()) for ut in UNIT_TYPES}


def build(counts, plate_tol, unit_tol, time_limit=45.0) -> v2b.StackingConfig:
    types = [v2b.UnitType(ut, target_area=UNIT_AREA[ut],
                          fixed_count=counts[ut], area_tol=unit_tol)
             for ut in UNIT_TYPES]
    plates = [v2b.PlateClass(n, a, f, area_tol=plate_tol) for n, a, f in PLATES]
    return v2b.StackingConfig(
        types=types, plates=plates, n_zones=K, min_floors_per_zone=2,
        max_count_per_type_per_floor=15, enumeration_max_count_total=20,
        deviation_weight=1000.0, pattern_simplicity_weight=1e-6,
        time_limit=time_limit, solver_msg=False, solver_options=HIGHS)


def feasible(sol) -> bool:
    return sol.status.startswith("Optimal") and bool(sol.zones)


def main() -> None:
    write_machine_json("case_study_real_envelope.py")
    counts = delivered_counts()
    total = sum(counts.values())
    target_area = sum(counts[u] * UNIT_AREA[u] for u in UNIT_TYPES)
    envelope = sum(a * f for _, a, f in PLATES)
    print(f"Posed program (fixed_count): {counts}  total={total}")
    print(f"Program target area {target_area:.0f} m^2 vs plate envelope "
          f"{envelope:.0f} m^2 ({target_area/envelope - 1:+.2%})\n")

    rows = []
    first_infeasible = None

    print("Axis 1: tighten plate tolerance (unit_tol = 5%), K=3:")
    for ptol in (0.02, 0.015, 0.01, 0.005, 0.0):
        sol = v2b.solve(build(counts, ptol, 0.05))
        ok = feasible(sol)
        print(f"  plate_tol={ptol*100:4.1f}%  unit_tol=5.0%  -> {sol.status:18s}"
              f"  zones={sol.n_zones_used}  ({sol.solve_time_sec:.1f}s)")
        rows.append(dict(axis="plate_tol", plate_tol_pct=round(ptol*100, 1),
                         unit_tol_pct=5.0, status=sol.status, feasible=ok,
                         zones_used=sol.n_zones_used,
                         solve_time_sec=round(sol.solve_time_sec, 2)))
        if not ok and first_infeasible is None:
            first_infeasible = (ptol, 0.05)

    if rows[-1]["feasible"]:
        print("\nAxis 2: plates locked (0%), tighten unit tolerance, K=3:")
        for utol in (0.05, 0.04, 0.03, 0.02, 0.01):
            sol = v2b.solve(build(counts, 0.0, utol))
            ok = feasible(sol)
            print(f"  plate_tol=0.0%  unit_tol={utol*100:4.1f}%  -> {sol.status:18s}"
                  f"  zones={sol.n_zones_used}  ({sol.solve_time_sec:.1f}s)")
            rows.append(dict(axis="unit_tol", plate_tol_pct=0.0,
                             unit_tol_pct=round(utol*100, 1), status=sol.status,
                             feasible=ok, zones_used=sol.n_zones_used,
                             solve_time_sec=round(sol.solve_time_sec, 2)))
            if not ok and first_infeasible is None:
                first_infeasible = (0.0, utol)

    with open(out_path("case_study_real_envelope.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path('case_study_real_envelope.csv')}.")

    if first_infeasible is None:
        print("\nFRONTIER: the posed program stays feasible across the whole "
              "swept envelope (to locked plates AND unit_tol=1%). Honest result: "
              "comfortable headroom on this instance; no single-knob diagnostic "
              "to run. (See discussion: real plates are gentler than the "
              "synthetic stress instance by design.)")
        return

    ptol, utol = first_infeasible
    print(f"\nFirst infeasible corner: plate_tol={ptol*100:.1f}%, "
          f"unit_tol={utol*100:.1f}%. Running single-knob diagnostic:\n")
    probes = v2b.diagnose_infeasibility(
        build(counts, ptol, utol, time_limit=45.0),
        plate_tol_step=0.005, unit_tol_step=0.02, zone_step=1,
        count_window_step=0, time_limit_per_probe=45.0, verbose=True)
    drows = [dict(knob=p.knob, description=p.description, delta=p.delta,
                  status=p.status, unblocks=p.status.startswith("Optimal"),
                  solve_time_sec=round(p.solve_time_sec, 2)) for p in probes]
    with open(out_path("case_study_real_diag.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
        w.writeheader()
        w.writerows(drows)
    print(f"\nWrote {out_path('case_study_real_diag.csv')}.")


if __name__ == "__main__":
    main()
