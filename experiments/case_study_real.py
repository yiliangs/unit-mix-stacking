"""Real-building case study: De Piek Waterfront Tower (KCAP, Rotterdam, 2025).

Companion to the SYNTHETIC case study (case_study_ksweep.py). Where the
synthetic instance is a controlled instrument (engineered Diophantine
blocker, K-sweep, diagnostic), this one is the external-validity demo:
real traced plates, a locally-realistic posed program, a feasible buildable
stacking rendered by fig14_real_case_stacking.py.

Building facts (traced from the published ArchDaily section, areas net of
the vertical core+corridor, balconies excluded):

    floors 1-2   amenity / lobby   (NON-rentable, not in the instance)
    floors 3-4   garage            (NON-rentable, not in the instance)
    floors 5-7   zone 1   net 940 m^2/floor   (3 floors)
    floors 8-15  zone 2   net 850 m^2/floor   (8 floors)
    floors 16-23 zone 3   net 640 m^2/floor   (8 floors)

The three rentable plates taper (940 > 850 > 640), so this exercises V2b's
heterogeneous-plate mode, not the uniform-plate v2 reduction.

Program (POSED, not the as-built KCAP mix): a representative Rotterdam
waterfront brief, fixed_count (157 units): ~29% in the affordable band
(studio + 1-bed; sociale huur / middenhuur), a 2-bed-heavy family middle,
large 3-beds and penthouses up top. Sized so the nominal area is ~0.2% over
the plate envelope (units near nominal, comfortable tolerance), not pinned to
the building's 142 (we do not claim to reproduce the real stacking). Unit
areas are NIA (net of core+corridor), consistent with the plate net areas.

Outputs:
    experiments/results/case_study_real_stacking.csv   (per-floor stacking)
    experiments/results/case_study_real_zones.csv      (per-zone summary)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make _common importable

from _common import out_path, write_machine_json
import unit_mix_solver_v2b as v2b


# --- Posed program (proportional). NIA areas, net of core+corridor. ---------
def build_instance(K: int, time_limit: float = 60.0) -> v2b.StackingConfig:
    # Posed program, fixed_count (the "delivered program" framing, as the
    # synthetic case). Sized so the nominal area (14,772 m^2) sits ~0.2% over the
    # plate envelope (14,740 m^2): units sit near their nominal sizes with
    # comfortable +/-5% slack, not crammed against tolerance. ~29% affordable
    # (studio+1B, Rotterdam sociale huur / middenhuur), 2B-dominant family middle,
    # 3B+PH premium crown. Counts decompose cleanly across the 3/8/8 floor plates
    # (lower 4 studio+11 1B; mid 8 2B+1 3B; upper 4 3B+1 PH, per floor). NIA areas.
    types = [
        v2b.UnitType("studio", target_area=50.0,  fixed_count=12, area_tol=0.05),
        v2b.UnitType("1B",     target_area=68.0,  fixed_count=33, area_tol=0.05),
        v2b.UnitType("2B",     target_area=92.0,  fixed_count=64, area_tol=0.05),
        v2b.UnitType("3B",     target_area=120.0, fixed_count=40, area_tol=0.05),
        v2b.UnitType("PH",     target_area=155.0, fixed_count=8,  area_tol=0.05),
    ]
    # Traced plates, ±2% tolerance (absorbs tracing error per the envelope sweep).
    plates = [
        v2b.PlateClass("lower", 940.0, 3, area_tol=0.02),
        v2b.PlateClass("mid",   850.0, 8, area_tol=0.02),
        v2b.PlateClass("upper", 640.0, 8, area_tol=0.02),
    ]
    return v2b.StackingConfig(
        types=types, plates=plates, n_zones=K,
        min_floors_per_zone=2,
        proportion_hard_band=0.05,
        max_count_per_type_per_floor=14,
        enumeration_max_count_total=18,
        deviation_weight=1000.0,
        pattern_simplicity_weight=1e-6,
        solver_objective="vertical_sort",   # compact + larger units higher (as F13)
        solver_backend="gurobi",             # F13's stacking also used Gurobi for the clean optimum
        time_limit=time_limit, solver_msg=False,
        solver_options={},
    )


def _renumber_zones_bottom_to_top(sol, target_area: dict) -> None:
    """Order zones in physical stacking order (plate bottom-to-top, then mean
    unit size within a plate) and renumber 1..n from the base, so the emitted
    per-floor table is monotonic in unit size from the ground up. Mirrors
    case_study_k8_stacking._renumber_zones_bottom_to_top (F13)."""
    def mean_target_size(z) -> float:
        units = sum(z.counts.values())
        if not units:
            return 0.0
        return sum(z.counts[n] * target_area[n] for n in z.counts) / units
    sol.zones.sort(key=lambda z: (z.plate_index, mean_target_size(z)))
    for new_index, z in enumerate(sol.zones, start=1):
        z.zone_index = new_index


def main() -> None:
    write_machine_json("case_study_real.py")
    # Single canonical run: K cap = 6, vertical_sort compacts to the fewest
    # templates and orders larger units higher. The figure reads this CSV.
    cfg = build_instance(K=6, time_limit=90.0)
    env = sum(p.net_area * p.n_floors for p in cfg.plates)
    nominal = sum((t.fixed_count or 0) * t.target_area for t in cfg.types)
    print("Real case study (De Piek), fixed-count program, vertical_sort, Gurobi:")
    print(f"  plate envelope {env:.0f} m^2; program nominal {nominal:.0f} m^2 "
          f"({nominal/env - 1:+.2%})\n")
    sol = v2b.solve(cfg)
    if not (sol.status.startswith("Optimal") and sol.zones):
        raise SystemExit(f"No usable stacking (status={sol.status}).")
    _renumber_zones_bottom_to_top(sol, {t.name: t.target_area for t in cfg.types})
    max_dev = max((abs(d) for d in sol.proportion_deviation.values()), default=0.0)
    print(f"  {sol.status}  zones_used={sol.n_zones_used} units={sol.total_units} "
          f"area={sol.total_area:.0f} max_dev={max_dev*100:.2f}pp "
          f"time={sol.solve_time_sec:.2f}s\n")
    sol.floors_dataframe().to_csv(out_path("case_study_real_stacking.csv"), index=False)
    sol.zones_dataframe().to_csv(out_path("case_study_real_zones.csv"), index=False)
    print(sol.zones_dataframe().to_string(index=False))
    print("\nProportion check (target -> achieved):")
    print(sol.summary_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
