"""Case-study K=8 stacking output for F13.

Solves the delivered consultant program (fixed_count, 814 units, the same
program as case_study_ksweep / Table tab:case-units) at K=8 and baseline
plate tolerance, under the VERTICAL-SORT objective so larger units stack
higher (the zone-ordering extension of Sec. 5.4). The total is pinned by
fixed-count mode, so vertical sorting is a cost-free reordering of the
delivered program rather than a density trade-off.

This is deliberately a different question from the program-headroom figure
(fig:case-headroom), which uses bounded_count + maximize_total_units to ask
"how many MORE units fit." Here we ask "what does a good stacking of the
delivered program look like."

Zones are re-numbered bottom-to-top (by plate, then mean unit size) so the
emitted tables read in physical stacking order.

Output:
    experiments/results/case_study_k8_stacking.csv
    experiments/results/case_study_k8_zones.csv
"""

import argparse
from pathlib import Path

from _common import out_path, write_machine_json   # noqa: F401 -- side-effect: sys.path
import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance         # fixed-count 814 program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args()

    write_machine_json("case_study_k8_stacking.py")

    cfg = build_instance(K=args.K, time_limit=args.time_limit)
    cfg.solver_objective = "vertical_sort"   # compact + larger units higher
    # F13-only overrides (do NOT touch the shared ksweep/headroom runs):
    #  - lift the per-floor unit caps so the large Main plate can hold a dense
    #    affordable mix (~21 small units/floor to fill 1380 m^2); at the
    #    default cap of 18 the Main plate is forced into large units and small
    #    units strand on the small upper plates, inverting the gradient;
    #  - use Gurobi, which reaches the compact (4-template) structural optimum
    #    that HiGHS leaves budget-bound at this instance (cf. Sec. 7.2 cross-
    #    check). Clear the HiGHS-only options so they are not passed to Gurobi.
    cfg.enumeration_max_count_total = 26
    cfg.max_count_per_type_per_floor = 12
    cfg.min_distinct_types_per_template = 2   # every zone reads as a mix
    cfg.solver_backend = "gurobi"
    cfg.solver_options = {}
    # Per-plate program availability: bar small units from the upper plates
    # ("no studios above the podium, no 1-BR in the crown"). This guarantees a
    # premium top and forces the affordable units to the base, independent of
    # the search; 3-BR stays available on every plate because 144 units cannot
    # fit in Trans+Crown alone.
    excluded = {"M": (), "T": ("studio",), "C": ("studio", "1B")}
    cfg.plates = [
        v2b.PlateClass(p.name, p.net_area, p.n_floors, area_tol=p.area_tol,
                       excluded_types=excluded.get(p.name, ()))
        for p in cfg.plates
    ]
    print(f"Solving K={args.K} (fixed-count, vertical_sort) ...")
    sol = v2b.solve(cfg)
    print(f"  {sol.status}  ({sol.solve_time_sec:.1f}s)  "
          f"units={sol.total_units}  zones_used={sol.n_zones_used}")

    if not sol.status.startswith("Optimal") or not sol.zones:
        raise SystemExit(
            f"Solve did not reach a usable optimum (status={sol.status}, "
            f"zones={sol.n_zones_used}); leaving existing CSVs untouched."
        )

    target_area = {t.name: t.target_area for t in cfg.types}
    _renumber_zones_bottom_to_top(sol, target_area)

    floors = sol.floors_dataframe()
    csv_path = out_path("case_study_k8_stacking.csv")
    floors.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(floors)} floors).")

    zones_csv = out_path("case_study_k8_zones.csv")
    sol.zones_dataframe().to_csv(zones_csv, index=False)
    print(f"Wrote {zones_csv} ({sol.n_zones_used} zones).")


def _renumber_zones_bottom_to_top(sol: v2b.Solution,
                                  target_area: dict) -> None:
    """Order zones in physical stacking order and renumber 1..n from the base.

    Plates already stack bottom-to-top by plate_index; within a plate (equal
    elevation in the objective) we order by mean unit size so the smallest
    units sit at the bottom of the plate and the largest at the top. This
    makes the per-floor table monotonic in unit size from floor 1 upward.

    Mean size uses TARGET areas (not the solver's realized areas) so the
    ordering matches the unit-size encoding F13 draws.
    """
    def mean_target_size(z: v2b.ZoneAssignment) -> float:
        units = sum(z.counts.values())
        if not units:
            return 0.0
        return sum(z.counts[n] * target_area[n] for n in z.counts) / units

    sol.zones.sort(key=lambda z: (z.plate_index, mean_target_size(z)))
    for new_index, z in enumerate(sol.zones, start=1):
        z.zone_index = new_index


if __name__ == "__main__":
    main()
