"""Empirical LP-relaxation tightness of the V2b configuration IP,
under the BOUNDED-COUNT mode with a maximize-units objective.

Why bounded + maximize, not proportional + minimize-deviation:

  In proportional mode the LP can drive deviation to exactly zero by
  fractional assignment of count vectors to floors (z_LP = 0), while
  the MILP must absorb integer rounding (z_MILP > 0). The relative
  gap g = (z_MILP - z_LP) / |z_MILP| then pins to 1.0 by definition
  whenever z_MILP > 0, irrespective of how good the relaxation
  actually is. The metric is degenerate, not the method.

  Bounded-count mode with a maximize-units objective (the case-study
  framing of Section 7.4) gives both LP and MILP positive objective
  values, so g actually measures relaxation quality.

For each instance, solve both the LP relaxation and the full MILP,
compute the integrality gap

    g = (z_LP - z_MILP) / (|z_MILP| + epsilon),

(LP upper-bounds the maximization MILP, so z_LP >= z_MILP and g >= 0
by construction; we report it as a positive fraction).

Output: experiments/results/lp_gap.csv
"""

import argparse
import csv
import math
import time
from typing import Dict, List

from _common import out_path, write_machine_json

import pulp
import unit_mix_solver_v2b as v2b
from v2b_scaling import make_plates, make_types

EPSILON = 1e-9


def build_cfg(num_plates: int, F: int, T: int, K: int,
              time_limit: float) -> v2b.StackingConfig:
    """Bounded-count instance: each type has min_count = its proportional
    share of F * occupancy_per_floor as a contractual floor, max_count
    open; objective = maximize total units.

    occupancy_per_floor is approximated from typical plate area / mean
    unit area, yielding a non-trivial but achievable per-type floor.
    """
    types_base = make_types(T)
    plates = make_plates(num_plates, F)

    total_area = sum(p.net_area * p.n_floors for p in plates)
    mean_unit_area = sum(t.target_area for t in types_base) / T
    approx_total_units = int(total_area / mean_unit_area)
    # Floor each type at ~60% of its uniform-share allocation, so the
    # constraint is binding but feasible.
    per_type_floor = int(0.6 * approx_total_units / T)

    types = [
        v2b.UnitType(
            name=t.name,
            target_area=t.target_area,
            min_count=per_type_floor,
            area_tol=t.area_tol,
        )
        for t in types_base
    ]
    return v2b.StackingConfig(
        types=types, plates=plates, n_zones=K,
        min_floors_per_zone=2,
        max_count_per_type_per_floor=8,
        enumeration_max_count_total=18,
        pattern_simplicity_weight=1e-6,
        solver_objective="maximize_total_units",
        time_limit=time_limit, solver_msg=False,
        solver_options={
            "mip_heuristic_effort": 0.3,
            "presolve": "on",
        },
    )


def solve_relaxed(cfg: v2b.StackingConfig) -> float:
    """Solve the LP relaxation by monkey-patching LpVariable categories
    to Continuous during model construction. Returns the LP objective."""
    original_init = pulp.LpVariable.__init__

    def patched_init(self, name, lowBound=None, upBound=None,
                     cat="Continuous", *args, **kwargs):
        if cat == "Binary":
            cat = "Continuous"
            if lowBound is None:
                lowBound = 0
            if upBound is None:
                upBound = 1
        elif cat == "Integer":
            cat = "Continuous"
        return original_init(self, name=name, lowBound=lowBound,
                             upBound=upBound, cat=cat, *args, **kwargs)

    pulp.LpVariable.__init__ = patched_init
    try:
        sol = v2b.solve(cfg)
    finally:
        pulp.LpVariable.__init__ = original_init
    return sol.objective if sol.objective is not None else math.nan


def run_one(num_plates: int, F: int, T: int, K: int,
            time_limit: float) -> Dict:
    cfg = build_cfg(num_plates, F, T, K, time_limit)
    print(f"  (P={num_plates}, F={F}, T={T}, K={K}) ... ", end="", flush=True)

    sol_milp = v2b.solve(cfg)
    z_milp = sol_milp.objective  # = -total_units (we minimize -units)

    t0 = time.time()
    z_lp = solve_relaxed(cfg)
    lp_time = time.time() - t0

    # Convert to positive unit counts for legibility (objective is -units).
    u_milp = -z_milp if z_milp is not None else None
    u_lp = -z_lp if (z_lp is not None and not math.isnan(z_lp)) else None

    if u_milp is None or u_lp is None or math.isnan(u_lp):
        gap = math.nan
    else:
        # LP upper bound on max-units; gap is fraction by which LP overshoots MILP.
        gap = (u_lp - u_milp) / (abs(u_milp) + EPSILON)

    if u_milp is not None and u_lp is not None:
        print(f"MILP={u_milp} LP={u_lp:.1f} gap={gap:.4f} "
              f"({sol_milp.solve_time_sec:.1f}s + {lp_time:.1f}s LP)")
    elif u_milp is not None:
        print(f"MILP={u_milp} LP=nan ({sol_milp.solve_time_sec:.1f}s)")
    else:
        print(f"MILP={sol_milp.status} (no objective)")

    return {
        "num_plates": num_plates, "F": F, "T": T, "K": K,
        "units_milp": u_milp,
        "units_lp": u_lp,
        "integrality_gap": gap,
        "milp_time_sec": round(sol_milp.solve_time_sec, 3),
        "lp_time_sec": round(lp_time, 3),
        "milp_status": sol_milp.status,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--time-limit", type=float, default=120.0)
    args = parser.parse_args()

    if args.quick:
        battery = [(2, 40, 3, 4), (2, 55, 4, 5), (3, 55, 4, 5)]
    else:
        # 4 plates x 2 floor counts x 3 type counts x 2 K-offsets = 48 cells.
        # Skip P=1 with low T because pool too small after bound floors.
        battery = []
        for P in (1, 2, 3, 4):
            for F in (40, 70):
                for T in (3, 4, 5):
                    for k_offset in (0, 4):
                        battery.append((P, F, T, P + k_offset))

    write_machine_json("lp_tightness_battery.py")

    rows: List[Dict] = []
    for inst in battery:
        rows.append(run_one(*inst, args.time_limit))

    csv_path = out_path("lp_gap.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
