"""LP-relaxation tightness battery under Gurobi.

Mirror of lp_tightness_battery.py with the solver swapped to Gurobi.
Gurobi's barrier + simplex LP is widely considered more numerically
reliable than HiGHS in tight regimes; this rerun cross-checks the
§6.3 integrality-gap numbers.

Same instances, same objective (maximize_total_units in bounded mode),
same LP-relaxation monkeypatch.

Output: experiments/results/lp_gap_gurobi.csv
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

GUROBI_OPTIONS = {
    "Heuristics": 0.3,
    "Presolve": 2,
    "Threads": 4,
    "NumericFocus": 2,
}


def build_cfg(num_plates: int, F: int, T: int, K: int,
              time_limit: float) -> v2b.StackingConfig:
    types_base = make_types(T)
    plates = make_plates(num_plates, F)

    total_area = sum(p.net_area * p.n_floors for p in plates)
    mean_unit_area = sum(t.target_area for t in types_base) / T
    approx_total_units = int(total_area / mean_unit_area)
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
        solver_backend="gurobi",
        solver_options=dict(GUROBI_OPTIONS),
    )


def solve_relaxed(cfg: v2b.StackingConfig) -> float:
    """LP relaxation via the same Binary/Integer -> Continuous monkeypatch
    used in lp_tightness_battery.py."""
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
    z_milp = sol_milp.objective

    t0 = time.time()
    z_lp = solve_relaxed(cfg)
    lp_time = time.time() - t0

    u_milp = -z_milp if z_milp is not None else None
    u_lp = -z_lp if (z_lp is not None and not math.isnan(z_lp)) else None

    if u_milp is None or u_lp is None or math.isnan(u_lp):
        gap = math.nan
    else:
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
        "solver": "Gurobi",
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
        battery = []
        for P in (1, 2, 3, 4):
            for F in (40, 70):
                for T in (3, 4, 5):
                    for k_offset in (0, 4):
                        battery.append((P, F, T, P + k_offset))

    write_machine_json("gurobi_lp_gap.py")

    rows: List[Dict] = []
    for inst in battery:
        rows.append(run_one(*inst, args.time_limit))

    csv_path = out_path("lp_gap_gurobi.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
