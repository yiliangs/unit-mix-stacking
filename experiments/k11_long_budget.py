"""K=11 with a 1800s budget on the case-study instance.

Single cell. Question: does the K=11 result we already have (Optimal at the
300s budget, 814 units) close to provable optimality, or does it stay
budget-limited? Outcome is one row, one number; if it closes, we can
upgrade the K-sweep narrative from "open-source tooling boundary" to
"budget-limited primal search" at the high-K tail.
"""

import csv

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance


def main():
    write_machine_json("k11_long_budget.py")
    cfg = build_instance(K=11, time_limit=1800.0)
    sol = v2b.solve(cfg)
    print(
        f"K=11 (1800s): status={sol.status} time={sol.solve_time_sec:.1f}s "
        f"zones_used={sol.n_zones_used} units={sol.total_units} "
        f"dual_bound={sol.mip_dual_bound} gap_rel={sol.mip_gap_rel}"
    )
    row = {
        "K": 11,
        "time_limit": 1800.0,
        "status": sol.status,
        "solve_time_sec": round(sol.solve_time_sec, 3),
        "zones_used": sol.n_zones_used,
        "total_units": sol.total_units,
        "total_area": round(sol.total_area, 1),
        "objective": sol.objective,
        "mip_dual_bound": sol.mip_dual_bound,
        "mip_gap_rel": sol.mip_gap_rel,
        "mip_node_count": sol.mip_node_count,
    }
    p = out_path("k11_long_budget.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"Wrote {p}.")


if __name__ == "__main__":
    main()
