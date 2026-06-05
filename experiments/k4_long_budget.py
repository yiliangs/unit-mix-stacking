"""K=4 with a 1800s budget on the case-study instance.

Reviewer-requested follow-up to k11_long_budget. Two outcomes both
informative:

  - K=4 closes to feasibility at 1800s: only K=3 is structurally
    infeasible; the entire K >= 4 band is just budget-limited under
    the 300s HiGHS regime. The single-knob diagnostic's "n_zones +1"
    probe becomes a budget-limited false negative.

  - K=4 remains Not Solved at 1800s: stronger tooling-boundary
    diagnosis for the middle band. The K=11 long-budget closure to
    4 zones is then reached because LP relaxation gives more pruning
    headroom at larger K-cap, not because the integer feasibility
    problem itself is easier.
"""

import csv

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance


def main():
    write_machine_json("k4_long_budget.py")
    cfg = build_instance(K=4, time_limit=1800.0)
    sol = v2b.solve(cfg)
    print(
        f"K=4 (1800s): status={sol.status} time={sol.solve_time_sec:.1f}s "
        f"zones_used={sol.n_zones_used} units={sol.total_units} "
        f"dual_bound={sol.mip_dual_bound} gap_rel={sol.mip_gap_rel}"
    )
    row = {
        "K": 4,
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
    p = out_path("k4_long_budget.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"Wrote {p}.")


if __name__ == "__main__":
    main()
