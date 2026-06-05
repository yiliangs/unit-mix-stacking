"""K=11 case-study cell under Gurobi at the 1800s budget.

Parallels k11_long_budget.py. The HiGHS K=11 row currently terminates
with a primal solution but no provably tight dual bound. Question:
does Gurobi close the gap inside 1800s?

Output: experiments/results/gurobi_k11.csv
"""

import csv

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance


GUROBI_OPTIONS = {
    "Threads": 4,
    "Symmetry": 2,
    "Presolve": 2,
    # NumericFocus left at default; raising it slowed easy cells dramatically
    # in an earlier run for no real numeric benefit.
}


def main():
    write_machine_json("gurobi_k11.py")
    cfg = build_instance(K=11, time_limit=1800.0)
    cfg.solver_backend = "gurobi"
    cfg.solver_options = dict(GUROBI_OPTIONS)
    sol = v2b.solve(cfg)
    print(
        f"Gurobi K=11 (1800s): status={sol.status} time={sol.solve_time_sec:.1f}s "
        f"zones_used={sol.n_zones_used} units={sol.total_units} "
        f"dual_bound={sol.mip_dual_bound} gap_rel={sol.mip_gap_rel}"
    )
    row = {
        "K": 11,
        "solver": "Gurobi",
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
    p = out_path("gurobi_k11.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"Wrote {p}.")


if __name__ == "__main__":
    main()
