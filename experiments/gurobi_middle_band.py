"""Gurobi on the K in {4,5,6} cells where HiGHS times out.

Direct parallel to scip_middle_band.py: same instance from
case_study_ksweep.build_instance(), only solver_backend changes.

Gurobi parameter naming differs from HiGHS:
    HiGHS mip_heuristic_effort   -> Gurobi Heuristics (0..1)
    HiGHS presolve "on"          -> Gurobi Presolve 2 (aggressive)
    HiGHS mip_detect_symmetry    -> Gurobi Symmetry 2 (aggressive)
    HiGHS parallel/threads       -> Gurobi Threads
Defaults kept light: Gurobi is generally smart enough that aggressive
parameter tuning would over-fit the experiment to this instance.
"""

import csv

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance


GUROBI_OPTIONS = {
    "Threads": 4,
    "Symmetry": 2,   # templates within a plate are exchangeable
    "Presolve": 2,
    # NumericFocus left at default. An earlier run set NumericFocus=2 because
    # Gurobi warns about the model's [1e-13, 1e+03] matrix range, but that
    # turned K=8 (smoke: Optimal in 30s) into a 600s TimeLimit — the cure was
    # worse than the disease. Gurobi handles the range fine at default.
}


def main():
    write_machine_json("gurobi_middle_band.py")
    rows = []
    for K in (4, 5, 6):
        cfg = build_instance(K=K, time_limit=600.0)
        cfg.solver_backend = "gurobi"
        cfg.solver_options = dict(GUROBI_OPTIONS)
        print(f"  Gurobi K={K} ... ", end="", flush=True)
        sol = v2b.solve(cfg)
        print(
            f"{sol.status} ({sol.solve_time_sec:.1f}s) "
            f"units={sol.total_units} gap={sol.mip_gap_rel}"
        )
        rows.append({
            "K": K,
            "solver": "Gurobi",
            "time_limit": 600.0,
            "status": sol.status,
            "solve_time_sec": round(sol.solve_time_sec, 3),
            "zones_used": sol.n_zones_used,
            "total_units": sol.total_units,
            "total_area": round(sol.total_area, 1),
            "objective": sol.objective,
            "mip_dual_bound": sol.mip_dual_bound,
            "mip_gap_rel": sol.mip_gap_rel,
            "mip_node_count": sol.mip_node_count,
        })
    p = out_path("gurobi_middle_band.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {p}.")


if __name__ == "__main__":
    main()
