"""K=4 case-study cell deep probe under Gurobi.

K=4 is the only cell in the K-sweep where neither HiGHS (300s) nor
Gurobi (600s, default heuristic effort) finds a feasible primal. This
script asks: is K=4 truly infeasible at the tooling-tolerance
configuration, or is it merely extraordinarily hard for default
branch-and-bound heuristics to locate a primal?

Tactic: 1800s budget with MIPFocus=1 (emphasize finding feasible
solutions), Heuristics=0.9 (spend ~90% of node time on primal
heuristics), NoRelHeurTime=300 (300s of no-relaxation heuristics
before B&B starts).

If Gurobi still finds no primal here, the §7.2 "K=4 boundary" claim
is empirically very strong: even an aggressive commercial primal
search at 1800s cannot find a feasible point.

Output: experiments/results/gurobi_k4_probe.csv
"""

import csv

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance


PROBE_OPTIONS = {
    "Threads": 4,
    "Symmetry": 2,
    "Presolve": 2,
    "MIPFocus": 1,         # prioritize finding feasible solutions
    "Heuristics": 0.9,     # default 0.05; crank way up
    "NoRelHeurTime": 300,  # 5 minutes of no-relaxation heuristics first
}


def main():
    write_machine_json("gurobi_k4_probe.py")
    cfg = build_instance(K=4, time_limit=1800.0)
    cfg.solver_backend = "gurobi"
    cfg.solver_options = dict(PROBE_OPTIONS)
    print("K=4 deep primal probe (1800s, MIPFocus=1, Heuristics=0.9, "
          "NoRelHeurTime=300s):")
    sol = v2b.solve(cfg)
    print(
        f"  status={sol.status} time={sol.solve_time_sec:.1f}s "
        f"zones_used={sol.n_zones_used} units={sol.total_units} "
        f"gap={sol.mip_gap_rel} dual_bound={sol.mip_dual_bound}"
    )
    row = {
        "K": 4,
        "probe": "MIPFocus=1, Heuristics=0.9, NoRelHeurTime=300",
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
    p = out_path("gurobi_k4_probe.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"Wrote {p}.")


if __name__ == "__main__":
    main()
