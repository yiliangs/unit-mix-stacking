"""SCIP on the K in {4,5,6} cells where HiGHS times out.

Reviewer ask: install an alternative open-source MIP solver (SCIP via
pyscipopt) and report what happens. If SCIP closes the middle band, the
"open-source toolchain times out" framing in section 7.2/8.1 is
strengthened or softened depending on the result. Either way, an honest
data point.

We use the same instance build_instance() from case_study_ksweep.py;
only solver_backend changes. SCIP_PY does not accept the HiGHS-specific
tuning options (mip_heuristic_effort etc.), so solver_options is wiped.
"""

import csv

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance


def main():
    write_machine_json("scip_middle_band.py")
    rows = []
    for K in (4, 5, 6):
        cfg = build_instance(K=K, time_limit=600.0)
        cfg.solver_backend = "scip"
        cfg.solver_options = None
        print(f"  SCIP K={K} ... ", end="", flush=True)
        sol = v2b.solve(cfg)
        print(f"{sol.status} ({sol.solve_time_sec:.1f}s) units={sol.total_units}")
        rows.append({
            "K": K,
            "solver": "SCIP",
            "time_limit": 600.0,
            "status": sol.status,
            "solve_time_sec": round(sol.solve_time_sec, 3),
            "zones_used": sol.n_zones_used,
            "total_units": sol.total_units,
            "total_area": round(sol.total_area, 1),
            "objective": sol.objective,
        })
    p = out_path("scip_middle_band.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {p}.")


if __name__ == "__main__":
    main()
