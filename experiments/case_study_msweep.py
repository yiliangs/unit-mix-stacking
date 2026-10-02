"""Minimum-run sweep on the synthetic 55-floor case study.

Companion to case_study_ksweep.py, which is the m = 2 row and is not rerun
here. Same instance, baseline plate tolerance (0.5%), and HiGHS settings
(4 threads per solve); only the minimum run per template m changes. For each
m in {3, 4, 5, 6}, K runs from 3 to 11 and stops at the first K with a
feasible stacking.

Outcomes stay distinct: feasible, proven infeasible ("Infeasible" or
"no_count_vectors..."), and undetermined (anything else at the time limit).

Resumable: every cell is appended to the CSV as it completes; a rerun reuses
those rows unless --fresh is passed.

Outputs:
    experiments/results/case_study_msweep.csv
    experiments/results/machine_case_study_msweep.json
"""

import argparse
import csv
import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from _common import out_path, write_machine_json
from case_study_ksweep import build_instance

import unit_mix_solver_v2b as v2b


M_LEVELS = (3, 4, 5, 6)
K_LADDER = range(3, 12)
TIME_LIMIT_PER_CELL = 300.0
CSV_NAME = "case_study_msweep.csv"
FIELDS = ["m", "K", "status", "outcome", "zones_used", "floors_per_zone",
          "solve_time_sec", "objective", "mip_dual_bound", "mip_gap_rel",
          "mip_node_count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="Discard existing cells.")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT_PER_CELL,
                    help=f"Per-cell time limit (default {TIME_LIMIT_PER_CELL}s).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent solver processes, one per m (default 1).")
    args = ap.parse_args()

    write_machine_json("case_study_msweep.py")
    if args.fresh:
        out_path(CSV_NAME).unlink(missing_ok=True)
    done = _read_rows()
    if not out_path(CSV_NAME).exists():
        with open(out_path(CSV_NAME), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(m_ladder, m, args.time_limit,
                               [r for r in done if r["m"] == str(m)])
                   for m in M_LEVELS]
        for fut in as_completed(futures):
            m, rows = fut.result()
            print(f"m={m}: {len(rows)} new cells", flush=True)

    rows = _read_rows()
    rows.sort(key=lambda r: (int(r["m"]), int(r["K"])))
    with open(out_path(CSV_NAME), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path(CSV_NAME)}.")


def m_ladder(m: int, time_limit: float, done: List[Dict]):
    """Climb K for one m, appending each new cell to the CSV as it completes."""
    cache = {int(r["K"]): r for r in done}
    new = []
    for K in K_LADDER:
        row = cache.get(K)
        if row is None:
            row = _solve_cell(m, K, time_limit)
            _append_row(row)
            new.append(row)
            print(f"  m={m} K={K}: {row['status']} ({row['solve_time_sec']}s)",
                  flush=True)
        if row["outcome"] == "feasible":
            break
    return m, new


def _solve_cell(m: int, K: int, time_limit: float) -> Dict:
    cfg = dataclasses.replace(build_instance(K, time_limit=time_limit),
                              min_floors_per_zone=m)
    sol = v2b.solve(cfg)
    ok = sol.status.startswith("Optimal") and bool(sol.zones)
    if ok:
        outcome = "feasible"
    elif sol.status == "Infeasible" or sol.status.startswith("no_count_vectors"):
        outcome = "infeasible"
    else:
        outcome = "undetermined"
    return {
        "m": m, "K": K, "status": sol.status, "outcome": outcome,
        "zones_used": sol.n_zones_used,
        "floors_per_zone": "|".join(f"{z.plate_name}:{z.n_floors}"
                                    for z in sol.zones) if ok else "",
        "solve_time_sec": round(sol.solve_time_sec, 3),
        "objective": sol.objective,
        "mip_dual_bound": sol.mip_dual_bound,
        "mip_gap_rel": sol.mip_gap_rel,
        "mip_node_count": sol.mip_node_count,
    }


def _read_rows() -> List[Dict]:
    path = out_path(CSV_NAME)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_row(row: Dict) -> None:
    with open(out_path(CSV_NAME), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


if __name__ == "__main__":
    main()
