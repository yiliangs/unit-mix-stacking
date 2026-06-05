"""V1 (McCormick direct MILP) versus V2 (count-vector configuration IP).

Tier 1.4 revision: per-cell replication across solver random seeds and
sensitivity sweep over per-type tolerance sigma_t in {0.02, 0.05, 0.10}.

Per-cell run shape:
  - 3 solver random seeds (HiGHS `random_seed`)
  - sigma_t in {0.02, 0.05, 0.10}, applied uniformly to all types
  - V1 dual gap at timeout reported when status is Optimal_TimeLimit

Output: experiments/results/v1_vs_v2.csv (one row per
(T, F, K, sigma, seed) cell, with V1 and V2 paired).

The paper's headline contrast remains the per-cell median; the full
replication and sigma sensitivity is supplementary material.
"""

import argparse
import csv
import time
from typing import Dict, List

from _common import out_path, write_machine_json

import unit_mix_solver_v1 as v1
import unit_mix_solver_v2 as v2


# ---------------------------------------------------------------------------
# Instance battery
# ---------------------------------------------------------------------------

def type_set(T: int) -> List:
    if T == 2:
        spec = [("studio", 40.0), ("2B", 80.0)]
    elif T == 3:
        spec = [("studio", 40.0), ("1B", 60.0), ("2B", 90.0)]
    elif T == 4:
        spec = [("studio", 40.0), ("1B", 55.0), ("2B", 80.0), ("3B", 120.0)]
    elif T == 5:
        spec = [("studio", 40.0), ("1B", 55.0), ("2B", 78.0),
                ("3B", 115.0), ("4B", 180.0)]
    elif T == 6:
        spec = [("studio", 40.0), ("1B", 55.0), ("2B", 78.0),
                ("2B+D", 92.0), ("3B", 118.0), ("4B", 195.0)]
    else:
        raise ValueError(f"unsupported T={T}")
    p = 1.0 / T
    return spec, p


def make_v1_cfg(T: int, F: int, K: int, sigma: float, seed: int,
                time_limit: float = 300.0):
    spec, p = type_set(T)
    types = [v1.UnitType(n, target_area=a, target_proportion=p, area_tol=sigma)
             for (n, a) in spec]
    return v1.StackingConfig(
        types=types, n_floors=F, floor_net_area=700.0,
        floor_area_tol=0.02, n_zones=K,
        min_floors_per_zone=2, proportion_hard_band=0.05,
        max_count_per_type_per_zone=12,
        deviation_weight=1000.0, area_weight=1.0,
        time_limit=time_limit, solver_msg=False,
        solver_options={"random_seed": seed, "presolve": "on", "threads": 1},
    )


def make_v2_cfg(T: int, F: int, K: int, sigma: float, seed: int,
                time_limit: float = 300.0):
    spec, p = type_set(T)
    types = [v2.UnitType(n, target_area=a, target_proportion=p, area_tol=sigma)
             for (n, a) in spec]
    return v2.StackingConfig(
        types=types, n_floors=F, floor_net_area=700.0,
        floor_area_tol=0.02, n_zones=K,
        min_floors_per_zone=2, proportion_hard_band=0.05,
        max_count_per_type_per_floor=12,
        deviation_weight=1000.0, time_limit=time_limit,
        solver_msg=False,
        solver_options={"random_seed": seed, "presolve": "on", "threads": 1},
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(T: int, F: int, K: int, sigma: float, seed: int,
            time_limit: float) -> Dict:
    print(f"  (T={T}, F={F}, K={K}, sigma={sigma:.2f}, seed={seed})", flush=True)
    row = {"T": T, "F": F, "K": K, "sigma": sigma, "seed": seed,
           "time_limit": time_limit}

    # V1
    sol_v1 = v1.solve(make_v1_cfg(T, F, K, sigma, seed, time_limit=time_limit))
    row["v1_status"] = sol_v1.status
    row["v1_time_sec"] = round(sol_v1.solve_time_sec, 3)
    row["v1_objective"] = sol_v1.objective
    row["v1_zones_used"] = sol_v1.n_zones_used
    row["v1_total_units"] = sol_v1.total_units
    row["v1_mip_dual_bound"] = sol_v1.mip_dual_bound
    row["v1_mip_gap_rel"] = sol_v1.mip_gap_rel
    print(f"    V1 {sol_v1.status} in {sol_v1.solve_time_sec:.1f}s "
          f"(gap={sol_v1.mip_gap_rel})", flush=True)

    # V2
    sol_v2 = v2.solve(make_v2_cfg(T, F, K, sigma, seed, time_limit=time_limit))
    row["v2_status"] = sol_v2.status
    row["v2_time_sec"] = round(sol_v2.solve_time_sec, 3)
    row["v2_objective"] = sol_v2.objective
    row["v2_zones_used"] = sol_v2.n_zones_used
    row["v2_total_units"] = sol_v2.total_units
    row["v2_cv_count"] = sol_v2.n_count_vectors_enumerated
    row["v2_mip_dual_bound"] = sol_v2.mip_dual_bound
    row["v2_mip_gap_rel"] = sol_v2.mip_gap_rel
    print(f"    V2 {sol_v2.status} in {sol_v2.solve_time_sec:.1f}s", flush=True)

    if sol_v1.solve_time_sec > 0 and sol_v2.solve_time_sec > 0:
        row["speedup"] = round(sol_v1.solve_time_sec / sol_v2.solve_time_sec, 2)
    else:
        row["speedup"] = None
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test subset (T in {2,3,4}, single seed, sigma=0.05).")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="Per-solve time limit, seconds.")
    parser.add_argument("--T-list", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--F-list", type=int, nargs="+", default=[70])
    parser.add_argument("--K-list", type=int, nargs="+", default=[3])
    parser.add_argument("--sigma-list", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=str, default="v1_vs_v2.csv",
                        help="Output CSV filename (in results/).")
    args = parser.parse_args()

    if args.quick:
        battery = [(T, 30, 3, 0.05, 42) for T in (2, 3, 4)]
    else:
        # 5 T x 1 F x 1 K x 3 sigma x 3 seeds = 45 cells. Each V1 cell at
        # 300s timeout = 13.5k sec max for V1 alone. V2 adds <5min total.
        # ~3.75h wall for the full battery with V1 mostly timing out.
        battery = [(T, F, K, sigma, seed)
                   for T in args.T_list
                   for F in args.F_list
                   for K in args.K_list
                   for sigma in args.sigma_list
                   for seed in args.seeds]

    write_machine_json("v1_vs_v2_benchmark.py")

    rows: List[Dict] = []
    for cell in battery:
        rows.append(run_one(*cell, args.time_limit))
        # Stream partial results to disk so a crash doesn't lose work
        csv_path = out_path(args.output)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\nWrote {out_path(args.output)} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
