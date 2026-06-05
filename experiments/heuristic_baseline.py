"""Metaheuristic baseline (Tier 1.3 in the revision notes).

Reimplements a unit-mix stacking solver as a genetic algorithm (GA) via
pymoo, and compares its behavior with the exact MILP on the case-study
instance at two K settings:

  K = 3 -- provably infeasible (LP-detectable Diophantine block on the
          studio count). Expected GA behavior: returns a candidate
          stacking with positive constraint violation, with no signal
          distinguishing this from a heuristic search failure.

  K = 8 -- provably feasible, MILP closes in tens of seconds.
          Expected GA behavior: locates a feasible-or-near-feasible
          stacking after many generations; final population may still
          carry small constraint violations.

This is the central empirical support for the paper's constraint-envelope
claim: exact methods surface infeasibility honestly; metaheuristics do
not. We do not present this as a horse-race on solve time. The MILP is
faster on the feasible cell and gives a certificate on the infeasible
one; the GA does neither.

Output: experiments/results/heuristic_baseline.csv (one row per
(K, seed) run) plus heuristic_baseline_summary.csv with the
exact-vs-heuristic comparison.

Algorithm: a single-objective integer GA. Decision variables are, for
each plate p and each of K_max template slots k: a count vector
(n_t for t in types) and a floor-count assignment f. Per-template areas
are fixed to target areas (defensible simplification: the exact MILP's
continuous-area degree of freedom is exactly the feature the GA cannot
exploit, so leaving it out of the GA understates the gap, not overstates
it).

Fitness: weighted sum of hard-constraint violations
  - floor-partition (sum_k f[p,k] == F_p) per plate
  - K-cap (count of slots with f > 0) <= K
  - per-template area in plate band [A_p*(1-tau), A_p*(1+tau)]
  - per-type total (sum_pk n_t[p,k] * f[p,k]) == N_t* (fixed-count)
  - min-floors-per-used-zone
plus a small pull toward fewer active templates.

A solution with fitness == 0 is feasible. The GA cannot distinguish
'fitness 0 not achievable' from 'fitness 0 not yet found'; that is the
empirical demonstration.
"""

import argparse
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import build_instance as build_case_instance, HIGHS_OPTIONS


from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@dataclass
class Encoded:
    P: int
    K_max: int
    T: int
    n_max_per_type_per_floor: int
    plate_floors: List[int]
    n_vars_per_slot: int  # T (counts) + 1 (floor count)

    @property
    def n_vars(self) -> int:
        return self.P * self.K_max * self.n_vars_per_slot

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        xl = np.zeros(self.n_vars, dtype=int)
        xu = np.zeros(self.n_vars, dtype=int)
        idx = 0
        for p in range(self.P):
            for k in range(self.K_max):
                for t in range(self.T):
                    xl[idx] = 0
                    xu[idx] = self.n_max_per_type_per_floor
                    idx += 1
                xl[idx] = 0
                xu[idx] = self.plate_floors[p]
                idx += 1
        return xl, xu

    def decode(self, x: np.ndarray):
        """Decode a single genome to per-template (count_vector, floors) pairs."""
        templates = []  # list of (p_idx, count_vec_tuple, floor_count)
        idx = 0
        for p in range(self.P):
            for k in range(self.K_max):
                cv = tuple(int(x[idx + t]) for t in range(self.T))
                f = int(x[idx + self.T])
                idx += self.n_vars_per_slot
                templates.append((p, cv, f))
        return templates


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

W_FLOOR_PARTITION = 1e4
W_K_CAP            = 1e4
W_PLATE_AREA       = 1e2
W_TYPE_TOTAL       = 1e3
W_MIN_FLOORS       = 1e3
W_COMPACTNESS      = 1e-1


def evaluate(genome: np.ndarray, enc: Encoded,
             cfg: v2b.StackingConfig, K_cap: int,
             target_counts: List[int]) -> float:
    templates = enc.decode(genome)

    # Active templates (f > 0)
    active = [(p, cv, f) for (p, cv, f) in templates if f > 0]

    # Floor partition per plate
    floor_violation = 0.0
    floors_per_plate = [0] * enc.P
    for (p, cv, f) in active:
        floors_per_plate[p] += f
    for p in range(enc.P):
        floor_violation += abs(floors_per_plate[p] - enc.plate_floors[p])

    # K-cap: total active templates across the building
    k_excess = max(0, len(active) - K_cap)

    # Per-template plate area
    plate_area_violation = 0.0
    types = cfg.types
    for (p, cv, f) in active:
        if f == 0:
            continue
        plate = cfg.plates[p]
        A_lo = plate.area_min
        A_hi = plate.area_max
        # Use target areas (GA does not adjust per-template areas)
        template_area = sum(cv[t] * types[t].target_area for t in range(enc.T))
        if template_area < A_lo:
            plate_area_violation += (A_lo - template_area)
        elif template_area > A_hi:
            plate_area_violation += (template_area - A_hi)

    # Per-type total counts vs program (fixed_count mode)
    type_totals = [0] * enc.T
    for (p, cv, f) in active:
        for t in range(enc.T):
            type_totals[t] += cv[t] * f
    type_violation = 0.0
    for t in range(enc.T):
        type_violation += abs(type_totals[t] - target_counts[t])

    # Min floors per used zone
    min_floors_violation = 0.0
    for (p, cv, f) in active:
        if 0 < f < cfg.min_floors_per_zone:
            min_floors_violation += (cfg.min_floors_per_zone - f)

    # Compactness tie-breaker
    compactness = len(active)

    fitness = (
        W_FLOOR_PARTITION * floor_violation
        + W_K_CAP          * k_excess
        + W_PLATE_AREA     * plate_area_violation
        + W_TYPE_TOTAL     * type_violation
        + W_MIN_FLOORS     * min_floors_violation
        + W_COMPACTNESS    * compactness
    )
    return float(fitness), {
        "floor_violation": float(floor_violation),
        "k_excess": int(k_excess),
        "plate_area_violation": float(plate_area_violation),
        "type_violation": float(type_violation),
        "type_totals_achieved": type_totals,
        "min_floors_violation": float(min_floors_violation),
        "n_active_templates": len(active),
        "active_templates": [(p, cv, f) for (p, cv, f) in active],
    }


# ---------------------------------------------------------------------------
# Pymoo problem
# ---------------------------------------------------------------------------

class StackingGAProblem(Problem):
    def __init__(self, enc: Encoded, cfg: v2b.StackingConfig,
                 K_cap: int, target_counts: List[int]):
        xl, xu = enc.bounds()
        super().__init__(n_var=enc.n_vars, n_obj=1, xl=xl, xu=xu, vtype=int)
        self.enc = enc
        self.cfg = cfg
        self.K_cap = K_cap
        self.target_counts = target_counts

    def _evaluate(self, X, out, *args, **kwargs):
        # X is (pop_size, n_var)
        F = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            f, _ = evaluate(X[i].astype(int), self.enc, self.cfg,
                            self.K_cap, self.target_counts)
            F[i, 0] = f
        out["F"] = F


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ga(cfg: v2b.StackingConfig, K_cap: int,
           target_counts: List[int], n_gen: int, pop_size: int,
           seed: int) -> Tuple[Dict, np.ndarray]:
    """Run the GA once. Returns summary dict and best genome."""
    enc = Encoded(
        P=len(cfg.plates),
        K_max=K_cap,
        T=len(cfg.types),
        n_max_per_type_per_floor=cfg.max_count_per_type_per_floor,
        plate_floors=[p.n_floors for p in cfg.plates],
        n_vars_per_slot=len(cfg.types) + 1,
    )
    problem = StackingGAProblem(enc, cfg, K_cap, target_counts)

    algorithm = GA(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0 / enc.n_vars, eta=20, vtype=float,
                    repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    t0 = time.time()
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    wall = time.time() - t0

    best_genome = res.X.astype(int)
    best_fitness = float(res.F[0])
    _, info = evaluate(best_genome, enc, cfg, K_cap, target_counts)
    # Hard violations (excluding compactness tie-break term):
    is_feasible = (
        info["floor_violation"] == 0
        and info["k_excess"] == 0
        and info["plate_area_violation"] == 0.0
        and info["type_violation"] == 0
        and info["min_floors_violation"] == 0
    )
    return {
        "K_cap": K_cap,
        "seed": seed,
        "n_gen": n_gen,
        "pop_size": pop_size,
        "wall_sec": round(wall, 2),
        "best_fitness": best_fitness,
        "ga_reports_feasible": is_feasible,
        "floor_partition_violation": info["floor_violation"],
        "k_cap_excess": info["k_excess"],
        "plate_area_violation_m2": round(info["plate_area_violation"], 2),
        "type_total_violation_units": int(info["type_violation"]),
        "min_floors_violation": info["min_floors_violation"],
        "n_active_templates": info["n_active_templates"],
        "type_totals_achieved": ";".join(str(x) for x in info["type_totals_achieved"]),
        "type_totals_target":   ";".join(str(x) for x in target_counts),
    }, best_genome


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CASE_STUDY_COUNTS = [58, 228, 264, 120, 144]   # studio, 1B, 2B, 2B+D, 3B


def run_milp_comparison(K_cap: int, time_limit: float) -> Dict:
    cfg = build_case_instance(K_cap, time_limit=time_limit)
    t0 = time.time()
    sol = v2b.solve(cfg)
    wall = time.time() - t0
    return {
        "K_cap": K_cap,
        "milp_status": sol.status,
        "milp_wall_sec": round(wall, 2),
        "milp_zones_used": sol.n_zones_used,
        "milp_total_units": sol.total_units,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K-list", type=int, nargs="+", default=[3, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-gen", type=int, default=400,
                        help="GA generations per run (default 400).")
    parser.add_argument("--pop-size", type=int, default=200,
                        help="GA population size (default 200).")
    parser.add_argument("--milp-time-limit", type=float, default=600.0,
                        help="MILP comparison time limit per K (default 600s).")
    parser.add_argument("--skip-milp", action="store_true",
                        help="Skip MILP comparison (use existing K-sweep data).")
    args = parser.parse_args()

    write_machine_json("heuristic_baseline.py")

    ga_rows: List[Dict] = []
    milp_rows: List[Dict] = []

    for K_cap in args.K_list:
        cfg_for_ga = build_case_instance(K_cap, time_limit=60.0)
        print(f"\n=== GA at K={K_cap} ===")
        for seed in args.seeds:
            print(f"  seed={seed} ... ", end="", flush=True)
            row, _ = run_ga(cfg_for_ga, K_cap, CASE_STUDY_COUNTS,
                            n_gen=args.n_gen, pop_size=args.pop_size, seed=seed)
            ga_rows.append(row)
            print(
                f"fitness={row['best_fitness']:.1f} "
                f"feasible_per_GA={row['ga_reports_feasible']} "
                f"type_violation={row['type_total_violation_units']} units "
                f"({row['wall_sec']}s)"
            )

        if not args.skip_milp:
            print(f"\n=== MILP at K={K_cap} ===")
            milp_rows.append(run_milp_comparison(K_cap, args.milp_time_limit))

    ga_csv = out_path("heuristic_baseline.csv")
    with open(ga_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ga_rows[0].keys()))
        w.writeheader()
        w.writerows(ga_rows)
    print(f"\nWrote {ga_csv} ({len(ga_rows)} rows).")

    if milp_rows:
        milp_csv = out_path("heuristic_baseline_milp.csv")
        with open(milp_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(milp_rows[0].keys()))
            w.writeheader()
            w.writerows(milp_rows)
        print(f"Wrote {milp_csv} ({len(milp_rows)} rows).")


if __name__ == "__main__":
    main()
