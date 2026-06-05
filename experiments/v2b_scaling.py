"""V2b scaling characterization.

Sweeps plate count, total floor count, type count, and K-cap on V2b and
records enumeration time, MILP solve time, total time, and pool size.
Output: experiments/results/v2b_scaling.csv. Generates the data behind
Table 2 and Figure 2 (Section 6.2).
"""

import argparse
import csv
import time
from typing import Dict, List

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b


PLATE_AREAS = [1400.0, 700.0, 580.0, 580.0, 580.0]


def make_plates(num_plates: int, total_floors: int) -> List:
    """Distribute total_floors across num_plates with the first plate
    taking the majority share."""
    if num_plates == 1:
        return [v2b.PlateClass("M", PLATE_AREAS[0], total_floors, area_tol=0.02)]
    # First plate: 60-70%, remainder split evenly across the others
    main_floors = int(round(0.65 * total_floors))
    remaining = total_floors - main_floors
    per_other = remaining // (num_plates - 1)
    leftover = remaining - per_other * (num_plates - 1)
    plates = [v2b.PlateClass("M", PLATE_AREAS[0], main_floors, area_tol=0.02)]
    for i in range(1, num_plates):
        f_i = per_other + (1 if i <= leftover else 0)
        plates.append(v2b.PlateClass(f"S{i}", PLATE_AREAS[i], f_i, area_tol=0.02))
    return plates


def make_types(T: int):
    spec = [
        ("studio", 40.0),
        ("1B", 55.0),
        ("2B", 78.0),
        ("2B+D", 92.0),
        ("3B", 118.0),
        ("4B", 195.0),
    ][:T]
    p = 1.0 / T
    return [v2b.UnitType(n, target_area=a, target_proportion=p, area_tol=0.05)
            for (n, a) in spec]


def run_one(num_plates: int, F: int, T: int, K: int,
            time_limit: float) -> Dict:
    print(f"  (P={num_plates}, F={F}, T={T}, K={K})", flush=True)
    types = make_types(T)
    plates = make_plates(num_plates, F)
    # pattern_simplicity_weight=0.0 keeps the objective as pure deviation, so
    # solve-time measurements are not confounded by a template-count
    # tie-breaker. The case study (fixed_count mode) uses 1e-6 instead.
    # max_count_per_type_per_floor=8 and enumeration_max_count_total=18
    # keep the count-vector pool tractable across plate sizes; without
    # these caps the 1400 m^2 plate at T=6 generates pools of 10^5+ vectors.
    cfg = v2b.StackingConfig(
        types=types, plates=plates, n_zones=K,
        min_floors_per_zone=2, proportion_hard_band=0.05,
        max_count_per_type_per_floor=8,
        enumeration_max_count_total=18,
        deviation_weight=1000.0, pattern_simplicity_weight=0.0,
        time_limit=time_limit, solver_msg=False,
    )

    t_enum_start = time.time()
    # Enumeration is performed inside solve(), so we measure end-to-end and
    # separately re-run enumeration alone for a clean timing.
    from unit_mix_solver_v2b import _enumerate_count_vectors_for_plate
    pool_sizes = []
    for plate in plates:
        if plate.n_floors == 0:
            pool_sizes.append(0)
            continue
        cvs = _enumerate_count_vectors_for_plate(cfg, plate)
        pool_sizes.append(len(cvs))
    enum_time = time.time() - t_enum_start
    pool_total = sum(pool_sizes)

    sol = v2b.solve(cfg)

    return {
        "num_plates": num_plates, "F": F, "T": T, "K": K,
        "status": sol.status,
        "enum_time_sec": round(enum_time, 3),
        "milp_time_sec": round(max(0.0, sol.solve_time_sec - enum_time), 3),
        "total_time_sec": round(sol.solve_time_sec, 3),
        "pool_total": pool_total,
        "pool_sizes": ";".join(str(s) for s in pool_sizes),
        "zones_used": sol.n_zones_used,
        "total_units": sol.total_units,
        "objective": sol.objective,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--time-limit", type=float, default=120.0)
    args = parser.parse_args()

    if args.quick:
        battery = [
            (1, 40, 3, 3),
            (2, 55, 4, 4),
            (3, 55, 4, 5),
        ]
    else:
        # Trimmed battery: full P sweep and T sweep at floor and K extremes.
        # 5 plates x 2 floor counts x 4 type counts x 2 K-offsets = 80 instances.
        # Covers all four scaling dimensions; F=55 and K-offset=2 are
        # intermediate values that would only fill in curves.
        battery = []
        for P in (1, 2, 3, 4, 5):
            for F in (40, 70):
                for T in (3, 4, 5, 6):
                    for k_offset in (0, 4):
                        K = P + k_offset
                        battery.append((P, F, T, K))

    write_machine_json("v2b_scaling.py")

    rows: List[Dict] = []
    for instance in battery:
        rows.append(run_one(*instance, args.time_limit))

    csv_path = out_path("v2b_scaling.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
