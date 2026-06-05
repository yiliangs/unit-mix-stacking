"""Program-headroom battery across synthetic instances (§7.5).

Hardens the single-instance §7.4 headroom result against the reviewer
objection that "9.3-13.5%" is one data point. We sweep a small
distribution of synthetic residential-tower instances and report the
headroom each one admits, holding the workflow fixed.

Two axes of variation:
  - Building archetype: three plausible tower massings with distinct
    plate areas and floor divisors (so the Diophantine landscape
    differs, not just the magnitudes).
  - Program tightness rho: the consultant program's target-area
    footprint as a fraction of the nominal plate envelope, rho in
    {0.90, 0.94, 0.98}. Tighter programs (rho -> 1) leave less
    leftover envelope, hence less headroom; this dependence is the
    result, not a nuisance.

Semantics are identical to case_study_headroom.py: the consultant
counts become per-type minimums (bounded_count), the objective is
maximize_total_units, and headroom = (placed - consultant_floor) /
consultant_floor at the baseline plate tolerance.

The canonical case-study instance (exact 58/228/264/120/144 counts) is
included verbatim as a continuity anchor; it should reproduce the ~9.3%
baseline figure reported in §7.4.

Output: experiments/results/case_study_headroom_battery.csv
"""

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

from _common import out_path, write_machine_json

import unit_mix_solver_v2b as v2b
from case_study_ksweep import PLATE_TOL_BASELINE, HIGHS_OPTIONS


# Mix proportions and per-type target areas, fixed across all instances.
# Proportions are the case-study consultant mix (total 814 units); the
# area-weighted mean is ~78 m^2/unit.
MIX: List[Tuple[str, float, float]] = [
    # (name, target_area_m2, proportion)
    ("studio", 40.0,  58 / 814),
    ("1B",     55.0,  228 / 814),
    ("2B",     78.0,  264 / 814),
    ("2B+D",   92.0,  120 / 814),
    ("3B",     118.0, 144 / 814),
]
AREA_TOL = 0.05


@dataclass(frozen=True)
class Archetype:
    name: str
    plates: Tuple[Tuple[str, float, int], ...]  # (name, net_area, n_floors)

    @property
    def envelope(self) -> float:
        return sum(net * n for _, net, n in self.plates)


ARCHETYPES = [
    # Setback profile (the §7 case-study geometry): street-frontage main
    # plate, transition, crown. Envelope ~62,940 m^2.
    Archetype("setback", (("M", 1380.0, 37), ("T", 700.0, 12), ("C", 580.0, 6))),
    # Slab: larger uniform body, short transition and crown. Coarser
    # floor divisors (45/8/4). Envelope ~59,500 m^2.
    Archetype("slab", (("M", 1100.0, 45), ("T", 900.0, 8), ("C", 700.0, 4))),
    # Tapered tower: three substantial plate classes, finer crown
    # divisor (30/15/10). Envelope ~66,950 m^2.
    Archetype("tapered", (("M", 1500.0, 30), ("T", 1050.0, 15), ("C", 620.0, 10))),
]

RHO_VALUES = [0.90, 0.94, 0.98]


def consultant_counts(arch: Archetype, rho: float) -> Dict[str, int]:
    """Scale the fixed mix to a target footprint = rho * envelope."""
    mean_area = sum(area * prop for _, area, prop in MIX)
    n_total = rho * arch.envelope / mean_area
    return {name: max(1, round(prop * n_total)) for name, area, prop in MIX}


def build_instance(arch: Archetype, counts: Dict[str, int], K: int,
                   tau: float, time_limit: float) -> v2b.StackingConfig:
    types = [
        v2b.UnitType(name, target_area=area, min_count=counts[name], area_tol=AREA_TOL)
        for name, area, prop in MIX
    ]
    plates = [
        v2b.PlateClass(pname, net, nfl, area_tol=tau)
        for pname, net, nfl in arch.plates
    ]
    return v2b.StackingConfig(
        types=types, plates=plates, n_zones=K,
        min_floors_per_zone=2,
        max_count_per_type_per_floor=8,
        enumeration_max_count_total=18,
        pattern_simplicity_weight=1e-6,
        solver_objective="maximize_total_units",
        time_limit=time_limit, solver_msg=False,
        solver_options=HIGHS_OPTIONS,
    )


def run_one(label: str, arch: Archetype, counts: Dict[str, int], K: int,
            tau: float, rho: float, time_limit: float) -> Dict:
    cfg = build_instance(arch, counts, K, tau, time_limit)
    base = sum(counts.values())
    print(f"  {label:28s} rho={rho:.2f} floor={base:4d} ... ", end="", flush=True)
    sol = v2b.solve(cfg)
    placed = sol.total_units or 0
    headroom = placed - base if placed else 0
    print(f"{sol.status:18s} placed={placed:4d}  +{headroom:4d} "
          f"({headroom / base * 100:+.1f}%)  {sol.solve_time_sec:.1f}s")
    return {
        "label": label,
        "archetype": arch.name,
        "envelope_m2": round(arch.envelope, 0),
        "rho_target": rho,
        "K": K,
        "tau": tau,
        "consultant_floor": base,
        "status": sol.status,
        "feasible": sol.status.startswith("Optimal"),
        "placed_units": placed,
        "headroom_units": headroom,
        "headroom_pct": round(headroom / base * 100.0, 2) if base else 0.0,
        "total_area_m2": round(sol.total_area, 1) if placed else 0.0,
        "zones_used": sol.n_zones_used,
        "solve_time_sec": round(sol.solve_time_sec, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--tau", type=float, default=PLATE_TOL_BASELINE)
    parser.add_argument("--time-limit", type=float, default=180.0)
    args = parser.parse_args()

    write_machine_json("case_study_headroom_battery.py")

    print(f"Headroom battery (K={args.K}, tau={args.tau}, "
          f"maximize_total_units, {args.time_limit:.0f}s/instance):")

    rows: List[Dict] = []

    # Continuity anchor: the exact §7.4 consultant program on the setback
    # geometry. Should reproduce ~+9.3% at baseline tolerance.
    canonical = {"studio": 58, "1B": 228, "2B": 264, "2B+D": 120, "3B": 144}
    setback = ARCHETYPES[0]
    rho_canon = sum(canonical[n] * a for n, a, _ in MIX) / setback.envelope
    rows.append(run_one("canonical-setback", setback, canonical, args.K,
                        args.tau, round(rho_canon, 3), args.time_limit))

    # Battery: archetype x tightness.
    for arch in ARCHETYPES:
        for rho in RHO_VALUES:
            counts = consultant_counts(arch, rho)
            label = f"{arch.name}-rho{int(rho * 100)}"
            rows.append(run_one(label, arch, counts, args.K, args.tau,
                                rho, args.time_limit))

    csv_path = out_path("case_study_headroom_battery.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    feas = [r for r in rows if r["feasible"]]
    if feas:
        pcts = [r["headroom_pct"] for r in feas]
        print(f"\n{len(feas)}/{len(rows)} feasible. "
              f"Headroom range: {min(pcts):+.1f}% to {max(pcts):+.1f}% "
              f"(median {sorted(pcts)[len(pcts) // 2]:+.1f}%).")
    print(f"Wrote {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
