"""
Unit Mix Stacking Solver v2-b — Heterogeneous Floor Plates
==========================================================

Extension of v2 that allows the building to be composed of multiple
plate classes, each with its own net area and floor count. For example:

    plates = [
        PlateClass("base",       net_area=720.0, n_floors=45),
        PlateClass("transition", net_area=650.0, n_floors=5),
        PlateClass("crown",      net_area=580.0, n_floors=10),
    ]

Mathematical invariants preserved from v2
-----------------------------------------
* Count-vector enumeration linearizes the bilinear n*a coupling.
* No Big-M product expansions are required.
* Enumeration is performed independently per plate; the MILP grows
  linearly in the number of plate classes.

What changes vs v2
------------------
* Templates are plate-specific: a (plate i, pattern p) pair.
* Floor-count constraint applies per plate.
* Area band is per plate; each plate's count vectors are enumerated
  against its own area limits.
* K (distinct-template cap) applies building-wide:
    sum_{(i,p)} y_{i,p}  <=  K
* Building proportion totals aggregate across all plates.

Backward compatibility
----------------------
make_uniform_config() builds a single-plate StackingConfig that
reproduces v2 input/output exactly.

Solver: PuLP; HiGHS by default (CBC is PuLP's bundled fallback, not used
for any reported number).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pulp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitType:
    """A unit type with target area and one of three count specifications.

    Count modes (mutually exclusive, all types in a StackingConfig share one):

      1. PROPORTIONAL — set `target_proportion`. Solver chooses total units;
         soft L1 deviation from ratio + hard band (`proportion_hard_band`).

      2. FIXED_COUNT — set `fixed_count`. Hard equality on building total.

      3. BOUNDED_COUNT — set any of `min_count`, `max_count`, `target_count`.
         Building total must satisfy `min_count <= total <= max_count` (either
         bound optional, defaulting to 0 / unbounded). If `target_count` is
         given, soft L1 deviation from target inside the band is penalized.

    Architectural note: bounded mode is the right framing for most real
    projects — "at least 70 3B units (inclusionary), at most 320 2B units
    (market saturation), ideally 310" — and was added because fixed-count
    mode is too brittle when plate areas are tightly locked.
    """
    name: str
    target_area: float

    # PROPORTIONAL mode
    target_proportion: Optional[float] = None

    # FIXED_COUNT mode (shorthand for min=max=target)
    fixed_count: Optional[int] = None

    # BOUNDED_COUNT mode (any combination)
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    target_count: Optional[int] = None  # soft pull inside [min, max]

    area_tol: float = 0.05

    @property
    def area_min(self) -> float:
        return self.target_area * (1.0 - self.area_tol)

    @property
    def area_max(self) -> float:
        return self.target_area * (1.0 + self.area_tol)

    @property
    def _has_proportional(self) -> bool:
        return self.target_proportion is not None

    @property
    def _has_fixed(self) -> bool:
        return self.fixed_count is not None

    @property
    def _has_bounded(self) -> bool:
        return any(v is not None for v in (self.min_count, self.max_count, self.target_count))


@dataclass(frozen=True)
class PlateClass:
    """A contiguous block of floors sharing a net area."""
    name: str
    net_area: float
    n_floors: int
    area_tol: float = 0.02
    # Unit-type names barred from this plate (per-plate program availability,
    # e.g. no studios in the crown). Empty = all types allowed. Their count is
    # forced to zero in this plate's count-vector enumeration.
    excluded_types: Tuple[str, ...] = ()

    @property
    def area_min(self) -> float:
        return self.net_area * (1.0 - self.area_tol)

    @property
    def area_max(self) -> float:
        return self.net_area * (1.0 + self.area_tol)


@dataclass
class StackingConfig:
    types: List[UnitType]
    plates: List[PlateClass]
    n_zones: int = 3                        # K; cap on # distinct templates building-wide
    enforce_exact_K: bool = False
    min_floors_per_zone: int = 2
    proportion_hard_band: float = 0.05       # used only in PROPORTIONAL mode
    max_count_per_type_per_floor: int = 15
    # Minimum number of distinct unit types per template (count vector). 1 = no
    # restriction; 2 forbids single-type floors so every zone reads as a mix.
    min_distinct_types_per_template: int = 1
    deviation_weight: float = 1000.0
    pattern_simplicity_weight: float = 0.0
    solver_msg: bool = False
    time_limit: Optional[float] = 120.0
    enumeration_max_count_total: int = 25
    solver_backend: str = "highs"  # "highs" (default) or "cbc"
    # "minimize_deviation" (default) uses the mode-appropriate deviation
    # objective. "maximize_total_units" minimizes -total_units instead,
    # subject to the mode's constraints; only meaningful in bounded_count
    # mode where the total is not pinned by the constraint set.
    solver_objective: str = "minimize_deviation"
    # Optional HiGHS solver options passed to pulp.HiGHS(**solver_options).
    # Useful keys: mip_heuristic_effort (0..1, default 0.05),
    # mip_feasibility_tolerance, presolve ("on"/"off"/"choose"),
    # mip_detect_symmetry (bool), parallel ("on"/"off"/"choose").
    solver_options: Optional[Dict[str, object]] = None

    def __post_init__(self):
        # Mode is determined by which count fields are populated.
        # All types must use the SAME mode.
        flags = [(t._has_proportional, t._has_fixed, t._has_bounded) for t in self.types]
        has_prop = [f[0] for f in flags]
        has_fixed = [f[1] for f in flags]
        has_bound = [f[2] for f in flags]

        # Reject ambiguous combinations on any single type
        for t, (p, e, b) in zip(self.types, flags):
            if sum([p, e, b]) > 1:
                raise ValueError(
                    f"UnitType '{t.name}' specifies more than one count mode "
                    f"(proportional={p}, fixed={e}, bounded={b}). Pick one."
                )

        modes_used = [name for name, used in
                      [("proportional", any(has_prop)),
                       ("fixed_count", any(has_fixed)),
                       ("bounded_count", any(has_bound))] if used]
        if len(modes_used) > 1:
            raise ValueError(
                f"Mixed modes not supported across types: {modes_used}. "
                f"All UnitTypes must use the same count mode."
            )
        if not modes_used:
            raise ValueError(
                "No count specification given. Set one of target_proportion, "
                "fixed_count, or (min_count/max_count/target_count) on every UnitType."
            )

        # Per-mode validation
        if all(has_fixed):
            self._mode = "fixed_count"
            for t in self.types:
                if t.fixed_count < 0:
                    raise ValueError(f"UnitType '{t.name}': fixed_count < 0.")

        elif all(has_prop):
            self._mode = "proportional"
            s = sum(t.target_proportion for t in self.types)
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"Target proportions must sum to 1.0, got {s:.4f}")

        elif all(has_bound):
            self._mode = "bounded_count"
            for t in self.types:
                lo = t.min_count if t.min_count is not None else 0
                hi = t.max_count if t.max_count is not None else None
                tg = t.target_count
                if lo < 0:
                    raise ValueError(f"UnitType '{t.name}': min_count < 0.")
                if hi is not None and hi < lo:
                    raise ValueError(f"UnitType '{t.name}': max_count < min_count.")
                if tg is not None:
                    if tg < lo or (hi is not None and tg > hi):
                        raise ValueError(
                            f"UnitType '{t.name}': target_count {tg} outside "
                            f"[min={lo}, max={hi}]."
                        )
        else:
            missing = [t.name for t, ok in zip(self.types, [
                p or e or b for p, e, b in flags
            ]) if not ok]
            raise ValueError(
                f"Mode could not be determined: not all types share a single "
                f"count mode. Types missing any count spec: {missing}"
            )

        if not self.plates:
            raise ValueError("At least one plate class is required.")
        for plate in self.plates:
            if plate.n_floors < 0:
                raise ValueError(f"Plate '{plate.name}' has negative n_floors.")
            if 0 < plate.n_floors < self.min_floors_per_zone:
                raise ValueError(
                    f"Plate '{plate.name}' has {plate.n_floors} floors, "
                    f"but min_floors_per_zone is {self.min_floors_per_zone}. "
                    f"Plate is too small to host any zone."
                )

    @property
    def mode(self) -> str:
        """One of 'proportional', 'fixed_count', 'bounded_count'."""
        return self._mode

    @property
    def total_floors(self) -> int:
        return sum(p.n_floors for p in self.plates)

    @property
    def n_active_plates(self) -> int:
        return sum(1 for p in self.plates if p.n_floors > 0)


@dataclass(frozen=True)
class CountVector:
    counts: Tuple[int, ...]

    def total(self) -> int:
        return sum(self.counts)


@dataclass
class ZoneAssignment:
    zone_index: int
    plate_name: str
    plate_index: int
    n_floors: int
    counts: Dict[str, int]
    areas: Dict[str, float]
    template_total_area: float
    template_total_units: int


@dataclass
class Solution:
    status: str
    objective: Optional[float]
    zones: List[ZoneAssignment] = field(default_factory=list)
    achieved_counts: Dict[str, int] = field(default_factory=dict)
    achieved_proportions: Dict[str, float] = field(default_factory=dict)
    target_proportions: Dict[str, float] = field(default_factory=dict)
    proportion_deviation: Dict[str, float] = field(default_factory=dict)
    total_units: int = 0
    total_area: float = 0.0
    n_count_vectors_enumerated: int = 0
    n_count_vectors_per_plate: Dict[str, int] = field(default_factory=dict)
    n_zones_used: int = 0
    n_zones_requested: int = 0
    solve_time_sec: float = 0.0
    mip_dual_bound: Optional[float] = None
    mip_gap_rel: Optional[float] = None
    mip_node_count: Optional[int] = None

    def zones_dataframe(self) -> pd.DataFrame:
        if not self.zones:
            return pd.DataFrame()
        type_names = list(self.target_proportions.keys())
        rows = []
        for z in self.zones:
            row = {
                "zone": z.zone_index,
                "plate": z.plate_name,
                "n_floors": z.n_floors,
                "template_total_units": z.template_total_units,
                "template_total_area": round(z.template_total_area, 2),
            }
            for n in type_names:
                row[f"{n}_count"] = z.counts.get(n, 0)
                row[f"{n}_area"] = round(z.areas.get(n, 0.0), 2) if z.counts.get(n, 0) > 0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    def floors_dataframe(self) -> pd.DataFrame:
        """One row per floor, grouped by plate then zone."""
        if not self.zones:
            return pd.DataFrame()
        type_names = list(self.target_proportions.keys())
        zones_by_plate: Dict[int, List[ZoneAssignment]] = defaultdict(list)
        for z in self.zones:
            zones_by_plate[z.plate_index].append(z)
        rows = []
        floor_idx = 1
        for plate_idx in sorted(zones_by_plate.keys()):
            for z in zones_by_plate[plate_idx]:
                for _ in range(z.n_floors):
                    row = {
                        "floor": floor_idx,
                        "plate": z.plate_name,
                        "zone": z.zone_index,
                        "total_area": round(z.template_total_area, 2),
                    }
                    for n in type_names:
                        row[f"{n}_count"] = z.counts.get(n, 0)
                        row[f"{n}_area"] = round(z.areas.get(n, 0.0), 2) if z.counts.get(n, 0) > 0 else 0.0
                    rows.append(row)
                    floor_idx += 1
        return pd.DataFrame(rows)

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for name in self.target_proportions:
            rows.append({
                "type": name,
                "target_pct": self.target_proportions[name] * 100,
                "achieved_pct": self.achieved_proportions.get(name, 0.0) * 100,
                "deviation_pp": self.proportion_deviation.get(name, 0.0) * 100,
                "achieved_count": self.achieved_counts.get(name, 0),
            })
        return pd.DataFrame(rows)

    def plates_dataframe(self) -> pd.DataFrame:
        """Floors and units per plate."""
        if not self.zones:
            return pd.DataFrame()
        agg = defaultdict(lambda: {"n_floors": 0, "n_units": 0, "n_zones": 0})
        for z in self.zones:
            agg[z.plate_name]["n_floors"] += z.n_floors
            agg[z.plate_name]["n_units"] += z.template_total_units * z.n_floors
            agg[z.plate_name]["n_zones"] += 1
        return pd.DataFrame([
            {"plate": k, **v} for k, v in agg.items()
        ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_uniform_config(
    types: List[UnitType],
    n_floors: int,
    floor_net_area: float,
    floor_area_tol: float = 0.02,
    **kwargs,
) -> StackingConfig:
    """Build a single-plate StackingConfig (reproduces v2 behavior)."""
    plates = [PlateClass(
        name="main",
        net_area=floor_net_area,
        n_floors=n_floors,
        area_tol=floor_area_tol,
    )]
    return StackingConfig(types=types, plates=plates, **kwargs)


def solve(cfg: StackingConfig) -> Solution:
    return _solve_with_k(cfg, cfg.n_zones)


def solve_min_k(
    cfg: StackingConfig,
    k_min: int = 1,
    k_max: int = 5,
    deviation_threshold: float = 0.0,
    early_stop: bool = True,
    verbose: bool = True,
) -> Tuple[Solution, int]:
    """Sweep K and find smallest with max deviation <= threshold.

    Note: For heterogeneous plates with P active plates, K < P is generally
    infeasible (each plate needs at least one used pattern). The sweep skips
    K values that cannot satisfy this.
    """
    best: Optional[Solution] = None
    best_k: Optional[int] = None
    p_active = cfg.n_active_plates
    k_min_effective = max(k_min, p_active)
    if verbose:
        print(f"  Sweeping K from {k_min_effective} to {k_max} "
              f"(threshold={deviation_threshold*100:.3f}pp, "
              f"{p_active} active plates → effective k_min = {k_min_effective})")
    for k in range(k_min_effective, k_max + 1):
        sol = _solve_with_k(cfg, k)
        if sol.status in ("Optimal", "Optimal_TimeLimit"):
            max_dev = max(abs(d) for d in sol.proportion_deviation.values())
            tag = "(TIME LIMIT)" if sol.status == "Optimal_TimeLimit" else ""
            if verbose:
                print(
                    f"    K={k}: zones_used={sol.n_zones_used}, "
                    f"max_dev={max_dev*100:.4f}pp, "
                    f"area={sol.total_area:>7.0f} m2, "
                    f"units={sol.total_units}, "
                    f"time={sol.solve_time_sec:.2f}s {tag}"
                )
            if best is None or max_dev < max(abs(d) for d in best.proportion_deviation.values()):
                best = sol
                best_k = k
            if max_dev <= deviation_threshold + 1e-9 and early_stop:
                return sol, k
        else:
            if verbose:
                print(f"    K={k}: status={sol.status}, time={sol.solve_time_sec:.2f}s")
    if best is None:
        raise RuntimeError("No feasible solution at any K.")
    return best, best_k


# ---------------------------------------------------------------------------
# Infeasibility diagnostic
# ---------------------------------------------------------------------------

@dataclass
class RelaxationProbe:
    """One probe: relax a single knob by a defined step and report what happens."""
    knob: str           # e.g. "plate:1669.area_tol", "unit:2B.area_tol", "n_zones", "bounded:2B.window"
    description: str    # human-readable
    delta: str          # e.g. "+0.02", "+5 zones", "±2 units"
    status: str         # solver status
    solve_time_sec: float


def diagnose_infeasibility(
    cfg: StackingConfig,
    plate_tol_step: float = 0.02,
    unit_tol_step: float = 0.02,
    zone_step: int = 5,
    count_window_step: int = 5,
    time_limit_per_probe: float = 30.0,
    verbose: bool = True,
) -> List[RelaxationProbe]:
    """Try relaxing one knob at a time; report which single relaxation restores feasibility.

    This is a lightweight (non-IIS) diagnostic. It explores a one-step
    neighborhood of the infeasible instance and reports which probes flipped
    to Optimal. Useful for telling the designer:
       "lock plates if you want, but you'll need K >= 10"
       "you can keep K=5, but plate 1669 needs +1% tolerance"

    Args mirror the relevant knobs:
        plate_tol_step: how much to widen each plate's area_tol when probing it
        unit_tol_step:  how much to widen each unit's area_tol when probing it
        zone_step:      how much to bump n_zones when probing it
        count_window_step: in bounded_count mode, expand min/max by ±this many

    Returns the full list of probes in the order tried. Caller can filter by
    status to see which knobs unblock the problem.
    """
    probes: List[RelaxationProbe] = []

    def run_probe(knob: str, description: str, delta: str, new_cfg: StackingConfig):
        if verbose:
            print(f"  probing {knob} ({delta})... ", end="", flush=True)
        # Cap time tightly to keep diagnostic responsive
        new_cfg.time_limit = time_limit_per_probe
        sol = solve(new_cfg)
        probe = RelaxationProbe(
            knob=knob, description=description, delta=delta,
            status=sol.status, solve_time_sec=sol.solve_time_sec,
        )
        probes.append(probe)
        if verbose:
            marker = "[OK]" if sol.status.startswith("Optimal") else "[--]"
            print(f"{marker} {sol.status} ({sol.solve_time_sec:.1f}s)")

    # Probe 1: widen each plate's area_tol by one step
    for i, plate in enumerate(cfg.plates):
        new_plates = [_replace_plate_tol(p, p.area_tol + plate_tol_step) if j == i else p
                      for j, p in enumerate(cfg.plates)]
        new_cfg = _rebuild_cfg(cfg, plates=new_plates)
        run_probe(
            knob=f"plate:{plate.name}.area_tol",
            description=f"Widen plate '{plate.name}' floor-area tolerance",
            delta=f"+{plate_tol_step:.2%} (now {plate.area_tol + plate_tol_step:.2%})",
            new_cfg=new_cfg,
        )

    # Probe 2: widen each unit's area_tol by one step
    for ti, t in enumerate(cfg.types):
        new_types = [_replace_unit_tol(u, u.area_tol + unit_tol_step) if j == ti else u
                     for j, u in enumerate(cfg.types)]
        new_cfg = _rebuild_cfg(cfg, types=new_types)
        run_probe(
            knob=f"unit:{t.name}.area_tol",
            description=f"Widen unit '{t.name}' area tolerance",
            delta=f"+{unit_tol_step:.2%} (now {t.area_tol + unit_tol_step:.2%})",
            new_cfg=new_cfg,
        )

    # Probe 3: bump n_zones
    new_cfg = _rebuild_cfg(cfg, n_zones=cfg.n_zones + zone_step)
    run_probe(
        knob="n_zones",
        description="Allow more distinct templates building-wide",
        delta=f"+{zone_step} zones (now {cfg.n_zones + zone_step})",
        new_cfg=new_cfg,
    )

    # Probe 4: in bounded_count mode, widen each type's count window
    if cfg.mode == "bounded_count":
        for ti, t in enumerate(cfg.types):
            new_types = [
                _widen_count_window(u, count_window_step) if j == ti else u
                for j, u in enumerate(cfg.types)
            ]
            new_cfg = _rebuild_cfg(cfg, types=new_types)
            run_probe(
                knob=f"bounded:{t.name}.window",
                description=f"Widen unit '{t.name}' count window",
                delta=f"±{count_window_step} units",
                new_cfg=new_cfg,
            )

    if verbose:
        wins = [p for p in probes if p.status.startswith("Optimal")]
        print()
        if wins:
            print(f"Result: {len(wins)} single-knob relaxation(s) restore feasibility:")
            for p in wins:
                print(f"  • {p.knob}: {p.description} ({p.delta})")
        else:
            print("Result: no single-knob relaxation restores feasibility.")
            print("        The instance is likely infeasible by more than one degree.")
            print("        Try combining relaxations or rethinking the program.")

    return probes


def _replace_plate_tol(plate: PlateClass, new_tol: float) -> PlateClass:
    return PlateClass(plate.name, plate.net_area, plate.n_floors,
                      area_tol=max(0.0, new_tol),
                      excluded_types=plate.excluded_types)


def _replace_unit_tol(t: UnitType, new_tol: float) -> UnitType:
    return UnitType(
        name=t.name, target_area=t.target_area,
        target_proportion=t.target_proportion,
        fixed_count=t.fixed_count,
        min_count=t.min_count, max_count=t.max_count, target_count=t.target_count,
        area_tol=max(0.0, new_tol),
    )


def _widen_count_window(t: UnitType, step: int) -> UnitType:
    """In bounded mode, widen min/max symmetrically by `step` units."""
    new_min = max(0, (t.min_count if t.min_count is not None else 0) - step)
    new_max = (t.max_count if t.max_count is not None else 0) + step
    return UnitType(
        name=t.name, target_area=t.target_area,
        target_proportion=t.target_proportion, fixed_count=t.fixed_count,
        min_count=new_min, max_count=new_max, target_count=t.target_count,
        area_tol=t.area_tol,
    )


def _rebuild_cfg(cfg: StackingConfig, **overrides) -> StackingConfig:
    """Construct a fresh StackingConfig overriding specific fields. Trips
    __post_init__ to re-validate the new combo."""
    fields = dict(
        types=cfg.types, plates=cfg.plates, n_zones=cfg.n_zones,
        enforce_exact_K=cfg.enforce_exact_K,
        min_floors_per_zone=cfg.min_floors_per_zone,
        proportion_hard_band=cfg.proportion_hard_band,
        max_count_per_type_per_floor=cfg.max_count_per_type_per_floor,
        deviation_weight=cfg.deviation_weight,
        pattern_simplicity_weight=cfg.pattern_simplicity_weight,
        solver_msg=cfg.solver_msg,
        time_limit=cfg.time_limit,
        enumeration_max_count_total=cfg.enumeration_max_count_total,
        solver_backend=cfg.solver_backend,
        solver_objective=cfg.solver_objective,
        solver_options=cfg.solver_options,
    )
    fields.update(overrides)
    return StackingConfig(**fields)


# ---------------------------------------------------------------------------
# Solver (private)
# ---------------------------------------------------------------------------

def _make_solver(cfg: "StackingConfig"):
    """Construct the PuLP solver object indicated by cfg.solver_backend.

    HiGHS is the default; it is dramatically faster than CBC on tight
    integer programs (notably the fixed_count case study). CBC is kept
    selectable for backwards compatibility with older runs.
    """
    backend = (cfg.solver_backend or "highs").lower()
    opts = dict(cfg.solver_options or {})
    if backend == "highs":
        return pulp.HiGHS(msg=cfg.solver_msg, timeLimit=cfg.time_limit, **opts)
    if backend == "cbc":
        return pulp.PULP_CBC_CMD(msg=cfg.solver_msg, timeLimit=cfg.time_limit)
    if backend == "scip":
        return pulp.SCIP_PY(msg=cfg.solver_msg, timeLimit=cfg.time_limit)
    if backend == "gurobi":
        # PuLP's GUROBI class drives gurobipy in-process; no external file
        # round-trip. solver_options are forwarded as Gurobi parameters
        # (e.g. {"MIPGap": 0.01, "Threads": 8, "Seed": 0}).
        return pulp.GUROBI(msg=cfg.solver_msg, timeLimit=cfg.time_limit, **opts)
    raise ValueError(f"Unknown solver_backend: {cfg.solver_backend!r}")


def _solve_with_k(cfg: StackingConfig, K: int) -> Solution:
    t_start = time.time()
    types = cfg.types
    T = len(types)
    type_names = [t.name for t in types]
    plates = cfg.plates
    P = len(plates)

    if cfg.mode == "fixed_count":
        fixed_counts = [t.fixed_count for t in types]
        total_target_units = sum(fixed_counts)
        targets_p = [c / total_target_units for c in fixed_counts] if total_target_units > 0 else [0.0] * T
        min_counts = max_counts = target_counts = None
    elif cfg.mode == "bounded_count":
        min_counts = [t.min_count if t.min_count is not None else 0 for t in types]
        # Use a large but safe upper bound when max_count is None
        big_max = sum(p.n_floors for p in plates) * cfg.max_count_per_type_per_floor
        max_counts = [t.max_count if t.max_count is not None else big_max for t in types]
        target_counts = [t.target_count for t in types]   # may contain None
        # Derive a reporting proportion from target_count if provided, else from band midpoint
        midpoints = [
            target_counts[t] if target_counts[t] is not None
            else 0.5 * (min_counts[t] + max_counts[t])
            for t in range(T)
        ]
        s = sum(midpoints)
        targets_p = [m / s for m in midpoints] if s > 0 else [0.0] * T
        fixed_counts = None
        total_target_units = None
    else:
        targets_p = [t.target_proportion for t in types]
        fixed_counts = min_counts = max_counts = target_counts = None
        total_target_units = None

    # Enumerate count vectors per plate
    cvs_per_plate: List[List[CountVector]] = []
    cvs_count_by_plate: Dict[str, int] = {}
    for i, plate in enumerate(plates):
        if plate.n_floors == 0:
            cvs_per_plate.append([])
            cvs_count_by_plate[plate.name] = 0
            continue
        cvs = _enumerate_count_vectors_for_plate(cfg, plate)
        cvs_per_plate.append(cvs)
        cvs_count_by_plate[plate.name] = len(cvs)
        if not cvs:
            return Solution(
                status=f"no_count_vectors_for_plate:{plate.name}",
                objective=None,
                n_count_vectors_enumerated=0,
                n_count_vectors_per_plate=cvs_count_by_plate,
                n_zones_requested=K,
                target_proportions=dict(zip(type_names, targets_p)),
                solve_time_sec=time.time() - t_start,
            )

    total_cv = sum(len(c) for c in cvs_per_plate)

    prob = pulp.LpProblem("UnitMixZoning_v2b", pulp.LpMinimize)

    # Variables, indexed by (plate_idx, pattern_idx_within_plate)
    x: Dict[Tuple[int, int], pulp.LpVariable] = {}
    y: Dict[Tuple[int, int], pulp.LpVariable] = {}
    a: Dict[Tuple[int, int, int], pulp.LpVariable] = {}

    for i, plate in enumerate(plates):
        for p_idx in range(len(cvs_per_plate[i])):
            x[(i, p_idx)] = pulp.LpVariable(
                f"x_{i}_{p_idx}", lowBound=0, upBound=plate.n_floors, cat="Integer"
            )
            y[(i, p_idx)] = pulp.LpVariable(f"y_{i}_{p_idx}", cat="Binary")
            for t in range(T):
                a[(i, p_idx, t)] = pulp.LpVariable(
                    f"a_{i}_{p_idx}_{t}",
                    lowBound=types[t].area_min,
                    upBound=types[t].area_max,
                )

    # Constraint 1: Per-plate floor count
    for i, plate in enumerate(plates):
        if plate.n_floors == 0:
            continue
        prob += (
            pulp.lpSum(x[(i, p_idx)] for p_idx in range(len(cvs_per_plate[i])))
            == plate.n_floors
        ), f"PlateFloors_{i}"

    # Constraint 2: Link x and y; min floors per used template
    for i, plate in enumerate(plates):
        for p_idx in range(len(cvs_per_plate[i])):
            prob += x[(i, p_idx)] <= plate.n_floors * y[(i, p_idx)], f"XYlink_hi_{i}_{p_idx}"
            prob += x[(i, p_idx)] >= cfg.min_floors_per_zone * y[(i, p_idx)], f"XYlink_lo_{i}_{p_idx}"

    # Constraint 3: Building-wide K cap
    all_y = [y[(i, p_idx)] for i in range(P) for p_idx in range(len(cvs_per_plate[i]))]
    if cfg.enforce_exact_K:
        prob += pulp.lpSum(all_y) == K, "ZoneCountExact"
    else:
        prob += pulp.lpSum(all_y) <= K, "ZoneCountMax"

    # Constraint 4: Per-template area band (only active when y=1)
    for i, plate in enumerate(plates):
        A_lo_i = plate.area_min
        A_hi_i = plate.area_max
        for p_idx in range(len(cvs_per_plate[i])):
            cv = cvs_per_plate[i][p_idx]
            area_expr = pulp.lpSum(cv.counts[t] * a[(i, p_idx, t)] for t in range(T))
            prob += area_expr >= A_lo_i * y[(i, p_idx)], f"AreaLo_{i}_{p_idx}"
            max_possible_area = sum(cv.counts[t] * types[t].area_max for t in range(T))
            prob += area_expr <= A_hi_i * y[(i, p_idx)] + max_possible_area * (1 - y[(i, p_idx)]), \
                f"AreaHi_{i}_{p_idx}"

    # Building-wide type totals
    type_total = [
        pulp.lpSum(
            cvs_per_plate[i][p_idx].counts[t] * x[(i, p_idx)]
            for i in range(P)
            for p_idx in range(len(cvs_per_plate[i]))
        )
        for t in range(T)
    ]
    total_units = pulp.lpSum(type_total)

    # Constraint 5/6: Type-total constraints — branch on mode
    if cfg.mode == "fixed_count":
        # Hard equality
        d_pos = []
        d_neg = []
        for t in range(T):
            prob += type_total[t] == fixed_counts[t], f"FixedCount_{t}"

    elif cfg.mode == "bounded_count":
        # Hard min/max band on each type
        for t in range(T):
            prob += type_total[t] >= min_counts[t], f"MinCount_{t}"
            prob += type_total[t] <= max_counts[t], f"MaxCount_{t}"
        # Soft L1 toward target_count (only for types with a target)
        d_pos = []
        d_neg = []
        for t in range(T):
            if target_counts[t] is not None:
                dp = pulp.LpVariable(f"dp_{t}", lowBound=0)
                dn = pulp.LpVariable(f"dn_{t}", lowBound=0)
                d_pos.append(dp)
                d_neg.append(dn)
                prob += type_total[t] - target_counts[t] == dp - dn, f"BoundDev_{t}"

    else:  # proportional
        d_pos = [pulp.LpVariable(f"dp_{t}", lowBound=0) for t in range(T)]
        d_neg = [pulp.LpVariable(f"dn_{t}", lowBound=0) for t in range(T)]
        for t in range(T):
            prob += type_total[t] - targets_p[t] * total_units == d_pos[t] - d_neg[t], f"DevDef_{t}"
        band = cfg.proportion_hard_band
        for t in range(T):
            lo = max(0.0, targets_p[t] - band)
            hi = min(1.0, targets_p[t] + band)
            prob += type_total[t] - hi * total_units <= 0, f"HardHi_{t}"
            prob += type_total[t] - lo * total_units >= 0, f"HardLo_{t}"

    # Objective
    if cfg.solver_objective == "vertical_sort":
        # Compact, vertically-sorted stacking (the zone-ordering extension of
        # Sec. 5.4, "larger units higher up"). Two lexicographic objectives:
        #
        #   primary   : fewest distinct templates (compactness, consistent
        #               with the K-sweep structural optimum);
        #   secondary : place larger units higher, by maximizing the
        #               elevation-weighted unit size-rank. Each plate i carries
        #               a mid-storey elevation e_i; large-unit templates are
        #               pulled to the crown and small-unit templates to the
        #               base.
        #
        # In fixed_count mode the building-wide total is pinned, so the
        # secondary term reorders the delivered program at no cost in units;
        # intra-plate order (zones at equal e_i) is resolved by the caller
        # when laying floors out. Designed for fixed_count mode; in bounded
        # mode the secondary term would also pull the total toward the
        # per-type minima, which is usually not intended.
        cum = 0.0
        elev: List[float] = []
        for plate in plates:
            elev.append(cum + plate.n_floors / 2.0)
            cum += plate.n_floors
        # Size weight: squared rank by ascending target area (1, 4, 9, ...).
        # Squaring makes the weight CONVEX, so a few large units outscore many
        # medium units at a given elevation; a plain rank sum would fill an
        # upper plate with medium units (more rank-units) instead of the
        # largest type. Rank (not area) is the base because count * area ~ the
        # plate net area is nearly fixed per floor, so an area weight would be
        # near-constant and could not distinguish a small-unit floor from a
        # large-unit one.
        order = sorted(range(T), key=lambda t: types[t].target_area)
        w = [0] * T
        for r, t in enumerate(order):
            w[t] = (r + 1) ** 2
        # Secondary objective: MAXIMIZE the elevation-weighted size weight,
        # which pulls the largest units to the crown and the smallest to the
        # base.
        size_elev = pulp.lpSum(
            elev[i] * sum(w[t] * cvs_per_plate[i][p_idx].counts[t]
                          for t in range(T)) * x[(i, p_idx)]
            for i in range(P)
            for p_idx in range(len(cvs_per_plate[i]))
        )
        # The 2e-7 scale keeps the secondary term strictly below one template
        # (size_elev <= total_floors * max_weight * total_units), so
        # compactness always wins first and the size ordering only breaks ties
        # among the minimum-template stacks.
        obj = pulp.lpSum(all_y) - 2e-7 * size_elev
    elif cfg.solver_objective == "maximize_total_units":
        # Maximize total units (PuLP minimizes, so we minimize the negative).
        # Only meaningful in bounded_count mode where the total is not pinned
        # by an exact-equality constraint; the per-type min/max bounds shape
        # the feasible region, and the soft target_count term (if set) is
        # ignored in this objective branch.
        if cfg.mode != "bounded_count":
            raise ValueError(
                "solver_objective='maximize_total_units' requires "
                "mode='bounded_count' (use min_count/max_count on types)."
            )
        obj = -total_units
        # Light tie-breaker: among equally-dense solutions prefer fewer zones.
        tb = cfg.pattern_simplicity_weight if cfg.pattern_simplicity_weight > 0 else 1e-6
        obj += tb * pulp.lpSum(all_y)
    elif cfg.mode == "fixed_count":
        # No deviation possible — counts pinned. Light pattern-simplicity tie-breaker.
        if cfg.pattern_simplicity_weight > 0:
            obj = cfg.pattern_simplicity_weight * pulp.lpSum(all_y)
        else:
            obj = 1e-6 * pulp.lpSum(all_y)
    elif cfg.mode == "bounded_count":
        # Soft L1 toward target_count where set, plus optional pattern simplicity
        if d_pos or d_neg:
            obj = cfg.deviation_weight * (pulp.lpSum(d_pos) + pulp.lpSum(d_neg))
        else:
            obj = 1e-6 * pulp.lpSum(all_y)
        if cfg.pattern_simplicity_weight > 0:
            obj += cfg.pattern_simplicity_weight * pulp.lpSum(all_y)
    else:
        obj = cfg.deviation_weight * (pulp.lpSum(d_pos) + pulp.lpSum(d_neg))
        if cfg.pattern_simplicity_weight > 0:
            obj += cfg.pattern_simplicity_weight * pulp.lpSum(all_y)
    prob += obj, "Objective"

    # Solve
    solver = _make_solver(cfg)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    elapsed = time.time() - t_start
    backend = (cfg.solver_backend or "highs").lower()

    # PuLP reports "Not Solved" when Gurobi hits TIME_LIMIT even if a feasible
    # primal exists (Gurobi only flags status=Optimal once the dual bound
    # certifies optimality). Detect the primal-without-certificate case and
    # extract it instead of discarding the solution.
    has_primal = prob.status == 1
    if (
        not has_primal
        and backend == "gurobi"
        and getattr(prob, "solverModel", None) is not None
    ):
        try:
            if prob.solverModel.SolCount > 0:
                has_primal = True
                status = "Optimal_TimeLimit"
        except Exception:
            pass

    if (
        cfg.time_limit is not None
        and status == "Optimal"
        and elapsed >= cfg.time_limit * 0.95
    ):
        status = "Optimal_TimeLimit"

    sol = Solution(
        status=status,
        objective=pulp.value(prob.objective) if has_primal else None,
        n_count_vectors_enumerated=total_cv,
        n_count_vectors_per_plate=cvs_count_by_plate,
        n_zones_requested=K,
        target_proportions=dict(zip(type_names, targets_p)),
        solve_time_sec=elapsed,
    )

    if getattr(prob, "solverModel", None) is not None:
        if backend == "highs":
            try:
                hi = prob.solverModel.getInfo()
                sol.mip_dual_bound = float(hi.mip_dual_bound)
                sol.mip_gap_rel = float(hi.mip_gap)
                sol.mip_node_count = int(hi.mip_node_count)
            except Exception:
                pass
        elif backend == "gurobi":
            try:
                gm = prob.solverModel
                sol.mip_dual_bound = float(gm.ObjBound)
                sol.mip_gap_rel = float(gm.MIPGap)
                sol.mip_node_count = int(gm.NodeCount)
            except Exception:
                pass

    if not has_primal:
        return sol

    # Extract solution
    achieved = {nm: 0 for nm in type_names}
    total_units_actual = 0
    total_area_actual = 0.0

    for i, plate in enumerate(plates):
        for p_idx in range(len(cvs_per_plate[i])):
            x_val = int(round(x[(i, p_idx)].value() or 0))
            y_val = int(round(y[(i, p_idx)].value() or 0))
            if x_val == 0 or y_val == 0:
                continue
            cv = cvs_per_plate[i][p_idx]
            counts_dict = {type_names[t]: cv.counts[t] for t in range(T)}
            areas_dict = {type_names[t]: float(a[(i, p_idx, t)].value() or 0.0) for t in range(T)}
            template_area = sum(cv.counts[t] * areas_dict[type_names[t]] for t in range(T))
            template_units = cv.total()
            sol.zones.append(ZoneAssignment(
                zone_index=len(sol.zones) + 1,
                plate_name=plate.name,
                plate_index=i,
                n_floors=x_val,
                counts=counts_dict,
                areas=areas_dict,
                template_total_area=template_area,
                template_total_units=template_units,
            ))
            for t in range(T):
                achieved[type_names[t]] += cv.counts[t] * x_val
            total_units_actual += template_units * x_val
            total_area_actual += template_area * x_val

    sol.n_zones_used = len(sol.zones)
    sol.achieved_counts = achieved
    sol.total_units = total_units_actual
    sol.total_area = total_area_actual
    if total_units_actual > 0:
        sol.achieved_proportions = {
            nm: achieved[nm] / total_units_actual for nm in type_names
        }
    else:
        sol.achieved_proportions = {nm: 0.0 for nm in type_names}
    sol.proportion_deviation = {
        nm: sol.achieved_proportions[nm] - dict(zip(type_names, targets_p))[nm]
        for nm in type_names
    }
    return sol


# ---------------------------------------------------------------------------
# Enumeration (private helper)
# ---------------------------------------------------------------------------

def _enumerate_count_vectors_for_plate(
    cfg: StackingConfig, plate: PlateClass
) -> List[CountVector]:
    """Enumerate count vectors feasible against this plate's area band."""
    types = cfg.types
    T = len(types)
    A_lo = plate.area_min
    A_hi = plate.area_max
    a_min = [t.area_min for t in types]
    a_max = [t.area_max for t in types]
    Nmax = cfg.max_count_per_type_per_floor

    per_type_max = [min(Nmax, int(A_hi // a_min[t])) for t in range(T)]

    # Per-plate program availability: force barred types to zero count.
    if plate.excluded_types:
        excluded = set(plate.excluded_types)
        per_type_max = [0 if types[t].name in excluded else per_type_max[t]
                        for t in range(T)]

    results: List[CountVector] = []
    counts = [0] * T

    def recurse(i: int, sum_min: float, sum_max: float):
        if i == T:
            tot = sum(counts)
            if tot == 0:
                return
            if tot > cfg.enumeration_max_count_total:
                return
            if sum(1 for c in counts if c > 0) < cfg.min_distinct_types_per_template:
                return
            if sum_max >= A_lo - 1e-9 and sum_min <= A_hi + 1e-9:
                results.append(CountVector(counts=tuple(counts)))
            return

        for n_i in range(0, per_type_max[i] + 1):
            new_min = sum_min + n_i * a_min[i]
            new_max = sum_max + n_i * a_max[i]
            remaining_max_potential = sum(
                per_type_max[j] * a_max[j] for j in range(i + 1, T)
            )
            if new_max + remaining_max_potential < A_lo - 1e-9:
                continue
            if new_min > A_hi + 1e-9:
                break
            counts[i] = n_i
            recurse(i + 1, new_min, new_max)
        counts[i] = 0

    recurse(0, 0.0, 0.0)
    return results