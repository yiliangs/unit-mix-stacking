"""
Unit Mix Stacking Solver v1 — Zone-Based Formulation
=====================================================

Building partitioned into K vertical zones. Each zone has:
  - f_z floors (>= 2 if used, or 0 if unused)
  - n_{t,z} count of each unit type t (integer)
  - a_{t,z} continuous area per type t, within [0.95 a_t*, 1.05 a_t*]

All floors within a zone share an identical template.

Bilinear term n_{t,z} * a_{t,z} is exactly linearized via binary
selection of count value combined with a continuous area variable.

Solver: PuLP; HiGHS by default (CBC is PuLP's bundled fallback, not used
for any reported number).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pulp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitType:
    name: str
    target_area: float          # a_t*
    target_proportion: float    # p_t* in [0, 1]
    area_tol: float = 0.05      # +/- fraction; default 5%

    @property
    def area_min(self) -> float:
        return self.target_area * (1.0 - self.area_tol)

    @property
    def area_max(self) -> float:
        return self.target_area * (1.0 + self.area_tol)


@dataclass
class StackingConfig:
    types: List[UnitType]
    n_floors: int                       # F
    floor_net_area: float               # A^net
    floor_area_tol: float = 0.02        # +/- fraction on floor area
    n_zones: int = 3                    # K (default 3)
    min_floors_per_zone: int = 2        # avoid orphans (apply only to used zones)
    proportion_hard_band: float = 0.05  # +/- on each type's proportion
    max_count_per_type_per_zone: int = 12
    deviation_weight: float = 1000.0    # alpha
    area_weight: float = 1.0            # eta (V1-only area term; paper renames beta->eta)
    pattern_simplicity_weight: float = 0.0  # gamma; 0 = off
    solver_msg: bool = False
    time_limit: Optional[float] = 120.0
    solver_backend: str = "highs"  # "highs" (default) or "cbc"
    solver_options: Optional[Dict[str, object]] = None

    def __post_init__(self):
        s = sum(t.target_proportion for t in self.types)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Target proportions must sum to 1.0, got {s:.4f}")


@dataclass
class ZoneAssignment:
    zone_index: int
    n_floors: int
    counts: Dict[str, int]
    areas: Dict[str, float]
    template_total_area: float
    template_total_units: int

    def describe(self) -> str:
        parts = [
            f"{name}:{c}@{self.areas[name]:.2f}"
            for name, c in self.counts.items() if c > 0
        ]
        return (
            f"Zone {self.zone_index}: {self.n_floors} floors | "
            f"{', '.join(parts)} | {self.template_total_area:.1f} m2/floor"
        )


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
        """Expand zones into one row per floor."""
        if not self.zones:
            return pd.DataFrame()
        type_names = list(self.target_proportions.keys())
        rows = []
        floor_idx = 1
        for z in self.zones:
            for _ in range(z.n_floors):
                row = {"floor": floor_idx, "zone": z.zone_index}
                for n in type_names:
                    row[f"{n}_count"] = z.counts.get(n, 0)
                    row[f"{n}_area"] = round(z.areas.get(n, 0.0), 2) if z.counts.get(n, 0) > 0 else 0.0
                row["total_area"] = round(z.template_total_area, 2)
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


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------

def _solve_with_k(cfg: StackingConfig, K: int) -> Solution:
    """Solve the zone-based MILP with K zones."""
    t_start = time.time()
    types = cfg.types
    T = len(types)
    type_names = [t.name for t in types]
    targets_p = [t.target_proportion for t in types]

    A_lo = cfg.floor_net_area * (1 - cfg.floor_area_tol)
    A_hi = cfg.floor_net_area * (1 + cfg.floor_area_tol)
    F = cfg.n_floors
    Nmax = cfg.max_count_per_type_per_zone

    prob = pulp.LpProblem("UnitMixZoning", pulp.LpMinimize)

    # ----- Decision variables -----
    # f[z]: number of floors in zone z. 0 if unused, else >= min_floors_per_zone.
    # used[z]: 1 if zone z is used.
    f = {z: pulp.LpVariable(f"f_{z}", lowBound=0, upBound=F, cat="Integer") for z in range(K)}
    used = {z: pulp.LpVariable(f"used_{z}", cat="Binary") for z in range(K)}

    # n[t,z]: integer count of type t in zone z (0 .. Nmax)
    n = {(t, z): pulp.LpVariable(f"n_{t}_{z}", lowBound=0, upBound=Nmax, cat="Integer")
         for t in range(T) for z in range(K)}

    # a[t,z]: continuous area of type t in zone z, within type's range
    a = {(t, z): pulp.LpVariable(
            f"a_{t}_{z}",
            lowBound=types[t].area_min,
            upBound=types[t].area_max,
        )
        for t in range(T) for z in range(K)}

    # xi[t,z,k]: binary, =1 iff n[t,z] == k
    # u[t,z,k]: continuous, = xi[t,z,k] * a[t,z]   (linearizes bilinear)
    xi = {}
    u = {}
    for t in range(T):
        a_max_t = types[t].area_max
        for z in range(K):
            for k in range(Nmax + 1):
                xi[(t, z, k)] = pulp.LpVariable(f"xi_{t}_{z}_{k}", cat="Binary")
                u[(t, z, k)] = pulp.LpVariable(f"u_{t}_{z}_{k}", lowBound=0, upBound=a_max_t)

    # Zone total area aux variable (per-floor template area for zone z)
    zone_area = {z: pulp.LpVariable(f"zone_area_{z}", lowBound=0, upBound=A_hi * 2)
                 for z in range(K)}

    # Per-type total count across building (sum over zones of n[t,z] * f[z])
    # This is also bilinear (n times f). We introduce counts_floor[t,z] = n[t,z] * f[z]
    # and linearize via another big-M scheme.
    # Bound: counts_floor[t,z] <= Nmax * F
    counts_floor = {(t, z): pulp.LpVariable(f"counts_floor_{t}_{z}", lowBound=0, upBound=Nmax * F)
                    for t in range(T) for z in range(K)}

    # ----- Constraints -----

    # 1. f[z] sums to F
    prob += pulp.lpSum(f[z] for z in range(K)) == F, "TotalFloors"

    # 2. f[z] is 0 if not used, else within [min_floors_per_zone, F]
    for z in range(K):
        prob += f[z] <= F * used[z], f"FloorsUseLink_hi_{z}"
        prob += f[z] >= cfg.min_floors_per_zone * used[z], f"FloorsUseLink_lo_{z}"

    # 3. xi: exactly one count value selected per (t, z)
    for t in range(T):
        for z in range(K):
            prob += pulp.lpSum(xi[(t, z, k)] for k in range(Nmax + 1)) == 1, f"XiSum_{t}_{z}"
            prob += n[(t, z)] == pulp.lpSum(k * xi[(t, z, k)] for k in range(Nmax + 1)), f"NDef_{t}_{z}"

    # 4. McCormick exact for u[t,z,k] = xi[t,z,k] * a[t,z]
    # Since xi is binary and a is bounded [a_min_t, a_max_t]:
    #   u <= a_max * xi
    #   u >= a_min * xi
    #   u <= a - a_min * (1 - xi)
    #   u >= a - a_max * (1 - xi)
    for t in range(T):
        a_min_t = types[t].area_min
        a_max_t = types[t].area_max
        for z in range(K):
            for k in range(Nmax + 1):
                prob += u[(t, z, k)] <= a_max_t * xi[(t, z, k)], f"u_hi1_{t}_{z}_{k}"
                prob += u[(t, z, k)] >= a_min_t * xi[(t, z, k)], f"u_lo1_{t}_{z}_{k}"
                prob += u[(t, z, k)] <= a[(t, z)] - a_min_t * (1 - xi[(t, z, k)]), f"u_hi2_{t}_{z}_{k}"
                prob += u[(t, z, k)] >= a[(t, z)] - a_max_t * (1 - xi[(t, z, k)]), f"u_lo2_{t}_{z}_{k}"

    # 5. Zone template area: zone_area[z] = sum_{t,k} k * u[t,z,k]
    for z in range(K):
        prob += zone_area[z] == pulp.lpSum(
            k * u[(t, z, k)] for t in range(T) for k in range(Nmax + 1)
        ), f"ZoneAreaDef_{z}"

    # 6. Zone area within band IF zone is used; if unused, zone_area can be 0
    for z in range(K):
        prob += zone_area[z] >= A_lo * used[z], f"ZoneArea_lo_{z}"
        prob += zone_area[z] <= A_hi * used[z] + (1 - used[z]) * 0, f"ZoneArea_hi_{z}"
        # When used[z]=0, the n[t,z] should also be 0 (else nonsense); enforce:
        for t in range(T):
            prob += n[(t, z)] <= Nmax * used[z], f"CountUseLink_{t}_{z}"

    # 7. counts_floor[t,z] = n[t,z] * f[z] linearization (big-M).
    # We need: counts_floor = n * f.  n is bounded by Nmax, f by F.
    # Equivalent: introduce binary expansion of n via xi, and write
    #   counts_floor[t,z] = sum_k k * v[t,z,k]
    # where v[t,z,k] = xi[t,z,k] * f[z] (binary * integer).
    v = {}
    for t in range(T):
        for z in range(K):
            for k in range(Nmax + 1):
                v[(t, z, k)] = pulp.LpVariable(
                    f"v_{t}_{z}_{k}", lowBound=0, upBound=F, cat="Integer"
                )

    # McCormick exact for v = xi * f
    for t in range(T):
        for z in range(K):
            for k in range(Nmax + 1):
                prob += v[(t, z, k)] <= F * xi[(t, z, k)], f"v_hi1_{t}_{z}_{k}"
                prob += v[(t, z, k)] <= f[z], f"v_hi2_{t}_{z}_{k}"
                prob += v[(t, z, k)] >= f[z] - F * (1 - xi[(t, z, k)]), f"v_lo_{t}_{z}_{k}"

    for t in range(T):
        for z in range(K):
            prob += counts_floor[(t, z)] == pulp.lpSum(
                k * v[(t, z, k)] for k in range(Nmax + 1)
            ), f"CountsFloorDef_{t}_{z}"

    # 8. Total count per type; total units
    type_total = [
        pulp.lpSum(counts_floor[(t, z)] for z in range(K)) for t in range(T)
    ]
    total_units = pulp.lpSum(type_total)

    # 9. Soft proportion deviation (L1)
    d_pos = [pulp.LpVariable(f"dp_{t}", lowBound=0) for t in range(T)]
    d_neg = [pulp.LpVariable(f"dn_{t}", lowBound=0) for t in range(T)]
    for t in range(T):
        prob += type_total[t] - targets_p[t] * total_units == d_pos[t] - d_neg[t], f"DevDef_{t}"

    # 10. Hard proportion bounds
    band = cfg.proportion_hard_band
    for t in range(T):
        lo = max(0.0, targets_p[t] - band)
        hi = min(1.0, targets_p[t] + band)
        prob += type_total[t] - hi * total_units <= 0, f"HardHi_{t}"
        prob += type_total[t] - lo * total_units >= 0, f"HardLo_{t}"

    # 11. Total-area term for the objective tie-breaker. The exact total area,
    # sum_{t,z} counts_floor[t,z] * a[t,z], is bilinear; since area only breaks
    # ties here and is not a primary signal, approximate it with the per-type
    # target areas weighted by the building-wide type totals.
    approx_total_area = pulp.lpSum(
        types[t].target_area * type_total[t] for t in range(T)
    )

    # ----- Objective -----
    obj = (
        cfg.deviation_weight * (pulp.lpSum(d_pos) + pulp.lpSum(d_neg))
        - cfg.area_weight * approx_total_area
    )
    if cfg.pattern_simplicity_weight > 0:
        obj += cfg.pattern_simplicity_weight * pulp.lpSum(used[z] for z in range(K))
    prob += obj, "Objective"

    # ----- Solve -----
    backend = (getattr(cfg, "solver_backend", "highs") or "highs").lower()
    opts = dict(getattr(cfg, "solver_options", None) or {})
    if backend == "highs":
        solver = pulp.HiGHS(msg=cfg.solver_msg, timeLimit=cfg.time_limit, **opts)
    elif backend == "cbc":
        solver = pulp.PULP_CBC_CMD(msg=cfg.solver_msg, timeLimit=cfg.time_limit)
    elif backend == "gurobi":
        solver = pulp.GUROBI(msg=cfg.solver_msg, timeLimit=cfg.time_limit, **opts)
    else:
        raise ValueError(f"Unknown solver_backend: {cfg.solver_backend!r}")
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    elapsed = time.time() - t_start

    # Gurobi TIME_LIMIT with a primal: PuLP reports "Not Solved" even though
    # solverModel.SolCount > 0. Detect and treat as Optimal_TimeLimit.
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

    # Detect probable time-limit termination: if status is "Optimal" but time
    # is at the limit (within 5%), CBC may have stopped without proving optimality.
    if (
        cfg.time_limit is not None
        and status == "Optimal"
        and elapsed >= cfg.time_limit * 0.95
    ):
        status = "Optimal_TimeLimit"  # feasible solution, optimality not proven

    sol = Solution(
        status=status,
        objective=pulp.value(prob.objective) if has_primal else None,
        n_zones_requested=K,
        target_proportions={t.name: t.target_proportion for t in types},
        solve_time_sec=elapsed,
    )

    # Capture MIP dual bound / gap / node count if the backend exposes it.
    if getattr(prob, "solverModel", None) is not None:
        if backend == "highs":
            try:
                hi = prob.solverModel.getInfo()
                # V1 uses LpMinimize with obj = dev*1000 - area. PuLP applies
                # obj_mult = 1 for minimize, so HiGHS's primal/dual are in the
                # original sense.
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
    for z in range(K):
        nf = int(round(f[z].value() or 0))
        if nf == 0:
            continue
        counts = {types[t].name: int(round(n[(t, z)].value() or 0)) for t in range(T)}
        areas = {types[t].name: float(a[(t, z)].value() or 0.0) for t in range(T)}
        template_total_area = sum(counts[types[t].name] * areas[types[t].name] for t in range(T))
        template_total_units = sum(counts.values())
        sol.zones.append(ZoneAssignment(
            zone_index=len(sol.zones) + 1,
            n_floors=nf,
            counts=counts,
            areas=areas,
            template_total_area=template_total_area,
            template_total_units=template_total_units,
        ))

    sol.n_zones_used = len(sol.zones)

    # Aggregate
    achieved_counts = {nm: 0 for nm in type_names}
    total_units_actual = 0
    total_area_actual = 0.0
    for z in sol.zones:
        for nm, c in z.counts.items():
            achieved_counts[nm] += c * z.n_floors
        total_units_actual += z.template_total_units * z.n_floors
        total_area_actual += z.template_total_area * z.n_floors

    sol.achieved_counts = achieved_counts
    sol.total_units = total_units_actual
    sol.total_area = total_area_actual
    if total_units_actual > 0:
        sol.achieved_proportions = {
            nm: achieved_counts[nm] / total_units_actual for nm in type_names
        }
    else:
        sol.achieved_proportions = {nm: 0.0 for nm in type_names}
    sol.proportion_deviation = {
        nm: sol.achieved_proportions[nm] - dict(zip(type_names, targets_p))[nm]
        for nm in type_names
    }
    return sol


def solve(cfg: StackingConfig) -> Solution:
    """Solve at configured K (cfg.n_zones)."""
    return _solve_with_k(cfg, cfg.n_zones)


def solve_min_k(
    cfg: StackingConfig,
    k_min: int = 1,
    k_max: int = 5,
    deviation_threshold: float = 0.0,
    early_stop: bool = True,
    verbose: bool = True,
) -> Tuple[Solution, int]:
    """Sweep K and find smallest with max deviation <= threshold."""
    best: Optional[Solution] = None
    best_k: Optional[int] = None
    if verbose:
        print(f"  Sweeping K from {k_min} to {k_max} (threshold={deviation_threshold*100:.3f}pp)")
    for k in range(k_min, k_max + 1):
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