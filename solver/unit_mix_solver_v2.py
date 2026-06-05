"""
Unit Mix Stacking Solver v2 — Count-Vector Templates + Continuous Areas
=======================================================================

Formulation idea
----------------
Pre-enumerate all feasible *count vectors* (n_1, ..., n_T), where a count
vector is feasible iff there EXIST areas a_t in [a_t_min, a_t_max] such that
sum_t n_t * a_t lies in the floor area band.

For each used count vector p, attach:
  - x_p (integer >= 0): number of floors using this template
  - y_p (binary): whether template is used
  - a_{p,t} (continuous): area for type t on this template

Because n is constant per template, all bilinear terms are eliminated.
Floor area constraint sum_t n_{p,t} * a_{p,t} is linear in a.
Building total of type t = sum_p n_{p,t} * x_p is linear in x.

Number of zones K = sum_p y_p (interpreted as # of distinct templates used).

Solver: PuLP; HiGHS by default (CBC is PuLP's bundled fallback). Pure
MILP, no Big-M product expansions.
"""

from __future__ import annotations

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
    target_area: float
    target_proportion: float
    area_tol: float = 0.05

    @property
    def area_min(self) -> float:
        return self.target_area * (1.0 - self.area_tol)

    @property
    def area_max(self) -> float:
        return self.target_area * (1.0 + self.area_tol)


@dataclass
class StackingConfig:
    types: List[UnitType]
    n_floors: int
    floor_net_area: float
    floor_area_tol: float = 0.02
    n_zones: int = 3                   # K; <= cap on # distinct templates
    enforce_exact_K: bool = False      # if False, K is upper bound (solver may use fewer)
    min_floors_per_zone: int = 2
    proportion_hard_band: float = 0.05
    max_count_per_type_per_floor: int = 15
    deviation_weight: float = 1000.0
    pattern_simplicity_weight: float = 0.0  # gamma; 0 = off
    solver_msg: bool = False
    time_limit: Optional[float] = 120.0
    enumeration_max_count_total: int = 25  # cap total units per floor for enum sanity
    solver_backend: str = "highs"  # "highs" (default) or "cbc"
    solver_options: Optional[Dict[str, object]] = None

    def __post_init__(self):
        s = sum(t.target_proportion for t in self.types)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Target proportions must sum to 1.0, got {s:.4f}")


@dataclass(frozen=True)
class CountVector:
    counts: Tuple[int, ...]

    def total(self) -> int:
        return sum(self.counts)


@dataclass
class ZoneAssignment:
    zone_index: int
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
        if not self.zones:
            return pd.DataFrame()
        type_names = list(self.target_proportions.keys())
        rows = []
        floor_idx = 1
        for z in self.zones:
            for _ in range(z.n_floors):
                row = {"floor": floor_idx, "zone": z.zone_index, "total_area": round(z.template_total_area, 2)}
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


# ---------------------------------------------------------------------------
# Count-vector enumeration
# ---------------------------------------------------------------------------

def enumerate_count_vectors(cfg: StackingConfig) -> List[CountVector]:
    """
    Enumerate count vectors (n_1, ..., n_T) such that there exist
    a_t in [a_t_min, a_t_max] with sum n_t * a_t in [A_lo, A_hi].

    Equivalent feasibility check (no need to solve LP per vector):
        sum n_t * a_t_max >= A_lo   (max area side)
        sum n_t * a_t_min <= A_hi   (min area side)
    """
    types = cfg.types
    T = len(types)
    A_lo = cfg.floor_net_area * (1 - cfg.floor_area_tol)
    A_hi = cfg.floor_net_area * (1 + cfg.floor_area_tol)
    a_min = [t.area_min for t in types]
    a_max = [t.area_max for t in types]
    Nmax = cfg.max_count_per_type_per_floor

    # Per-type max count given smallest area: n_t * a_t_min <= A_hi -> n_t <= A_hi / a_t_min
    per_type_max = [min(Nmax, int(A_hi // a_min[t])) for t in range(T)]

    results: List[CountVector] = []
    counts = [0] * T

    def recurse(i: int, sum_min: float, sum_max: float):
        if i == T:
            tot = sum(counts)
            if tot == 0:
                return
            if tot > cfg.enumeration_max_count_total:
                return
            # Final feasibility
            if sum_max >= A_lo - 1e-9 and sum_min <= A_hi + 1e-9:
                results.append(CountVector(counts=tuple(counts)))
            return

        for n_i in range(0, per_type_max[i] + 1):
            new_min = sum_min + n_i * a_min[i]
            new_max = sum_max + n_i * a_max[i]
            # Prune: even maxing remaining types' areas, can't reach A_lo
            remaining_max_potential = sum(
                per_type_max[j] * a_max[j] for j in range(i + 1, T)
            )
            if new_max + remaining_max_potential < A_lo - 1e-9:
                continue
            # Prune: even minning remaining types (zero contribution), already over A_hi
            if new_min > A_hi + 1e-9:
                break
            counts[i] = n_i
            recurse(i + 1, new_min, new_max)
        counts[i] = 0

    recurse(0, 0.0, 0.0)
    return results


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------

def _solve_with_k(cfg: StackingConfig, K: int) -> Solution:
    t_start = time.time()
    types = cfg.types
    T = len(types)
    type_names = [t.name for t in types]
    targets_p = [t.target_proportion for t in types]
    A_lo = cfg.floor_net_area * (1 - cfg.floor_area_tol)
    A_hi = cfg.floor_net_area * (1 + cfg.floor_area_tol)
    F = cfg.n_floors

    cvs = enumerate_count_vectors(cfg)
    n_cv = len(cvs)
    if n_cv == 0:
        return Solution(
            status="no_count_vectors",
            objective=None,
            n_count_vectors_enumerated=0,
            n_zones_requested=K,
            target_proportions={t.name: t.target_proportion for t in types},
            solve_time_sec=time.time() - t_start,
        )

    prob = pulp.LpProblem("UnitMixZoning_v2", pulp.LpMinimize)

    # Variables
    x = [pulp.LpVariable(f"x_{p}", lowBound=0, upBound=F, cat="Integer") for p in range(n_cv)]
    y = [pulp.LpVariable(f"y_{p}", cat="Binary") for p in range(n_cv)]
    a = {}  # (p, t) -> continuous area
    for p in range(n_cv):
        for t in range(T):
            a[(p, t)] = pulp.LpVariable(
                f"a_{p}_{t}",
                lowBound=types[t].area_min,
                upBound=types[t].area_max,
            )

    # Constraints
    # 1. Total floors
    prob += pulp.lpSum(x) == F, "TotalFloors"

    # 2. Link x and y; min floors per used template (avoid orphans)
    for p in range(n_cv):
        prob += x[p] <= F * y[p], f"XYlink_hi_{p}"
        prob += x[p] >= cfg.min_floors_per_zone * y[p], f"XYlink_lo_{p}"

    # 3. Number of distinct templates used <= K (or == K if enforce_exact_K)
    if cfg.enforce_exact_K:
        prob += pulp.lpSum(y) == K, "ZoneCountExact"
    else:
        prob += pulp.lpSum(y) <= K, "ZoneCountMax"

    # 4. Per-template floor area band: only active when y_p = 1
    # sum_t n_{p,t} * a_{p,t} in [A_lo * y_p, A_hi]
    # When y_p = 0, the a values are free within their bounds but x_p = 0,
    # so they don't enter the building totals. Still, we want them inactive.
    # Formulation:
    #   sum_t n_{p,t} * a_{p,t} >= A_lo * y_p
    #   sum_t n_{p,t} * a_{p,t} <= A_hi * y_p + BigArea * (1 - y_p)
    # The first ensures used templates respect lower band. Upper band only
    # enforced when used.
    for p in range(n_cv):
        cv = cvs[p]
        area_expr = pulp.lpSum(cv.counts[t] * a[(p, t)] for t in range(T))
        # When unused (y_p=0), area_expr can be any value the bounds on a allow;
        # we only need it within band when used.
        prob += area_expr >= A_lo * y[p], f"AreaLo_{p}"
        # Upper bound when used
        max_possible_area = sum(cv.counts[t] * types[t].area_max for t in range(T))
        prob += area_expr <= A_hi * y[p] + max_possible_area * (1 - y[p]), f"AreaHi_{p}"

    # 5. Building totals (linear since n is constant per p)
    type_total = [
        pulp.lpSum(cvs[p].counts[t] * x[p] for p in range(n_cv)) for t in range(T)
    ]
    total_units = pulp.lpSum(type_total)

    # 6. Soft proportion (L1)
    d_pos = [pulp.LpVariable(f"dp_{t}", lowBound=0) for t in range(T)]
    d_neg = [pulp.LpVariable(f"dn_{t}", lowBound=0) for t in range(T)]
    for t in range(T):
        prob += type_total[t] - targets_p[t] * total_units == d_pos[t] - d_neg[t], f"DevDef_{t}"

    # 7. Hard proportion bounds
    band = cfg.proportion_hard_band
    for t in range(T):
        lo = max(0.0, targets_p[t] - band)
        hi = min(1.0, targets_p[t] + band)
        prob += type_total[t] - hi * total_units <= 0, f"HardHi_{t}"
        prob += type_total[t] - lo * total_units >= 0, f"HardLo_{t}"

    # Objective
    obj = cfg.deviation_weight * (pulp.lpSum(d_pos) + pulp.lpSum(d_neg))
    if cfg.pattern_simplicity_weight > 0:
        obj += cfg.pattern_simplicity_weight * pulp.lpSum(y)
    prob += obj, "Objective"

    # Solve. HiGHS default; CBC selectable via cfg.solver_backend = "cbc".
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

    if (
        cfg.time_limit is not None
        and status == "Optimal"
        and elapsed >= cfg.time_limit * 0.95
    ):
        status = "Optimal_TimeLimit"

    sol = Solution(
        status=status,
        objective=pulp.value(prob.objective) if has_primal else None,
        n_count_vectors_enumerated=n_cv,
        n_zones_requested=K,
        target_proportions={t.name: t.target_proportion for t in types},
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

    # Extract
    achieved = {nm: 0 for nm in type_names}
    total_units_actual = 0
    total_area_actual = 0.0
    for p in range(n_cv):
        x_val = int(round(x[p].value() or 0))
        y_val = int(round(y[p].value() or 0))
        if x_val == 0 or y_val == 0:
            continue
        cv = cvs[p]
        counts_dict = {type_names[t]: cv.counts[t] for t in range(T)}
        areas_dict = {type_names[t]: float(a[(p, t)].value() or 0.0) for t in range(T)}
        template_area = sum(cv.counts[t] * areas_dict[type_names[t]] for t in range(T))
        template_units = cv.total()
        sol.zones.append(ZoneAssignment(
            zone_index=len(sol.zones) + 1,
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