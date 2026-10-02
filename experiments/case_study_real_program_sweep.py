"""Real-case program-variation sweep: De Piek.

Companion to case_study_real_envelope.py. The plate geometry is the De Piek
baseline; the unit program is the variable. Programs are generated plate-blind
by rule, with no hand-tuning:

  posed      the 157-unit posed program (reference row);
  count      each type at +1 and at -1 against the posed program;
  share      affordable share s (studio + 1B, by unit count) in
             {0.15, 0.20, 0.25, 0.35, 0.40, 0.45}, posed ratios kept within
             the affordable group (12:33) and the rest (64:40:8), total
             n = round(envelope / mean unit area);
  tightness  posed proportions, n = round(rho * envelope / mean unit area)
             for rho in {0.96, 0.98, 1.00, 1.02, 1.04}.

Integer counts come from largest-remainder rounding of shares times n.

Per program:
  1. K ladder in fixed-count mode, K = 3 .. 9, stopping at the first feasible K;
  2. single-knob diagnostic at K = 3 when K = 3 is not feasible;
  3. count slack at K = 3 (bounded-count mode, window +/- d for d = 1, 2, 3,
     stopping at the first feasible d) when K = 3 is not feasible;
  4. minimum run m = 3, 4 at K = 3 and K = 9 (posed program only).

Every solve has three distinct outcomes: feasible, proven infeasible
("Infeasible" or "no_count_vectors..."), and undetermined (anything else at the
time limit). A configuration the solver rejects before solving is recorded as
config_error with its message.

Resumable: every solve is appended to the long-format cells CSV as it
completes; a rerun reuses those rows unless --fresh is passed.

Backend: HiGHS via PuLP, 4 threads per solve. Outputs in experiments/results/:
    case_study_real_program_sweep.csv   one row per program
    case_study_real_program_cells.csv   every solve, long format
    case_study_real_program_diag.csv    every diagnostic probe
    case_study_real_program_mrun.csv    minimum-run rows (posed program)
    machine_case_study_real_program_sweep.json
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from _common import out_path, write_machine_json
from case_study_real_envelope import HIGHS, PLATES, UNIT_AREA, UNIT_TYPES

import unit_mix_solver_v2b as v2b


POSED = {"studio": 12, "1B": 33, "2B": 64, "3B": 40, "PH": 8}
ENVELOPE = sum(a * f for _, a, f in PLATES)          # 14,740 m^2
PLATE_TOL = 0.02
UNIT_TOL = 0.05
MIN_RUN = 2
K_LADDER = range(3, 10)
SLACK_WINDOWS = (1, 2, 3)
MRUN_CELLS = ((3, 3), (4, 3), (3, 9), (4, 9))        # (m, K)
CELL_TIME = 120.0
DIAG_TIME = 45.0
SHARE_LEVELS = (0.15, 0.20, 0.25, 0.35, 0.40, 0.45)
RHO_LEVELS = (0.96, 0.98, 1.00, 1.02, 1.04)
HIGHS_OPTIONS = {**HIGHS, "threads": 4}

CELLS_CSV = "case_study_real_program_cells.csv"
DIAG_CSV = "case_study_real_program_diag.csv"
SWEEP_CSV = "case_study_real_program_sweep.csv"
MRUN_CSV = "case_study_real_program_mrun.csv"

CELL_FIELDS = ["program_id", "phase", "K", "m", "d", "status", "outcome",
               "zones_used", "floors_per_zone", "achieved_counts",
               "abs_deviation", "solve_time_sec"]
DIAG_FIELDS = ["program_id", "knob", "description", "delta", "status",
               "unblocks", "solve_time_sec"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="Discard existing cells and diagnostic rows.")
    ap.add_argument("--only", default="",
                    help="Comma-separated program ids to run (default: all).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent solver processes (default 1).")
    args = ap.parse_args()

    write_machine_json("case_study_real_program_sweep.py")
    programs = generate_programs()
    wanted = [p for p in programs
              if not args.only or p["program_id"] in args.only.split(",")]

    if args.fresh:
        for name in (CELLS_CSV, DIAG_CSV):
            out_path(name).unlink(missing_ok=True)
    cells = _read_rows(CELLS_CSV)
    diags = _read_rows(DIAG_CSV)
    for name, fields in ((CELLS_CSV, CELL_FIELDS), (DIAG_CSV, DIAG_FIELDS)):
        if not out_path(name).exists():
            _write_rows(name, fields, [])

    jobs = []
    for p in wanted:
        pid = p["program_id"]
        jobs.append((p, [r for r in cells if r["program_id"] == pid],
                     any(r["program_id"] == pid for r in diags)))

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_program, *job) for job in jobs]
        for fut in as_completed(futures):
            pid, new_cells, new_diags = fut.result()
            print(f"{pid}: {len(new_cells)} new solves, "
                  f"{len(new_diags)} new probes", flush=True)

    write_summaries(programs)


def generate_programs() -> List[Dict]:
    progs = [_program("posed", "posed", "", dict(POSED))]
    for ut in UNIT_TYPES:
        for step in (+1, -1):
            counts = dict(POSED)
            counts[ut] += step
            progs.append(_program(f"count_{ut}{step:+d}", "count",
                                  f"{ut}{step:+d}", counts))
    afford = POSED["studio"] + POSED["1B"]
    rest = POSED["2B"] + POSED["3B"] + POSED["PH"]
    for s in SHARE_LEVELS:
        shares = {"studio": s * POSED["studio"] / afford,
                  "1B": s * POSED["1B"] / afford}
        for ut in ("2B", "3B", "PH"):
            shares[ut] = (1 - s) * POSED[ut] / rest
        n = round(ENVELOPE / _mean_area(shares))
        progs.append(_program(f"share_{s:.2f}", "share", f"{s:.2f}",
                              _largest_remainder(shares, n)))
    total = sum(POSED.values())
    posed_shares = {ut: POSED[ut] / total for ut in UNIT_TYPES}
    for rho in RHO_LEVELS:
        n = round(rho * ENVELOPE / _mean_area(posed_shares))
        progs.append(_program(f"tightness_{rho:.2f}", "tightness", f"{rho:.2f}",
                              _largest_remainder(posed_shares, n)))
    return progs


def run_program(prog: Dict, done: List[Dict], diag_done: bool
                ) -> Tuple[str, List[Dict], List[Dict]]:
    """Run every measurement for one program, reusing completed cells."""
    pid, counts = prog["program_id"], prog["counts"]
    cache = {_key(r): r for r in done}
    new_cells: List[Dict] = []

    def cell(phase, K, m, d, cfg_fn):
        key = (phase, str(K), str(m), str(d))
        if key in cache:
            return cache[key]
        row = _solve_row(pid, phase, K, m, d, cfg_fn, counts)
        _append_rows(CELLS_CSV, CELL_FIELDS, [row])
        cache[key] = row
        new_cells.append(row)
        return row

    k3 = None
    for K in K_LADDER:
        r = cell("ladder", K, MIN_RUN, "",
                 lambda K=K: build_fixed(counts, K, MIN_RUN, CELL_TIME))
        if K == 3:
            k3 = r
        if r["outcome"] == "feasible":
            break

    new_diags: List[Dict] = []
    if k3["outcome"] != "feasible":
        if not diag_done:
            probes = v2b.diagnose_infeasibility(
                build_fixed(counts, 3, MIN_RUN, DIAG_TIME),
                plate_tol_step=0.005, unit_tol_step=0.02, zone_step=1,
                count_window_step=0, time_limit_per_probe=DIAG_TIME,
                verbose=False)
            new_diags = [dict(program_id=pid, knob=p.knob,
                              description=p.description, delta=p.delta,
                              status=p.status,
                              unblocks=p.status.startswith("Optimal"),
                              solve_time_sec=round(p.solve_time_sec, 2))
                         for p in probes]
            _append_rows(DIAG_CSV, DIAG_FIELDS, new_diags)
        for d in SLACK_WINDOWS:
            r = cell("slack", 3, MIN_RUN, d,
                     lambda d=d: build_bounded(counts, d, 3, CELL_TIME))
            if r["outcome"] == "feasible":
                break

    if pid == "posed":
        for m, K in MRUN_CELLS:
            cell("mrun", K, m, "",
                 lambda m=m, K=K: build_fixed(counts, K, m, CELL_TIME))
    return pid, new_cells, new_diags


def build_fixed(counts: Dict[str, int], K: int, m: int,
                time_limit: float) -> v2b.StackingConfig:
    types = [v2b.UnitType(ut, target_area=UNIT_AREA[ut],
                          fixed_count=counts[ut], area_tol=UNIT_TOL)
             for ut in UNIT_TYPES]
    return _config(types, K, m, time_limit)


def build_bounded(counts: Dict[str, int], d: int, K: int,
                  time_limit: float) -> v2b.StackingConfig:
    types = [v2b.UnitType(ut, target_area=UNIT_AREA[ut],
                          min_count=max(0, counts[ut] - d),
                          max_count=counts[ut] + d, target_count=counts[ut],
                          area_tol=UNIT_TOL)
             for ut in UNIT_TYPES]
    return _config(types, K, MIN_RUN, time_limit)


def write_summaries(programs: List[Dict]) -> None:
    cells = _read_rows(CELLS_CSV)
    diags = _read_rows(DIAG_CSV)
    rows = []
    for p in programs:
        pid = p["program_id"]
        ladder = [r for r in cells if r["program_id"] == pid
                  and r["phase"] == "ladder"]
        if not ladder:
            continue
        ladder.sort(key=lambda r: int(r["K"]))
        k3 = next((r for r in ladder if r["K"] == "3"), None)
        feas = [int(r["K"]) for r in ladder if r["outcome"] == "feasible"]
        inf = [r["K"] for r in ladder if r["outcome"] == "infeasible"]
        und = [r["K"] for r in ladder if r["outcome"] == "undetermined"]
        unblock = [d["description"] for d in diags
                   if d["program_id"] == pid and d["unblocks"] == "True"]
        slack = sorted((r for r in cells if r["program_id"] == pid
                        and r["phase"] == "slack"), key=lambda r: int(r["d"]))
        slack_ok = next((r for r in slack if r["outcome"] == "feasible"), None)
        rows.append({
            "program_id": pid, "axis": p["axis"], "param": p["param"],
            **{ut: p["counts"][ut] for ut in UNIT_TYPES},
            "total_units": p["total_units"], "nominal_area": p["nominal_area"],
            "area_ratio": p["area_ratio"],
            "k3_outcome": k3["outcome"] if k3 else "",
            "k3_status": k3["status"] if k3 else "",
            "k3_zones_used": k3["zones_used"] if k3 else "",
            "smallest_feasible_K": min(feas) if feas else "none proven",
            "infeasible_K": ";".join(inf),
            "undetermined_K": ";".join(und),
            "diag_unblocking_at_K3": " | ".join(unblock),
            "slack_d_tried": ";".join(r["d"] for r in slack),
            "smallest_slack_d": slack_ok["d"] if slack_ok else
                                ("none proven" if slack else ""),
            "slack_delivered_counts": slack_ok["achieved_counts"] if slack_ok else "",
            "slack_abs_deviation": slack_ok["abs_deviation"] if slack_ok else "",
        })
    _write_rows(SWEEP_CSV, list(rows[0].keys()) if rows else ["program_id"], rows)

    mrun = [{k: r[k] for k in ("m", "K", "status", "outcome", "zones_used",
                               "floors_per_zone", "solve_time_sec")}
            for r in cells if r["program_id"] == "posed" and r["phase"] == "mrun"]
    mrun.sort(key=lambda r: (int(r["K"]), int(r["m"])))
    _write_rows(MRUN_CSV, ["m", "K", "status", "outcome", "zones_used",
                           "floors_per_zone", "solve_time_sec"], mrun)
    print(f"Wrote {out_path(SWEEP_CSV)} and {out_path(MRUN_CSV)}.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _program(pid: str, axis: str, param: str, counts: Dict[str, int]) -> Dict:
    area = sum(counts[ut] * UNIT_AREA[ut] for ut in UNIT_TYPES)
    return {"program_id": pid, "axis": axis, "param": param, "counts": counts,
            "total_units": sum(counts.values()), "nominal_area": round(area, 1),
            "area_ratio": round(area / ENVELOPE, 4)}


def _mean_area(shares: Dict[str, float]) -> float:
    return sum(shares[ut] * UNIT_AREA[ut] for ut in UNIT_TYPES)


def _largest_remainder(shares: Dict[str, float], n: int) -> Dict[str, int]:
    raw = {ut: shares[ut] * n for ut in UNIT_TYPES}
    out = {ut: int(raw[ut]) for ut in UNIT_TYPES}
    left = n - sum(out.values())
    order = sorted(UNIT_TYPES, key=lambda ut: raw[ut] - out[ut], reverse=True)
    for ut in order[:left]:
        out[ut] += 1
    return out


def _key(row: Dict) -> Tuple[str, str, str, str]:
    return (row["phase"], str(row["K"]), str(row["m"]), str(row["d"]))


def _solve_row(pid, phase, K, m, d, cfg_fn, counts) -> Dict:
    row = dict(program_id=pid, phase=phase, K=K, m=m, d=d, status="",
               outcome="", zones_used="", floors_per_zone="",
               achieved_counts="", abs_deviation="", solve_time_sec="")
    try:
        cfg = cfg_fn()
    except ValueError as e:
        row.update(status=f"config_error: {e}", outcome="config_error")
        return row
    sol = v2b.solve(cfg)
    ok = sol.status.startswith("Optimal") and bool(sol.zones)
    row.update(status=sol.status, outcome=_outcome(sol.status, ok),
               zones_used=sol.n_zones_used,
               solve_time_sec=round(sol.solve_time_sec, 2))
    if ok:
        row["floors_per_zone"] = "|".join(f"{z.plate_name}:{z.n_floors}"
                                          for z in sol.zones)
        ach = {ut: int(sol.achieved_counts.get(ut, 0)) for ut in UNIT_TYPES}
        row["achieved_counts"] = json.dumps(ach)
        row["abs_deviation"] = sum(abs(ach[ut] - counts[ut]) for ut in UNIT_TYPES)
    return row


def _outcome(status: str, ok: bool) -> str:
    if ok:
        return "feasible"
    if status == "Infeasible" or status.startswith("no_count_vectors"):
        return "infeasible"
    return "undetermined"


def _config(types, K, m, time_limit) -> v2b.StackingConfig:
    plates = [v2b.PlateClass(n, a, f, area_tol=PLATE_TOL) for n, a, f in PLATES]
    return v2b.StackingConfig(
        types=types, plates=plates, n_zones=K, min_floors_per_zone=m,
        max_count_per_type_per_floor=15, enumeration_max_count_total=20,
        deviation_weight=1000.0, pattern_simplicity_weight=1e-6,
        time_limit=time_limit, solver_msg=False, solver_options=HIGHS_OPTIONS)


def _read_rows(name: str) -> List[Dict]:
    path = out_path(name)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_rows(name: str, fields: List[str], rows: List[Dict]) -> None:
    """Append to a CSV whose header main() has already written."""
    if not rows:
        return
    with open(out_path(name), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerows(rows)


def _write_rows(name: str, fields: List[str], rows: List[Dict]) -> None:
    with open(out_path(name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
