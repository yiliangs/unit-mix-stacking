# Unit-Mix Stacking for Residential Towers: A Count-Vector Formulation

Reference implementation and reproducibility package for the exact
MILP-based unit-mix stacking solver described in the accompanying paper.
All instances are synthetic or drawn from public architectural
documentation.

## What this is

- `solver/` — the V1 (direct McCormick), V2 (single-plate count-vector),
  and V2b (heterogeneous-plate) formulations. V2b is the deployed model;
  V1 is kept runnable as the benchmark baseline.
- `experiments/` — scripts that produce the tables and figures in the
  paper; each writes CSV results and a `machine_*.json` environment
  record to `experiments/results/`.

## Environment

All headline numbers run on **HiGHS** via PuLP. **Gurobi 13.0.2** and
**SCIP** are used only for cross-checks. Exact Python, PuLP, HiGHS, and
Gurobi versions, OS, CPU, and RAM for each run are recorded per script in
`experiments/results/machine_<script>.json`.

```
pip install pulp pandas highspy   # Gurobi/SCIP optional, licensed, cross-checks only
python experiments/<script>.py     # writes results/*.csv + machine_<script>.json
```

Figures in the paper are plotted from the CSVs in `experiments/results/`;
the deposited data is the figure source of truth.

## What reproduces what

| Result (paper section) | Script(s) | Output |
|---|---|---|
| V1 vs V2 benchmark (§6.1) | `v1_vs_v2_benchmark.py`, `gurobi_v1_vs_v2.py` | `v1_vs_v2*.csv` |
| V2b scaling (§6.2) | `v2b_scaling.py` | `v2b_scaling.csv` |
| LP-relaxation tightness (§6.3) | `lp_tightness_battery.py`, `gurobi_lp_gap.py` | `lp_gap*.csv` |
| Metaheuristic (GA) baseline (§6.4) | `heuristic_baseline.py` | `heuristic_baseline*.csv` |
| Synthetic case: K-sweep + diagnostic (§7.2) | `case_study_ksweep.py` | `case_study_ksweep.csv`, `case_study_diag.csv` |
| Synthetic case: envelope (§7.3) | `case_study_envelope.py` | `case_study_envelope.csv` |
| Synthetic case: headroom + battery (§7.4–7.5) | `case_study_headroom.py`, `case_study_headroom_battery.py` | `case_study_headroom*.csv` |
| Synthetic case: stacking figure | `case_study_k8_stacking.py` | `case_study_k8_stacking.csv` |
| Real case (De Piek) stacking figure | `case_study_real.py` | `case_study_real_stacking.csv` |
| Real case: feasibility envelope | `case_study_real_envelope.py` | `case_study_real_envelope.csv`, `case_study_real_diag.csv` |
| Solver cross-checks (Gurobi/SCIP, extended budgets) | `gurobi_*.py`, `scip_middle_band.py`, `k11_long_budget.py`, `k4_long_budget.py` | `*_gurobi.csv`, `scip_*.csv`, `k*_long_budget.csv` |

The two case-study stacking figures are arranged under a vertical-sort
objective computed with Gurobi; all reported timings, statuses,
feasibility verdicts, and counts are HiGHS.

## Real-case data provenance

The real-building study (De Piek, Rotterdam) uses only publicly available
architectural documentation: the published, scaled section and per-level
plans cited in the paper. Plate areas were traced from those scaled
drawings; the unit program is a posed (illustrative) brief, not an
as-built mix. No proprietary project data is used.

## License

TODO: add a license before making this repository public.
