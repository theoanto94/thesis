# Dutch Healthcare Costs by Postcode Area (2011–2023)

Thesis codebase analysing Vektis health-insurance claim costs at the
**3-digit postal code × age band × gender × year** grain, with two Plotly Dash
dashboards on top:

| Dashboard | Entry point | Port | Content |
|---|---|---|---|
| Phase 1 — EDA | `app.py` | 8050 | Temporal trends, demographics, geography (+ 2024–2030 projections if the temporal track has been run) |
| Phase 2/3 — Model results | `app_models.py` | 8051 | Model comparison, prediction diagnostics, segmented performance, learning curves, SHAP |

Everything downstream reads from **`identifier.db`** (DuckDB, gitignored) and
**`artifacts/`** (gitignored). Both are rebuilt from the CSVs in `data/` by the
commands below — no manual data wrangling required.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Dependency/venv manager. macOS: `brew install uv` |
| Python 3.9 | Pinned via `.python-version`; `uv` installs it for you |
| OpenMP runtime | **macOS only** — XGBoost and LightGBM need it and `uv` does not ship it: `brew install libomp` (keg-only; the wheels load it from `/opt/homebrew/opt/libomp/lib`) |
| ~15 GB free disk | `identifier.db` is ~200 MB; `artifacts/` a few GB with all models and SHAP matrices |

Linux: install OpenMP through your package manager instead (`libgomp1` on
Debian/Ubuntu, usually already present). Windows: the wheels bundle OpenMP, no
extra step.

## 2. Environment setup

```bash
git clone <repo-url>
cd thesis

uv sync            # creates .venv and installs the locked dependency graph
```

Every command below is prefixed with `uv run`, so no manual venv activation is
needed. Verify the toolchain:

```bash
uv run python -c "import duckdb, lightgbm, xgboost, catboost, shap, dash; print('ok')"
```

If this fails on macOS with `Library not loaded: @rpath/libomp.dylib`, run
`brew install libomp` and retry.

## 3. What lives in `data/`

| File(s) | Role | Source |
|---|---|---|
| `2011_postcode3_en.csv` … `2023_postcode3_en.csv` | The claims panel, one file per year, semicolon-delimited, **English headers** | [Vektis Open Data](https://www.vektis.nl/open-data) (translated — see §7) |
| `cpi_price_index_health_cbs_2011_2025.csv` | CBS CPI series `060000 Health`, used to deflate all costs to 2020 constant euros | CBS StatLine |
| `NL_full.csv` | Postcode → municipality + lat/lon lookup | Open postcode dataset |
| `Gebieden_in_Nederland_2023.csv` | Municipality → `GM####` code lookup | CBS "Gebieden in Nederland 2023" |
| `gemeente_2023.geojson` | Choropleth geometry | [cartomap](https://cartomap.github.io/nl/wgs84/gemeente_2023.geojson) (auto-downloaded by `app.py` if missing) |

Raw CSVs are **never modified** — every derived artefact is written elsewhere.

## 4. Build the database

```bash
uv run python preprocessing.py            # writes ./identifier.db
# or: uv run python preprocessing.py <db_path>
```

This runs seven steps: load the claims CSVs (year parsed from the filename) →
load CPI → postcode lookup → municipality codes → drop null-gender rows →
aggregate into 5-year age bands and deflate to 2020 euros (`costs_*_real`) →
enrich with `gementee`, `gementee_code`, `latitude`, `longitude`.

Takes a few minutes. **The EDA dashboard is usable immediately after this step.**

```bash
uv run python app.py                      # http://127.0.0.1:8050
```

## 5. Phase 2/3 — modeling pipeline

Run the whole thing in one go:

```bash
uv run python -m modeling.run_all
```

…or stage by stage (tuned params are persisted and reused, so stages can be
re-run independently):

```bash
uv run python -m modeling.tune            # 1. grid search on the total target   (~75 min)
uv run python -m modeling.train           # 2. fit 8 base models × 24 targets     (~90 min)
uv run python -m modeling.tier4_stacking  # 3. OOF meta-features + Ridge stack
uv run python -m modeling.evaluate        # 4. held-out 2021–2023 metrics, learning curves
```

Then the two optional analysis stages the dashboard and thesis tables consume:

```bash
uv run python -m modeling.shap_compute    # TreeSHAP for Tier 2–3 → artifacts/shap/*.parquet
uv run python -m modeling.bootstrap_ci    # percentile CIs → artifacts/metrics/bootstrap_ci.json
```

Useful variants:

```bash
uv run python -m modeling.train xgboost   # retrain a single model across all targets
```

Everything lands under `artifacts/` (`models/`, `params/`, `metrics/`,
`predictions/`, `learning_curves/`, `timings/`, `shap/`). The dashboard reads
these artefacts only — **it never trains at runtime**.

```bash
uv run python app_models.py               # http://127.0.0.1:8051
```

### Split discipline

Training and tuning use **2011–2020 only**; **2021–2023** is a temporal
hold-out read exactly once, by `modeling/evaluate.py`. Do not add stages that
fit anything on the test years.

## 6. Temporal forecasting track (optional — enables 2024–2030 in the EDA app)

```bash
uv run python -m modeling.temporal.run_all
```

Stage 1 runs the COVID-treatment experiments (7 configs, recursive evaluation)
and writes the winner + leaderboard; stage 2 refits the *deployed* config
(pinned in `modeling/temporal/config.py`) and projects 2024–2030, writing the
`claims_forecast` table and the `claims_extended` view into `identifier.db`.

> **Stage 2 opens `identifier.db` for writing — stop both dashboards first.**

Individual stages:

```bash
uv run python -m modeling.temporal.experiments
uv run python -m modeling.temporal.forecast
uv run python -m modeling.temporal.forecast --model lightgbm --treatment no_covid
```

`app.py` detects `claims_forecast` on start-up and exposes the projected years
(flagged `is_projected`); without it the dashboard simply stays at 2023.

## 7. Adding a newly released Vektis year

Vektis publishes one postcode3 CSV per year here:
**<https://www.vektis.nl/open-data>** (files named
`Vektis Open Databestand Zorgverzekeringswet <year> - postcode3.csv`).

The header translation in `preprocessing.py` (`TRANSLATION_MAP`) is the single
source of truth for the Dutch→English mapping and already covers the historical
variants, including the truncated 2023 names and the year-specific GGZ columns —
so a new year usually needs no code change.

1. **Download** the postcode3 CSV for the new year.
2. **Rename it** to `<year>_postcode3.csv` (e.g. `2024_postcode3.csv`) and drop
   it in `data/`. The name matters twice: the year is parsed from the leading
   digits of the filename, and the build glob is `data/*_postcode3_en.csv`.
3. **Translate the headers** (non-destructive; originals untouched, data rows
   copied verbatim, `<stem>_en.csv` written beside each input; already-`_en`
   files are skipped, so it is idempotent):

   ```bash
   uv run python preprocessing.py translate 'data/*_postcode3.csv'
   ```

4. **Read the output.** Unknown columns are *left as-is and warned about*, never
   dropped:

   ```
   WARNING: 2024_postcode3.csv: 1 untranslated header(s) left as-is: ['kosten_nieuwe_categorie']
   ```

   If you see this, Vektis changed or added a column. Add the mapping to
   `TRANSLATION_MAP`, and if it is a new **cost** category also add it to
   `COST_CATEGORIES` in `modeling/config.py` and `COST_CATEGORIES` in `app.py` so it
   flows into the totals and the dashboards. If it belongs to mental health, add
   it to the `costs_mental_health_total` roll-up in `preprocessing.py::_load_claims`
   rather than as a standalone category, to avoid double counting.
5. **Check CPI coverage.** `data/cpi_price_index_health_cbs_2011_2025.csv`
   currently runs to 2025; a year beyond that needs a new CBS row appended
   (series `060000 Health`, 2015=100), otherwise the deflation join drops the year.
6. **Rebuild and re-run** whatever you need:

   ```bash
   uv run python preprocessing.py          # rebuild identifier.db
   uv run python -m modeling.run_all       # only if the models should learn the new year
   ```

Translate-and-build in one step:

```bash
uv run python preprocessing.py build-from 'data/*_postcode3.csv' [db_path]
```

Build a throwaway DB from an explicit set of globs (no translation):

```bash
uv run python preprocessing.py build-glob /tmp/scratch.db 'data/*_en.csv' 'elsewhere/*_en.csv'
```

> **Scope note.** The thesis panel is 2011–2023 and every modelling decision is
> justified from those years alone. A later release (e.g. 2024) must not feed
> model selection or tuning. To sanity-check the deployed forecast against a
> later year *after* the configuration is frozen, put the translated file in
> `data_for_verification_temporal/` (outside `data/`, so the thesis build cannot
> pick it up) and run the out-of-scope check:
>
> ```bash
> uv run python -m modeling.temporal.verify_external --year 2024
> ```

## 8. Tests

```bash
uv run pytest
```

Covers the temporal lag construction and the COVID treatment splits
(`tests/test_temporal_lags.py`, `tests/test_temporal_treatments.py`).

## 9. Repository layout

```
data/                        raw CSVs + lookups (never modified)
identifier.db                DuckDB build output (gitignored)
preprocessing.py             header translation + DB build (Steps 0–7)
app.py                       Phase 1 EDA dashboard        (port 8050)
app_models.py                Phase 2/3 results dashboard  (port 8051)
eda_analysis.py              Marimo exploration notebook
modeling/
  config.py                  single source of truth: paths, RANDOM_STATE, targets, grids
  data.py encoders.py        loading + train-only encoders
  tune.py train.py           Tiers 1–3: tuning and fitting
  tier1_linear.py … tier4_stacking.py
  evaluate.py metrics.py     held-out evaluation + weighted metrics
  shap_compute.py            offline TreeSHAP matrices
  bootstrap_ci.py            bootstrap CIs for the test metrics
  run_all.py                 Phase 2 orchestrator
  temporal/                  lag-based forecasting track (2024–2030 projection)
artifacts/                   models, params, metrics, predictions, SHAP (gitignored)
tests/                       pytest suite
```

Conventions worth knowing before editing: a global `RANDOM_STATE` (42) drives
every model and split; all cost rates use **`insured_years`** as the denominator
(never `number_of_insured`); only deflated `costs_*_real` columns are modelled;
and the components of the target must never appear as features.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Library not loaded: @rpath/libomp.dylib` | `brew install libomp` |
| `IO Error: Could not set lock on file "identifier.db"` | A dashboard or notebook holds the DB. Stop it, then re-run the write stage. |
| Model dashboard starts but tabs are empty | `artifacts/` is missing or partial — run `uv run python -m modeling.run_all` (and `shap_compute` for the SHAP tab). |
| Choropleth says "could not load gemeente_2023.geojson" | No network on first run; download it manually into `data/` from cartomap. |
| A new year loads but its costs are zero/NULL | Untranslated header or missing CPI row — see §7 steps 4–5. |
