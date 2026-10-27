# Dutch Healthcare Costs by Postcode Area (2011–2023)

Code for the Master's thesis *Analysis and Prediction of Dutch Healthcare
Costs Using Machine Learning: A Longitudinal Study of Vektis Open Data
(2011–2023)*. It analyses Vektis health-insurance claim costs at the
**3-digit postal code × age band × gender × year** grain, in four parts:

1. **Describe:** a Phase 1 EDA dashboard over the 2011–2023 panel.
2. **Predict:** a four-tier model system (linear → trees → boosting →
   stacking) trained on 2011–2020 and evaluated once on the held-out years
   2021–2023 (RQ2).
3. **Explain:** pre-computed TreeSHAP for the tree models (RQ3).
4. **Project:** a lag-based temporal track that rolls the cost rate forward
   to 2030, with each cell's population growing by its CBS age × sex
   forecast.

| Dashboard | Entry point | Port | Content |
|---|---|---|---|
| Phase 1 — EDA | `app.py` | 8050 | Temporal trends, demographics, geography (+ 2024–2030 once the temporal track has been run) |
| Phase 2/3 — Model results | `app_models.py` | 8051 | Model comparison, prediction diagnostics, segmented performance, learning curves, SHAP |

Everything downstream reads from **`identifier.db`** (DuckDB) and
**`artifacts/`**, both gitignored and both rebuilt from the files in `data/`
by the commands below. No manual data wrangling is needed.

**Where each thesis table and figure comes from:** see
[§9 Reproducing the thesis](#9-reproducing-the-thesis).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Dependency and virtual-environment manager. macOS: `brew install uv` |
| Python 3.9 | Pinned in `.python-version`; `uv` installs it if needed |
| OpenMP runtime | **macOS only.** XGBoost and LightGBM need it and `uv` does not ship it: `brew install libomp` |
| ~5 GB free disk | `identifier.db` ~200 MB, `artifacts/` ~3 GB (mostly trained models), `.venv` ~1 GB |

Linux: install OpenMP through the package manager (`libgomp1` on
Debian/Ubuntu, usually already present). Windows: the wheels bundle OpenMP.

All results in the thesis were produced on an Apple Mac mini (M4, 24 GB),
CPU only.

## 2. Environment setup

```bash
git clone https://github.com/theoanto94/thesis.git
cd thesis

uv sync            # creates .venv and installs the locked dependency graph
```

`uv sync` installs the runtime dependencies plus the `dev` group (pytest,
ruff). Every command below is prefixed with `uv run`, so the virtual
environment never needs activating. To check the toolchain:

```bash
uv run python -c "import duckdb, lightgbm, xgboost, catboost, shap, dash; print('ok')"
```

If this fails on macOS with `Library not loaded: @rpath/libomp.dylib`, run
`brew install libomp` and retry.

## 3. Data

| File(s) | Role | Source |
|---|---|---|
| `data/2011_postcode3_en.csv` … `2023_postcode3_en.csv` | The claims panel, one file per year, semicolon-delimited, English headers | [Vektis Open Data](https://www.vektis.nl/open-data) (headers translated, see §8) |
| `data/cpi_price_index_health_cbs_2011_2025.csv` | CBS CPI series `060000 Health`, used to deflate all costs to 2020 euros | CBS StatLine |
| `data/NL_full.csv` | Postcode → municipality + latitude/longitude lookup | Open postcode dataset |
| `data/Gebieden_in_Nederland_2023.csv` | Municipality → `GM####` code lookup | CBS "Gebieden in Nederland 2023" |
| `data/gemeente_2023.geojson` | Municipality geometry for the choropleth | [cartomap](https://cartomap.github.io/nl/wgs84/gemeente_2023.geojson) (downloaded by `app.py` if missing) |
| `data/external/cbs_population_age_sex.csv` | Population by sex and 5-year age band, observed 2023–2024 and forecast 2025–2031, for the projection | CBS [7461BEV](https://www.cbs.nl/nl-nl/cijfers/detail/7461BEV) and kernprognose [86041NED](https://www.cbs.nl/nl-nl/cijfers/detail/86041NED); re-downloaded by `modeling.temporal.population` if missing |

The raw files are **never modified**; every derived file is written elsewhere.

### Usage rule for costs

Every cost rate uses `insured_years` (exposure: persons weighted by their
enrolled fraction of the year) as the denominator, never `number_of_insured`:

```
cost_rate_real = costs_X_real / insured_years
weighted_avg   = Σ costs_X_real / Σ insured_years        (across cells)
```

Only the deflated `costs_*_real` columns are modelled, and the cost columns
that sum into the target never appear as features.

## 4. Build the database

```bash
uv run python preprocessing.py            # writes ./identifier.db
# or: uv run python preprocessing.py <db_path>
```

The build loads the claims CSVs (year parsed from the filename), loads the
CPI, joins the postcode and municipality lookups, drops null-gender rows,
aggregates to 5-year age bands, deflates to 2020 euros (`costs_*_real`), and
adds `gementee`, `gementee_code`, `latitude` and `longitude`. It takes a few
minutes.

**The EDA dashboard works as soon as this step is done:**

```bash
uv run python app.py                      # http://127.0.0.1:8050
```

## 5. Phase 2/3 — modelling pipeline

All at once:

```bash
uv run python -m modeling.run_all         # tune -> train -> stack -> evaluate
```

Or stage by stage. Tuned parameters are saved and reused, so any stage can be
re-run on its own:

```bash
uv run python -m modeling.tune            # 1. grid search on the total target        (~75 min)
uv run python -m modeling.train           # 2. fit 9 base models x 24 targets          (~90 min)
uv run python -m modeling.tier4_stacking  # 3. out-of-fold meta-features + Ridge stack
uv run python -m modeling.evaluate        # 4. held-out 2021-2023 metrics + learning curves
```

The nine base models are Linear, Ridge, Lasso and **Ridge-AG** (Tier 1; a
Ridge on joint age × gender one-hots, the "fair" linear baseline), Random
Forest, GBM and XGBoost (Tier 2), and LightGBM and CatBoost (Tier 3). The
Tier 4 stack combines XGBoost, LightGBM, CatBoost and Random Forest.

Any of the stages accepts model keys to work on a subset, e.g.

```bash
uv run python -m modeling.tune ridge_ag
uv run python -m modeling.train ridge_ag
uv run python -m modeling.evaluate ridge_ag --no-learning-curves
```

`evaluate` updates only the rows of the models it evaluated, so the metric
files keep the other models.

Then the analysis stages that the dashboard and thesis tables read:

```bash
uv run python -m modeling.shap_compute    # TreeSHAP for Tiers 2-3 -> artifacts/shap/*.parquet
uv run python -m modeling.bootstrap_ci    # paired bootstrap CIs, pooled and per test year
uv run python -m modeling.per_year        # metrics + mean signed residual per test year
```

Outputs go under `artifacts/` (`models/`, `params/`, `metrics/`,
`predictions/`, `learning_curves/`, `timings/`, `shap/`). The results
dashboard only reads these files; **it never trains.**

```bash
uv run python app_models.py               # http://127.0.0.1:8051
```

### Split discipline

Training, tuning, feature selection and every encoder fit use **2011–2020
only**. **2021–2023** is a temporal hold-out, read once by
`modeling/evaluate.py`. All cross-validation runs inside the training years.

## 6. Data audits

```bash
uv run python -m modeling.descriptives    # per-year cells, exposure, total cost, cost rate
uv run python -m modeling.coverage_audit  # category coverage changes + fixed-basket growth
```

Both are read-only on `identifier.db` and write to `artifacts/descriptives/`
and `artifacts/coverage/`.

## 7. Temporal forecasting track (enables 2024–2030 in the EDA app)

```bash
uv run python -m modeling.temporal.run_all
```

Stage 1 runs the experiments: persistence plus Ridge, LightGBM and XGBoost,
each with and without the COVID years (seven configurations), scored by
recursive prediction. Stage 2 refits the *deployed* configuration
(Ridge/`base`, pinned in `modeling/temporal/config.py` with the reasoning)
through 2023 and projects 2024–2030. Each 2023 cell's insured-years and
headcount grow by its own CBS age × sex factor (`modeling/temporal/population.py`),
so the projected population ages as CBS forecasts. Stage 2 writes the
`claims_forecast` table and the `claims_extended` view into `identifier.db`.

> **Stage 2 opens `identifier.db` for writing. Stop both dashboards first.**

Then the evaluation tables of the temporal track:

```bash
uv run python -m modeling.temporal.evaluate_extra   # bootstrap CIs, per-year errors,
                                                    # same-row comparison with Phase 2,
                                                    # deployed Ridge coefficients
uv run python -m modeling.temporal.population       # print the CBS exposure factors
```

Individual stages, and the non-deployed LightGBM roll-forward reported beside
the deployed one:

```bash
uv run python -m modeling.temporal.experiments
uv run python -m modeling.temporal.forecast                                      # deployed config
uv run python -m modeling.temporal.forecast --model lightgbm --treatment no_covid
```

> Running `forecast` with `--model` **replaces** the deployed projection in
> `artifacts/temporal/forecast/` and `identifier.db`. Run it once more without
> arguments afterwards to restore the deployed projection.

`app.py` detects `claims_forecast` on start-up and adds the projected years
(flagged `is_projected`); without it the dashboard stays at 2023.

## 8. Adding a newly released Vektis year

Vektis publishes one postcode3 CSV per year at
**<https://www.vektis.nl/open-data>** (`Vektis Open Databestand
Zorgverzekeringswet <year> - postcode3.csv`). The header translation in
`preprocessing.py` (`TRANSLATION_MAP`) is the single source of truth for the
Dutch→English mapping and covers the historical variants, including the
truncated 2023 names and the year-specific mental-health columns, so a new
year usually needs no code change.

1. **Download** the postcode3 CSV for the new year.
2. **Rename it** to `<year>_postcode3.csv` (e.g. `2024_postcode3.csv`) and put
   it in `data/`. The year is parsed from the leading digits of the filename.
3. **Translate the headers.** This is non-destructive: originals stay
   untouched, data rows are copied verbatim, and `<stem>_en.csv` is written
   beside each input. Files already ending in `_en` are skipped.

   ```bash
   uv run python preprocessing.py translate 'data/*_postcode3.csv'
   ```

4. **Read the output.** Unknown columns are left as-is with a warning, never
   dropped:

   ```
   WARNING: 2024_postcode3.csv: 1 untranslated header(s) left as-is: ['kosten_nieuwe_categorie']
   ```

   If this appears, Vektis changed or added a column. Add it to
   `TRANSLATION_MAP`. If it is a new **cost** category, also add it to
   `COST_CATEGORIES` in `modeling/config.py` and in `app.py`. If it belongs to
   mental health, add it to the `costs_mental_health_total` roll-up in
   `preprocessing.py::_load_claims` instead, to avoid double counting.
5. **Check CPI coverage.** `data/cpi_price_index_health_cbs_2011_2025.csv`
   runs to 2025. A later year needs a new CBS row (series `060000 Health`,
   2015=100), otherwise the deflation join drops it.
6. **Rebuild** and re-run what is needed:

   ```bash
   uv run python preprocessing.py          # rebuild identifier.db
   uv run python -m modeling.run_all       # only if the models should learn the new year
   ```

Translate and build in one step, or build a separate database from explicit
globs:

```bash
uv run python preprocessing.py build-from 'data/*_postcode3.csv' [db_path]
uv run python preprocessing.py build-glob /tmp/scratch.db 'data/*_en.csv' 'elsewhere/*_en.csv'
```

> **Scope.** The thesis panel is 2011–2023, and every modelling decision in it
> is justified from those years alone. A later release must not feed model
> selection or tuning of the thesis models.

## 9. Reproducing the thesis

Numbers refer to the submitted thesis. "Dashboard" means a screenshot of the
named card, taken with the filters at their defaults unless stated. Figures
3.1 and 4.1 are diagrams and are not generated by code.

### Chapters 3–5: data and methods

| Thesis | Content | Command | Output |
|---|---|---|---|
| Table 3.1 | The 23 cost categories | `modeling/config.py` (`COST_CATEGORIES`) | — |
| Table 3.2, §3.6 | Coverage changes; fixed-basket and age-standardised growth | `modeling.coverage_audit` | `artifacts/coverage/category_coverage.csv`, `growth.json` |
| Table 3.3 | Dutch→English header map | `preprocessing.py` (`TRANSLATION_MAP`) | — |
| Table 3.4 | Per-year descriptive statistics | `modeling.descriptives` | `artifacts/descriptives/by_year.csv` |
| Tables 4.1–4.2 | Feature set; encoding per model family | `modeling/config.py`, `modeling/encoders.py` | — |
| Table 4.3 | Geographic feature ablation | Development-time selection run, not part of the pipeline | — |
| Table 4.4 | Stack composition selection | Development-time selection run, not part of the pipeline; the result is `STACK_BASE_LEARNERS` in `modeling/config.py` | — |
| Table 4.5, Eq. 4.2 | Temporal model inputs and specification | `modeling/temporal/data.py`, `forecast.py` | — |
| Table 5.1 | Temporal hold-out split | `modeling/config.py` (`TRAIN_YEARS`, `TEST_YEARS`) | — |
| Tables 5.2–5.3 | Grids, selected hyperparameters, grid-search summary | `modeling.tune` | `artifacts/params/<model>.json` |
| Table 5.4 | Resolved library versions | `uv.lock` | — |

### Chapter 6: Phase 1 (EDA)

| Thesis | Content | Source |
|---|---|---|
| Fig. 6.1 | National real cost per insured-year | `app.py`, Temporal tab, card 1.1 |
| Fig. 6.2 | Year-on-year growth | `app.py`, Temporal tab, card 1.4 |
| Fig. 6.3 | Category change 2019→2020 | `app.py`, Temporal tab, card 1.5, years 2019–2020 |
| Fig. 6.4 | Cost rate by care category | `app.py`, Temporal tab, card 1.3 |
| Fig. 6.5 | Cost by age band, 2023 | `app.py`, Demographics tab, card 2.3, year 2023 |
| Fig. 6.6 | Cost by gender | `app.py`, Demographics tab, card 2.2 |
| Fig. 6.7 | Age band × year heatmap | `app.py`, Demographics tab, card 2.1 |
| Table 6.1 | Top and bottom municipalities | `app.py`, Geographic tab, card 3.3 |
| Fig. 6.8 | Municipality choropleth, 2023 | `app.py`, Geographic tab, card 3.1, year 2023 |
| Fig. 6.9 | Distribution of municipal cost rates | `app.py`, Geographic tab, card 3.4 |

### Chapter 6: Phase 2 (prediction)

| Thesis | Content | Command | Output |
|---|---|---|---|
| Table 6.2 | Test metrics with bootstrap CIs | `modeling.evaluate`, then `modeling.bootstrap_ci` | `artifacts/metrics/overall.parquet`, `bootstrap_ci.json` |
| Table 6.3 | Metrics and mean signed residual per test year | `modeling.per_year` | `artifacts/metrics/by_year.parquet` |
| Table 6.4 | Training time and memory | `modeling.train` | `artifacts/timings/<model>.json` |
| Table 6.5 | Metrics by age band, incl. Ridge-AG within-band R² | `modeling.evaluate` | `artifacts/metrics/by_age_group.parquet` |
| Table 6.6 | Stack metrics per target | `modeling.evaluate` | `artifacts/metrics/overall.parquet` |
| Figs. 6.10–6.11 | Actual vs predicted; residuals | `app_models.py`, Prediction Diagnostics tab (stack; Ridge for Fig. 6.10 right) | — |
| Figs. 6.12–6.13 | RMSE by age band; by cost category | `app_models.py`, Segmented Performance tab | — |
| Fig. 6.14 | Learning curves of the stack's base learners | `modeling.evaluate`; `app_models.py`, Learning Curves tab, model = stack | `artifacts/learning_curves/` |

### Chapter 6: Phase 3 (SHAP)

| Thesis | Content | Command | Output |
|---|---|---|---|
| Table 6.7 | Mean \|SHAP\| per feature per model | `modeling.shap_compute` | `artifacts/shap/<model>.parquet` |
| Figs. 6.15–6.16 | Beeswarm and importance; comparison across models | `app_models.py`, SHAP tab, charts 6.1–6.3 | — |
| Figs. 6.17–6.21 | SHAP dependence (age, gender, longitude, headcount, year) | `app_models.py`, SHAP tab, chart 6.4, model LightGBM | — |

### Chapter 6: temporal track and projection

| Thesis | Content | Command | Output |
|---|---|---|---|
| Tables 6.8–6.10 | Protocol A and B results with CIs; per-year errors | `modeling.temporal.run_all`, then `modeling.temporal.evaluate_extra` | `artifacts/temporal/metrics/experiments.parquet`, `bootstrap_ci.json`, `by_year.parquet` |
| §6.4.1, §7.1 | Same-row comparison with the Phase 2 stack | `modeling.temporal.evaluate_extra` | `artifacts/temporal/metrics/phase2_same_rows_ci.json` |
| §6.4.2 | Fitted ρ and θ of the deployed Ridge | `modeling.temporal.evaluate_extra` | `artifacts/temporal/metrics/ridge_coefficients.json` |
| Table 6.11 | Seven-year roll-forward, LightGBM vs deployed Ridge | `modeling.temporal.forecast --model lightgbm --treatment no_covid`, then `modeling.temporal.forecast` (see §7) | `artifacts/temporal/forecast/forecast_summary.json` |
| Table 6.12 | Deployed 2024–2030 projection | `modeling.temporal.forecast` | `artifacts/temporal/forecast/forecast_summary.json` |
| Fig. 6.22 | Projected cost rate and total cost to 2030 | `app.py`, Temporal tab, cards 1.1 and 1.2, full 2011–2030 range, after the forecast | — |

All randomness (model seeds, CV splits, subsamples, bootstrap draws, the SHAP
sample) is driven by `RANDOM_STATE = 42` in `modeling/config.py`.

## 10. Tests and linting

```bash
uv run pytest           # 18 tests, synthetic data, no database needed
uv run ruff check .
```

| File | Covers |
|---|---|
| `tests/test_temporal_lags.py` | Lag construction: gaps, entity boundaries, row count |
| `tests/test_temporal_treatments.py` | COVID treatment splits and the recursive prediction chain |
| `tests/test_bootstrap_population.py` | Paired bootstrap core; CBS exposure factors reproduce the CBS totals |
| `tests/test_encoders_evaluate.py` | Ridge-AG's age × gender encoder; metric files keep other models on update |

## 11. Repository layout

```
data/                        raw CSVs + lookups (never modified)
  external/                  CBS population extract for the projection
preprocessing.py             header translation + database build
app.py                       Phase 1 EDA dashboard           (port 8050)
app_models.py                Phase 2/3 results dashboard     (port 8051)
eda_analysis.py              Marimo exploration notebook     (uv run marimo edit eda_analysis.py)
modeling/
  config.py                  single source of truth: paths, RANDOM_STATE, targets, features, grids
  data.py                    loading, temporal split, leakage assertion
  encoders.py                per-family preprocessing, fitted on training folds only
  factory.py                 uniform build/fit/predict across model families
  tier1_linear.py            Linear, Ridge, Lasso, Ridge-AG
  tier2_trees.py             Random Forest, GBM, XGBoost
  tier3_boosting.py          LightGBM, CatBoost
  tier4_stacking.py          out-of-fold predictions + Ridge meta-learner
  tune.py train.py           grid search; fitting across the 24 targets
  evaluate.py metrics.py     held-out evaluation; population-weighted metrics
  bootstrap_ci.py per_year.py  confidence intervals; per-test-year metrics and bias
  shap_compute.py            offline TreeSHAP matrices
  descriptives.py coverage_audit.py  data audits for Chapter 3
  run_all.py                 Phase 2 orchestrator
  temporal/                  lag-based track: experiments, forecast, CBS population,
                             extra evaluation, orchestrator
artifacts/                   models, params, metrics, predictions, SHAP (gitignored)
tests/                       pytest suite
```

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `Library not loaded: @rpath/libomp.dylib` | `brew install libomp` |
| `IO Error: Could not set lock on file "identifier.db"` | A dashboard or notebook holds the database. Stop it, then re-run the writing stage. |
| Results dashboard starts but tabs are empty | `artifacts/` is missing or partial. Run `uv run python -m modeling.run_all` (and `shap_compute` for the SHAP tab). |
| Choropleth says "could not load gemeente_2023.geojson" | No network on first run; download it into `data/` from cartomap. |
| A new year loads but its costs are zero or NULL | Untranslated header or missing CPI row; see §8 steps 4–5. |
| EDA app stops at 2023 | The temporal track has not been run; see §7. |
