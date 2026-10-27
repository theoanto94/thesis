"""Phase 2 Results Dashboard — model comparison & prediction diagnostics.

A separate Dash app from the EDA dashboard in app.py. It only reads the
artifacts modeling/evaluate.py persisted and never trains anything at runtime.
All filters live in a single shared dcc.Store.

    uv run python app_models.py   # http://127.0.0.1:8051
"""

from __future__ import annotations

import functools
import json

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from modeling import config
from modeling.metrics import weighted_metrics
from modeling.tier4_stacking import STACK_KEY

PORT = 8051

# Reuse the EDA dashboard's visual language for a consistent look.
PALETTE = {
    "primary": "#1c4778", "accent": "#dc6803", "muted": "#6b7280",
    "bg": "#f4f6f8", "card": "#ffffff", "grid": "#e5e7eb", "ink": "#1f2937",
    "good": "#16a34a", "bad": "#dc2626",
}
PLOTLY_LAYOUT = dict(
    template="simple_white",
    margin=dict(l=55, r=20, t=50, b=45),
    font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=12, color=PALETTE["ink"]),
    colorway=[PALETTE["primary"], PALETTE["accent"], "#22c55e", "#a855f7",
              "#ef4444", "#06b6d4", "#84cc16", "#eab308", "#64748b"],
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=PALETTE["grid"]),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
)

MODEL_ORDER = [k for k in config.MODEL_KEYS]
METRIC_OPTIONS = [
    {"label": "R²", "value": "r2"},
    {"label": "RMSE (€)", "value": "rmse"},
    {"label": "MAE (€)", "value": "mae"},
    {"label": "MAPE (%)", "value": "mape"},
]
# Drives the highlighting and ranking direction.
LOWER_IS_BETTER = {"rmse", "mae", "mape"}


# Missing files are tolerated so the app still boots without artifacts.

def _read_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_overall() -> pd.DataFrame:
    """Per-(model, target) overall test metrics."""
    return _read_parquet(config.METRICS_DIR / "overall.parquet")


@functools.lru_cache(maxsize=None)
def load_predictions(model_key: str) -> pd.DataFrame:
    """Row-level test predictions for one model (cached)."""
    return _read_parquet(config.PREDICTIONS_DIR / f"{model_key}.parquet")


def load_timings() -> pd.DataFrame:
    """Computational-cost records (one JSON per model)."""
    rows = []
    for key in config.MODEL_KEYS:
        path = config.TIMINGS_DIR / f"{key}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return pd.DataFrame(rows)


def load_learning_curve(model_key: str) -> pd.DataFrame:
    """Saved train/CV learning curve of *model_key* (empty if missing)."""
    return _read_parquet(config.LEARNING_CURVES_DIR / f"{model_key}.parquet")


# Feature labels for the SHAP tab.
SHAP_FEATURE_LABELS = {
    "age_group": "Age band", "gender": "Gender", "postal_code_3": "Postcode area (PC3)",
    "year": "Year", "insured_years": "Insured years (exposure)",
    "number_of_insured": "Insured headcount", "latitude": "Latitude", "longitude": "Longitude",
}
# Gender is stored M/V.
GENDER_LABELS = {"M": "Male", "V": "Female"}


def _humanize(feature: str, raw_value) -> str:
    """Pretty-print a raw categorical value."""
    if feature == "gender":
        return GENDER_LABELS.get(str(raw_value), str(raw_value))
    return str(raw_value)


def _feature_plot_values(df: pd.DataFrame, feature: str):
    """Return ``(x_values, ticks)`` for plotting a feature on an axis.

    Low-cardinality categoricals get a label-ordered code from the raw
    companion column, so the axis reads young→old regardless of how the model
    encoded the category internally, and ticks maps those codes to labels.
    Everything else plots the stored model-input value, with no ticks.
    """
    if feature in config.LOW_CARD_FEATURES and feature in df.columns:
        cats = sorted(df[feature].astype(str).unique())
        code = {c: i for i, c in enumerate(cats)}
        x = df[feature].astype(str).map(code).to_numpy(dtype=float)
        ticks = (list(range(len(cats))), [_humanize(feature, c) for c in cats])
        return x, ticks
    return df[f"val_{feature}"].to_numpy(dtype=float), None


@functools.lru_cache(maxsize=None)
def load_shap(model_key: str) -> pd.DataFrame:
    """Pre-computed per-instance SHAP matrix for one model (cached)."""
    return _read_parquet(config.SHAP_DIR / f"{model_key}.parquet")


def shap_models() -> list[str]:
    """In-scope SHAP models (Tier 2-3) that actually have a saved matrix."""
    return [m for m in config.SHAP_MODELS if not load_shap(m).empty]


def _filter_shap(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Slice a SHAP matrix to the active region/age/year filters.

    Cost category is ignored on purpose — these matrices explain the single
    total cost-rate target and are not per-category.
    """
    if df.empty:
        return df
    df = df[df["year"].astype(int).between(f["year_min"], f["year_max"])]
    if f.get("regions"):
        df = df[df["gementee"].isin(f["regions"])]
    if f.get("age_groups"):
        df = df[df["age_group"].isin(f["age_groups"])]
    return df


def _weighted_mean_abs_shap(df: pd.DataFrame) -> pd.Series:
    """Population-weighted mean(|SHAP|) per feature, descending."""
    w = df["weight"].to_numpy(dtype=float)
    wsum = w.sum()
    out = {}
    for fkey in config.FEATURES:
        a = np.abs(df[f"shap_{fkey}"].to_numpy(dtype=float))
        out[fkey] = float((a * w).sum() / wsum) if wsum > 0 else float("nan")
    return pd.Series(out).sort_values(ascending=False)


def available_models() -> list[str]:
    """Models that actually have an overall-metrics row, in tier order."""
    ov = load_overall()
    present = set(ov["model"].unique()) if not ov.empty else set()
    return [m for m in MODEL_ORDER if m in present] or MODEL_ORDER


def available_regions() -> list[str]:
    """Gemeenten present in the saved test predictions, for the region filter."""
    ov_model = available_models()[0]
    preds = load_predictions(ov_model)
    if preds.empty:
        return []
    return sorted(preds["gementee"].dropna().astype(str).unique())


# Plot helpers

def style(fig, **overrides):
    """Apply the shared Plotly layout (with *overrides*) and axis styling."""
    fig.update_layout(**{**PLOTLY_LAYOUT, **overrides})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


def empty_fig(message: str = "No data for current filters") -> go.Figure:
    """Blank figure carrying a centred *message*."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=PALETTE["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig)


def _filter_predictions(model_key: str, f: dict) -> pd.DataFrame:
    """Slice one model's predictions to the active filter state."""
    df = load_predictions(model_key)
    if df.empty:
        return df
    df = df[df["target"] == f["target"]]
    df = df[df["year"].between(f["year_min"], f["year_max"])]
    if f.get("regions"):
        df = df[df["gementee"].isin(f["regions"])]
    if f.get("age_groups"):
        df = df[df["age_group"].isin(f["age_groups"])]
    return df


def _wm(df: pd.DataFrame) -> dict:
    if df.empty:
        return {k: float("nan") for k in ("r2", "rmse", "mae", "mape", "mape_coverage")}
    return weighted_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(),
                            df["weight"].to_numpy())


# Layout

def _target_options():
    # "Total" first, then the 23 categories.
    keys = [config.TARGET_TOTAL] + [k for k in config.TARGETS if k != config.TARGET_TOTAL]
    return [{"label": config.TARGET_LABELS[k], "value": k} for k in keys]


def filter_bar() -> dbc.Card:
    """Persistent top bar holding the global filters."""
    models = available_models()
    regions = available_regions()
    age_opts = [{"label": a, "value": a} for a in sorted(
        load_predictions(models[0])["age_group"].dropna().astype(str).unique()
    )] if not load_predictions(models[0]).empty else []

    return dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([
            html.Label("Model", className="small fw-semibold text-muted"),
            dcc.Dropdown(id="f-model", options=[{"label": config.MODEL_LABELS[m], "value": m}
                                                 for m in models],
                         value=models[-1] if models else None, clearable=False),
        ], md=2),
        dbc.Col([
            html.Label("Cost category", className="small fw-semibold text-muted"),
            dcc.Dropdown(id="f-target", options=_target_options(),
                         value=config.TARGET_TOTAL, clearable=False),
        ], md=3),
        dbc.Col([
            html.Label("Test years", className="small fw-semibold text-muted"),
            dcc.RangeSlider(id="f-years", min=2021, max=2023, step=1, value=[2021, 2023],
                            marks={y: str(y) for y in range(2021, 2024)}),
        ], md=2),
        dbc.Col([
            html.Label("Region (gemeente)", className="small fw-semibold text-muted"),
            dcc.Dropdown(id="f-regions", options=[{"label": r, "value": r} for r in regions],
                         multi=True, placeholder="All regions"),
        ], md=3),
        dbc.Col([
            html.Label("Age group", className="small fw-semibold text-muted"),
            dcc.Dropdown(id="f-ages", options=age_opts, multi=True, placeholder="All ages"),
        ], md=2),
    ], className="g-2 align-items-end")), className="border-0 shadow-sm mb-3")


def header() -> html.Header:
    """Page header."""
    return html.Header(dbc.Container(html.Div([
        html.Div([
            html.H2("Healthcare Cost — Model Results", className="mb-0",
                    style={"color": PALETTE["primary"], "fontWeight": 600}),
            html.Div("Phase 2 · predicting real cost rate (€2020 / insured year) · "
                     "trained 2011–2020 · tested 2021–2023",
                     className="text-muted small mt-1"),
        ]),
        dbc.Badge("Model dashboard", color="light", text_color="dark"),
    ], className="d-flex justify-content-between align-items-center py-3"), fluid=True),
        style={"backgroundColor": PALETTE["card"], "borderBottom": f"1px solid {PALETTE['grid']}"})


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Healthcare Model Results",
           # Tab bodies and their dropdowns are built by the tab-router
           # callback, so they aren't in the initial layout.
           suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id="filters"),
    header(),
    dbc.Container([
        filter_bar(),
        dbc.Tabs([
            dbc.Tab(label="Model Comparison", tab_id="t1"),
            dbc.Tab(label="Prediction Diagnostics", tab_id="t2"),
            dbc.Tab(label="Segmented Performance", tab_id="t3"),
            dbc.Tab(label="Learning Curves", tab_id="t4"),
            dbc.Tab(label="SHAP Explainability", tab_id="t5"),
        ], id="tabs", active_tab="t1", className="mb-3"),
        html.Div(id="tab-content"),
    ], fluid=True, className="py-3"),
], style={"backgroundColor": PALETTE["bg"], "minHeight": "100vh"})


# Filter store

@app.callback(
    Output("filters", "data"),
    Input("f-model", "value"), Input("f-target", "value"), Input("f-years", "value"),
    Input("f-regions", "value"), Input("f-ages", "value"),
)
def _sync_filters(model, target, years, regions, ages):
    years = years or [2021, 2023]
    return {"model": model, "target": target,
            "year_min": years[0], "year_max": years[1],
            "regions": regions or [], "age_groups": ages or []}


# Tab router

@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"), Input("filters", "data"))
def _render_tab(active, f):
    if not f:
        return dbc.Alert("Loading…", color="light")
    if load_overall().empty:
        return dbc.Alert(
            "No model artifacts found. Run `uv run python -m modeling.train`, "
            "`-m modeling.tier4_stacking`, then `-m modeling.evaluate` first.",
            color="warning")
    return {"t1": tab_comparison, "t2": tab_diagnostics,
            "t3": tab_segmented, "t4": tab_learning, "t5": tab_shap}[active](f)


@app.callback(
    Output("f-target", "disabled"), Output("f-target", "className"),
    Input("tabs", "active_tab"),
)
def _toggle_target_filter(active):
    """Grey out the cost-category filter on the SHAP tab.

    SHAP explains the single total cost-rate target, so a per-category
    selection would do nothing there.
    """
    if active == "t5":
        return True, "opacity-50"
    return False, ""


# Tab 1 — Model Comparison

def tab_comparison(f: dict):
    """Tab 1: metrics table, metric bar chart and computational cost table."""
    ov = load_overall()
    sub = ov[ov["target"] == f["target"]].copy()
    sub = sub.set_index("model").reindex(
        [m for m in MODEL_ORDER if m in set(sub["model"]) | set(sub.index)]
    ).dropna(how="all").reset_index()
    sub["Model"] = sub["model"].map(config.MODEL_LABELS)

    # 1.1 metrics table with best-per-column highlighting.
    table_cols = [("r2", "R²", "{:.3f}"), ("rmse", "RMSE (€)", "{:,.0f}"),
                  ("mae", "MAE (€)", "{:,.0f}"), ("mape", "MAPE (%)", "{:.1f}")]
    records, styles = [], []
    for _, r in sub.iterrows():
        records.append({"Model": r["Model"],
                        **{disp: fmt.format(r[c]) for c, disp, fmt in table_cols}})
    for c, disp, _ in table_cols:
        if sub[c].notna().any():
            best = sub[c].idxmax() if c not in LOWER_IS_BETTER else sub[c].idxmin()
            best_model = sub.loc[best, "Model"]
            styles.append({"if": {"filter_query": f"{{Model}} = '{best_model}'", "column_id": disp},
                           "backgroundColor": "#e8f0fe", "fontWeight": "700"})
    metrics_table = dash_table.DataTable(
        data=records, columns=[{"name": "Model", "id": "Model"}]
        + [{"name": d, "id": d} for _, d, _ in table_cols],
        sort_action="native", style_as_list_view=True,
        style_cell={"padding": "8px 12px", "fontFamily": "Inter, Arial", "fontSize": "13px"},
        style_header={"fontWeight": "700", "backgroundColor": PALETTE["bg"]},
        style_data_conditional=styles)

    # 1.3 computational cost table.
    timings = load_timings()
    if not timings.empty:
        timings = timings.copy()
        timings["Model"] = timings["model"].map(config.MODEL_LABELS)
        tcols = {"Model": "Model", "train_seconds_total": "Train time (s, all targets)",
                 "train_seconds_per_target": "Train time (s/target)", "model_size_mb": "Model size (MB)"}
        timings = timings[[c for c in tcols if c in timings.columns]].rename(columns=tcols)
        cost_table = dash_table.DataTable(
            data=timings.to_dict("records"),
            columns=[{"name": c, "id": c} for c in timings.columns],
            sort_action="native", style_as_list_view=True,
            style_cell={"padding": "8px 12px", "fontFamily": "Inter, Arial", "fontSize": "13px"},
            style_header={"fontWeight": "700", "backgroundColor": PALETTE["bg"]})
    else:
        cost_table = html.Div("No timing records.", className="text-muted small")

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6(f"Test metrics — {config.TARGET_LABELS[f['target']]}", className="text-muted"),
            metrics_table]), className="border-0 shadow-sm mb-3"), md=7),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([html.H6("Metric comparison", className="text-muted d-inline me-3"),
                      dcc.Dropdown(id="cmp-metric", options=METRIC_OPTIONS, value="rmse",
                                   clearable=False, style={"width": "160px", "display": "inline-block"})]),
            dcc.Graph(id="cmp-bar", config={"displayModeBar": False})]),
            className="border-0 shadow-sm mb-3"), md=5),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Computational cost (Orji & Ukwandu 2024, Table 7 style)", className="text-muted"),
            cost_table]), className="border-0 shadow-sm"), md=12),
    ])


@app.callback(Output("cmp-bar", "figure"), Input("cmp-metric", "value"), Input("filters", "data"))
def _comparison_bar(metric, f):
    if not f:
        return empty_fig()
    ov = load_overall()
    sub = ov[ov["target"] == f["target"]].copy()
    if sub.empty or metric not in sub:
        return empty_fig()
    sub["Model"] = sub["model"].map(config.MODEL_LABELS)
    sub = sub.sort_values(metric, ascending=metric in LOWER_IS_BETTER)
    fig = go.Figure(go.Bar(x=sub[metric], y=sub["Model"], orientation="h",
                           marker_color=PALETTE["primary"]))
    label = next(o["label"] for o in METRIC_OPTIONS if o["value"] == metric)
    return style(fig, height=360, xaxis_title=label, yaxis_title=None,
                 yaxis=dict(autorange="reversed", type="category"))


# Tab 2 — Prediction Diagnostics

def tab_diagnostics(f: dict):
    """Tab 2: actual vs predicted, residuals vs fitted and residual histogram."""
    df = _filter_predictions(f["model"], f)
    if df.empty:
        return empty_fig_card("No predictions for current filters.")

    yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    m = _wm(df)

    # 2.1 actual vs predicted (sampled for rendering) + identity + best-fit + R².
    n = len(df)
    sample = df.sample(min(n, 6000), random_state=config.RANDOM_STATE)
    lim = float(np.nanpercentile(np.concatenate([yt, yp]), 99.5))
    scatter = go.Figure()
    scatter.add_trace(go.Scattergl(x=sample["y_true"], y=sample["y_pred"], mode="markers",
                                   marker=dict(size=4, color=PALETTE["primary"], opacity=0.35),
                                   name="cells", hovertemplate="actual €%{x:,.0f}<br>pred €%{y:,.0f}<extra></extra>"))
    scatter.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="y = x",
                                 line=dict(color=PALETTE["muted"], dash="dash")))
    if n >= 2:
        b1, b0 = np.polyfit(yt, yp, 1)
        scatter.add_trace(go.Scatter(x=[0, lim], y=[b0, b0 + b1 * lim], mode="lines",
                                     name="best fit", line=dict(color=PALETTE["accent"])))
    scatter.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False,
                           align="left", font=dict(size=13, color=PALETTE["ink"]),
                           text=f"R² = {m['r2']:.3f}<br>RMSE = €{m['rmse']:,.0f}<br>n = {n:,}")
    style(scatter, height=420, xaxis_title="Actual (€/insured yr)",
          yaxis_title="Predicted (€/insured yr)",
          xaxis=dict(range=[0, lim]), yaxis=dict(range=[0, lim]))

    # 2.3 residuals vs fitted.
    resid = yt - yp
    rsample = sample.assign(resid=sample["y_true"] - sample["y_pred"])
    res_fit = go.Figure()
    res_fit.add_trace(go.Scattergl(x=rsample["y_pred"], y=rsample["resid"], mode="markers",
                                   marker=dict(size=4, color=PALETTE["primary"], opacity=0.35)))
    res_fit.add_hline(y=0, line=dict(color=PALETTE["muted"], dash="dash"))
    style(res_fit, height=320, xaxis_title="Fitted (predicted €)", yaxis_title="Residual (actual − pred)")

    # 2.4 residual distribution.
    rlim = float(np.nanpercentile(np.abs(resid), 99))
    hist = go.Figure(go.Histogram(x=np.clip(resid, -rlim, rlim), nbinsx=60,
                                  marker_color=PALETTE["primary"]))
    hist.add_vline(x=0, line=dict(color=PALETTE["accent"], dash="dash"))
    style(hist, height=320, xaxis_title="Residual (€)", yaxis_title="Count")

    return html.Div([
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
            html.H6(f"Actual vs Predicted — {config.MODEL_LABELS[f['model']]} · "
                    f"{config.TARGET_LABELS[f['target']]}", className="text-muted"),
            dcc.Graph(figure=scatter, config={"displayModeBar": False})]),
            className="border-0 shadow-sm mb-3"), md=12)),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Residuals vs fitted", className="text-muted"),
                    dcc.Graph(figure=res_fit, config={"displayModeBar": False})]),
                    className="border-0 shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Residual distribution", className="text-muted"),
                    dcc.Graph(figure=hist, config={"displayModeBar": False})]),
                    className="border-0 shadow-sm"), md=6),
        ]),
    ])


# Tab 3 — Segmented Performance

def tab_segmented(f: dict):
    """Tab 3: metrics by segment and the best model per segment."""
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([html.Span("Segment by ", className="text-muted small me-2"),
                dcc.Dropdown(id="seg-dim", clearable=False, value="age_group",
                             options=[{"label": "Age group", "value": "age_group"},
                                      {"label": "Region (gemeente)", "value": "gementee"},
                                      {"label": "Cost category", "value": "target"}],
                             style={"width": "220px", "display": "inline-block"})],
                className="d-flex align-items-center"), md=6),
            dbc.Col(html.Div([html.Span("Error metric ", className="text-muted small me-2"),
                dcc.Dropdown(id="seg-metric", clearable=False, value="rmse",
                             options=METRIC_OPTIONS,
                             style={"width": "160px", "display": "inline-block"})],
                className="d-flex align-items-center justify-content-end"), md=6),
        ], className="mb-3"),
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Per-model metrics (current filters)", className="text-muted"),
            html.Div(id="seg-table")]), className="border-0 shadow-sm mb-3"), md=12)),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Error by segment (selected model)", className="text-muted"),
                    dcc.Graph(id="seg-bar", config={"displayModeBar": False})]),
                    className="border-0 shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Best model per segment", className="text-muted"),
                    dcc.Graph(id="seg-rank", config={"displayModeBar": False})]),
                    className="border-0 shadow-sm"), md=6),
        ]),
    ])


def _segmented_table_for_all_models(f: dict) -> pd.DataFrame:
    """Recompute weighted metrics per model under the active filters."""
    rows = []
    for m in available_models():
        df = _filter_predictions(m, f)
        met = _wm(df)
        met.update({"Model": config.MODEL_LABELS[m], "n": int(len(df))})
        rows.append(met)
    return pd.DataFrame(rows)


@app.callback(Output("seg-table", "children"), Input("filters", "data"))
def _seg_table(f):
    if not f:
        return ""
    df = _segmented_table_for_all_models(f)
    if df.empty:
        return html.Div("No data.", className="text-muted small")
    show = df[["Model", "r2", "rmse", "mae", "mape", "n"]].copy()
    for c, fmt in [("r2", "{:.3f}"), ("rmse", "€{:,.0f}"), ("mae", "€{:,.0f}"), ("mape", "{:.1f}%")]:
        show[c] = df[c].map(lambda v: fmt.format(v) if pd.notna(v) else "—")
    show.columns = ["Model", "R²", "RMSE", "MAE", "MAPE", "n cells"]
    return dash_table.DataTable(
        data=show.to_dict("records"), columns=[{"name": c, "id": c} for c in show.columns],
        sort_action="native", style_as_list_view=True,
        style_cell={"padding": "8px 12px", "fontFamily": "Inter, Arial", "fontSize": "13px"},
        style_header={"fontWeight": "700", "backgroundColor": PALETTE["bg"]})


@app.callback(Output("seg-bar", "figure"), Input("seg-dim", "value"),
              Input("seg-metric", "value"), Input("filters", "data"))
def _seg_bar(dim, metric, f):
    if not f:
        return empty_fig()
    df = _filter_predictions(f["model"], f) if dim != "target" else load_predictions(f["model"])
    if dim == "target":  # category view ignores the single-target filter
        df = df[df["year"].between(f["year_min"], f["year_max"])]
        if f.get("regions"):
            df = df[df["gementee"].isin(f["regions"])]
        if f.get("age_groups"):
            df = df[df["age_group"].isin(f["age_groups"])]
    if df.empty:
        return empty_fig()
    rows = []
    for g, sub in df.groupby(dim, observed=True):
        met = weighted_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), sub["weight"].to_numpy())
        label = config.TARGET_LABELS.get(g, g) if dim == "target" else g
        rows.append({"seg": label, "val": met[metric]})
    d = pd.DataFrame(rows).dropna().sort_values("val", ascending=metric in LOWER_IS_BETTER)
    d = d.head(25)
    # Categorical axis, or Plotly reads age bands like "10-14" as dates.
    fig = go.Figure(go.Bar(x=d["val"], y=d["seg"].astype(str), orientation="h",
                           marker_color=PALETTE["primary"]))
    label = next(o["label"] for o in METRIC_OPTIONS if o["value"] == metric)
    return style(fig, height=max(320, 18 * len(d)), xaxis_title=label,
                 yaxis=dict(autorange="reversed", type="category"))


@app.callback(Output("seg-rank", "figure"), Input("seg-dim", "value"),
              Input("seg-metric", "value"), Input("filters", "data"))
def _seg_rank(dim, metric, f):
    if not f or dim == "target":
        return empty_fig("Best-model-per-segment uses age or region segments.")
    frames = []
    for m in available_models():
        df = _filter_predictions(m, f)
        if df.empty:
            continue
        for g, sub in df.groupby(dim, observed=True):
            met = weighted_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), sub["weight"].to_numpy())
            frames.append({"seg": str(g), "model": m, "val": met[metric],
                           "exposure": float(sub["weight"].sum())})
    if not frames:
        return empty_fig()
    d = pd.DataFrame(frames).dropna(subset=["val"])
    idx = (d.groupby("seg")["val"].idxmin() if metric in LOWER_IS_BETTER
           else d.groupby("seg")["val"].idxmax())
    best = d.loc[idx].copy()
    # With ~330 regions, keep the most populous and show them in segment order.
    best = best.sort_values("exposure", ascending=False).head(25).sort_values("seg")

    label = next(o["label"] for o in METRIC_OPTIONS if o["value"] == metric)
    # One trace per winning model, so the bars carry a model legend.
    fig = go.Figure()
    for m in [k for k in MODEL_ORDER if k in set(best["model"])]:
        bm = best[best["model"] == m]
        fig.add_trace(go.Bar(y=bm["seg"], x=bm["val"], orientation="h",
                             name=config.MODEL_LABELS[m],
                             hovertemplate="%{y}<br>%{x:,.1f}<extra>" + config.MODEL_LABELS[m] + "</extra>"))
    return style(fig, height=max(320, 24 * len(best)), barmode="stack",
                 xaxis_title=f"Best {label} in segment (bar = winning model)",
                 yaxis=dict(type="category", autorange="reversed"),
                 legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0))


# Tab 4 — Learning Curves

def tab_learning(f: dict):
    """Tab 4: learning curve of the selected model (base learners for the stack)."""
    if f["model"] == STACK_KEY:
        # The meta-model has no learning curve of its own, so overlay its base
        # learners' CV curves.
        fig = go.Figure()
        has_data = False
        for k in config.STACK_BASE_LEARNERS:
            lc = load_learning_curve(k)
            if lc.empty:
                continue
            has_data = True
            fig.add_trace(go.Scatter(x=lc["train_size"], y=lc["cv_rmse"],
                                     mode="lines+markers", name=config.MODEL_LABELS[k]))
        if not has_data:
            return empty_fig_card("No learning-curve artifacts found.")
        style(fig, height=420, xaxis_title="Training set size (rows)",
              yaxis_title="Weighted CV RMSE (€)")
        return dbc.Card(dbc.CardBody([
            html.H6("Learning curves — stacked ensemble base learners (CV RMSE, total cost rate)",
                    className="text-muted"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            dbc.Alert("The stacked meta-model has no standalone learning curve; shown are the "
                      "CV curves of its base learners. Select a base model above for its full "
                      "train-vs-CV curve.", color="light", className="mt-2 mb-0 py-2 small"),
        ]), className="border-0 shadow-sm")
    lc = load_learning_curve(f["model"])
    if lc.empty:
        return empty_fig_card("No learning-curve artifact for this model.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lc["train_size"], y=lc["train_rmse"], mode="lines+markers",
                             name="Train RMSE", line=dict(color=PALETTE["primary"])))
    fig.add_trace(go.Scatter(x=lc["train_size"], y=lc["cv_rmse"], mode="lines+markers",
                             name="CV RMSE", line=dict(color=PALETTE["accent"])))
    style(fig, height=420, xaxis_title="Training set size (rows)", yaxis_title="Weighted RMSE (€)")

    # 5.2 auto plateau flag.
    cv = lc.sort_values("train_size")["cv_rmse"].to_numpy()
    note, color = "Curve still improving — more data may help.", "warning"
    if len(cv) >= 2:
        rel = abs(cv[-1] - cv[-2]) / (abs(cv[-2]) + 1e-9)
        if rel < 0.01:
            note, color = ("CV error has plateaued (<1% change at the last step) — "
                           "diminishing returns from more data."), "success"
    return dbc.Card(dbc.CardBody([
        html.H6(f"Learning curve — {config.MODEL_LABELS[f['model']]} (total cost rate)",
                className="text-muted"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dbc.Alert(note, color=color, className="mt-2 mb-0 py-2 small"),
    ]), className="border-0 shadow-sm")


# Tab 5 — SHAP Explainability (pre-computed matrices only)

def _active_shap_model(f: dict) -> str | None:
    """The selected model if it's in SHAP scope, else a sensible default."""
    models = shap_models()
    if not models:
        return None
    return f["model"] if f.get("model") in models else models[0]


def _beeswarm_fig(df: pd.DataFrame, height: int = 460, max_points: int = 5000) -> go.Figure:
    """Manual SHAP beeswarm: one jittered strip per feature, coloured by value.

    Ordered by population-weighted mean(|SHAP|), subsampled for display, and
    normalised per feature so one colour scale reads low→high across them all.
    """
    if df.empty:
        return empty_fig("No instances for current filters.")
    order = _weighted_mean_abs_shap(df).index.tolist()
    n_feat = len(order)
    if len(df) > max_points:
        df = df.sample(max_points, random_state=config.RANDOM_STATE)
    rng = np.random.RandomState(config.RANDOM_STATE)

    xs, ys, colors = [], [], []
    for i, fkey in enumerate(order):
        yrow = n_feat - 1 - i  # most important feature at the top
        s = df[f"shap_{fkey}"].to_numpy(dtype=float)
        v = df[f"val_{fkey}"].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
        vn = (np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo
              else np.full_like(v, 0.5, dtype=float))
        xs.append(s)
        ys.append(yrow + rng.uniform(-0.33, 0.33, size=len(s)))
        colors.append(vn)

    fig = go.Figure(go.Scattergl(
        x=np.concatenate(xs), y=np.concatenate(ys), mode="markers",
        marker=dict(size=4, color=np.concatenate(colors), colorscale="RdYlBu_r",
                    opacity=0.55, cmin=0, cmax=1,
                    colorbar=dict(title="Feature<br>value", tickvals=[0, 1],
                                  ticktext=["low", "high"])),
        hovertemplate="SHAP %{x:+.0f} €<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=PALETTE["muted"], dash="dot"))
    style(fig, height=height,
          xaxis_title="SHAP value (€ contribution to predicted cost rate)",
          yaxis=dict(tickmode="array",
                     tickvals=[n_feat - 1 - i for i in range(n_feat)],
                     ticktext=[SHAP_FEATURE_LABELS[k] for k in order],
                     range=[-0.6, n_feat - 0.4]))
    return fig


def _importance_bar(mabs: pd.Series, height: int = 460) -> go.Figure:
    """Horizontal mean(|SHAP|) bar, same ordering as the paired beeswarm."""
    if mabs.empty:
        return empty_fig()
    order = mabs.index.tolist()[::-1]  # most important ends up on top
    fig = go.Figure(go.Bar(
        x=[mabs[k] for k in order], y=[SHAP_FEATURE_LABELS[k] for k in order],
        orientation="h", marker_color=PALETTE["primary"],
        hovertemplate="%{y}: €%{x:,.1f}<extra></extra>"))
    return style(fig, height=height, yaxis=dict(type="category"),
                 xaxis_title="mean(|SHAP|) €  ·  population-weighted")


def _cross_model_fig(height: int = 460) -> go.Figure:
    """Grouped bar: mean(|SHAP|) per feature, one bar group per model.

    Deliberately unfiltered, so the comparison across model classes is stable.
    """
    models = shap_models()
    if not models:
        return empty_fig()
    per = {m: _weighted_mean_abs_shap(load_shap(m)) for m in models}
    avg = pd.DataFrame(per).mean(axis=1).sort_values()  # smallest first → bottom
    order = avg.index.tolist()
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Bar(
            y=[SHAP_FEATURE_LABELS[k] for k in order],
            x=[per[m][k] for k in order], orientation="h",
            name=config.MODEL_LABELS[m]))
    return style(fig, height=height, barmode="group",
                 yaxis=dict(type="category"), xaxis_title="mean(|SHAP|) €")


def tab_shap(f: dict):
    """Tab 5: SHAP charts from the pre-computed matrices, sliced by the filters."""
    models = shap_models()
    if not models:
        return empty_fig_card(
            "No SHAP artifacts found. Run `uv run python -m modeling.shap_compute` first.")
    model_key = _active_shap_model(f)
    full = load_shap(model_key)
    filt = _filter_shap(full, f)
    mabs_full = _weighted_mean_abs_shap(full)

    notes = []
    if f.get("model") not in models:
        notes.append(dbc.Alert(
            f"SHAP covers Tier 2-3 base learners only (Random Forest, XGBoost, "
            f"LightGBM, CatBoost). “{config.MODEL_LABELS.get(f.get('model'), '—')}” "
            f"is out of scope — showing {config.MODEL_LABELS[model_key]} instead.",
            color="info", className="py-2 small mb-2"))
    notes.append(dbc.Alert(
        f"SHAP explains the total real cost rate (the cost-category filter is "
        f"disabled on this tab). Exact TreeSHAP on a shared {len(full):,}-instance "
        f"held-out sample.", color="light", className="py-2 small mb-3"))

    feat_opts = [{"label": SHAP_FEATURE_LABELS[k], "value": k} for k in mabs_full.index]
    color_opts = [{"label": "None", "value": "none"}] + feat_opts

    return html.Div([
        *notes,
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(f"6.1 · SHAP summary (beeswarm) — {config.MODEL_LABELS[model_key]}",
                        className="text-muted"),
                dcc.Graph(figure=_beeswarm_fig(full), config={"displayModeBar": False})]),
                className="border-0 shadow-sm"), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("6.2 · Relative feature importance", className="text-muted"),
                dcc.Graph(figure=_importance_bar(mabs_full), config={"displayModeBar": False})]),
                className="border-0 shadow-sm"), md=5),
        ], className="mb-3"),
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("6.3 · Feature importance across models (mean|SHAP|, full sample)",
                    className="text-muted"),
            dcc.Graph(figure=_cross_model_fig(), config={"displayModeBar": False})]),
            className="border-0 shadow-sm mb-3"), md=12)),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("6.4 · SHAP dependence", className="text-muted"),
                html.Div([
                    html.Span("Feature", className="text-muted small me-1"),
                    dcc.Dropdown(id="shap-dep-feature", clearable=False,
                                 value=mabs_full.index[0], options=feat_opts,
                                 style={"width": "190px", "display": "inline-block"}),
                    html.Span("colour by", className="text-muted small mx-1"),
                    dcc.Dropdown(id="shap-dep-color", clearable=False,
                                 value=(mabs_full.index[1] if len(mabs_full) > 1 else "none"),
                                 options=color_opts,
                                 style={"width": "190px", "display": "inline-block"}),
                ], className="d-flex align-items-center mb-2 flex-wrap"),
                dcc.Graph(id="shap-dep", config={"displayModeBar": False})]),
                className="border-0 shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("6.5 · Segment-filtered beeswarm (active filters)",
                        className="text-muted"),
                dcc.Graph(figure=_beeswarm_fig(filt), config={"displayModeBar": False})]),
                className="border-0 shadow-sm"), md=6),
        ]),
    ])


@app.callback(Output("shap-dep", "figure"),
              Input("shap-dep-feature", "value"), Input("shap-dep-color", "value"),
              Input("filters", "data"))
def _shap_dependence(feature, color_feat, f):
    if not f or feature is None:
        return empty_fig()
    model_key = _active_shap_model(f)
    if model_key is None:
        return empty_fig()
    df = _filter_shap(load_shap(model_key), f)
    if df.empty:
        return empty_fig("No instances for current filters.")
    if len(df) > 6000:
        df = df.sample(6000, random_state=config.RANDOM_STATE)
    x, x_ticks = _feature_plot_values(df, feature)
    y = df[f"shap_{feature}"].to_numpy(dtype=float)

    if x_ticks is not None:
        hov = [_humanize(feature, v) for v in df[feature].tolist()]
        x_hover = "%{customdata}"
    else:
        hov = None
        x_hover = "%{x:,.0f}"

    marker = dict(size=5, opacity=0.55)
    if color_feat and color_feat != "none":
        c_vals, c_ticks = _feature_plot_values(df, color_feat)
        marker.update(color=c_vals, colorscale="RdYlBu_r",
                      colorbar=dict(title=SHAP_FEATURE_LABELS[color_feat]))
        if c_ticks is not None:  # label the colourbar (e.g. Female / Male)
            marker["colorbar"].update(tickvals=c_ticks[0], ticktext=c_ticks[1])
    else:
        marker.update(color=PALETTE["primary"], opacity=0.5)

    fig = go.Figure(go.Scattergl(
        x=x, y=y, mode="markers", marker=marker, customdata=hov,
        hovertemplate=f"{SHAP_FEATURE_LABELS[feature]}: {x_hover}"
                      "<br>SHAP %{y:+,.0f} €<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=PALETTE["muted"], dash="dot"))

    x_title = (SHAP_FEATURE_LABELS[feature] if x_ticks is not None
               else f"{SHAP_FEATURE_LABELS[feature]} (model-input value)")
    style(fig, height=420, xaxis_title=x_title,
          yaxis_title=f"SHAP for {SHAP_FEATURE_LABELS[feature]} (€)")
    if x_ticks is not None:  # show band / gender labels instead of codes
        fig.update_xaxes(tickmode="array", tickvals=x_ticks[0], ticktext=x_ticks[1])
    return fig


def empty_fig_card(message: str):
    """Card holding a blank figure with *message*."""
    return dbc.Card(dbc.CardBody(dcc.Graph(figure=empty_fig(message),
                    config={"displayModeBar": False})), className="border-0 shadow-sm")


if __name__ == "__main__":
    app.run(debug=True, port=PORT)
