"""
Dutch Healthcare Cost Explorer — Plotly Dash dashboard.

Reads from `identifier.db` (built by `preprocessing.py`).
Run with: `uv run python app.py` then open http://127.0.0.1:8050
"""

import json
import os
import urllib.request
from typing import Optional

import dash_bootstrap_components as dbc
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

# Configuration

DB_PATH       = "identifier.db"
GEOJSON_PATH  = "data/gemeente_2023.geojson"
GEOJSON_URL   = "https://cartomap.github.io/nl/wgs84/gemeente_2023.geojson"
GEOJSON_KEY   = "properties.statcode"  # cartomap encodes GM#### as `statcode`

# Display label → DuckDB column
COST_CATEGORIES = {
    "Total (all categories)":             "ALL",
    "Medical specialist care":            "costs_medical_specialist_care_real",
    "Pharmacy":                           "costs_pharmacy_real",
    "GP — registration fee":              "costs_gp_registration_fee_real",
    "GP — consultation":                  "costs_gp_consultation_real",
    "GP — multidisciplinary care":        "costs_gp_multidisciplinary_care_real",
    "GP — other":                         "costs_gp_other_real",
    "Medical aids":                       "costs_medical_aids_real",
    "Dental care":                        "costs_dental_care_real",
    "Physiotherapy":                      "costs_physiotherapy_real",
    "Paramedical (other)":                "costs_paramedical_other_real",
    "Patient transport — seated":         "costs_patient_transport_seated_real",
    "Patient transport — ambulance":      "costs_patient_transport_ambulance_real",
    "Maternity care":                     "costs_maternity_care_real",
    "Obstetric care":                     "costs_obstetric_care_real",
    "Cross-border care":                  "costs_cross_border_care_real",
    "Primary care support":               "costs_primary_care_support_real",
    "Geriatric rehabilitation":           "costs_geriatric_rehabilitation_real",
    "Primary care residence":             "costs_primary_care_residence_real",
    "Nursing care":                       "costs_nursing_care_real",
    "Medical care (specific patients)":   "costs_medical_care_specific_patients_real",
    "Integrated birth care":              "costs_integrated_birth_care_real",
    "Mental health (total)":              "costs_mental_health_total_real",
    "Other":                              "costs_other_real",
}
COST_COLUMNS = [c for c in COST_CATEGORIES.values() if c != "ALL"]
TOTAL_EXPR   = " + ".join(f"COALESCE({c}, 0)" for c in COST_COLUMNS)

PALETTE = {
    "primary":   "#1c4778",   # CBS-style deep blue
    "accent":    "#dc6803",
    "muted":     "#6b7280",
    "bg":        "#f4f6f8",
    "card":      "#ffffff",
    "grid":      "#e5e7eb",
    "ink":       "#1f2937",
}

# Projected years (written to the DB by modeling.temporal.forecast) all share
# one dashed/faded style, distinct from the actuals.
PROJECTED_COLOR        = "#7f9cc0"                   # lighter primary
PROJECTED_ACCENT       = "rgba(220, 104, 3, 0.45)"   # faded accent
PROJECTED_LABEL_STYLE  = {"fontSize": "10px", "color": "#9ca3af", "fontStyle": "italic"}

PLOTLY_LAYOUT = dict(
    template="simple_white",
    margin=dict(l=50, r=20, t=40, b=40),
    font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=12, color=PALETTE["ink"]),
    colorway=[PALETTE["primary"], PALETTE["accent"], "#22c55e", "#a855f7",
              "#ef4444", "#06b6d4", "#84cc16", "#eab308"],
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=PALETTE["grid"]),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
)


# Data access

def _con() -> duckdb.DuckDBPyConnection:
    """Open a read-only connection. Cheap; can be called per callback."""
    return duckdb.connect(DB_PATH, read_only=True)


def _forecast_available() -> bool:
    """True when `claims_forecast` and the `claims_extended` view have rows.

    Any failure degrades to actuals-only, ending the slider at the last actual
    year.
    """
    try:
        con = _con()
    except Exception:
        return False
    try:
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()}
        if not {"claims_forecast", "claims_extended"} <= tables:
            return False
        return con.execute("SELECT count(*) FROM claims_forecast").fetchone()[0] > 0
    except Exception:
        return False
    finally:
        con.close()


HAS_FORECAST = _forecast_available()
CLAIMS_TABLE = "claims_extended" if HAS_FORECAST else "claims"


def _load_metadata():
    con = _con()
    years      = con.execute(f"SELECT DISTINCT year FROM {CLAIMS_TABLE} ORDER BY year").fetchdf()["year"].tolist()
    actual_max = con.execute("SELECT max(year) FROM claims").fetchone()[0]
    age_groups = con.execute(
        "SELECT DISTINCT age_group FROM claims WHERE age_group IS NOT NULL ORDER BY age_group"
    ).fetchdf()["age_group"].tolist()
    gementees  = con.execute(
        "SELECT DISTINCT gementee FROM claims WHERE gementee IS NOT NULL ORDER BY gementee"
    ).fetchdf()["gementee"].tolist()
    con.close()
    return years, int(actual_max), age_groups, gementees


def _load_geojson() -> Optional[dict]:
    """Load cartomap municipality polygons; download on first run."""
    if not os.path.exists(GEOJSON_PATH):
        try:
            print(f"Downloading municipality GeoJSON from {GEOJSON_URL}...")
            os.makedirs(os.path.dirname(GEOJSON_PATH), exist_ok=True)
            urllib.request.urlretrieve(GEOJSON_URL, GEOJSON_PATH)
        except Exception as exc:
            print(f"  WARNING: could not download GeoJSON ({exc}); choropleth will be disabled.")
            return None
    try:
        with open(GEOJSON_PATH) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  WARNING: could not parse GeoJSON ({exc}); choropleth will be disabled.")
        return None


YEARS, ACTUAL_MAX_YEAR, AGE_GROUPS, GEMENTEES = _load_metadata()
GEOJSON = _load_geojson()


# Query helpers

def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_where(f: dict, extra: Optional[list] = None) -> str:
    clauses = [f"year BETWEEN {f['year_min']} AND {f['year_max']}"]
    if f.get("gender") and f["gender"] != "ALL":
        clauses.append(f"gender = {_quote(f['gender'])}")
    if f.get("age_groups"):
        bands = ", ".join(_quote(a) for a in f["age_groups"])
        clauses.append(f"age_group IN ({bands})")
    if f.get("gementees"):
        names = ", ".join(_quote(g) for g in f["gementees"])
        clauses.append(f"gementee IN ({names})")
    if extra:
        clauses.extend(extra)
    return " AND ".join(clauses)


def cost_expr(category: str) -> str:
    return TOTAL_EXPR if category == "ALL" else f"COALESCE({category}, 0)"


def category_label(category: str) -> str:
    for label, col in COST_CATEGORIES.items():
        if col == category:
            return label
    return category


# Projection helpers — years past ACTUAL_MAX_YEAR are model-projected.

def shows_projection(f: dict) -> bool:
    """True when the active year range reaches into projected years."""
    return HAS_FORECAST and f["year_max"] > ACTUAL_MAX_YEAR


def year_tier(y: int) -> str:
    """Evidential tier of a year: Actual or Projected."""
    if not HAS_FORECAST or y <= ACTUAL_MAX_YEAR:
        return "Actual"
    return "Projected"


def split_actual_projected(df: pd.DataFrame, year_col: str = "year"):
    """Split into (actual, projected) with the boundary year in both halves,
    so line traces connect visually across the actual/projected seam."""
    return (df[df[year_col] <= ACTUAL_MAX_YEAR],
            df[df[year_col] >= ACTUAL_MAX_YEAR])


def add_projection_marker(fig, f: dict):
    """Dotted boundary line + note on year-axis charts."""
    if not shows_projection(f):
        return fig
    if f["year_min"] <= ACTUAL_MAX_YEAR:
        fig.add_vline(x=ACTUAL_MAX_YEAR + 0.5, line_dash="dot",
                      line_width=1.5, line_color=PALETTE["muted"])
    fig.add_annotation(
        text="→ projected", xref="paper", yref="paper", x=1, y=1.06,
        xanchor="right", showarrow=False,
        font=dict(size=11, color=PROJECTED_COLOR),
    )
    return fig


def projection_badge(fig, f: dict, text: str = None):
    """Corner badge for charts without a year axis (maps, shares)."""
    if not shows_projection(f):
        return fig
    if text is None:
        lo = max(f["year_min"], ACTUAL_MAX_YEAR + 1)
        text = f"Includes projected years {lo}–{f['year_max']}"
    fig.add_annotation(
        text=text, xref="paper", yref="paper", x=0.01, y=0.99,
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=11, color=PALETTE["ink"]),
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor=PROJECTED_COLOR,
        borderwidth=1, borderpad=4,
    )
    return fig


def add_split_line_trace(fig, df, y_col: str, name: str, color: str,
                         projected_color: str = None, **trace_kwargs):
    """Add a year-line as solid actual + dashed projected.

    Both traces share a legend group (one legend entry). The projected trace
    is only drawn when the frame actually reaches into it.
    """
    actual, projected = split_actual_projected(df)
    common = dict(mode="lines+markers", name=name, legendgroup=name, **trace_kwargs)
    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual["year"], y=actual[y_col],
            line=dict(width=3, color=color), marker=dict(size=8, color=color),
            **common,
        ))
    pcolor = projected_color or color
    # A single point carries no line.
    if len(projected) >= (2 if not actual.empty else 1):
        fig.add_trace(go.Scatter(
            x=projected["year"], y=projected[y_col],
            line=dict(width=3, color=pcolor, dash="dash"),
            marker=dict(size=8, color=pcolor, symbol="circle-open"),
            showlegend=False, **common,
        ))
    return fig


def style(fig, **overrides):
    layout = {**PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


def empty_fig(message: str = "No data for current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=PALETTE["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig)


def fmt_eur(n: float) -> str:
    n = float(n or 0)
    if n >= 1e9: return f"€{n/1e9:,.2f} B"
    if n >= 1e6: return f"€{n/1e6:,.1f} M"
    if n >= 1e3: return f"€{n/1e3:,.0f} K"
    return f"€{n:,.0f}"


def fmt_num(n: float) -> str:
    n = float(n or 0)
    if n >= 1e9: return f"{n/1e9:,.2f} B"
    if n >= 1e6: return f"{n/1e6:,.2f} M"
    if n >= 1e3: return f"{n/1e3:,.1f} K"
    return f"{n:,.0f}"


def kpi_card(label: str, value: str, subtitle: str = "", emphasize: bool = True) -> dbc.Card:
    value_style = {
        "color": PALETTE["primary"],
        "fontSize": "30px" if emphasize else "18px",
        "fontWeight": "700",
        "lineHeight": "1.15",
    }
    body_children = [
        html.Div(label, className="small text-uppercase text-muted fw-semibold"),
        html.Div(value, className="mt-1", style=value_style),
    ]
    if subtitle:
        body_children.append(html.Div(subtitle, className="small text-muted mt-1"))
    return dbc.Card(dbc.CardBody(body_children), className="border-0 shadow-sm h-100")


# Layout

def header() -> html.Header:
    if HAS_FORECAST:
        subtitle = (f"Insured-population claims · PC3 resolution · "
                    f"{YEARS[0]}–{ACTUAL_MAX_YEAR} actual · "
                    f"{ACTUAL_MAX_YEAR + 1}–{YEARS[-1]} model-projected · "
                    f"inflation-adjusted to €2020")
    else:
        subtitle = (f"Insured-population claims · PC3 resolution · "
                    f"{YEARS[0]}–{YEARS[-1]} · inflation-adjusted to €2020")
    badges = [
        dbc.Badge("EDA dashboard", color="light", text_color="dark", className="me-2"),
        dbc.Badge(f"{len(GEMENTEES)} gemeenten", color="light", text_color="dark"),
    ]
    if HAS_FORECAST:
        badges.append(dbc.Badge("incl. projections", color="light",
                                text_color="secondary", className="ms-2"))
    return html.Header(
        dbc.Container(
            html.Div([
                html.Div([
                    html.H2("Dutch Healthcare Cost Explorer", className="mb-0",
                            style={"color": PALETTE["primary"], "fontWeight": 600}),
                    html.Div(subtitle, className="text-muted small mt-1"),
                ]),
                html.Div(badges, className="d-flex align-items-center"),
            ], className="d-flex justify-content-between align-items-center py-3"),
            fluid=True,
        ),
        style={"backgroundColor": PALETTE["card"], "borderBottom": f"1px solid {PALETTE['grid']}"},
    )


def _year_option_label(y: int) -> str:
    """Dropdown label tagging projected years."""
    if not HAS_FORECAST or y <= ACTUAL_MAX_YEAR:
        return str(y)
    return f"{y} (proj.)"


def _year_marks() -> dict:
    """Slider marks; with 20 years label alternate ones (always first, last
    actual, last), and style projected-year labels distinctly."""
    label_all = len(YEARS) <= 15
    anchors = (YEARS[0], YEARS[-1], ACTUAL_MAX_YEAR)
    marks = {}
    for y in YEARS:
        show = label_all or y % 2 == 1 or y in anchors
        if not HAS_FORECAST or y <= ACTUAL_MAX_YEAR:
            style_ = {"fontSize": "10px"}
        else:
            style_ = PROJECTED_LABEL_STYLE
        marks[y] = {"label": str(y) if show else "", "style": style_}
    return marks


def filter_bar() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Year range", className="form-label fw-semibold small text-uppercase text-muted"),
                    dcc.RangeSlider(
                        id="f-year",
                        min=YEARS[0], max=YEARS[-1], step=1,
                        value=[YEARS[0], YEARS[-1]],
                        marks=_year_marks(),
                        tooltip={"placement": "bottom", "always_visible": False},
                        allowCross=False,
                    ),
                ], xs=12, lg=6),
                dbc.Col([
                    html.Label("Gender", className="form-label fw-semibold small text-uppercase text-muted"),
                    dcc.Dropdown(
                        id="f-gender",
                        options=[
                            {"label": "All",    "value": "ALL"},
                            {"label": "Female", "value": "V"},
                            {"label": "Male",   "value": "M"},
                        ],
                        value="ALL", clearable=False,
                    ),
                ], xs=6, lg=3),
                dbc.Col([
                    html.Label("Age group", className="form-label fw-semibold small text-uppercase text-muted"),
                    dcc.Dropdown(
                        id="f-age",
                        options=[{"label": ag, "value": ag} for ag in AGE_GROUPS],
                        value=[], multi=True,
                        placeholder="All ages (pick to narrow…)",
                    ),
                ], xs=6, lg=3),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Care category", className="form-label fw-semibold small text-uppercase text-muted"),
                    dcc.Dropdown(
                        id="f-category",
                        options=[{"label": k, "value": v} for k, v in COST_CATEGORIES.items()],
                        value="ALL", clearable=False,
                    ),
                ], xs=12, lg=4),
                dbc.Col([
                    html.Label("Region / Gemeente", className="form-label fw-semibold small text-uppercase text-muted"),
                    dcc.Dropdown(
                        id="f-gementee",
                        options=[{"label": g, "value": g} for g in GEMENTEES],
                        value=[], multi=True,
                        placeholder="All gemeenten (type to search…)",
                    ),
                ], xs=12, lg=8),
            ], className="g-3"),
        ]),
        className="border-0 shadow-sm mb-4",
    )


def graph_card(title: str, graph_id: str, height: int = 380, extra_header=None) -> dbc.Card:
    header_children = [html.Span(title, className="fw-semibold")]
    if extra_header is not None:
        header_children.append(extra_header)
    return dbc.Card([
        dbc.CardHeader(
            html.Div(header_children, className="d-flex justify-content-between align-items-center"),
            style={"backgroundColor": PALETTE["card"], "borderBottom": f"1px solid {PALETTE['grid']}"},
        ),
        dbc.CardBody(dcc.Graph(id=graph_id, config={"displayModeBar": False, "responsive": True},
                               style={"height": f"{height}px"})),
    ], className="border-0 shadow-sm h-100")


def table_card(title: str, table_id: str) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Span(title, className="fw-semibold"),
                       style={"backgroundColor": PALETTE["card"], "borderBottom": f"1px solid {PALETTE['grid']}"}),
        dbc.CardBody(html.Div(id=table_id)),
    ], className="border-0 shadow-sm h-100")


def tab_temporal() -> dbc.Container:
    return dbc.Container([
        dbc.Row([
            dbc.Col(graph_card("1.1 · Real cost per insured year over time", "chart-1-1"),
                    xs=12, lg=6, className="mb-3"),
            dbc.Col(graph_card("1.2 · Total real cost per year", "chart-1-total"),
                    xs=12, lg=6, className="mb-3"),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(graph_card("1.3 · Cost breakdown by care category", "chart-1-2", height=460),
                    width=12, className="mb-3"),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(graph_card("1.4 · Year-on-year growth rate (real terms)", "chart-1-3"),
                    xs=12, lg=6, className="mb-3"),
            dbc.Col(graph_card("1.5 · Category growth over filter range", "chart-1-catgrowth", height=460),
                    xs=12, lg=6, className="mb-3"),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


def tab_demographics() -> dbc.Container:
    year_selector = html.Div([
        html.Span("Year:", className="small text-muted me-2"),
        dcc.Dropdown(
            id="age-grad-year",
            options=[
                {"label": _year_option_label(y), "value": y}
                for y in YEARS
            ],
            value=ACTUAL_MAX_YEAR, clearable=False,
            style={"width": "130px"},
        ),
    ], className="d-flex align-items-center")

    return dbc.Container([
        dbc.Row([
            dbc.Col(graph_card("2.1 · Cost by age band × year", "chart-2-1", height=460),
                    width=12, className="mb-3"),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(graph_card("2.2 · Gender comparison over time", "chart-2-2"),
                    xs=12, lg=6, className="mb-3"),
            dbc.Col(graph_card("2.3 · Age–cost gradient", "chart-2-3", extra_header=year_selector),
                    xs=12, lg=6, className="mb-3"),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(graph_card("2.4 · Share of total cost by age band", "chart-2-concentration", height=420),
                    width=12, className="mb-3"),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


def tab_geographic() -> dbc.Container:
    return dbc.Container([
        dbc.Row([
            dbc.Col(graph_card("3.1 · Real cost per insured year by gemeente", "chart-3-1", height=620),
                    width=12, className="mb-3"),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(graph_card("3.2 · Cost at PC3 resolution", "chart-3-2", height=560),
                    xs=12, lg=7, className="mb-3"),
            dbc.Col(table_card("3.3 · Top / bottom 10 gemeenten", "table-3-3"),
                    xs=12, lg=5, className="mb-3"),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(graph_card("3.4 · Distribution of cost rates across gemeenten", "chart-3-distribution", height=440),
                    width=12, className="mb-3"),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


# App

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Dutch Healthcare Cost Explorer",
)
server = app.server  # for WSGI deployment

app.layout = html.Div([
    dcc.Store(id="filter-store"),
    header(),
    dbc.Container([
        filter_bar(),
        html.Div(id="kpi-strip", className="mb-4"),
        dbc.Tabs([
            dbc.Tab(tab_temporal(),     label="Temporal Trends", tab_id="t1"),
            dbc.Tab(tab_demographics(), label="Demographics",    tab_id="t2"),
            dbc.Tab(tab_geographic(),   label="Geographic",      tab_id="t3"),
        ], id="tabs", active_tab="t1"),
        html.Footer(
            html.Div("Source: Vektis postcode-3 healthcare claims · CBS Gebieden in Nederland 2023 · CBS CPI",
                     className="text-muted small text-center py-3"),
            className="mt-4",
        ),
    ], fluid=True, className="pb-3"),
], style={"backgroundColor": PALETTE["bg"], "minHeight": "100vh",
          "fontFamily": "Inter, Helvetica Neue, Arial, sans-serif"})


# Shared filter state

@app.callback(
    Output("filter-store", "data"),
    Input("f-year", "value"),
    Input("f-gender", "value"),
    Input("f-age", "value"),
    Input("f-category", "value"),
    Input("f-gementee", "value"),
)
def update_filter_store(years, gender, age_groups, category, gementees):
    return {
        "year_min":   years[0],
        "year_max":   years[1],
        "gender":     gender,
        "age_groups": age_groups or [],
        "category":   category,
        "gementees":  gementees or [],
    }


# KPI summary strip

@app.callback(Output("kpi-strip", "children"), Input("filter-store", "data"))
def kpi_strip(f):
    if not f:
        return None
    cat_sums = ",\n        ".join(f"SUM(COALESCE({c}, 0)) AS {c}" for c in COST_COLUMNS)
    sql = f"""
        SELECT
            SUM(insured_years)       AS total_iy,
            SUM(number_of_insured)   AS total_n,
            {cat_sums}
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)}
    """
    row = _con().execute(sql).fetchdf().iloc[0]
    total_iy = float(row["total_iy"] or 0)
    total_n  = float(row["total_n"]  or 0)
    cat_totals = {c: float(row[c] or 0) for c in COST_COLUMNS}
    total_real = sum(cat_totals.values())
    avg_rate   = total_real / total_iy if total_iy > 0 else 0
    top_col, top_total = max(cat_totals.items(), key=lambda x: x[1]) if cat_totals else ("", 0)
    top_share = (top_total / total_real * 100) if total_real > 0 else 0

    cost_subtitle = "in €2020"
    if shows_projection(f):
        lo = max(f["year_min"], ACTUAL_MAX_YEAR + 1)
        cost_subtitle = f"in €2020 · incl. {lo}–{f['year_max']} projections"
    return dbc.Row([
        dbc.Col(kpi_card("Total real cost", fmt_eur(total_real), cost_subtitle),
                xs=6, md=3),
        dbc.Col(kpi_card("Insured years", fmt_num(total_iy), f"{fmt_num(total_n)} insured persons"),
                xs=6, md=3),
        dbc.Col(kpi_card("Avg cost rate", fmt_eur(avg_rate), "€ per insured year"),
                xs=6, md=3),
        dbc.Col(kpi_card("Top category", category_label(top_col), f"{top_share:.1f}% of total", emphasize=False),
                xs=6, md=3),
    ], className="g-3")


# Tab 1 — Temporal Trends

@app.callback(Output("chart-1-1", "figure"), Input("filter-store", "data"))
def chart_1_1(f):
    if not f:
        return empty_fig()
    sql = f"""
        SELECT year,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS cost_rate
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)}
        GROUP BY year
        ORDER BY year
    """
    df = _con().execute(sql).fetchdf()
    if df.empty or df["cost_rate"].dropna().empty:
        return empty_fig()
    fig = go.Figure()
    add_split_line_trace(fig, df, "cost_rate", "Cost rate", PALETTE["primary"],
                         projected_color=PROJECTED_COLOR,
                         hovertemplate="%{x}: €%{y:,.0f}/iy<extra></extra>")
    add_projection_marker(fig, f)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Real € / insured year",
        yaxis_tickformat=",.0f",
        showlegend=False,
    )
    return style(fig)


@app.callback(Output("chart-1-2", "figure"), Input("filter-store", "data"))
def chart_1_2(f):
    if not f:
        return empty_fig()
    select_cols = ",\n        ".join(
        f"SUM(COALESCE({c}, 0)) / NULLIF(SUM(insured_years), 0) AS {c}"
        for c in COST_COLUMNS
    )
    sql = f"""
        SELECT year, {select_cols}
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)}
        GROUP BY year
        ORDER BY year
    """
    df = _con().execute(sql).fetchdf()
    if df.empty:
        return empty_fig()
    long = df.melt(id_vars="year", var_name="category_col", value_name="cost_rate")
    long["category"] = long["category_col"].map({v: k for k, v in COST_CATEGORIES.items() if v != "ALL"})
    long = long.dropna(subset=["cost_rate"])
    fig = px.bar(long, x="year", y="cost_rate", color="category",
                 labels={"year": "Year", "cost_rate": "Real € / insured year", "category": ""},
                 color_discrete_sequence=px.colors.qualitative.Safe)
    if shows_projection(f):
        for tr in fig.data:
            tr.marker.opacity = [
                {"Actual": 1.0, "Projected": 0.55}[year_tier(int(x))]
                for x in tr.x
            ]
        add_projection_marker(fig, f)
    fig.update_layout(yaxis_tickformat=",.0f")
    return style(fig,
                 legend=dict(orientation="v", yanchor="top", y=1, x=1.02),
                 margin=dict(r=260, l=50, t=40, b=40))


@app.callback(Output("chart-1-3", "figure"), Input("filter-store", "data"))
def chart_1_3(f):
    if not f:
        return empty_fig()
    sql = f"""
        WITH per_year AS (
            SELECT year,
                   SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS cost_rate
            FROM {CLAIMS_TABLE}
            WHERE {build_where(f)}
            GROUP BY year
            ORDER BY year
        )
        SELECT year,
               cost_rate,
               LAG(cost_rate) OVER (ORDER BY year) AS prev,
               (cost_rate - LAG(cost_rate) OVER (ORDER BY year))
                 / NULLIF(LAG(cost_rate) OVER (ORDER BY year), 0) * 100 AS yoy_pct
        FROM per_year
    """
    df = _con().execute(sql).fetchdf().dropna(subset=["yoy_pct"])
    if df.empty:
        return empty_fig()

    def _bar_color(year, v):
        if year_tier(year) == "Projected":
            return "rgba(127, 156, 192, 0.9)" if v >= 0 else PROJECTED_ACCENT
        return PALETTE["primary"] if v >= 0 else PALETTE["accent"]

    colors = [_bar_color(y, v) for y, v in zip(df["year"], df["yoy_pct"])]
    fig = go.Figure(go.Bar(
        x=df["year"], y=df["yoy_pct"], marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["yoy_pct"]], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    add_projection_marker(fig, f)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="YoY change (%)",
        yaxis_tickformat=".1f",
    )
    fig.add_hline(y=0, line_color=PALETTE["muted"], line_width=1)
    return style(fig)


@app.callback(Output("chart-1-total", "figure"), Input("filter-store", "data"))
def chart_1_total(f):
    if not f:
        return empty_fig()
    sql = f"""
        SELECT year, SUM({cost_expr(f['category'])}) AS total_real
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)}
        GROUP BY year
        ORDER BY year
    """
    df = _con().execute(sql).fetchdf()
    if df.empty or df["total_real"].dropna().empty:
        return empty_fig()
    actual, projected = split_actual_projected(df)
    fig = go.Figure()
    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual["year"], y=actual["total_real"],
            mode="lines+markers",
            line=dict(width=3, color=PALETTE["primary"]),
            marker=dict(size=8, color=PALETTE["primary"]),
            fill="tozeroy", fillcolor="rgba(28, 71, 120, 0.10)",
            hovertemplate="%{x}: €%{y:,.0f}<extra></extra>",
        ))
    if len(projected) >= (2 if not actual.empty else 1):
        fig.add_trace(go.Scatter(
            x=projected["year"], y=projected["total_real"],
            mode="lines+markers",
            line=dict(width=3, color=PROJECTED_COLOR, dash="dash"),
            marker=dict(size=8, color=PROJECTED_COLOR, symbol="circle-open"),
            fill="tozeroy", fillcolor="rgba(28, 71, 120, 0.04)",
            hovertemplate="%{x}: €%{y:,.0f} (projected)<extra></extra>",
        ))
    add_projection_marker(fig, f)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Total real € (2020 base)",
        yaxis_tickformat=",.2s",  # auto SI suffix: 15.2G, 182M
        showlegend=False,
    )
    return style(fig)


@app.callback(Output("chart-1-catgrowth", "figure"), Input("filter-store", "data"))
def chart_1_catgrowth(f):
    if not f:
        return empty_fig()
    if f["year_min"] == f["year_max"]:
        return empty_fig("Pick a year range (≥2 years) to compute growth.")

    cat_rates = ",\n        ".join(
        f"SUM(COALESCE({c}, 0)) / NULLIF(SUM(insured_years), 0) AS {c}"
        for c in COST_COLUMNS
    )
    sql = f"""
        SELECT year, {cat_rates}
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)} AND year IN ({f['year_min']}, {f['year_max']})
        GROUP BY year
        ORDER BY year
    """
    df = _con().execute(sql).fetchdf()
    if len(df) < 2:
        return empty_fig("Not enough data to compute growth.")
    first, last = df.iloc[0], df.iloc[-1]

    rows = []
    for c in COST_COLUMNS:
        f_val = float(first[c]) if pd.notna(first[c]) else 0.0
        l_val = float(last[c])  if pd.notna(last[c])  else 0.0
        if f_val > 0:
            rows.append({"category": category_label(c),
                         "growth": (l_val - f_val) / f_val * 100})
    if not rows:
        return empty_fig()

    gdf = pd.DataFrame(rows).sort_values("growth", ascending=True)
    colors = [PALETTE["primary"] if g >= 0 else PALETTE["accent"] for g in gdf["growth"]]
    fig = go.Figure(go.Bar(
        x=gdf["growth"], y=gdf["category"], orientation="h",
        marker_color=colors,
        text=[f"{g:+.1f}%" for g in gdf["growth"]], textposition="outside",
        hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
    ))
    endpoint_note = " (endpoint projected)" if shows_projection(f) else ""
    fig.update_layout(
        xaxis_title=f"% change in €/insured year, {f['year_min']} → {f['year_max']}{endpoint_note}",
        xaxis_tickformat=".0f",
    )
    fig.add_vline(x=0, line_color=PALETTE["muted"], line_width=1)
    return style(fig, margin=dict(l=210, r=60, t=20, b=40))


# Tab 2 — Demographics

@app.callback(Output("chart-2-1", "figure"), Input("filter-store", "data"))
def chart_2_1(f):
    if not f:
        return empty_fig()
    # Show all bands regardless of the age filter.
    f2 = {**f, "age_groups": []}
    sql = f"""
        SELECT year,
               age_group,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS cost_rate
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f2)} AND age_group IS NOT NULL
        GROUP BY year, age_group
        ORDER BY age_group, year
    """
    df = _con().execute(sql).fetchdf()
    if df.empty:
        return empty_fig()
    pivot = df.pivot(index="age_group", columns="year", values="cost_rate").sort_index()
    fig = px.imshow(
        pivot,
        labels=dict(x="Year", y="Age group", color="€ / insured year"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_traces(hovertemplate="Year %{x}<br>Age %{y}<br>€%{z:,.0f}/iy<extra></extra>")
    fig.update_layout(
        coloraxis_colorbar=dict(tickformat=",.0f"),
        # or age bands like '00-04' get parsed as dates
        yaxis=dict(type="category"),
        xaxis=dict(type="category"),
    )
    if shows_projection(f):
        years_shown = list(pivot.columns)
        if ACTUAL_MAX_YEAR in years_shown and years_shown[-1] > ACTUAL_MAX_YEAR:
            fig.add_vline(x=years_shown.index(ACTUAL_MAX_YEAR) + 0.5,
                          line_color="white", line_width=3)
            note = f"columns > {ACTUAL_MAX_YEAR} are projected"
        else:
            note = "all years shown are projected"
        fig.add_annotation(text=note, xref="paper", yref="paper",
                           x=1, y=1.06, xanchor="right", showarrow=False,
                           font=dict(size=11, color=PALETTE["muted"]))
    return style(fig)


@app.callback(Output("chart-2-2", "figure"), Input("filter-store", "data"))
def chart_2_2(f):
    if not f:
        return empty_fig()
    # Both lines, whatever the gender filter says.
    f2 = {**f, "gender": "ALL"}
    sql = f"""
        SELECT year,
               CASE gender WHEN 'V' THEN 'Female' WHEN 'M' THEN 'Male' ELSE gender END AS gender_label,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS cost_rate
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f2)} AND gender IN ('V', 'M')
        GROUP BY year, gender_label
        ORDER BY year, gender_label
    """
    df = _con().execute(sql).fetchdf()
    if df.empty:
        return empty_fig()
    fig = go.Figure()
    for label, color in (("Female", PALETTE["primary"]), ("Male", PALETTE["accent"])):
        sub = df[df["gender_label"] == label].sort_values("year")
        if sub.empty:
            continue
        add_split_line_trace(
            fig, sub, "cost_rate", label, color,
            hovertemplate=label + " %{x}: €%{y:,.0f}/iy<extra></extra>",
        )
    add_projection_marker(fig, f)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Real € / insured year",
        yaxis_tickformat=",.0f",
    )
    return style(fig)


@app.callback(
    Output("chart-2-3", "figure"),
    Input("filter-store", "data"),
    Input("age-grad-year", "value"),
)
def chart_2_3(f, year):
    if not f or year is None:
        return empty_fig()
    # The gradient covers every band for the picked year.
    f2 = {**f, "age_groups": [], "year_min": year, "year_max": year}
    sql = f"""
        SELECT age_group,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS cost_rate
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f2)} AND age_group IS NOT NULL
        GROUP BY age_group
        ORDER BY age_group
    """
    df = _con().execute(sql).fetchdf()
    if df.empty:
        return empty_fig()
    is_projected_year = HAS_FORECAST and year > ACTUAL_MAX_YEAR
    tier_color = PROJECTED_COLOR if is_projected_year else PALETTE["primary"]
    tier_label = "Projected" if is_projected_year else ""
    fig = go.Figure(go.Bar(
        x=df["age_group"], y=df["cost_rate"],
        marker_color=tier_color,
        hovertemplate="Age %{x}: €%{y:,.0f}/iy<extra></extra>",
    ))
    if is_projected_year:
        fig.add_annotation(text=tier_label, xref="paper", yref="paper",
                           x=1, y=1.06, xanchor="right", showarrow=False,
                           font=dict(size=11, color=tier_color))
    fig.update_layout(
        xaxis_title="Age group",
        yaxis_title=(f"Real € / insured year ({year}"
                     f"{', ' + tier_label.lower() if is_projected_year else ''})"),
        yaxis_tickformat=",.0f",
        xaxis=dict(type="category"),  # age bands are categorical, not dates
    )
    return style(fig)


@app.callback(Output("chart-2-concentration", "figure"), Input("filter-store", "data"))
def chart_2_concentration(f):
    if not f:
        return empty_fig()
    f2 = {**f, "age_groups": []}
    sql = f"""
        SELECT age_group, SUM({cost_expr(f['category'])}) AS total
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f2)} AND age_group IS NOT NULL
        GROUP BY age_group
        ORDER BY age_group
    """
    df = _con().execute(sql).fetchdf()
    if df.empty or df["total"].sum() <= 0:
        return empty_fig()
    grand = df["total"].sum()
    df["share"] = df["total"] / grand * 100
    df["total_fmt"] = df["total"].apply(fmt_eur)

    fig = go.Figure(go.Bar(
        x=df["age_group"], y=df["share"],
        marker_color=PALETTE["primary"],
        text=[f"{s:.1f}%" for s in df["share"]],
        textposition="outside",
        customdata=df[["total_fmt"]].values,
        hovertemplate="Age %{x}<br>Share: %{y:.2f}%<br>Total: %{customdata[0]}<extra></extra>",
    ))
    if shows_projection(f):
        fig.add_annotation(text="shares include projected years",
                           xref="paper", yref="paper", x=1, y=1.06,
                           xanchor="right", showarrow=False,
                           font=dict(size=11, color=PALETTE["muted"]))
    fig.update_layout(
        xaxis_title="Age group",
        yaxis_title="Share of total real cost (%)",
        yaxis_ticksuffix="%",
        xaxis=dict(type="category"),  # age bands are categorical, not dates
    )
    return style(fig)


# Tab 3 — Geographic

@app.callback(Output("chart-3-1", "figure"), Input("filter-store", "data"))
def chart_3_1(f):
    if not f:
        return empty_fig()
    if GEOJSON is None:
        return empty_fig("Choropleth disabled — could not load gemeente_2023.geojson")
    sql = f"""
        SELECT gementee_code,
               any_value(gementee)                                                    AS gementee,
               SUM(number_of_insured)                                                 AS n_insured,
               SUM(insured_years)                                                     AS insured_years,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0)        AS cost_rate
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)} AND gementee_code IS NOT NULL
        GROUP BY gementee_code
    """
    df = _con().execute(sql).fetchdf().dropna(subset=["cost_rate"])
    if df.empty:
        return empty_fig()
    fig = px.choropleth_map(
        df,
        geojson=GEOJSON,
        locations="gementee_code",
        featureidkey=GEOJSON_KEY,
        color="cost_rate",
        color_continuous_scale="YlOrRd",
        range_color=(df["cost_rate"].quantile(0.05), df["cost_rate"].quantile(0.95)),
        hover_name="gementee",
        hover_data={
            "gementee_code":  True,
            "cost_rate":      ":,.0f",
            "insured_years":  ":,.2f",
            "n_insured":      ":,.0f",
        },
        map_style="carto-positron",
        center={"lat": 52.15, "lon": 5.3},
        zoom=6.2,
        opacity=0.78,
        labels={"cost_rate": "€ / iy", "insured_years": "Insured-years",
                "n_insured": "Insured persons", "gementee_code": "Code"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="€ / insured year", tickformat=",.0f"),
    )
    projection_badge(fig, f)
    return fig


@app.callback(Output("chart-3-2", "figure"), Input("filter-store", "data"))
def chart_3_2(f):
    if not f:
        return empty_fig()
    sql = f"""
        SELECT postal_code_3,
               any_value(gementee)                                              AS gementee,
               AVG(latitude)                                                    AS lat,
               AVG(longitude)                                                   AS lon,
               SUM(number_of_insured)                                           AS n_insured,
               SUM(insured_years)                                               AS insured_years,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS cost_rate
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)} AND latitude IS NOT NULL AND longitude IS NOT NULL
        GROUP BY postal_code_3
    """
    df = _con().execute(sql).fetchdf().dropna(subset=["cost_rate", "lat", "lon"])
    if df.empty:
        return empty_fig()
    fig = px.scatter_map(
        df, lat="lat", lon="lon",
        size="insured_years", color="cost_rate",
        color_continuous_scale="YlOrRd",
        range_color=(df["cost_rate"].quantile(0.05), df["cost_rate"].quantile(0.95)),
        size_max=20,
        zoom=6.2, center={"lat": 52.15, "lon": 5.3},
        map_style="carto-positron",
        hover_name="gementee",
        hover_data={
            "postal_code_3": True, "lat": False, "lon": False,
            "insured_years": ":,.2f", "n_insured": ":,.0f", "cost_rate": ":,.0f",
        },
        labels={"cost_rate": "€ / iy", "insured_years": "Insured-years",
                "n_insured": "Insured persons", "postal_code_3": "PC3"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="€ / insured year", tickformat=",.0f"),
    )
    projection_badge(fig, f)
    return fig


@app.callback(Output("table-3-3", "children"), Input("filter-store", "data"))
def table_3_3(f):
    if not f:
        return None
    sql = f"""
        SELECT gementee,
               gementee_code,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS cost_rate,
               SUM(insured_years)     AS insured_years,
               SUM(number_of_insured) AS n_insured
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)} AND gementee IS NOT NULL
        GROUP BY gementee, gementee_code
        HAVING SUM(insured_years) > 0
    """
    df = _con().execute(sql).fetchdf().dropna(subset=["cost_rate"])
    if df.empty:
        return html.Div("No data for current filters.", className="text-muted small")
    df = df.sort_values("cost_rate", ascending=False).reset_index(drop=True)
    top    = df.head(10).assign(group="Top 10 (highest)",   rank=range(1, min(11, len(df) + 1)))
    bottom = (df.tail(10).iloc[::-1]
                 .assign(group="Bottom 10 (lowest)",
                         rank=range(len(df), len(df) - min(10, len(df)), -1)))
    combined = pd.concat([top, bottom], ignore_index=True)
    combined["cost_rate"]     = combined["cost_rate"].round(0).astype(int)
    combined["insured_years"] = combined["insured_years"].astype(float).round(2)
    combined["n_insured"]     = combined["n_insured"].round(0).astype(int)

    note = None
    if shows_projection(f):
        lo = max(f["year_min"], ACTUAL_MAX_YEAR + 1)
        note = html.Div(
            f"Averages include projected years {lo}–{f['year_max']}.",
            className="small text-muted fst-italic mb-2",
        )

    table = dash_table.DataTable(
        data=combined[["group", "rank", "gementee", "gementee_code",
                       "cost_rate", "insured_years", "n_insured"]].to_dict("records"),
        columns=[
            {"name": "",                 "id": "group"},
            {"name": "Rank",             "id": "rank",          "type": "numeric"},
            {"name": "Gemeente",         "id": "gementee"},
            {"name": "Code",             "id": "gementee_code"},
            {"name": "€ / insured year", "id": "cost_rate",     "type": "numeric",
             "format": dash_table.Format.Format(group=",", precision=0, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Insured years",    "id": "insured_years", "type": "numeric",
             "format": dash_table.Format.Format(group=",", precision=2, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Insured persons", "id": "n_insured",      "type": "numeric",
             "format": dash_table.Format.Format(group=",", precision=0, scheme=dash_table.Format.Scheme.fixed)},
        ],
        sort_action="native",
        page_action="none",
        style_table={"overflowX": "auto", "maxHeight": "560px", "overflowY": "auto"},
        style_header={
            "backgroundColor": PALETTE["bg"], "fontWeight": "600",
            "border": "none", "borderBottom": f"2px solid {PALETTE['grid']}",
        },
        style_cell={
            "fontFamily": "Inter, Helvetica Neue, Arial, sans-serif",
            "fontSize": "13px", "padding": "8px 10px", "border": "none",
            "borderBottom": f"1px solid {PALETTE['grid']}",
        },
        style_data_conditional=[
            {"if": {"filter_query": '{group} contains "Top"'},
             "backgroundColor": "#fdecea", "color": "#922b21"},
            {"if": {"filter_query": '{group} contains "Bottom"'},
             "backgroundColor": "#e8f3ee", "color": "#1f5132"},
        ],
    )
    return html.Div([note, table]) if note is not None else table


@app.callback(Output("chart-3-distribution", "figure"), Input("filter-store", "data"))
def chart_3_distribution(f):
    if not f:
        return empty_fig()
    sql = f"""
        SELECT year, gementee,
               SUM({cost_expr(f['category'])}) / NULLIF(SUM(insured_years), 0) AS rate
        FROM {CLAIMS_TABLE}
        WHERE {build_where(f)} AND gementee IS NOT NULL
        GROUP BY year, gementee
        HAVING SUM(insured_years) > 0
    """
    df = _con().execute(sql).fetchdf().dropna(subset=["rate"])
    if df.empty:
        return empty_fig()
    df["kind"] = df["year"].map(year_tier)
    fig = px.box(
        df, x="year", y="rate",
        points="outliers",
        color="kind",
        color_discrete_map={"Actual": PALETTE["primary"],
                            "Projected": PROJECTED_COLOR},
        labels={"year": "Year", "rate": "€ / insured year", "kind": ""},
        hover_data={"gementee": True, "rate": ":,.0f", "year": False, "kind": False},
    )
    fig.update_traces(boxmean=True,
                      marker=dict(size=4, opacity=0.55, color=PALETTE["accent"]))
    _BOX_FILL = {
        "Actual":    ("rgba(28, 71, 120, 0.18)",  PALETTE["primary"]),
        "Projected": ("rgba(127, 156, 192, 0.18)", PROJECTED_COLOR),
    }
    for tr in fig.data:
        fill, line = _BOX_FILL.get(tr.name, _BOX_FILL["Actual"])
        tr.fillcolor = fill
        tr.line = dict(color=line)
    add_projection_marker(fig, f)
    fig.update_layout(
        yaxis_tickformat=",.0f",
        showlegend=shows_projection(f),
        xaxis=dict(tickmode="linear", dtick=1),
        boxmode="overlay",
    )
    return style(fig)



if __name__ == "__main__":
    app.run(debug=True, port=8050)
