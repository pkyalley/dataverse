"""
NHE Healthcare Financial Intelligence Dashboard
Run: streamlit run nhe_dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LEGACY_DIRS = [BASE_DIR / "nhe24_summary", BASE_DIR / "nhe24-tables"]
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BLUE = "#1B4F72"
BLUE2 = "#2E86C1"
RED = "#E74C3C"
ORANGE = "#F39C12"
GREEN = "#1E8449"
PURPLE = "#6C3483"
GREY = "#717D7E"


def resolve_data_file(filename: str) -> Path:
    for root in [DATA_DIR, *LEGACY_DIRS]:
        candidate = root / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in data or legacy folders.")


def parse_year_series(values, start_year=1960):
    series = pd.Series(values)
    series.index = range(start_year, start_year + len(series))
    return series.dropna()


def load_table(filename: str, yr_row_idx: int = 1, data_start: int = 3) -> dict[str, pd.Series]:
    df = pd.read_excel(resolve_data_file(filename), header=None)
    year_row = df.iloc[yr_row_idx]
    years, year_cols = [], []
    for col_idx, value in enumerate(year_row):
        text = str(value).replace(".0", "").strip()
        if text.isdigit() and 1960 <= int(text) <= 2030:
            years.append(int(text))
            year_cols.append(col_idx)

    out: dict[str, pd.Series] = {}
    for row_idx in range(data_start, len(df)):
        label = str(df.iloc[row_idx, 0]).strip()
        if not label or label == "nan":
            continue
        values = []
        for col_idx in year_cols:
            try:
                values.append(float(str(df.iloc[row_idx, col_idx]).replace(",", "")))
            except Exception:
                values.append(np.nan)
        out[label] = pd.Series(values, index=years).dropna()
    return out


@st.cache_data(show_spinner=False)
def load_data():
    summary = pd.read_csv(resolve_data_file("NHE24_Summary.csv"), header=None)

    def get_summary_series(keyword: str) -> pd.Series:
        for _, row in summary.iterrows():
            label = str(row.iloc[0])
            if keyword.lower() in label.lower():
                values = []
                for value in row.iloc[1:]:
                    try:
                        values.append(float(str(value).replace(",", "")))
                    except Exception:
                        values.append(np.nan)
                return parse_year_series(values).dropna()
        return pd.Series(dtype=float)

    nhe = get_summary_series("National Health Expenditures (Amount")
    gdp = get_summary_series("Gross Domestic Product2  (Amount")
    admin = get_summary_series("Government Administration and Non-Medical")
    phc = get_summary_series("Personal Health Care")
    pubh = get_summary_series("Government Public Health Activities")
    pc = get_summary_series("National Health Expenditures (Per Capita")
    invest = get_summary_series("Investment")

    nhe_gdp = (nhe / gdp * 100).dropna()
    admin_pct = (admin / nhe * 100).dropna()

    t03 = load_table("Table 03 National Health Expenditures, by Source of Funds.xlsx")
    t02 = load_table("Table 02 National Health Expenditures, Aggregate and Per Capita Amounts, by Type of Expenditure.xlsx")

    def pick(data: dict[str, pd.Series], keyword: str) -> pd.Series:
        for label, series in data.items():
            if keyword.lower() in label.lower():
                return series.dropna()
        return pd.Series(dtype=float)

    return {
        "nhe": nhe,
        "gdp": gdp,
        "admin": admin,
        "phc": phc,
        "pubh": pubh,
        "pc": pc,
        "invest": invest,
        "nhe_gdp": nhe_gdp,
        "admin_pct": admin_pct,
        "medicare": pick(t03, "Medicare"),
        "medicaid": pick(t03, "Medicaid"),
        "private_ins": pick(t03, "Private Health Insurance"),
        "out_of_pocket": pick(t03, "Out of pocket"),
        "hospital": pick(t02, "Hospital Care"),
        "physician": pick(t02, "Physician and Clinical"),
        "rx": pick(t02, "Prescription Drugs"),
        "nursing": pick(t02, "Nursing Care"),
        "home_health": pick(t02, "Home Health"),
    }


def years_in_range(series: pd.Series, start_year: int, end_year: int) -> list[int]:
    return [year for year in sorted(series.index) if start_year <= year <= end_year]


def format_trillion(value_billion: float) -> str:
    return f"${value_billion / 1000:.2f}T"


def format_billion(value_billion: float) -> str:
    return f"${value_billion:,.0f}B"


def format_int(value: float) -> str:
    return f"${value:,.0f}"


def metrics_row(col_defs, items):
    for col, item in zip(col_defs, items):
        with col:
            with st.container(border=True):
                st.metric(item[0], item[1], item[2] if len(item) > 2 else None)


def add_common_layout(title: str, subtitle: str):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0F2740 0%, #1B4F72 55%, #2E86C1 100%);
            color: white;
            padding: 28px 28px 24px 28px;
            border-radius: 24px;
            box-shadow: 0 18px 40px rgba(15, 39, 64, 0.18);
            margin-bottom: 1.25rem;
        ">
            <div style="font-size: 1.65rem; font-weight: 800; letter-spacing: -0.03em;">{title}</div>
            <div style="margin-top: 0.35rem; font-size: 0.98rem; opacity: 0.88; line-height: 1.45;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_template():
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


st.set_page_config(
    page_title="U.S. Healthcare Financial Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #F5F8FC 0%, #FFFFFF 38%, #F8FBFE 100%);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F2740 0%, #173A5A 100%);
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(15, 39, 64, 0.08);
            border-radius: 18px;
            padding: 8px 10px;
            box-shadow: 0 8px 20px rgba(15, 39, 64, 0.06);
        }
        h1, h2, h3, h4 {
            letter-spacing: -0.03em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

d = load_data()

with st.sidebar:
    st.markdown("## 🏥 NHE Dashboard")
    st.markdown("U.S. Healthcare Financial Intelligence")
    st.caption("CMS National Health Expenditure Accounts, 1960–2024")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Administrative Costs",
            "Payer Mix",
            "Service Categories",
            "Trends & Projections",
            "Data Explorer",
        ],
    )

    st.markdown("### Analysis Filters")
    yr_start, yr_end = st.slider("Year range", 1960, 2024, (1990, 2024))
    st.caption("Use the page controls for category-specific filters and comparisons.")
    st.markdown("---")
    st.caption("Source: CMS NHEA data files now stored in data/.")


def render_overview():
    add_common_layout(
        "U.S. Healthcare Financial Intelligence Dashboard",
        "A clean CMS NHEA view of spending, payer mix, service categories, administrative costs, and projections.",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_row(
        [c1, c2, c3, c4, c5],
        [
            ("Total NHE 2024", format_trillion(d["nhe"].get(2024, np.nan)), "+7.2% vs 2023"),
            ("NHE as % of GDP", f"{d['nhe_gdp'].get(2024, np.nan):.1f}%", "+0.3pp vs 2023"),
            ("Per Capita 2024", format_int(d["pc"].get(2024, np.nan)), "+6.8% vs 2023"),
            ("Admin Costs 2024", format_billion(d["admin"].get(2024, np.nan)), f"{d['admin_pct'].get(2024, np.nan):.1f}% of NHE"),
            ("Admin Growth 1990→2024", f"{((d['admin'].get(2024, np.nan) / d['admin'].get(1990, np.nan) - 1) * 100):.0f}%", "vs 385% hospital growth"),
        ],
    )

    left, right = st.columns([1.25, 1])
    with left:
        yrs = [year for year in years_in_range(d["nhe"], max(1960, yr_start), yr_end) if year in d["nhe_gdp"].index]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=yrs,
                y=[d["nhe"][year] / 1000 for year in yrs],
                name="Total NHE ($T)",
                mode="lines",
                line=dict(color=BLUE, width=3),
                fill="tozeroy",
                fillcolor="rgba(27, 79, 114, 0.14)",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=yrs,
                y=[d["nhe_gdp"][year] for year in yrs],
                name="NHE as % of GDP",
                mode="lines",
                line=dict(color=RED, width=3, dash="dash"),
            ),
            secondary_y=True,
        )
        fig.update_layout(**plot_template(), title="Total NHE and GDP Share")
        fig.update_yaxes(title_text="NHE ($ Trillions)", secondary_y=False)
        fig.update_yaxes(title_text="% of GDP", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        payer_year = st.selectbox("Payer snapshot year", sorted([year for year in range(1987, 2025) if year in d["medicare"].index or year in d["medicaid"].index]), index=-1)
        payer_values, payer_labels = [], []
        for label, series_key in [
            ("Medicare", "medicare"),
            ("Private Insurance", "private_ins"),
            ("Medicaid", "medicaid"),
            ("Out-of-Pocket", "out_of_pocket"),
        ]:
            series = d[series_key]
            if payer_year in series.index:
                payer_labels.append(label)
                payer_values.append(series[payer_year])
        if payer_values:
            fig = go.Figure(
                go.Pie(
                    labels=payer_labels,
                    values=payer_values,
                    hole=0.6,
                    textinfo="label+percent",
                    marker=dict(colors=[RED, BLUE2, GREEN, GREY], line=dict(color="white", width=2)),
                )
            )
            fig.update_layout(**plot_template(), title=f"Payer Mix Snapshot {payer_year}", height=420)
            fig.add_annotation(
                text=f"${sum(payer_values) / 1000:.1f}T<br>Total",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color=BLUE),
            )
            st.plotly_chart(fig, use_container_width=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    admin_years = years_in_range(d["admin"], max(1970, yr_start), yr_end)
    fig.add_trace(
        go.Bar(x=admin_years, y=[d["admin"][year] for year in admin_years], name="Admin Cost ($B)", marker_color="rgba(243,156,18,0.78)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=admin_years, y=[d["admin_pct"][year] for year in admin_years], name="Admin % of NHE", mode="lines", line=dict(color=RED, width=3)),
        secondary_y=True,
    )
    fig.add_hline(y=7, line_dash="dot", line_color=RED, opacity=0.45, secondary_y=True)
    fig.update_layout(**plot_template(), title="Administrative Cost Trend")
    fig.update_yaxes(title_text="Admin Cost ($ Billions)", secondary_y=False)
    fig.update_yaxes(title_text="% of NHE", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


def render_admin_costs():
    add_common_layout(
        "Administrative Cost Analysis",
        "Government administration and non-medical insurance costs compared with core clinical spending categories.",
    )
    c1, c2, c3, c4 = st.columns(4)
    metrics_row(
        [c1, c2, c3, c4],
        [
            ("Admin Cost 2024", format_billion(d["admin"].get(2024, np.nan))),
            ("Admin Cost 1990", format_billion(d["admin"].get(1990, np.nan))),
            ("Growth 1990→2024", f"{((d['admin'].get(2024, np.nan) / d['admin'].get(1990, np.nan) - 1) * 100):.0f}%"),
            ("Admin % of NHE", f"{d['admin_pct'].get(2024, np.nan):.1f}%"),
        ],
    )

    left, right = st.columns(2)
    with left:
        yrs = years_in_range(d["admin"], yr_start, yr_end)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yrs, y=[d["admin"][year] for year in yrs], name="Admin Cost ($B)", marker_color="rgba(243,156,18,0.8)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=yrs, y=[d["admin_pct"][year] for year in yrs], name="Admin % of NHE", mode="lines", line=dict(color=RED, width=3)), secondary_y=True)
        fig.add_hline(y=7, line_dash="dot", line_color=RED, opacity=0.4, secondary_y=True)
        fig.update_layout(**plot_template(), title=f"Admin Costs ({yr_start}–{yr_end})")
        fig.update_yaxes(title_text="Admin Cost ($ Billions)", secondary_y=False)
        fig.update_yaxes(title_text="% of NHE", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        growth_rows = ["Admin & Non-Medical Insurance", "Hospital Care", "Physician & Clinical Services", "Prescription Drugs"]
        growth_values = []
        for key in ["admin", "hospital", "physician", "rx"]:
            series = d[key]
            if 1990 in series.index and 2024 in series.index:
                growth_values.append((series[2024] / series[1990] - 1) * 100)
            else:
                growth_values.append(np.nan)
        fig = go.Figure(go.Bar(x=growth_rows, y=growth_values, marker_color=[RED, BLUE, ORANGE, GREEN], text=[f"{v:.0f}%" for v in growth_values], textposition="outside"))
        fig.update_layout(**plot_template(), title="Growth by Category, 1990→2024", yaxis_title="Growth (%)")
        st.plotly_chart(fig, use_container_width=True)

    yrs_all = years_in_range(d["admin_pct"], max(1970, yr_start), yr_end)
    fig = go.Figure(
        go.Scatter(
            x=yrs_all,
            y=[d["admin_pct"][year] for year in yrs_all],
            name="Admin % of NHE",
            fill="tozeroy",
            line=dict(color=RED, width=3),
            fillcolor="rgba(231, 76, 60, 0.15)",
        )
    )
    fig.add_hline(y=7, line_dash="dash", line_color=ORANGE, annotation_text="7% reference line")
    fig.update_layout(**plot_template(), title="Administrative Cost as a Share of NHE", yaxis_title="Admin % of NHE", xaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)


def render_payer_mix():
    add_common_layout(
        "Healthcare Payer Mix Analysis",
        "Source-of-funds view of Medicare, Medicaid, private insurance, and out-of-pocket spending.",
    )
    payer_map = {
        "Medicare": d["medicare"],
        "Private Insurance": d["private_ins"],
        "Medicaid": d["medicaid"],
        "Out-of-Pocket": d["out_of_pocket"],
    }
    common_years = sorted(set.intersection(*[set(series.index) for series in payer_map.values() if not series.empty]))
    common_years = [year for year in common_years if yr_start <= year <= yr_end]

    if not common_years:
        st.info("No overlapping payer years available for the selected window.")
        return

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        colors = [RED, BLUE, GREEN, GREY]
        for (label, series), color in zip(payer_map.items(), colors):
            fig.add_trace(go.Scatter(x=common_years, y=[series.get(year, 0) for year in common_years], name=label, stackgroup="one", line=dict(color=color, width=2)))
        fig.update_layout(**plot_template(), title=f"Payer Share Trend ({yr_start}–{yr_end})", yaxis_title="Share (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        snapshot_year = st.selectbox("Snapshot year", sorted(common_years, reverse=True), index=0)
        values = [series[snapshot_year] for series in payer_map.values() if snapshot_year in series.index]
        labels = [label for label, series in payer_map.items() if snapshot_year in series.index]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=[RED, BLUE, GREEN, GREY], line=dict(color="white", width=2)),
                textinfo="label+percent+value",
            )
        )
        fig.update_layout(**plot_template(), title=f"Payer Snapshot {snapshot_year}")
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for label, series, color in [
        ("Medicare", d["medicare"], RED),
        ("Private Insurance", d["private_ins"], BLUE),
        ("Medicaid", d["medicaid"], GREEN),
        ("Out-of-Pocket", d["out_of_pocket"], GREY),
    ]:
        yrs = [year for year in common_years if year in series.index]
        fig.add_trace(go.Bar(x=yrs, y=[series[year] for year in yrs], name=label, marker_color=color))
    fig.update_layout(**plot_template(), title=f"Absolute Spending by Payer ({yr_start}–{yr_end})", barmode="stack", yaxis_title="$ Billions")
    st.plotly_chart(fig, use_container_width=True)


def render_service_categories():
    add_common_layout(
        "Healthcare Spending by Service Category",
        "CMS NHE Table 2 shows how administrative and clinical categories evolve across the care system.",
    )
    svc_map = {
        "Hospital Care": d["hospital"],
        "Physician & Clinical": d["physician"],
        "Prescription Drugs": d["rx"],
        "Admin & Non-Medical Insurance": d["admin"],
        "Nursing Care": d["nursing"],
        "Home Health": d["home_health"],
        "Public Health": d["pubh"],
    }

    left, right = st.columns(2)
    with left:
        service_year = st.selectbox("Year for category breakdown", list(range(yr_end, yr_start - 1, -1)), index=0)
        svc_vals = {label: series[service_year] for label, series in svc_map.items() if service_year in series.index}
        svc_series = pd.Series(svc_vals).sort_values()
        if not svc_series.empty:
            colors = [RED if "Admin" in label else ORANGE if "Public" in label else BLUE for label in svc_series.index]
            fig = go.Figure(go.Bar(x=svc_series.values / 1000, y=svc_series.index, orientation="h", marker_color=colors, text=[f"{value / 1000:.2f}T" if value >= 1000 else f"${value:.0f}B" for value in svc_series.values], textposition="outside"))
            fig.update_layout(**plot_template(), title=f"Service Category Snapshot {service_year}", xaxis_title="$ Trillions")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        selected_services = st.multiselect(
            "Compare categories over time",
            list(svc_map.keys()),
            default=["Hospital Care", "Admin & Non-Medical Insurance", "Prescription Drugs"],
        )
        fig = go.Figure()
        palette = [BLUE, RED, ORANGE, GREEN, PURPLE, BLUE2, GREY]
        for index, label in enumerate(selected_services):
            series = svc_map[label]
            yrs = years_in_range(series, yr_start, yr_end)
            fig.add_trace(go.Scatter(x=yrs, y=[series[year] for year in yrs], name=label, mode="lines", line=dict(color=palette[index % len(palette)], width=2.5)))
        fig.update_layout(**plot_template(), title=f"Category Trends ({yr_start}–{yr_end})", yaxis_title="$ Billions")
        st.plotly_chart(fig, use_container_width=True)


def render_trends():
    add_common_layout(
        "Spending Trends and CMS Projections",
        "Compare historical growth, per-capita spending, and a simple CMS-style projection to 2033.",
    )
    left, right = st.columns(2)
    with left:
        base_year = st.selectbox("Base year for growth index", [1980, 1990, 2000, 2010], index=1)
        fig = go.Figure()
        for label, key, color in [
            ("Admin", "admin", RED),
            ("Hospital", "hospital", BLUE),
            ("Physician", "physician", ORANGE),
            ("Rx Drugs", "rx", GREEN),
        ]:
            series = d[key]
            if base_year in series.index:
                yrs = [year for year in years_in_range(series, base_year, 2024)]
                fig.add_trace(go.Scatter(x=yrs, y=[series[year] / series[base_year] * 100 for year in yrs], name=label, mode="lines", line=dict(color=color, width=2.5)))
        fig.add_hline(y=100, line_dash="dot", line_color=GREY)
        fig.update_layout(**plot_template(), title=f"Growth Index (Base = {base_year})", yaxis_title="Index")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        hist_years = [year for year in years_in_range(d["nhe"], max(2000, yr_start), 2024)]
        hist_values = [d["nhe"][year] / 1000 for year in hist_years]
        future_years = list(range(2025, 2034))
        forecast = [d["nhe"][2024] / 1000]
        for _ in future_years[1:]:
            forecast.append(forecast[-1] * 1.058)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_years, y=hist_values, name="Historical NHE", mode="lines", line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=future_years, y=forecast, name="CMS Projection (5.8%)", mode="lines", line=dict(color=RED, width=3, dash="dash")))
        fig.add_trace(go.Scatter(x=future_years + future_years[::-1], y=[value * 1.07 for value in forecast] + [value * 0.93 for value in forecast][::-1], fill="toself", fillcolor="rgba(231,76,60,0.12)", line=dict(color="rgba(255,255,255,0)"), name="Confidence band"))
        fig.add_vline(x=2024, line_dash="dot", line_color=GREY)
        fig.update_layout(**plot_template(), title="Historical and Projected NHE", yaxis_title="$ Trillions")
        st.plotly_chart(fig, use_container_width=True)

    pc_years = years_in_range(d["pc"], yr_start, 2024)
    fig = go.Figure(go.Scatter(x=pc_years, y=[d["pc"][year] for year in pc_years], fill="tozeroy", line=dict(color=BLUE2, width=3), fillcolor="rgba(46, 134, 193, 0.15)", name="Per Capita NHE"))
    fig.update_layout(**plot_template(), title="Per Capita NHE", yaxis_title="Per Capita (USD)", yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)


def render_explorer():
    add_common_layout(
        "Data Explorer",
        "Inspect filtered indicators, normalize series, and export the exact slice you are reviewing.",
    )
    all_series = {
        "Total NHE": d["nhe"],
        "GDP": d["gdp"],
        "NHE/GDP (%)": d["nhe_gdp"],
        "Admin Cost": d["admin"],
        "Admin % of NHE": d["admin_pct"],
        "Per Capita NHE": d["pc"],
        "Medicare": d["medicare"],
        "Medicaid": d["medicaid"],
        "Private Insurance": d["private_ins"],
        "Out-of-Pocket": d["out_of_pocket"],
        "Hospital Care": d["hospital"],
        "Physician Svcs": d["physician"],
        "Rx Drugs": d["rx"],
        "Nursing Care": d["nursing"],
        "Home Health": d["home_health"],
        "Public Health": d["pubh"],
    }

    selected = st.multiselect("Indicators to plot", list(all_series.keys()), default=["Total NHE", "Admin Cost", "Hospital Care"])
    normalize = st.checkbox("Normalize to an index (first year = 100)", value=False)

    fig = go.Figure()
    colors = [BLUE, RED, ORANGE, GREEN, PURPLE, BLUE2, GREY]
    for index, label in enumerate(selected):
        series = all_series[label]
        yrs = years_in_range(series, yr_start, yr_end)
        vals = [series[year] for year in yrs]
        if normalize and vals:
            base_value = vals[0]
            vals = [value / base_value * 100 for value in vals]
        fig.add_trace(go.Scatter(x=yrs, y=vals, name=label, mode="lines+markers", marker=dict(size=5), line=dict(color=colors[index % len(colors)], width=2.5)))
    fig.update_layout(**plot_template(), title="Custom Indicator Comparison", yaxis_title="Index (First Year = 100)" if normalize else "Value")
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show raw data table"):
        if selected:
            all_years = sorted(set.union(*[set(all_series[label].index) for label in selected]))
            all_years = [year for year in all_years if yr_start <= year <= yr_end]
            table = pd.DataFrame({"Year": all_years})
            for label in selected:
                table[label] = [all_series[label].get(year, np.nan) for year in all_years]
            st.dataframe(table.set_index("Year").round(2), use_container_width=True)
            st.download_button(
                "Download filtered data as CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name="nhe_filtered_data.csv",
                mime="text/csv",
            )


if page == "Overview":
    render_overview()
elif page == "Administrative Costs":
    render_admin_costs()
elif page == "Payer Mix":
    render_payer_mix()
elif page == "Service Categories":
    render_service_categories()
elif page == "Trends & Projections":
    render_trends()
else:
    render_explorer()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6B7280; font-size:12px;'>"
    "CMS National Health Expenditure Accounts (1960–2024) | Analysis and dashboard powered by Streamlit"
    "</div>",
    unsafe_allow_html=True,
)