import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random
import io

import pandas as pd



def _resample_consumption(df, granularity: str = "M"):
    """
    Resamples consumption dataframe based on chosen granularity.
    """
    if df is None or df.empty:
        return df

    # Try to find a datetime column
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "month" in col.lower() or "year" in col.lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"No suitable date column found. Columns are: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col])

    return (
        df.set_index(date_col)
          .resample(granularity)
          .sum()
          .reset_index()
    )

# ----------------------------
# Time Granularity Helpers (GLOBAL)
# ----------------------------
GRANULARITY_OPTIONS = ["Day level", "Week level", "Month level", "Quarter level", "Year level"]
_GRAN_TO_FREQ = {
    "Day level": "D",
    "Week level": "W",
    "Month level": "MS",
    "Quarter level": "QS",
    "Year level": "AS",
}
_GRAN_TO_TREND = {
    "Day level": "DoD",
    "Week level": "WoW",
    "Month level": "MoM",
    "Quarter level": "QoQ",
    "Year level": "YoY",
}

def _resample_sum(df: pd.DataFrame, date_col: str, value_col: str, granularity: str) -> pd.Series:
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.set_index(date_col)[value_col].sort_index()
    return s.resample(_GRAN_TO_FREQ[granularity]).sum()

def _resample_mean(df: pd.DataFrame, date_col: str, value_col: str, granularity: str) -> pd.Series:
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.set_index(date_col)[value_col].sort_index()
    return s.resample(_GRAN_TO_FREQ[granularity]).mean()

def _trend_pct(series: pd.Series) -> float:
    s = series.dropna().tail(2)
    if len(s) < 2:
        return 0.0
    prev, curr = float(s.iloc[0]), float(s.iloc[1])
    if abs(prev) < 1e-9:
        return 0.0
    return (curr - prev) / prev * 100.0

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Market Intelligence Platform - Transworld",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Design System (Palette, CSS, Components)
# ----------------------------
agriq_palette = {
    'main_green': '#1E8449',
    'light_green': '#D4EFDF',
    'chart_green': '#2ECC71',
    'accent_gold': '#F39C12',
    'dark_text': '#2C3E50',
    'light_text': '#ECF0F1',
    'background': '#FDFEFE',
    'red_negative': '#E74C3C',
    'positive_trend': '#28B463',
    'negative_trend': '#C0392B',
    'blue_import': '#5DADE2',
    'orange_export': '#85C74D',
}

st.markdown(
    f"""
    <style>
    .metric-card {{
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        text-align: center;
        margin-bottom: 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        border: 1px solid #eef2f7;
    }}
    .metric-value {{ font-size: 2rem; font-weight: 700; color: {agriq_palette['main_green']}; }}
    .metric-title {{ font-size: 0.95rem; font-weight: 600; color: {agriq_palette['dark_text']}; opacity: 0.85; white-space: normal; }}
    .metric-trend {{ font-size: 0.9rem; margin-top: 8px; }}
    .positive-trend {{ color: {agriq_palette['positive_trend']}; }}
    .negative-trend {{ color: {agriq_palette['negative_trend']}; }}
    .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; background:#f4f6f8; margin-right:6px; font-size:12px; color:{agriq_palette['dark_text']}; }}
    </style>
    """,
    unsafe_allow_html=True
)
# Add this CSS block after the existing styling code, around line 60
st.markdown(
    """
    <style>
    /* Fix header and tabs at top */
    .main > div {
        padding-top: 0rem;
    }
    
    /* Make main container flex */
    [data-testid="stAppViewContainer"] > .main {
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow: hidden;
    }
    
    /* Fix title at top */
    .main h1 {
        position: sticky;
        top: 0;
        background: white;
        z-index: 10;
        padding: 1rem 0 0.5rem 0;
        margin: 0;
        border-bottom: 1px solid #eee;
    }
    
    /* Fix tabs at top */
    [data-testid="stTabs"] {
        position: sticky;
        top: 4rem;
        background: white;
        z-index: 9;
        border-bottom: 1px solid #eee;
        margin-bottom: 0;
    }
    
    /* Make tab content scrollable */
    [data-testid="stTabContent"] {
        flex: 1;
        overflow-y: auto;
        padding: 1rem 0;
        height: calc(100vh - 8rem);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            overflow-y: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Make sidebar fit to screen height (no scrolling)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        max-height: 100vh;  /* full screen height */
        overflow-y: auto;   /* show all content */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Remove extra padding at top of sidebar so filters move up
# Force sidebar content to stick at very top (remove Streamlit default padding/margin)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] section[tabindex="0"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0rem;
        margin-top: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Time Granularity Helpers (NEW)
# ----------------------------
GRANULARITY_OPTIONS = ["Day level", "Week level", "Month level", "Quarter level", "Year level"]
_GRAN_TO_FREQ = {
    "Day level": "D",
    "Week level": "W",
    "Month level": "MS",
    "Quarter level": "QS",
    "Year level": "AS",
}
_GRAN_TO_TREND = {
    "Day level": "DoD",
    "Week level": "WoW",
    "Month level": "MoM",
    "Quarter level": "QoQ",
    "Year level": "YoY",
}

def _resample_sum(df: pd.DataFrame, date_col: str, value_col: str, granularity: str) -> pd.Series:
    """Sum aggregation for the given value column at selected granularity."""
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.set_index(date_col)[value_col].sort_index()
    freq = _GRAN_TO_FREQ[granularity]
    return s.resample(freq).sum()

def _resample_mean(df: pd.DataFrame, date_col: str, value_col: str, granularity: str) -> pd.Series:
    """Mean aggregation (useful for prices) at selected granularity."""
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.set_index(date_col)[value_col].sort_index()
    freq = _GRAN_TO_FREQ[granularity]
    return s.resample(freq).mean()

def _trend_pct(series: pd.Series) -> float:
    """Compute % change between the last two periods of a resampled series."""
    s = series.dropna().tail(2)
    if len(s) < 2:
        return 0.0
    prev, curr = float(s.iloc[0]), float(s.iloc[1])
    if abs(prev) < 1e-9:
        return 0.0
    return (curr - prev) / prev * 100.0


def create_kpi_card(title, value, unit, trend_value=None, trend_label="MoM"):
    trend_text = ""
    if trend_value is not None:
        icon = "▲" if trend_value >= 0 else "▼"
        klass = "positive-trend" if trend_value >= 0 else "negative-trend"
        trend_text = f"<span class='metric-trend {klass}'>{icon} {trend_value:,.1f}% {trend_label}</span>"
    return f"""
    <div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value'>{value:,.0f} {unit}</div>
        {trend_text}
    </div>
    """

# ----------------------------
# Session Bootstrap
# ----------------------------
def init_session_state():
    if "selected_products" not in st.session_state:
        st.session_state.selected_products = []
    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = []
    if "selected_region_groups" not in st.session_state:
        st.session_state.selected_region_groups = ["GCC"]
    if "date_range" not in st.session_state:
        st.session_state.date_range = [
            datetime.now() - timedelta(days=90),
            datetime.now(),
        ]
    if "alert_rules" not in st.session_state:
        st.session_state.alert_rules = []
    if "recent_exports" not in st.session_state:
        st.session_state.recent_exports = []
    # NEW: default time granularity
    if "time_granularity" not in st.session_state:
        st.session_state.time_granularity = "Month level"

# ----------------------------
# Static Taxonomies
# ----------------------------
@st.cache_data(show_spinner=False)
def load_product_data():
    return {
        "ETHNIC PRODUCTS": [
            "Ivy Gourd (Tindora)", "Punjabi Tinda", "Parwal (Pointed Gourd)",
            "Guvar (Cluster Beans)", "Arvi (Taro Root)", "Ponk (Green Sorghum)",
            "Tuvar Lilva (Pigeon Peas)", "Papdi Lilva (Green Flat Beans)",
            "Surti Papdi (Indian Beans)", "Methi (Fenugreek)",
            "Amla (Indian Gooseberry)", "Chikkoo (Sapota)",
            "Jamun (Indian Blackberry)", "Red Pappaya", "Falsa (Black Currant)",
            "Fresh Turmeric", "Ginger Paste", "Garlic Paste"
        ],
        "FRESH PACKED": [
            "Alphonso Mango", "Kesar Mango", "Green Grapes", "Pomegranate",
            "Banana", "Papaya", "Chikkoo (Mud Apple)", "Ginger",
            "Fresh Turmeric", "Fresh Mango Ginger(Curcuma Amada)",
            "Lime", "Fresh Potatoes"
        ],
        "PUREES, PULPS & CONCENTRATES": [
            "Alphonso Mango (Pulp/Concentrate)", "Totapuri Mango (Pulp/Concentrate)",
            "Tomato (Puree/Concentrate)", "Banana", "Papaya", "Pineapple",
            "Guava (Pink)", "Guava (White)"
        ],
        "INTERNATIONAL PRODUCTS": [
            "Sweet Corn", "Green Peas", "Mix Vegetables", "Okra", "Pumpkin",
            "Carrot", "Broccoli Florets", "French Beans (Cut)",
            "Cauliflower Florets", "Spinach", "Moringa Leaves",
            "Drum Stick (Cut)", "Bitter Gourd (Cut)", "Coconut",
            "Mango Alphonso", "Mango Totapuri", "Guava (White)",
            "Guava (Pink)", "Pomegranate", "Jack Fruit (Raw)",
            "Jack Fruit (Seeds)", "Banana", "Pineapple",
            "Green Capsicum", "Green Chilli", "Red Chilli", "Jalapeno",
            "Potato", "Onion", "Garlic", "Aloe Vera"
        ]
    }

@st.cache_data(show_spinner=False)
def load_region_data():
    return {
        "Gcc": ["Saudi Arabia", "Kuwait", "UAE", "Qatar", "Oman", "Behrain"],
        "North America": ["US", "Canada"],
        "Europe": ["UK", "Germany", "Holland"],
        "Australia": ["Australia", "NZ"],
        "Asia": ["India", "SE Asia"],
        "Africa": ["South Africa"],
    }

def get_flat_product_list():
    d = load_product_data()
    return [p for v in d.values() for p in v]

def get_flat_region_list():
    d = load_region_data()
    return [c for v in d.values() for c in v]

# ----------------------------
# Synthetic Data Generators
# ----------------------------
@st.cache_data(show_spinner=False)
def generate_enhanced_trade_data(products, regions, date_range):
    start, end = date_range
    dates = pd.date_range(start, end, freq="D")
    product_volumes = {
        "Alphonso Mango": {"min": 50, "max": 300, "seasonal_peak": [4, 5, 6]},
        "Basmati Rice": {"min": 500, "max": 2000, "seasonal_peak": [10, 11, 12]},
        "Carrot": {"min": 100, "max": 800, "seasonal_peak": [1, 2, 11, 12]},
        "Tomatoes": {"min": 150, "max": 1000, "seasonal_peak": [1, 2, 3, 10, 11, 12]},
        "IQF Vegetables": {"min": 200, "max": 600, "seasonal_peak": [6, 7, 8, 9]},
        "Dates": {"min": 80, "max": 400, "seasonal_peak": [9, 10, 11]},
        "Frozen Chicken": {"min": 300, "max": 1200, "seasonal_peak": [11, 12, 1]},
    }
    consignees = [
        "Al Rawabi Trading LLC",
        "Fresh Express International",
        "Gulf Food Distributors",
        "Emirates Food Group",
        "Lulu Hypermarket",
        "Carrefour UAE",
    ]
    exporters = [
        "AgroGlobal Exports",
        "GreenFields Produce",
        "IndoMills",
        "EuroFresh GmbH",
        "US Agro Foods",
        "VN Frozen Foods",
    ]
    ports = ["Jebel Ali Port", "Abu Dhabi Commercial Port", "Sharjah Port", "Port Rashid"]
    origins = ["India", "Vietnam", "Egypt", "Turkey", "USA", "UK", "Germany"]

    rows = []
    for product in products:
        vol_cfg = product_volumes.get(product, {"min": 100, "max": 500, "seasonal_peak": []})
        for region in regions:
            for d in dates:
                seasonal = 1.5 if d.month in vol_cfg["seasonal_peak"] else 1.0
                region_mult = {
                    "UAE": 1.2,
                    "Saudi Arabia": 1.0,
                    "Qatar": 0.8,
                    "Kuwait": 0.7,
                    "US": 1.5,
                    "UK": 1.1,
                    "Germany": 0.9,
                    "India": 0.6,
                }.get(region, 1.0)

                inbound = int(random.randint(vol_cfg["min"], vol_cfg["max"]) * seasonal * region_mult)
                outbound = int(inbound * random.uniform(0.5, 0.9))
                price_per_mt = random.uniform(600, 1400)
                value = price_per_mt * (inbound + outbound)

                rows.append(
                    {
                        "Date": d,
                        "Product": product,
                        "Region": region,
                        "Inbound_Volume_MT": inbound,
                        "Outbound_Volume_MT": outbound,
                        "Shipment_Value_USD": round(value, 2),
                        "Port": random.choice(ports),
                        "Consignee": random.choice(consignees),
                        "Exporter": random.choice(exporters),
                        "Origin_Country": random.choice(origins),
                    }
                )
    df = pd.DataFrame(rows)
    return df

@st.cache_data(show_spinner=False)
def generate_consumption_data(products, regions, date_range):
    start, end = date_range
    dates = pd.date_range(start, end, freq="D")
    channels = ["Retail", "Foodservice", "Institutional", "Re-Export"]
    rows = []
    for product in products:
        for region in regions:
            for d in dates:
                for ch in channels:
                    base = random.randint(1000, 6000)
                    wday = 1.0 + 0.05 * np.sin(d.weekday() / 7 * 2 * np.pi)
                    seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365)
                    ch_mult = {"Retail": 1.0, "Foodservice": 0.9, "Institutional": 0.7, "Re-Export": 0.5}[ch]
                    qty_kg = int(base * wday * seasonal * ch_mult)
                    price = round(random.uniform(0.8, 3.5), 2)
                    demand_score = round(random.uniform(50, 95), 1)
                    rows.append(
                        {
                            "Date": d,
                            "Product": product,
                            "Region": region,
                            "Channel": ch,
                            "Consumption_Kg": qty_kg,
                            "Price_USD_per_Kg": price,
                            "Demand_Score": demand_score,
                        }
                    )
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def generate_supplier_data(products, regions):
    certs = ["HACCP", "FSSC22000", "ISO22000", "BRCGS"]
    pay = ["Advance", "Net 15", "Net 30", "LC 30", "LC 60"]
    origins = ["India", "Vietnam", "Egypt", "Turkey", "USA", "UK", "Germany"]
    rows = []
    for i in range(120):
        rows.append(
            {
                "Supplier_Name": f"Supplier {i+1}",
                "Origin": random.choice(origins),
                "Products": ", ".join(random.sample(products, k=min(2, len(products)))),
                "Primary_Product": random.choice(products) if products else "",
                "Regions": ", ".join(random.sample(regions, k=min(2, len(regions)))),
                "Volume_MT": random.randint(50, 2000),
                "Rating": round(random.uniform(3.0, 5.0), 1),
                "MOQ_MT": random.choice([5, 10, 20, 25]),
                "Lead_Time_Days": random.choice([7, 10, 14, 21, 30]),
                "Payment_Terms": random.choice(pay),
                "Certifications": ", ".join(random.sample(certs, k=random.randint(1, 3))),
                "Last_Shipment": (datetime.now() - timedelta(days=random.randint(1, 120))).date(),
            }
        )
    return pd.DataFrame(rows)

# ----------------------------
# Sidebar / Navigation
# ----------------------------
def render_global_sidebar():

    products_dict = load_product_data()
    regions_dict = load_region_data()

    # Product Filters with Category and 'All' option
    st.sidebar.markdown("**Product**")
    product_categories = ["All"] + list(products_dict.keys())
    selected_product_category = st.sidebar.selectbox("Select Category", product_categories, key="product_category")
    
    if selected_product_category == "All":
        flat_products = [p for v in products_dict.values() for p in v]
    else:
        flat_products = products_dict[selected_product_category]

    selected_products = st.sidebar.multiselect(
        "Select Products",
        flat_products,
        default=st.session_state.selected_products if st.session_state.selected_products and all(item in flat_products for item in st.session_state.selected_products) else [],
        key="sidebar_products",
    )
    if not selected_products:
        st.session_state.selected_products = flat_products
    else:
        st.session_state.selected_products = selected_products

    st.sidebar.markdown("")  # Reduce spacing

    # Region Filters with Group and 'All' option
    st.sidebar.markdown("**Region**")
    region_groups = ["All"] + list(regions_dict.keys())
    selected_region_group = st.sidebar.selectbox("Select Group", region_groups, key="region_group")
    
    if selected_region_group == "All":
        flat_regions = [c for v in regions_dict.values() for c in v]
    else:
        flat_regions = regions_dict[selected_region_group]
    
    selected_regions = st.sidebar.multiselect(
        "Select Countries",
        flat_regions,
        default=st.session_state.selected_regions if st.session_state.selected_regions and all(item in flat_regions for item in st.session_state.selected_regions) else [],
        key="sidebar_regions",
    )
    if not selected_regions:
        st.session_state.selected_regions = flat_regions
    else:
        st.session_state.selected_regions = selected_regions

    st.sidebar.markdown("")  # Reduce spacing

    # Date range
    date_range = st.sidebar.date_input(
        "Date Range",
        value=tuple(st.session_state.date_range),
        key="sidebar_daterange",
    )
    # Convert to datetimes
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        st.session_state.date_range = [
            datetime.combine(date_range[0], datetime.min.time()),
            datetime.combine(date_range[1], datetime.min.time()),
        ]

    # ----------------------------
    # NEW: Time Granularity (global)
    # ----------------------------
    st.sidebar.markdown("**Time Granularity**")
    st.session_state.time_granularity = st.sidebar.selectbox(
        "Aggregate data by",
        GRANULARITY_OPTIONS,
        index=GRANULARITY_OPTIONS.index(st.session_state.time_granularity) if st.session_state.get("time_granularity") in GRANULARITY_OPTIONS else 2,
        key="time_granularity_select",
    )
    
# ----------------------------
# Tabs
# ----------------------------
def executive_summary_tab():
    st.header(" Executive Summary")
    st.markdown("High-level market intelligence across selected filters")

    if not st.session_state.selected_products or not st.session_state.selected_regions:
        st.warning(" Please select at least one product and one region from the sidebar.")
        return

    trade = generate_enhanced_trade_data(
        st.session_state.selected_products,
        st.session_state.selected_regions,
        st.session_state.date_range,
    )
    cons = generate_consumption_data(
        st.session_state.selected_products,
        st.session_state.selected_regions,
        st.session_state.date_range,
    )

    gran = st.session_state.time_granularity
    trend_label = _GRAN_TO_TREND[gran]

    # KPI Row (Styled Cards) + Filter Pills
    st.markdown(
        f"**Filters:** "
        f"<span class='pill'>Category: {st.session_state.product_category}</span>"
        f"<span class='pill'>Products: {', '.join(st.session_state.selected_products)}</span>"
        f"<span class='pill'>Region Group: {st.session_state.region_group}</span>"
        f"<span class='pill'>Regions: {', '.join(st.session_state.selected_regions)}</span>"
        f"<span class='pill'>Range: {st.session_state.date_range[0].date()} → {st.session_state.date_range[1].date()}</span>"
        f"<span class='pill'>Granularity: {gran}</span>",
        unsafe_allow_html=True,
    )

    # Aggregate series at selected granularity
    cons_series = _resample_sum(cons, "Date", "Consumption_Kg", gran) / 1000.0  # MT
    imp_series = _resample_sum(trade, "Date", "Inbound_Volume_MT", gran)
    exp_series = _resample_sum(trade, "Date", "Outbound_Volume_MT", gran)

    # KPIs (totals over selected period)
    consumption_mt = cons_series.sum()
    production_mt = consumption_mt * random.uniform(0.9, 1.1)
    imports_mt = imp_series.sum()
    exports_mt = exp_series.sum()
    balance_mt = production_mt + imports_mt - consumption_mt - exports_mt

    # Trends based on last two periods at selected granularity
    # Make production trend track consumption trend (proxy)
    trend_cons = _trend_pct(cons_series)
    trend_prod = trend_cons
    trend_imp = _trend_pct(imp_series)
    trend_exp = _trend_pct(exp_series)

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.markdown(create_kpi_card("Production", production_mt, "MT", trend_prod, trend_label), unsafe_allow_html=True)
    r1c2.markdown(create_kpi_card("Consumption", consumption_mt, "MT", trend_cons, trend_label), unsafe_allow_html=True)
    r1c3.markdown(create_kpi_card("Market Balance", balance_mt, "MT"), unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    r2c1.markdown(create_kpi_card("Imports", imports_mt, "MT", trend_imp, trend_label), unsafe_allow_html=True)
    r2c2.markdown(create_kpi_card("Exports", exports_mt, "MT", trend_exp, trend_label), unsafe_allow_html=True)

    st.subheader("Production vs Consumption")
    # production proxy curve: scaled + noise on consumption
    prod_series = cons_series * random.uniform(0.95, 1.05)
    prod_series = prod_series + pd.Series(np.random.randn(len(prod_series)) * 5, index=prod_series.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prod_series.index, y=prod_series.values, name="Production", line=dict(color=agriq_palette['chart_green'])))
    fig.add_trace(go.Scatter(x=cons_series.index, y=cons_series.values, name="Consumption", line=dict(color=agriq_palette['accent_gold'])))
    fig.update_layout(xaxis_title="Date", yaxis_title="MT")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Import vs Export Flows")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=imp_series.index, y=imp_series.values, name="Imports", line=dict(color=agriq_palette['blue_import'])))
    fig2.add_trace(go.Scatter(x=exp_series.index, y=exp_series.values, name="Exports", line=dict(color=agriq_palette['orange_export'])))
    fig2.update_layout(xaxis_title="Date", yaxis_title="MT")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Market Balance Waterfall")
    wf = go.Figure(go.Waterfall(
        name="Balance",
        orientation="v",
        measure=["relative", "relative", "relative", "relative"],
        x=["Production", "Imports", "- Consumption", "- Exports"],
        textposition="outside",
        y=[production_mt, imports_mt, -consumption_mt, -exports_mt],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    wf.update_layout(showlegend=False, yaxis_title="MT")
    st.plotly_chart(wf, use_container_width=True)


def product_analytics_tab():
    st.header(" Product Analytics")
    st.markdown("Deep-dive into product performance, pricing, and trade partners")

    if not st.session_state.selected_products or not st.session_state.selected_regions:
        st.warning(" Please select products and regions.")
        return

    product = st.selectbox(
        "Select Product",
        st.session_state.selected_products,
        key="product_detail_selector",
    )

    trade = generate_enhanced_trade_data([product], st.session_state.selected_regions, st.session_state.date_range)
    cons = generate_consumption_data([product], st.session_state.selected_regions, st.session_state.date_range)

    gran = st.session_state.time_granularity
    trend_label = _GRAN_TO_TREND[gran]

    # ---- KPIs (aggregated at chosen granularity) ----
    cons_series = _resample_sum(cons, "Date", "Consumption_Kg", gran) / 1000.0  # MT
    consumption_mt = cons_series.sum()
    production_mt = consumption_mt * random.uniform(0.9, 1.1)

    imp_series = _resample_sum(trade, "Date", "Inbound_Volume_MT", gran)
    exp_series = _resample_sum(trade, "Date", "Outbound_Volume_MT", gran)
    imports_mt = imp_series.sum()
    exports_mt = exp_series.sum()

    # Trends
    trend_prod = _trend_pct(cons_series)   # proxy same as consumption
    trend_cons = trend_prod
    trend_imp = _trend_pct(imp_series)
    trend_exp = _trend_pct(exp_series)

    # Price stats
    price_avg = cons["Price_USD_per_Kg"].mean()
    price_vol = cons["Price_USD_per_Kg"].std()
    demand_score = cons["Demand_Score"].mean()
    inventory_mt = max(0, production_mt * 0.2)

    # Price trend at selected granularity
    price_series = _resample_mean(cons, "Date", "Price_USD_per_Kg", gran)
    trend_price = _trend_pct(price_series)

    # ---- KPI Cards ----
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.markdown(create_kpi_card("Production", production_mt, "MT", trend_prod, trend_label), unsafe_allow_html=True)
    r1c2.markdown(create_kpi_card("Consumption", consumption_mt, "MT", trend_cons, trend_label), unsafe_allow_html=True)
    r1c3.markdown(create_kpi_card("Avg Price", price_avg, "$/kg", trend_price, trend_label), unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.markdown(create_kpi_card("Imports", imports_mt, "MT", trend_imp, trend_label), unsafe_allow_html=True)
    r2c2.markdown(create_kpi_card("Exports", exports_mt, "MT", trend_exp, trend_label), unsafe_allow_html=True)
    r2c3.markdown(create_kpi_card("Inventory (est.)", inventory_mt, "MT"), unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)
    r3c1.markdown(create_kpi_card("Price Volatility (σ)", price_vol, ""), unsafe_allow_html=True)
    r3c2.markdown(create_kpi_card("Demand Score", demand_score, ""), unsafe_allow_html=True)

    # ---- Charts ----
    st.subheader(f"{product}: Consumption Trend")
    cons_trend = cons_series
    fig = px.line(cons_trend.reset_index(), x="Date", y="Consumption_Kg", title=f"{product}: Consumption Trend")
    fig.update_yaxes(title="MT")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Price Distribution")
    fig = px.box(cons, x="Region", y="Price_USD_per_Kg", points="suspectedoutliers", title="Regional Price Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(" Demand Score by Region & Channel")
    treemap_df = cons.groupby(["Region", "Channel"]).agg(score=("Demand_Score", "mean")).reset_index()
    fig_t = px.treemap(treemap_df, path=["Region", "Channel"], values="score", title="Demand Score Treemap")
    st.plotly_chart(fig_t, use_container_width=True)

    st.subheader(" Top Consignees & Exporters")
    col1, col2 = st.columns(2)
    with col1:
        cons_tbl = (
            trade.groupby("Consignee")
            .agg(
                Inbound_Volume_MT=("Inbound_Volume_MT", "sum"),
                Shipments=("Consignee", "count"),
                Shipment_Value_USD=("Shipment_Value_USD", "sum"),
                Last_Transaction=("Date", "max"),
            )
            .sort_values("Inbound_Volume_MT", ascending=False)
            .head(5)
        )
        cons_tbl["Market_Share_%"] = (cons_tbl["Inbound_Volume_MT"] / cons_tbl["Inbound_Volume_MT"].sum() * 100).round(1)
        st.dataframe(cons_tbl, use_container_width=True)
    with col2:
        exp_tbl = (
            trade.groupby("Exporter")
            .agg(
                Outbound_Volume_MT=("Outbound_Volume_MT", "sum"),
                Shipments=("Exporter", "count"),
                Shipment_Value_USD=("Shipment_Value_USD", "sum"),
                Last_Transaction=("Date", "max"),
            )
            .sort_values("Outbound_Volume_MT", ascending=False)
            .head(5)
        )
        exp_tbl["Market_Share_%"] = (exp_tbl["Outbound_Volume_MT"] / exp_tbl["Outbound_Volume_MT"].sum() * 100).round(1)
        st.dataframe(exp_tbl, use_container_width=True)



def region_details_tab():
    st.header(" Region Details")
    st.markdown("Regional market intelligence and trade-flow analysis")

    if not st.session_state.selected_regions:
        st.warning(" Please select at least one region.")
        return

    region = st.selectbox(" Select Region", st.session_state.selected_regions, key="region_detail_selector")

    trade = generate_enhanced_trade_data(st.session_state.selected_products, [region], st.session_state.date_range)
    cons = generate_consumption_data(st.session_state.selected_products, [region], st.session_state.date_range)
    suppliers = generate_supplier_data(st.session_state.selected_products, [region])

    gran = st.session_state.time_granularity
    trend_label = _GRAN_TO_TREND[gran]

    # --- KPIs ---
    cons_series = _resample_sum(cons, "Date", "Consumption_Kg", gran) / 1000.0
    consumption_mt = cons_series.sum()
    production_mt = consumption_mt * random.uniform(0.9, 1.1)

    imp_series = _resample_sum(trade, "Date", "Inbound_Volume_MT", gran)
    exp_series = _resample_sum(trade, "Date", "Outbound_Volume_MT", gran)
    imports_mt = imp_series.sum()
    exports_mt = exp_series.sum()
    balance_mt = production_mt + imports_mt - consumption_mt - exports_mt
    active_suppliers = len(suppliers)
    active_consignees = trade["Consignee"].nunique()

    # --- Trends ---
    trend_prod = _trend_pct(cons_series)  # proxy
    trend_cons = trend_prod
    trend_imp = _trend_pct(imp_series)
    trend_exp = _trend_pct(exp_series)

    # --- KPI Layout (3 per row, centered) ---
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.markdown(create_kpi_card("Production", production_mt, "MT", trend_prod, trend_label), unsafe_allow_html=True)
    r1c2.markdown(create_kpi_card("Consumption", consumption_mt, "MT", trend_cons, trend_label), unsafe_allow_html=True)
    r1c3.markdown(create_kpi_card("Imports", imports_mt, "MT", trend_imp, trend_label), unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.markdown(create_kpi_card("Exports", exports_mt, "MT", trend_exp, trend_label), unsafe_allow_html=True)
    r2c2.markdown(create_kpi_card("Market Balance", balance_mt, "MT"), unsafe_allow_html=True)
    r2c3.markdown(create_kpi_card("Active Suppliers", active_suppliers, "count"), unsafe_allow_html=True)

    r3c1, _ , r3c2 = st.columns([1,0.2,1])  # only 2 centered
    r3c1.markdown(create_kpi_card("Active Consignees", active_consignees, "count"), unsafe_allow_html=True)

    # --- Charts ---
    st.subheader(f"{region}: Inbound vs Outbound")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=imp_series.index, y=imp_series.values, name="Inbound (MT)"))
    fig.add_trace(go.Scatter(x=exp_series.index, y=exp_series.values, name="Outbound (MT)"))
    fig.update_layout(xaxis_title="Date", yaxis_title="MT")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Port-wise Shipment Breakdown")
    port_df = trade.groupby("Port").agg(
        Inbound_Volume_MT=("Inbound_Volume_MT", "sum"),
        Outbound_Volume_MT=("Outbound_Volume_MT", "sum"),
    ).reset_index()
    figp = px.bar(port_df, x="Port", y=["Inbound_Volume_MT", "Outbound_Volume_MT"], barmode="group", title="Port-wise Shipment Breakdown")
    st.plotly_chart(figp, use_container_width=True)

    # --- Top Consignees & Exporters ---
    st.subheader(" Top Consignees & Exporters")
    c1, c2 = st.columns(2)
    with c1:
        cons_tbl = (
            trade.groupby("Consignee")
            .agg(
                Inbound_Volume_MT=("Inbound_Volume_MT", "sum"),
                Shipments=("Consignee", "count"),
                Last_Transaction=("Date", "max"),
            )
            .sort_values("Inbound_Volume_MT", ascending=False)
            .head(10)
        )
        st.dataframe(cons_tbl, use_container_width=True)
    with c2:
        exp_tbl = (
            trade.groupby("Exporter")
            .agg(
                Outbound_Volume_MT=("Outbound_Volume_MT", "sum"),
                Shipments=("Exporter", "count"),
                Last_Transaction=("Date", "max"),
            )
            .sort_values("Outbound_Volume_MT", ascending=False)
            .head(10)
        )
        st.dataframe(exp_tbl, use_container_width=True)

    # --- Channel insights ---
    st.subheader(" Channel Insights")
    ch = cons.groupby("Channel").agg(Kg=("Consumption_Kg", "sum")).reset_index()
    figc = px.bar(ch, x="Channel", y="Kg", title="Consumption by Channel (Kg)")
    st.plotly_chart(figc, use_container_width=True)

    # --- Supplier overview ---
    st.subheader(" Supplier Overview")
    sup_cols = ["Supplier_Name", "Origin", "Products", "Volume_MT", "Rating", "Last_Shipment", "MOQ_MT", "Lead_Time_Days", "Payment_Terms", "Certifications"]
    st.dataframe(suppliers[sup_cols].sort_values("Volume_MT", ascending=False).head(20), use_container_width=True)

    # --- Origins Map ---
    st.subheader(" Origins Map")
    origins_df = trade.groupby("Origin_Country").agg(Volume_MT=("Inbound_Volume_MT", "sum")).reset_index()
    if not origins_df.empty:
        figm = px.scatter_geo(
            origins_df,
            locations="Origin_Country",
            locationmode="country names",
            size="Volume_MT",
            hover_name="Origin_Country",
            projection="natural earth",
            title="Inbound Origins by Volume",
        )
        st.plotly_chart(figm, use_container_width=True)
    else:
        st.info("No trade origin data for the selected filters.")

def forecasting_demand_planning_tab():
    st.header(" Forecasting & Demand Planning")
    st.markdown("Historical vs projected demand and supply-demand balance")

    # 🔧 Step 1: Map human labels → Pandas codes
    granularity_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }

    # 🔧 Step 2: Let user choose
    granularity_label = st.selectbox(
        "Select Forecast Granularity",
        list(granularity_map.keys()),
        index=2  # default = Monthly
    )
    granularity = granularity_map[granularity_label]  # Pandas-friendly code

    # --- Check for product/region
    if not st.session_state.selected_products or not st.session_state.selected_regions:
        st.warning(" Please select products and regions.")
        return

    # --- User inputs
    product = st.selectbox("Product", st.session_state.selected_products, key="fc_product")
    region = st.selectbox("Region", st.session_state.selected_regions, key="fc_region")
    horizon = st.selectbox("Forecast Horizon", ["1M", "3M", "6M"], index=1, key="fc_horizon")

    # --- Generate data
    cons = generate_consumption_data([product], [region], st.session_state.date_range)
    hist = _resample_consumption(cons, granularity)
    hist.index = pd.to_datetime(hist.index)

    # --- Forecast logic
    window_map = {"D": 7, "W": 4, "M": 3, "Q": 2, "Y": 2}
    window = window_map.get(granularity, 3)  # fallback = 3

    avg = hist.tail(window)["Consumption_Kg"].mean()

    if granularity == "D":
        periods = {"1M": 30, "3M": 90, "6M": 180}[horizon]
        freq = "D"
    elif granularity == "W":
        periods = {"1M": 4, "3M": 12, "6M": 26}[horizon]
        freq = "W"
    elif granularity == "M":
        periods = {"1M": 1, "3M": 3, "6M": 6}[horizon]
        freq = "MS"
    elif granularity == "Q":
        periods = {"1M": 1, "3M": 1, "6M": 2}[horizon]  # rough proxy
        freq = "QS"
    else:  # Yearly
        periods = {"1M": 1, "3M": 1, "6M": 1}[horizon]
        freq = "YS"

    future_idx = pd.date_range(
        hist.index.max() + pd.tseries.frequencies.to_offset(freq),
        periods=periods,
        freq=freq
    )
    fc = pd.DataFrame(index=future_idx, data={"Consumption_Kg": avg})

    # --- KPIs
    forecast_kg = fc["Consumption_Kg"].sum()
    forecast_mt = forecast_kg / 1000

    trade = generate_enhanced_trade_data([product], [region], st.session_state.date_range)
    production_mt = hist["Consumption_Kg"].sum() / 1000 * 0.9  # proxy 90%
    imports_mt = trade["Inbound_Volume_MT"].sum()
    est_supply_mt = production_mt + imports_mt
    gap_mt = est_supply_mt - forecast_mt

    growth_rate = (
        (fc["Consumption_Kg"].mean() - hist.tail(window)["Consumption_Kg"].mean())
        / hist.tail(window)["Consumption_Kg"].mean() * 100
        if not hist.tail(window).empty else 0.0
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(create_kpi_card("Forecasted Demand", forecast_mt, "MT"), unsafe_allow_html=True)
    k2.markdown(create_kpi_card("Estimated Supply", est_supply_mt, "MT"), unsafe_allow_html=True)
    k3.markdown(create_kpi_card("Supply–Demand Gap", gap_mt, "MT"), unsafe_allow_html=True)
    k4.markdown(create_kpi_card("Growth Rate (naive)", growth_rate, "%"), unsafe_allow_html=True)

    # --- Plots
    st.subheader("History vs Forecast")
    histr = hist.copy()
    histr["Type"] = "History"
    fcr = fc.copy()
    fcr["Type"] = "Forecast"
    both = pd.concat([histr, fcr]).reset_index().rename(columns={"index": "Date"})
# 🔧 Ensure no duplicate column names
    both = both.loc[:, ~both.columns.duplicated()]

    fig = px.line(
        both, x="Date", y="Consumption_Kg", color="Type",
        title=f"{product} @ {region}: {granularity_label} Demand"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Channel-wise Forecast (naive)")
    ch = cons.groupby("Channel")["Consumption_Kg"].sum()
    ch_df = ch.reset_index().rename(columns={"Consumption_Kg": "Hist_Kg"})
    ch_df["Forecast_Kg"] = ch_df["Hist_Kg"] * (forecast_kg / max(1.0, hist["Consumption_Kg"].sum()))
    figc = px.bar(ch_df, x="Channel", y=["Hist_Kg", "Forecast_Kg"], barmode="group", title="Channel Allocation")
    st.plotly_chart(figc, use_container_width=True)

    st.subheader("Supply vs Forecast Demand")
    comp = pd.DataFrame(
        {"Metric": ["Estimated Supply (MT)", "Forecast Demand (MT)"],
         "MT": [est_supply_mt, forecast_mt]}
    )
    figb = px.bar(comp, x="Metric", y="MT", title="Supply vs Demand")
    st.plotly_chart(figb, use_container_width=True)



def alerts_tab():
    st.header("🚨 Alerts")
    st.markdown("Create and manage alerts for price, volume, new suppliers, and anomalies")

    with st.form("alert_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            a_type = st.selectbox("Alert Type", ["Price Threshold", "Volume Change", "New Supplier", "Anomaly"])
        with col2:
            product = st.selectbox("Product", st.session_state.selected_products or get_flat_product_list())
        with col3:
            region = st.selectbox("Region", st.session_state.selected_regions or get_flat_region_list())

        thresh = st.text_input("Threshold / Rule (e.g., price > 2.5, volume change > 20%)")
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        channels = st.multiselect("Channels", ["Email", "SMS", "Dashboard", "Push"], default=["Dashboard"])
        freq = st.selectbox("Frequency", ["Real-time", "Hourly", "Daily", "Weekly"], index=2)
        submitted = st.form_submit_button("Create Alert")

        if submitted:
            st.session_state.alert_rules.append(
                {
                    "type": a_type,
                    "product": product,
                    "region": region,
                    "threshold": thresh,
                    "priority": priority,
                    "channels": channels,
                    "frequency": freq,
                    "active": True,
                    "created_at": datetime.now(),
                }
            )
            st.success(" Alert created.")

    st.subheader("Active Alerts")
    if not st.session_state.alert_rules:
        st.info("No alerts configured yet.")
    else:
        for idx, rule in enumerate(list(st.session_state.alert_rules)):
            cols = st.columns([3, 3, 3, 1, 1])
            cols[0].markdown(f"**{rule['type']}** — {rule['product']} @ {rule['region']}")
            cols[1].markdown(f"Rule: `{rule['threshold']}` • Priority: **{rule['priority']}**")
            cols[2].markdown(f"Channels: {', '.join(rule['channels'])} • {rule['frequency']}")
            if cols[3].button(("Pause" if rule["active"] else "Resume"), key=f"toggle_{idx}"):
                st.session_state.alert_rules[idx]["active"] = not rule["active"]
            if cols[4].button("Delete", key=f"delete_{idx}"):
                st.session_state.alert_rules.pop(idx)
                st.experimental_rerun()

    st.subheader("Recent Notifications (Mock)")
    noti = pd.DataFrame(
        [
            {"Time": datetime.now() - timedelta(hours=2), "Severity": "High", "Message": "Price > $3.0/kg for Tomatoes in UAE"},
            {"Time": datetime.now() - timedelta(hours=5), "Severity": "Medium", "Message": "New supplier detected for IQF Vegetables (India)"},
            {"Time": datetime.now() - timedelta(days=1), "Severity": "Low", "Message": "Volume +12% for Basmati Rice imports"},
        ]
    )
    st.table(noti)


def reports_export_tab():
    st.header(" Reports & Export")
    st.markdown("Configure and download reports. (Exports are generated in-memory.)")

    rtype = st.selectbox("Report Type", ["Market Summary", "Trade Flow", "Price Trend"], index=0)
    start, end = st.date_input("Report Date Range", value=tuple(st.session_state.date_range))
    fmt = st.selectbox("Format", ["CSV", "XLSX", "PDF (mock)", "PPTX (mock)"], index=1)
    include_charts = st.checkbox("Include charts snapshot (where applicable)")
    include_raw = st.checkbox("Include raw data extract", value=True)

    # Preview
    st.subheader("Preview")
    st.write(
        f"**Type:** {rtype}  •  **Range:** {start} to {end}  •  **Format:** {fmt}  •  "
        f"**Include charts:** {include_charts}  •  **Include raw:** {include_raw}"
    )

    # Build datasets for export
    trade = generate_enhanced_trade_data(
        st.session_state.selected_products, st.session_state.selected_regions, st.session_state.date_range
    )
    cons = generate_consumption_data(
        st.session_state.selected_products, st.session_state.selected_regions, st.session_state.date_range
    )

    buffer = io.BytesIO()
    filename = f"{rtype.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mime = "text/csv"
    data_bytes = None

    if fmt == "CSV":
        if rtype == "Market Summary":
            df = cons.groupby(["Date", "Region"]).agg(Consumption_Kg=("Consumption_Kg", "sum")).reset_index()
        elif rtype == "Trade Flow":
            df = trade.copy()
        else:
            df = cons[["Date", "Product", "Region", "Price_USD_per_Kg"]].copy()
        data_bytes = df.to_csv(index=False).encode("utf-8")
        filename += ".csv"
        mime = "text/csv"
    elif fmt == "XLSX":
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            if rtype == "Market Summary":
                cons.groupby(["Date", "Product", "Region"]).agg(Consumption_Kg=("Consumption_Kg", "sum")).reset_index().to_excel(
                    writer, sheet_name="Market Summary", index=False
                )
            if rtype == "Trade Flow":
                trade.to_excel(writer, sheet_name="Trade Flow", index=False)
            if rtype == "Price Trend":
                cons[["Date", "Product", "Region", "Price_USD_per_Kg"]].to_excel(writer, sheet_name="Price", index=False)
        data_bytes = buffer.getvalue()
        filename += ".xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif fmt == "PDF (mock)":
        data_bytes = b"%PDF-1.4\n% Mock PDF content generated by Streamlit app.\n"
        filename += ".pdf"
        mime = "application/pdf"
    else:  # PPTX mock
        data_bytes = b"PPTX-MOCK-BYTES"
        filename += ".pptx"
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

    st.download_button("⬇ Download Report", data=data_bytes, file_name=filename, mime=mime)

    # Schedule (simple stub)
    st.subheader("Scheduled Reports (Stub)")
    s1, s2 = st.columns(2)
    with s1:
        sch_enabled = st.checkbox("Enable Scheduling")
    with s2:
        sch_freq = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=1)
    if sch_enabled:
        st.success(f"Scheduling enabled: {sch_freq}")

    # Recent exports history
    if st.button("Save to Export History"):
        st.session_state.recent_exports.append({"date": datetime.now(), "type": rtype, "format": fmt, "status": "Completed"})
    if st.session_state.recent_exports:
        st.subheader("Recent Exports")
        st.table(pd.DataFrame(st.session_state.recent_exports))

# ----------------------------
# App Entrypoint
# ----------------------------
def main():
    init_session_state()
    render_global_sidebar()
    st.title("Market Intelligence Platform")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary",
        "Product Analytics",
        "Region Details",
        "Forecasting & Demand",
        "Alerts",
        "Reports & Export",
    ])
    
    with tab1:
        executive_summary_tab()
    with tab2:
        product_analytics_tab()
    with tab3:
        region_details_tab()
    with tab4:
        forecasting_demand_planning_tab()
    with tab5:
        alerts_tab()
    with tab6:
        reports_export_tab()

if __name__ == "__main__":
    main()
