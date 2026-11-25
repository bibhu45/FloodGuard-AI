from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import zoom, distance_transform_edt

# -------------------------- App configuration --------------------------
st.set_page_config(
    page_title="FloodGuard AI",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "FloodGuard_Data"
RAIN_A = DATA / "01_rainfall" / "floodguard_synthetic_rain.npz"
RAIN_B = DATA / "11_second_city" / "city_B_synthetic_rain.npz"
STATIC_A = DATA / "02_terrain_and_landcover" / "synthetic_static_layers.npz"
STATIC_B = DATA / "11_second_city" / "city_B_synthetic_static_layers.npz"
EVENTS = DATA / "03_flood_spots_and_events" / "test_events_SYNTHETIC.csv"
WARDS = DATA / "08_city_boundary_and_wards" / "wards_SYNTHETIC.geojson"
SPOTS = DATA / "09_flood_labels" / "flood_spots_SYNTHETIC.csv"
GAUGES = DATA / "06_gauges_and_weather" / "gauge_weather_hourly_SYNTHETIC.csv"
METRICS_DIR = ROOT / "outputs" / "metrics"
FORECAST_DIR = ROOT / "outputs" / "forecasts"

# -------------------------- Styling --------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: linear-gradient(135deg, #07111f 0%, #0a1729 55%, #07111f 100%); }
[data-testid="stSidebar"] { background: #081323; border-right: 1px solid #1c314b; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #eaf3ff !important; }
p, label, span { color: #d5e2f2; }
div[data-testid="stMetric"] {
  background: rgba(17, 34, 55, 0.85); border: 1px solid #203b59;
  padding: 15px 18px; border-radius: 14px;
}
div[data-testid="stMetricLabel"] { color: #8da9c8 !important; }
div[data-testid="stMetricValue"] { color: #f3f8ff !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: #12243a; border-radius: 9px; }
.small-muted { color: #91a8c3; font-size: 0.88rem; }
.warning-box { background: #332819; border: 1px solid #8d682a; padding: 12px 16px; border-radius: 12px; }
.info-box { background: #10253b; border: 1px solid #244966; padding: 12px 16px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# -------------------------- Data loaders --------------------------
@st.cache_data(show_spinner=False)
def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_city(city_name):
    if city_name == "Mumbai-like synthetic city":
        rain_file, static_file = RAIN_A, STATIC_A
        wards_file = WARDS
        return load_npz(rain_file), load_npz(static_file), wards_file
    rain_file, static_file = RAIN_B, STATIC_B
    return load_npz(rain_file), load_npz(static_file), None

def warning_for_rain(value):
    if value < 2.5:
        return "Green", "No significant rain", "#25b87a"
    if value < 10:
        return "Yellow", "Moderate to heavy rain, be aware", "#e7bd42"
    if value < 25:
        return "Orange", "Heavy rain, be prepared", "#f08a35"
    return "Red", "Very heavy rain, take action", "#e94c55"

def risk_map(static, rain_rate):
    """Relative risk visualization based on synthetic terrain/land layers and rainfall.
    This is a prototype index, not flood depth or a calibrated hydrological model.
    """
    elev = np.asarray(static["elevation_m"], dtype=float)
    land = np.asarray(static["landcover"], dtype=float)
    river = np.asarray(static["river_mask"], dtype=bool)
    elev_norm = (elev - np.nanmin(elev)) / (np.nanmax(elev) - np.nanmin(elev) + 1e-6)
    low_elev = 1.0 - elev_norm
    built = (land == 1).astype(float)
    # distance-to-river score; if no river pixels, safely use zeros
    if river.any():
        dist = distance_transform_edt(~river)
        river_near = 1.0 - np.clip(dist / (dist.max() + 1e-6), 0, 1)
    else:
        river_near = np.zeros_like(elev)
    rain = np.asarray(rain_rate, dtype=float)
    rain = zoom(rain, (elev.shape[0] / rain.shape[0], elev.shape[1] / rain.shape[1]), order=1)
    rain = rain[:elev.shape[0], :elev.shape[1]]
    rain_norm = np.clip(rain / 50.0, 0, 1)
    score = 0.38 * rain_norm + 0.24 * low_elev + 0.20 * built + 0.18 * river_near
    score = np.clip(score, 0, 1)
    # Sea pixels are excluded from terrestrial risk display
    score[land == 0] = np.nan
    return score

def risk_class(score):
    if score < 0.25:
        return "Low"
    if score < 0.5:
        return "Moderate"
    if score < 0.75:
        return "High"
    return "Severe"

def map_rain(rain, lat, lon, title):
    fig = go.Figure(go.Heatmap(
        z=rain, x=lon, y=lat, colorscale=[
            [0.0, "#172b4d"], [0.08, "#2366a8"], [0.25, "#25b7d6"],
            [0.45, "#4bd18b"], [0.65, "#f2d84b"], [0.82, "#f28b38"], [1.0, "#e74652"]
        ],
        colorbar=dict(title="mm/hr"), zmin=0, zmax=max(25, float(np.nanmax(rain))),
        hovertemplate="Lon %{x:.3f}<br>Lat %{y:.3f}<br>Rain %{z:.1f} mm/hr<extra></extra>"
    ))
    fig.update_layout(title=title, height=500, margin=dict(l=5,r=5,t=45,b=5),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font_color="#dceaff", xaxis_title="Longitude", yaxis_title="Latitude")
    return fig

def map_risk(score, static, lat, lon, wards_path=None, spots_df=None,
             show_risk=True, show_river=True, show_spots=True, show_wards=True):
    fig = go.Figure()
    if show_risk:
        fig.add_trace(go.Heatmap(
            z=score, x=lon, y=lat,
            colorscale=[[0,"#2e8bdb"],[0.25,"#42c77a"],[0.5,"#f2d44a"],[0.75,"#f18a3a"],[1,"#e74b55"]],
            zmin=0, zmax=1, colorbar=dict(title="Relative risk"),
            hovertemplate="Lon %{x:.3f}<br>Lat %{y:.3f}<br>Risk %{z:.2f}<extra></extra>"
        ))
    # Overlay synthetic river mask
    river = np.asarray(static["river_mask"])
    ry, rx = np.where(river > 0)
    if show_river and len(rx):
        fig.add_trace(go.Scatter(x=lon[np.clip((rx/(river.shape[1]-1)*(len(lon)-1)).astype(int),0,len(lon)-1)],
                                y=lat[np.clip((ry/(river.shape[0]-1)*(len(lat)-1)).astype(int),0,len(lat)-1)],
                                mode="markers", marker=dict(color="#59d7ff", size=2), name="River mask",
                                hoverinfo="skip"))
    if show_spots and spots_df is not None and len(spots_df):
        fig.add_trace(go.Scatter(x=spots_df["longitude"], y=spots_df["latitude"], mode="markers",
                                marker=dict(color="white", size=9, symbol="x", line=dict(width=1)),
                                name="Synthetic flood spots",
                                hovertemplate="%{text}<extra></extra>",
                                text=spots_df["name"]))
    if show_wards and wards_path and Path(wards_path).exists():
        gj = load_geojson(wards_path)
        for feature in gj.get("features", []):
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})
            if geom.get("type") == "Polygon":
                for ring in geom.get("coordinates", []):
                    coords = np.asarray(ring)
                    fig.add_trace(go.Scatter(x=coords[:,0], y=coords[:,1], mode="lines",
                        line=dict(color="rgba(220,235,255,0.55)", width=1),
                        name=str(props.get("ward_id","Ward")), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title="Synthetic city flood-risk layer", height=600,
        margin=dict(l=5,r=5,t=45,b=5), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#dceaff",
        xaxis_title="Longitude", yaxis_title="Latitude")
    return fig

def get_event_dates():
    if EVENTS.exists():
        df = load_csv(EVENTS)
        return df["date"].astype(str).tolist()
    return []


def load_saved_forecast(date_string):
    """Optional saved ConvLSTM output: outputs/forecasts/forecast_YYYY-MM-DD.npz.
    Expected array key: rain_forecast with shape (6, 20, 20).
    """
    f = FORECAST_DIR / f"forecast_{date_string}.npz"
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as z:
            key = "rain_forecast" if "rain_forecast" in z.files else z.files[0]
            arr = np.asarray(z[key], dtype=float)
        if arr.shape != (6, len(lat), len(lon)):
            return None
        return arr
    except Exception:
        return None

def load_saved_metrics():
    """Optional computed metrics file; never create scores if the file is absent."""
    f = METRICS_DIR / "metrics.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
        required = {"model", "lead_minutes", "POD", "FAR", "CSI", "FSS"}
        if not required.issubset(df.columns):
            return None
        return df
    except Exception:
        return None

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.markdown("## 🌧️ FloodGuard AI")
    st.caption("AI-Based Flood Prediction System")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Rainfall Forecast", "Flood Risk Map", "Model Performance", "About FloodGuard AI"],
        index=0
    )
    city = st.selectbox("Synthetic city", ["Mumbai-like synthetic city", "Synthetic City B"])
    st.markdown("---")
    st.markdown("**Data status**")
    st.success("Synthetic datasets loaded")
    st.caption("No live weather feed connected")
    st.markdown("---")
    st.caption("Prototype • Simulated data only")

rain_data, static_data, wards_path = load_city(city)
rain_all = rain_data["rain_mm_per_hr"]
times = pd.to_datetime(rain_data["time"].astype(str))
lat = rain_data["lat"]
lon = rain_data["lon"]

# Choose event date from supplied replay events for city A; city B uses available 2024 dates
event_dates = get_event_dates() if city == "Mumbai-like synthetic city" else []
if not event_dates:
    event_dates = sorted(pd.Series(times[times.year == 2024].date.astype(str)).unique().tolist())
if not event_dates:
    event_dates = sorted(pd.Series(times.date.astype(str)).unique().tolist())

# Forecast step is a 6-frame timeline; use following synthetic frames as replay preview until model output exists.
# This is explicitly not labelled as ConvLSTM prediction.
def selected_event_frame(date_string):
    date_mask = times.date.astype(str) == date_string
    inds = np.where(date_mask)[0]
    if len(inds) == 0:
        inds = np.array([np.argmin(np.abs(times - pd.Timestamp(date_string)))])
    return int(inds[0])

# -------------------------- Page content --------------------------
st.markdown("<h1>FloodGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='small-muted'>Integrated rainfall early-warning and flood-risk prototype</p>", unsafe_allow_html=True)
st.markdown("<div class='warning-box'>⚠️ <b>SIMULATED DATA ONLY</b> — This is a synthetic event replay. It is not a real weather forecast. Flood-risk scores are relative risk, not flood depth.</div>", unsafe_allow_html=True)

if page == "Dashboard":
    st.subheader("City overview")
    c1,c2,c3,c4 = st.columns(4)
    date_sel = st.selectbox("Replay event", event_dates, key="dash_event")
    start_idx = selected_event_frame(date_sel)
    frame = rain_all[start_idx]
    mean_rain = float(np.nanmean(frame))
    max_rain = float(np.nanmax(frame))
    level, reason, color = warning_for_rain(max_rain)
    c1.metric("Maximum rainfall", f"{max_rain:.1f} mm/hr")
    c2.metric("Area mean rainfall", f"{mean_rain:.1f} mm/hr")
    c3.metric("Prototype warning", level)
    c4.metric("Forecast horizon", "3 hours")
    st.markdown(f"<div style='background:{color};padding:12px 16px;border-radius:12px;color:#081323;font-weight:700'>⚠ {level} Warning — {reason} (based on maximum grid-cell rain rate)</div>", unsafe_allow_html=True)
    left,right = st.columns([1.35,1])
    with left:
        st.plotly_chart(map_rain(frame, lat, lon, "Synthetic rainfall event map"), use_container_width=True)
    with right:
        score = risk_map(static_data, frame)
        rlat = static_data["lat"]
        rlon = static_data["lon"]
        spots = load_csv(SPOTS) if city == "Mumbai-like synthetic city" and SPOTS.exists() else None
        st.plotly_chart(map_risk(score, static_data, rlat, rlon, wards_path, spots), use_container_width=True)
    st.info("The dashboard is using a synthetic event replay. ConvLSTM forecast files will be connected when trained outputs are available.")

elif page == "Rainfall Forecast":
    st.subheader("Rainfall forecast & event replay")
    date_sel = st.selectbox("Select synthetic event", event_dates, key="rain_event")
    base_idx = selected_event_frame(date_sel)
    step = st.slider("Forecast timeline", min_value=1, max_value=6, value=1, format="Step %d")
    saved_forecast = load_saved_forecast(date_sel)
    if saved_forecast is not None:
        selected = saved_forecast[step - 1]
        idx = min(base_idx + step, len(rain_all)-1)
        forecast_source = "Saved ConvLSTM forecast output"
        model_label = "ConvLSTM"
    else:
        offset = step  # synthetic replay: six following 30-minute frames
        idx = min(base_idx + offset, len(rain_all)-1)
        selected = rain_all[idx]
        forecast_source = "Synthetic rainfall event replay"
        model_label = "Synthetic replay (not ConvLSTM)"
    lead_minutes = step * 30
    max_rain = float(np.nanmax(selected))
    mean_rain = float(np.nanmean(selected))
    level, reason, color = warning_for_rain(max_rain)
    m1,m2,m3 = st.columns(3)
    m1.metric("Selected lead time", f"+{lead_minutes} min")
    m2.metric("Maximum grid-cell rain", f"{max_rain:.1f} mm/hr")
    m3.metric("Warning level", level)
    st.markdown(f"<div style='background:{color};padding:12px 16px;border-radius:12px;color:#081323;font-weight:700'>{level}: {reason}</div>", unsafe_allow_html=True)
    st.plotly_chart(map_rain(selected, lat, lon, f"Synthetic rainfall sequence • +{lead_minutes} minutes"), use_container_width=True)
    st.markdown(f"<div class='info-box'><b>Data source:</b> {forecast_source}<br><b>Model label:</b> {model_label}</div>", unsafe_allow_html=True)
    st.caption(f"Dataset timestamp reference: {times[idx]}")

elif page == "Flood Risk Map":
    st.subheader("Flood-risk assessment")
    date_sel = st.selectbox("Select rainfall event", event_dates, key="risk_event")
    base_idx = selected_event_frame(date_sel)
    step = st.slider("Rainfall frame", 0, 6, 0, key="risk_step", format="Step %d")
    idx = min(base_idx + step, len(rain_all)-1)
    rain_frame = rain_all[idx]
    score = risk_map(static_data, rain_frame)
    rlat = static_data["lat"]
    rlon = static_data["lon"]
    spots = load_csv(SPOTS) if city == "Mumbai-like synthetic city" and SPOTS.exists() else None
    l1, l2, l3, l4 = st.columns(4)
    show_risk = l1.checkbox("Risk layer", True)
    show_river = l2.checkbox("River", True)
    show_spots = l3.checkbox("Flood spots", True)
    show_wards = l4.checkbox("Ward boundaries", True)
    st.plotly_chart(map_risk(score, static_data, rlat, rlon, wards_path, spots,
                             show_risk, show_river, show_spots, show_wards), use_container_width=True)
    valid = score[np.isfinite(score)]
    counts = {
        "Low": int(np.sum(valid < 0.25)),
        "Moderate": int(np.sum((valid >= 0.25) & (valid < 0.5))),
        "High": int(np.sum((valid >= 0.5) & (valid < 0.75))),
        "Severe": int(np.sum(valid >= 0.75)),
    }
    cols = st.columns(4)
    for col, name in zip(cols, ["Low","Moderate","High","Severe"]):
        col.metric(name, f"{counts[name]:,} cells")
    st.caption("Prototype risk index combines rainfall, relative elevation, built-up land cover, and proximity to river mask. It is a visual relative-risk layer, not a validated hydrological model or flood-depth estimate.")
    if city == "Mumbai-like synthetic city" and WARDS.exists():
        ward_df_path = DATA / "08_city_boundary_and_wards" / "ward_terrain_summary_SYNTHETIC.csv"
        if ward_df_path.exists():
            st.markdown("#### Ward terrain summary")
            st.dataframe(load_csv(ward_df_path), use_container_width=True, hide_index=True)

elif page == "Model Performance":
    st.subheader("Model performance")
    st.markdown("Compare ConvLSTM against the Persistence and Optical Flow baselines using the test season.")
    metrics = load_saved_metrics()
    if metrics is None:
        st.markdown("""
        <div class='info-box'>
        <b>Evaluation results not available yet.</b><br>
        The supplied dataset package does not include trained ConvLSTM outputs or computed test metrics.
        After evaluation, save the computed results as <code>outputs/metrics/metrics.csv</code>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Required evaluation")
        st.write("Test split: 2024 • Lead times: +30 minutes to +3 hours • Rain thresholds: multiple thresholds as specified in the SRS.")
    else:
        metric = st.selectbox("Metric", ["POD", "FAR", "CSI", "FSS"])
        threshold_col = "threshold_mmhr" if "threshold_mmhr" in metrics.columns else None
        lead_col = "lead_minutes"
        chart_df = metrics.copy()
        if threshold_col:
            thresholds = sorted(chart_df[threshold_col].dropna().unique().tolist())
            selected_threshold = st.selectbox("Rain threshold (mm/hr)", thresholds)
            chart_df = chart_df[chart_df[threshold_col] == selected_threshold]
        fig = px.line(chart_df, x=lead_col, y=metric, color="model", markers=True,
                      title=f"{metric} by forecast lead time")
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#dceaff", xaxis_title="Lead time (minutes)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.caption("Metrics are loaded from your computed file; confirm they were calculated on the 2024 test split.")

else:
    st.subheader("About FloodGuard AI")
    st.markdown("""
    **FloodGuard AI** is a student prototype for short-range rainfall nowcasting and relative flood-risk assessment.

    **Project goal:** forecast rainfall maps for the next 3 hours, issue colour-coded warning levels, and show areas with higher relative waterlogging risk.

    **Main model:** ConvLSTM  
    **Baselines:** Persistence and Optical Flow  
    **Dashboard:** Streamlit  
    **Data:** Synthetic computer-generated data
    """)
    st.markdown("### System architecture")
    st.code("""Synthetic datasets
       ↓
Data validation & preprocessing
       ↓
6 historical rainfall frames
       ↓
ConvLSTM + Persistence + Optical Flow
       ↓
6 rainfall outputs (+30 min to +3 h)
       ↓
Warning levels + relative flood-risk layer
       ↓
Interactive Streamlit dashboard""", language="text")
    st.markdown("### Dataset package")
    st.write("The supplied package includes rainfall, terrain and land cover, gauges and weather, NWP, wards, flood spots and labels, radar-like frames, and a second synthetic city.")
    st.markdown("### Limitations")
    st.warning("All datasets are synthetic. The prototype is not a live forecast system. The risk layer is not flood depth. Real satellite/radar/gauge integration and real-world validation are future work.")
    st.markdown("### Warning thresholds (project-defined)")
    st.dataframe(pd.DataFrame([
        ["Green", "< 2.5 mm/hr", "No significant rain"],
        ["Yellow", "2.5–<10 mm/hr", "Moderate to heavy rain, be aware"],
        ["Orange", "10–<25 mm/hr", "Heavy rain, be prepared"],
        ["Red", "≥25 mm/hr", "Very heavy rain, take action"],
    ], columns=["Level","Rain rate","Meaning"]), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("FloodGuard AI • Hack Kalp prototype • All data in this build are synthetic")
