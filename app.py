
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from engine.pipeline import (
    load_model,
    load_features,
    load_dataset,
    predict_engine,
)
from engine.analysis import analyze_engine

st.set_page_config(
    page_title="AeroTwin AI | Mission Control",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- THEME --------------------
css_path = BASE_DIR / "assets" / "aerotwin.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# -------------------- HEADER --------------------
st.markdown("""
<div class="aero-topbar">
  <div class="aero-brand">
    <div class="aero-brand-icon">✈</div>
    <div>
      <div class="aero-brand-name">AEROTWIN</div>
      <div class="aero-brand-sub">ENGINE INTELLIGENCE PLATFORM</div>
    </div>
  </div>
  <div class="aero-top-status">
    <span class="aero-status-dot"></span>
    HISTORICAL DATA MODE
  </div>
  <div class="aero-top-right">
    <span class="aero-top-label">AI PROGNOSTICS</span>
    <span class="aero-version">RESEARCH PROTOTYPE · v1.0</span>
  </div>
</div>
<div class="aero-eyebrow">AEROSPACE INTELLIGENCE SYSTEM</div>
""", unsafe_allow_html=True)

st.title("Mission Control")
st.markdown(
    '<div class="aero-subtitle">'
    'Turbofan engine monitoring · Sensor intelligence · '
    'Remaining useful life estimation'
    '</div>',
    unsafe_allow_html=True,
)

st.warning(
    "RESEARCH PROTOTYPE — Uses historical NASA C-MAPSS FD001 data. "
    "No live aircraft telemetry is connected. Predictions and sensor "
    "interpretations are not certified for maintenance or flight-safety decisions."
)

# -------------------- DATA LOADING --------------------
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_features():
    return load_features()

@st.cache_data
def get_data():
    return load_dataset()

try:
    with st.spinner("Initializing AeroTwin intelligence systems..."):
        model = get_model()
        feature_names = get_features()
        data = get_data()
except Exception as exc:
    st.error(f"Could not initialize AeroTwin: {exc}")
    st.stop()

if len(feature_names) != 44:
    st.error(f"Expected 44 model features; found {len(feature_names)}.")
    st.stop()

required_columns = {
    "engine_id", "cycle",
    *feature_names,
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_11", "sensor_12", "sensor_14",
    "sensor_17", "sensor_20", "sensor_21",
}
missing_columns = sorted(required_columns - set(data.columns))
if missing_columns:
    st.error(f"Dataset is missing required columns: {missing_columns}")
    st.stop()

# -------------------- SIDEBAR NAVIGATION --------------------
st.sidebar.markdown("## ✈️ AEROTWIN")
st.sidebar.caption("ENGINE INTELLIGENCE PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Mission Control",
        "Sensor Diagnostics",
        "Prognostics",
        "Data Laboratory",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("### Engine Selection")

engine_ids = sorted(data["engine_id"].unique().tolist())
default_index = engine_ids.index(97) if 97 in engine_ids else 0

selected_engine = st.sidebar.selectbox(
    "NASA engine ID",
    engine_ids,
    index=default_index,
)

sensor_columns = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_11", "sensor_12", "sensor_14",
    "sensor_17", "sensor_20", "sensor_21",
]

sensor_choice = st.sidebar.selectbox(
    "Sensor to inspect",
    sensor_columns,
)

st.sidebar.divider()
st.sidebar.caption("DATA SOURCE")
st.sidebar.write("NASA C-MAPSS · FD001")
st.sidebar.caption("Recorded run-to-failure research data")
st.sidebar.caption("Model input: 44 engineered features")

# -------------------- SELECTED ENGINE --------------------
history = (
    data[data["engine_id"] == selected_engine]
    .sort_values("cycle")
    .copy()
)

if history.empty:
    st.error("No recorded observations were found for this engine.")
    st.stop()

latest_cycle = int(history["cycle"].max())

# -------------------- ANALYSIS ACTION --------------------
st.markdown('<div class="aero-section-label">ENGINE WORKSPACE</div>',
            unsafe_allow_html=True)
st.subheader(f"Engine {int(selected_engine):03d} — Analysis Console")
st.caption(
    f"{len(history):,} recorded cycles · "
    f"Latest recorded cycle: {latest_cycle}"
)

analyze_clicked = st.button(
    "▶  ANALYZE ENGINE",
    type="primary",
    width="stretch",
)

if analyze_clicked:
    try:
        with st.spinner("Running model prediction and sensor analysis..."):
            result = predict_engine(
                model=model,
                data=data,
                feature_names=feature_names,
                engine_id=int(selected_engine),
            )
            report = analyze_engine(result, sensor_columns)

            st.session_state["aerotwin_result"] = result
            st.session_state["aerotwin_report"] = report
            st.session_state["aerotwin_engine"] = int(selected_engine)
    except Exception as exc:
        st.error(f"Engine analysis failed: {exc}")
        st.stop()

# Prevent a report for a previously selected engine from being displayed.
if st.session_state.get("aerotwin_engine") != int(selected_engine):
    st.session_state.pop("aerotwin_result", None)
    st.session_state.pop("aerotwin_report", None)
    st.session_state.pop("aerotwin_engine", None)

has_result = (
    "aerotwin_result" in st.session_state
    and "aerotwin_report" in st.session_state
)

# -------------------- OVERVIEW METRICS --------------------
st.markdown('<div class="aero-section-label">DATASET OVERVIEW</div>',
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Recorded Engines", f"{data['engine_id'].nunique():,}")
m2.metric("Dataset Observations", f"{len(data):,}")
m3.metric("Model Features", f"{len(feature_names)}/44")

if not has_result:
    st.info(
        "Choose an engine and press ANALYZE ENGINE to run the trained "
        "model and inspect the recorded sensor history."
    )

    if page == "Data Laboratory":
        st.subheader("Dataset Explorer")
        st.dataframe(
            data.head(100),
            width="stretch",
            hide_index=True,
        )
        st.caption("Showing the first 100 rows of the historical dataset.")

    elif page == "Sensor Diagnostics":
        st.subheader("Recorded Sensor Trend")
        fig = px.line(
            history,
            x="cycle",
            y=sensor_choice,
            title=f"{sensor_choice} — Engine {int(selected_engine):03d}",
            labels={
                "cycle": "Recorded operating cycle",
                sensor_choice: "Sensor value",
            },
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    elif page == "Prognostics":
        st.subheader("Remaining Useful Life")
        st.write(
            "Run engine analysis first to display the model's RUL estimate. "
            "An estimate is not a verified measurement of actual remaining life."
        )

    st.stop()

# -------------------- ANALYSIS RESULTS --------------------
result = st.session_state["aerotwin_result"]
report = st.session_state["aerotwin_report"]

prediction = float(report["predicted_rul"])
reported_latest_cycle = int(report["latest_cycle"])

st.divider()
st.markdown('<div class="aero-section-label">MODEL OUTPUTS</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL", f"{prediction:.2f} cycles")
c2.metric("Latest Recorded Cycle", str(reported_latest_cycle))
c3.metric("Model Input Features", f"{len(feature_names)}/44")

st.caption(
    "RUL is a model estimate. It is not ground truth, a confidence score, "
    "or a guarantee of engine service life."
)

# -------------------- RUL INTERPRETATION --------------------
st.subheader("Prognostics Summary")
st.write(report["rul_category"])

if prediction <= 15:
    st.warning(
        "The model returned a low RUL estimate. This is a research indicator, "
        "not an operational maintenance instruction."
    )
elif prediction <= 70:
    st.info(
        "The model returned a moderate RUL estimate. Validate performance "
        "against appropriate held-out test data before engineering conclusions."
    )
else:
    st.success(
        "The model returned a higher RUL estimate. This does not certify "
        "the engine as healthy or safe."
    )

# -------------------- SENSOR DIAGNOSTICS --------------------
if page in ("Mission Control", "Sensor Diagnostics"):
    st.divider()
    st.subheader("Sensor Trend Explorer")

    fig_sensor = px.line(
        history,
        x="cycle",
        y=sensor_choice,
        title=f"{sensor_choice} across recorded cycles",
        labels={
            "cycle": "Recorded operating cycle",
            sensor_choice: "Recorded sensor value",
        },
        template="plotly_dark",
    )
    fig_sensor.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig_sensor, width="stretch")

    st.subheader("Recent Sensor Analysis")
    st.dataframe(
        report["sensor_report"],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Largest Relative Changes vs Recent Readings")
    st.dataframe(
        report["largest_recent_changes"],
        width="stretch",
        hide_index=True,
    )
    st.caption(report["notice"])

    st.subheader("Latest Recorded Sensor Values")
    latest = result["latest"]
    available_sensors = [
        col for col in sensor_columns if col in latest.columns
    ]
    st.dataframe(
        latest[["cycle"] + available_sensors].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

# -------------------- PROGNOSTICS --------------------
if page in ("Mission Control", "Prognostics"):
    st.divider()
    st.subheader("Model Prediction Across Recorded History")

    # The plotted values are model estimates evaluated at recorded cycles.
    # They are not ground-truth RUL or independent future forecasts.
    history_features = history[feature_names]
    history_predictions = model.predict(history_features)
    history_predictions = pd.Series(history_predictions).clip(lower=0)

    trend = pd.DataFrame({
        "Cycle": history["cycle"].to_numpy(),
        "Predicted RUL (cycles)": history_predictions.to_numpy(),
    })

    fig_rul = px.line(
        trend,
        x="Cycle",
        y="Predicted RUL (cycles)",
        title="Model estimates at each recorded cycle",
        template="plotly_dark",
    )
    fig_rul.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig_rul, width="stretch")

    st.caption(
        "These are model outputs for recorded observations, not ground-truth "
        "remaining life and not a validated forecast of future engine behaviour."
    )

# -------------------- DATA LABORATORY --------------------
if page == "Data Laboratory":
    st.divider()
    st.subheader("Dataset Explorer")
    st.dataframe(
        history,
        width="stretch",
        hide_index=True,
    )

# -------------------- CSV EXPORT --------------------
st.divider()
st.subheader("Export Analysis")

export_df = report["sensor_report"].copy()
export_df.insert(0, "Engine ID", int(selected_engine))
export_df.insert(1, "Latest Recorded Cycle", reported_latest_cycle)
export_df["Model Predicted RUL (cycles)"] = round(prediction, 3)

st.download_button(
    "Download Sensor Analysis CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"aerotwin_engine_{int(selected_engine)}_analysis.csv",
    mime="text/csv",
    width="stretch",
)

st.caption(
    "AEROTWIN AI · Historical NASA C-MAPSS FD001 · Research prototype. "
    "Not approved for flight-safety or maintenance decisions."
)
