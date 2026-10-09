
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
    page_title="AeroTwin AI | Engine Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
css_path = BASE_DIR / "assets" / "aerotwin.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text()}</style>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="aero-eyebrow">AEROSPACE INTELLIGENCE SYSTEM</div>',
    unsafe_allow_html=True,
)
st.title("✈️ AeroTwin AI")
st.markdown(
    '<div class="aero-subtitle">'
    'Turbofan Engine Health • Remaining Useful Life • Sensor Intelligence'
    '</div>',
    unsafe_allow_html=True,
)
st.write("")

st.warning(
    "RESEARCH PROTOTYPE — This application analyzes historical NASA "
    "C-MAPSS FD001 data. It does not receive live aircraft telemetry. "
    "Predictions are not certified for aircraft maintenance or flight-safety decisions."
)

# ---------- Load data ----------
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
    with st.spinner("Initializing engine intelligence systems..."):
        model = get_model()
        feature_names = get_features()
        data = get_data()
except Exception as exc:
    st.error(f"Could not initialize AeroTwin: {exc}")
    st.stop()

if len(feature_names) != 44:
    st.error(f"Expected 44 features; found {len(feature_names)}.")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.markdown("## ✈️ AEROTWIN")
st.sidebar.caption("ENGINE ANALYSIS CONSOLE")
st.sidebar.divider()

engine_ids = sorted(data["engine_id"].unique().tolist())
selected_engine = st.sidebar.selectbox(
    "Select NASA engine",
    engine_ids,
    index=engine_ids.index(97) if 97 in engine_ids else 0,
)

sensor_columns = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_11", "sensor_12", "sensor_14",
    "sensor_17", "sensor_20", "sensor_21",
]
sensor_choice = st.sidebar.selectbox(
    "Sensor history",
    sensor_columns,
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("DATA SOURCE")
st.sidebar.write("NASA C-MAPSS · FD001")
st.sidebar.caption("Historical run-to-failure research dataset")

# ---------- Main analysis ----------
history = data[data["engine_id"] == selected_engine].copy()
if history.empty:
    st.error("No observations were found for this engine.")
    st.stop()

st.subheader(f"Engine {selected_engine:03d} — Analysis Console")
st.caption(
    f"{len(history)} recorded cycles available · "
    f"latest recorded cycle: {int(history['cycle'].max())}"
)

analyze_clicked = st.button(
    "▶  ANALYZE ENGINE",
    type="primary",
    width="stretch",
)

if analyze_clicked:
    with st.spinner("Running model prediction and sensor analysis..."):
        try:
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
            st.error(f"Analysis failed: {exc}")
            st.stop()

# Do not accidentally display a previous engine's report after selection changes.
if st.session_state.get("aerotwin_engine") != int(selected_engine):
    st.session_state.pop("aerotwin_result", None)
    st.session_state.pop("aerotwin_report", None)

if "aerotwin_result" not in st.session_state:
    st.info(
        "Select an engine in the sidebar, then press ANALYZE ENGINE "
        "to run the trained model and inspect its recorded sensor history."
    )
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Recorded Engines", f"{data['engine_id'].nunique()}")
    c2.metric("Dataset Observations", f"{len(data):,}")
    c3.metric("Model Features", f"{len(feature_names)}/44")
    st.stop()

result = st.session_state["aerotwin_result"]
report = st.session_state["aerotwin_report"]
prediction = report["predicted_rul"]
latest_cycle = report["latest_cycle"]

st.divider()

# ---------- Key metrics ----------
c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL", f"{prediction:.2f} cycles")
c2.metric("Latest Recorded Cycle", str(latest_cycle))
c3.metric("Model Input Features", f"{len(feature_names)}/44")

st.caption(
    "RUL means Remaining Useful Life. The number shown is a model estimate, "
    "not a guaranteed service-life measurement."
)

st.subheader("Predicted RUL Interpretation")
st.write(report["rul_category"])
if prediction <= 15:
    st.warning(
        "The model returned a low RUL estimate. This is a research-model "
        "indicator only, not an operational maintenance instruction."
    )
elif prediction <= 70:
    st.info(
        "The model returned a moderate RUL estimate. Validate it against "
        "appropriate test data before drawing engineering conclusions."
    )
else:
    st.success(
        "The model returned a higher RUL estimate. This does not certify "
        "the engine as healthy or safe."
    )

# ---------- Sensor trend ----------
st.divider()
st.subheader("Sensor Trend Explorer")

fig_sensor = px.line(
    history,
    x="cycle",
    y=sensor_choice,
    title=f"{sensor_choice} across recorded cycles",
    labels={
        "cycle": "Operating cycle",
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

# ---------- Prediction trend ----------
st.subheader("Model Prediction Across Recorded History")
history_features = history[feature_names]
history_predictions = model.predict(history_features).clip(min=0)

trend = pd.DataFrame({
    "Cycle": history["cycle"].to_numpy(),
    "Predicted RUL (cycles)": history_predictions,
})
fig_rul = px.line(
    trend,
    x="Cycle",
    y="Predicted RUL (cycles)",
    title="Predicted RUL at each observed cycle",
    template="plotly_dark",
)
fig_rul.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=55, b=10),
)
st.plotly_chart(fig_rul, width="stretch")

st.caption(
    "This curve shows model estimates at recorded cycles. It is not ground-truth "
    "remaining life and should not be interpreted as a validated forecast."
)

# ---------- Sensor table ----------
st.divider()
st.subheader("Recent Sensor Analysis")
st.dataframe(
    report["sensor_report"],
    width="stretch",
    hide_index=True,
)

st.markdown("#### Largest relative changes vs recent readings")
st.dataframe(
    report["largest_recent_changes"],
    width="stretch",
    hide_index=True,
)

st.caption(report["notice"])

# ---------- Latest readings ----------
st.subheader("Latest Recorded Sensor Values")
latest = result["latest"]
st.dataframe(
    latest[["cycle"] + sensor_columns].reset_index(drop=True),
    width="stretch",
    hide_index=True,
)

# ---------- Export ----------
st.divider()
st.subheader("Export Analysis")
export_df = report["sensor_report"].copy()
export_df.insert(0, "Engine ID", int(selected_engine))
export_df.insert(1, "Latest Cycle", latest_cycle)
export_df["Model Predicted RUL (cycles)"] = round(prediction, 3)

st.download_button(
    "Download Sensor Analysis CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"aerotwin_engine_{selected_engine}_analysis.csv",
    mime="text/csv",
    width="stretch",
)

st.caption(
    "Research prototype · NASA C-MAPSS FD001 · "
    "Not approved for flight-safety or maintenance decisions."
)
