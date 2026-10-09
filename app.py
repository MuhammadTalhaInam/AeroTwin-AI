
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================================
# PATHS AND PAGE CONFIGURATION
# ============================================================

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

# ============================================================
# CSS
# ============================================================

PREMIUM_CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 0%,
        rgba(8,145,178,.09), transparent 28%),
        #0b0f19;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}
[data-testid="stSidebar"] {
    background: #0d1320;
    border-right: 1px solid rgba(148,163,184,.17);
}
.aero-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 18px;
    padding: 18px 22px;
    margin-bottom: 20px;
    background: linear-gradient(120deg,#111827,#0f1f30);
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 16px;
}
.aero-brand {
    display: flex;
    align-items: center;
    gap: 13px;
}
.aero-brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: rgba(34,211,238,.12);
    border: 1px solid rgba(34,211,238,.3);
    color: #22d3ee;
    font-size: 25px;
}
.aero-brand-name {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 850;
    letter-spacing: 2px;
}
.aero-brand-sub {
    color: #94a3b8;
    font-size: 10px;
    letter-spacing: 1.5px;
    margin-top: 3px;
}
.aero-top-status {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 12px;
    border-radius: 9px;
    background: rgba(34,211,238,.07);
    border: 1px solid rgba(34,211,238,.19);
    color: #a5f3fc;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1px;
}
.aero-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22d3ee;
    display: inline-block;
}
.aero-top-right {
    display: flex;
    flex-direction: column;
    gap: 5px;
    text-align: right;
}
.aero-top-label {
    color: #e2e8f0;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.1px;
}
.aero-version {
    color: #94a3b8;
    font-size: 10px;
}
.aero-eyebrow, .aero-section-label {
    color: #22d3ee;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin: 8px 0 12px;
}
.aero-subtitle {
    color: #a8b7ca;
    font-size: 15px;
    margin-top: -8px;
    margin-bottom: 18px;
}
.aero-metric {
    min-height: 142px;
    padding: 18px;
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 14px;
    background: linear-gradient(145deg,#162030,#0e1623);
    position: relative;
    overflow: hidden;
}
.aero-metric::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: var(--metric-accent,#22d3ee);
}
.aero-metric-label {
    color: #9caec2;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: .9px;
    text-transform: uppercase;
}
.aero-metric-value {
    color: #f8fafc;
    font-size: clamp(23px,2.4vw,31px);
    font-weight: 800;
    margin-top: 15px;
    line-height: 1.15;
    overflow-wrap: anywhere;
}
.aero-metric-note {
    color: #8fa3b8;
    font-size: 11px;
    margin-top: 10px;
}
.aero-panel {
    padding: 18px;
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 14px;
    background: rgba(17,24,39,.7);
}
[data-testid="stMetric"] {
    background: rgba(17,24,39,.7);
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 12px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: #a7b7ca; }
[data-testid="stMetricValue"] { color: #f1f5f9; }
.stButton > button {
    border-radius: 10px;
    font-weight: 750;
    min-height: 42px;
}
div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(148,163,184,.1);
    border-radius: 12px;
    padding: 5px;
    background: rgba(17,24,39,.32);
}
hr { border-color: rgba(148,163,184,.15); }
@media(max-width:700px) {
    .aero-topbar { padding: 14px; }
    .aero-top-right { text-align: left; }
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# Load the project's existing stylesheet if available.
css_path = BASE_DIR / "assets" / "aerotwin.css"
if css_path.exists():
    existing_css = css_path.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{existing_css}</style>",
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER — USE st.html TO RENDER HTML DIRECTLY
# ============================================================

HEADER_HTML = """
<div class="aero-topbar"><div class="aero-brand"><div class="aero-brand-icon">✈</div><div><div class="aero-brand-name">AEROTWIN</div><div class="aero-brand-sub">ENGINE INTELLIGENCE PLATFORM</div></div></div><div class="aero-top-status"><span class="aero-status-dot"></span>HISTORICAL DATA MODE</div><div class="aero-top-right"><span class="aero-top-label">AI PROGNOSTICS</span><span class="aero-version">RESEARCH PROTOTYPE · v1.1</span></div></div><div class="aero-eyebrow">AEROSPACE INTELLIGENCE SYSTEM</div>
"""

# st.html renders the supplied HTML instead of displaying the tags as text.
st.html(HEADER_HTML)

st.title("Mission Control")

st.markdown(
    "Turbofan engine monitoring · Sensor intelligence · "
    "Remaining useful life estimation"
)

st.warning(
    "RESEARCH PROTOTYPE — Uses historical NASA C-MAPSS FD001 data. "
    "No live aircraft telemetry is connected. Predictions and sensor "
    "interpretations are not certified for maintenance or flight-safety decisions."
)

# ============================================================
# REUSABLE DISPLAY HELPERS
# ============================================================

def section_label(label):
    st.markdown(
        f'<div class="aero-section-label">{label}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, accent="#22d3ee"):
    # Escape values before inserting them into HTML.
    import html

    label = html.escape(str(label))
    value = html.escape(str(value))
    note = html.escape(str(note))

    st.markdown(
        f'<div class="aero-metric" style="--metric-accent:{accent};">'
        f'<div class="aero-metric-label">{label}</div>'
        f'<div class="aero-metric-value">{value}</div>'
        f'<div class="aero-metric-note">{note}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=12, r=12, t=58, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
    )
    return fig


# ============================================================
# LOAD THE EXISTING MODEL AND DATA
# ============================================================

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
    with st.spinner("Initializing AeroTwin..."):
        model = get_model()
        feature_names = get_features()
        data = get_data()
except Exception as exc:
    st.error(f"Could not initialize AeroTwin: {exc}")
    st.stop()

if len(feature_names) != 44:
    st.error(
        f"Expected 44 model features, but found {len(feature_names)}. "
        "Check the existing model feature file."
    )
    st.stop()

sensor_columns = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_11", "sensor_12", "sensor_14", "sensor_17",
    "sensor_20", "sensor_21",
]

required_columns = {"engine_id", "cycle", *feature_names, *sensor_columns}
missing_columns = sorted(required_columns - set(data.columns))

if missing_columns:
    st.error(f"Dataset is missing required columns: {missing_columns}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

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

sensor_choice = st.sidebar.selectbox(
    "Sensor to inspect",
    sensor_columns,
)

st.sidebar.divider()
st.sidebar.caption("DATA SOURCE")
st.sidebar.write("NASA C-MAPSS · FD001")
st.sidebar.caption("Recorded run-to-failure research data")
st.sidebar.caption("Model input: 44 engineered features")
st.sidebar.divider()
st.sidebar.caption("AEROTWIN AI · RESEARCH PROTOTYPE")
st.sidebar.caption("No live aircraft connection")

# ============================================================
# SELECTED ENGINE HISTORY
# ============================================================

history = (
    data[data["engine_id"] == selected_engine]
    .sort_values("cycle")
    .copy()
)

if history.empty:
    st.error("No recorded observations were found for this engine.")
    st.stop()

latest_cycle = int(history["cycle"].max())

section_label("ENGINE WORKSPACE")
st.subheader(f"Engine {int(selected_engine):03d} — Analysis Console")
st.caption(
    f"{len(history):,} recorded observations · "
    f"Latest recorded cycle: {latest_cycle}"
)

left_action, right_status = st.columns([1.4, 1])

with left_action:
    analyze_clicked = st.button(
        "▶ ANALYZE ENGINE",
        type="primary",
        use_container_width=True,
    )

with right_status:
    st.markdown(
        '<div class="aero-panel">'
        '<div class="aero-metric-label">DATA MODE</div>'
        '<div style="font-size:17px;font-weight:750;color:#67e8f9;'
        'margin-top:8px;">Historical Analysis</div>'
        '<div class="aero-metric-note">'
        'Recorded dataset · Not live telemetry</div></div>',
        unsafe_allow_html=True,
    )

# Clear previous results if the selected engine changes.
if st.session_state.get("aerotwin_engine") != int(selected_engine):
    st.session_state.pop("aerotwin_result", None)
    st.session_state.pop("aerotwin_report", None)
    st.session_state.pop("aerotwin_engine", None)

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

has_result = (
    "aerotwin_result" in st.session_state
    and "aerotwin_report" in st.session_state
)

# ============================================================
# DATASET OVERVIEW
# ============================================================

section_label("DATASET OVERVIEW")
overview_cols = st.columns(3, gap="medium")

with overview_cols[0]:
    metric_card(
        "Recorded Engines",
        f"{data['engine_id'].nunique():,}",
        "Engines in FD001 dataset",
        "#22d3ee",
    )

with overview_cols[1]:
    metric_card(
        "Dataset Observations",
        f"{len(data):,}",
        "Historical sensor records",
        "#818cf8",
    )

with overview_cols[2]:
    metric_card(
        "Model Features",
        f"{len(feature_names)}/44",
        "Expected engineered inputs",
        "#34d399",
    )

# ============================================================
# BEFORE ANALYSIS: SHOW RELEVANT PAGE CONTENT
# ============================================================

if not has_result:
    st.info(
        "Choose an engine and press ANALYZE ENGINE to run the trained "
        "model and inspect the recorded sensor history."
    )

    if page == "Sensor Diagnostics":
        section_label("SENSOR DIAGNOSTICS")
        fig = px.line(
            history,
            x="cycle",
            y=sensor_choice,
            title=f"{sensor_choice} — Engine {int(selected_engine):03d}",
            labels={
                "cycle": "Recorded operating cycle",
                sensor_choice: "Sensor value",
            },
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)
        st.caption("Historical readings only; this is not a live sensor feed.")

    elif page == "Prognostics":
        section_label("PROGNOSTICS")
        st.subheader("Remaining Useful Life")
        st.write(
            "Run engine analysis to display the trained model's estimate. "
            "The estimate is not verified actual remaining life."
        )

    elif page == "Data Laboratory":
        section_label("DATA LABORATORY")
        st.subheader("Dataset Explorer")
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)

        st.download_button(
            "Download Dataset Sample",
            data=data.head(100).to_csv(index=False).encode("utf-8"),
            file_name="aerotwin_dataset_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.stop()

# ============================================================
# RESULTS
# ============================================================

result = st.session_state["aerotwin_result"]
report = st.session_state["aerotwin_report"]

prediction = float(report["predicted_rul"])
reported_latest_cycle = int(report["latest_cycle"])

section_label("MODEL OUTPUTS")
output_cols = st.columns(3, gap="medium")

with output_cols[0]:
    metric_card(
        "Predicted RUL",
        f"{prediction:.2f} cycles",
        "Model estimate · Not verified actual life",
        "#22d3ee",
    )

with output_cols[1]:
    metric_card(
        "Latest Recorded Cycle",
        str(reported_latest_cycle),
        "Last observed cycle for this engine",
        "#818cf8",
    )

with output_cols[2]:
    metric_card(
        "Model Input Features",
        f"{len(feature_names)}/44",
        "Engineered model input features",
        "#34d399",
    )

st.caption(
    "RUL is a model estimate. It is not ground truth, a confidence score, "
    "or a guarantee of engine service life."
)

# ============================================================
# PROGNOSTICS SUMMARY
# ============================================================

section_label("PROGNOSTICS SUMMARY")
st.subheader("Model Interpretation")
st.write(report["rul_category"])

if prediction <= 15:
    st.warning(
        "The model returned a low RUL estimate. This is a research indicator, "
        "not an operational maintenance instruction."
    )
elif prediction <= 70:
    st.info(
        "The model returned a moderate RUL estimate. Validate model performance "
        "on appropriate held-out data before drawing engineering conclusions."
    )
else:
    st.success(
        "The model returned a higher RUL estimate. This does not certify "
        "the engine as healthy or safe."
    )

# ============================================================
# SENSOR DIAGNOSTICS
# ============================================================

if page in ("Mission Control", "Sensor Diagnostics"):
    st.divider()
    section_label("SENSOR INTELLIGENCE")
    st.subheader("Sensor Trend Explorer")

    fig_sensor = px.line(
        history,
        x="cycle",
        y=sensor_choice,
        title=f"{sensor_choice} Across Recorded Cycles",
        labels={
            "cycle": "Recorded operating cycle",
            sensor_choice: "Recorded sensor value",
        },
    )
    st.plotly_chart(style_chart(fig_sensor), use_container_width=True)

    st.caption(
        "Historical readings only. A change in sensor values does not, "
        "by itself, establish a component fault."
    )

    st.subheader("Recent Sensor Analysis")
    st.dataframe(
        report["sensor_report"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Largest Relative Changes vs Recent Readings")
    st.dataframe(
        report["largest_recent_changes"],
        use_container_width=True,
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
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# HISTORICAL RUL MODEL OUTPUTS
# ============================================================

if page in ("Mission Control", "Prognostics"):
    st.divider()
    section_label("RUL HISTORY")
    st.subheader("Model Prediction Across Recorded History")

    # These are predictions for recorded observations, not actual RUL labels.
    history_predictions = model.predict(history[feature_names])
    history_predictions = pd.Series(history_predictions).clip(lower=0)

    trend = pd.DataFrame({
        "Cycle": history["cycle"].to_numpy(),
        "Predicted RUL (cycles)": history_predictions.to_numpy(),
    })

    fig_rul = px.line(
        trend,
        x="Cycle",
        y="Predicted RUL (cycles)",
        title="Model Estimates at Each Recorded Cycle",
    )
    st.plotly_chart(style_chart(fig_rul), use_container_width=True)

    st.caption(
        "These are model outputs for recorded observations, not ground-truth "
        "remaining life and not a validated forecast of future behaviour."
    )

# ============================================================
# DATA LABORATORY
# ============================================================

if page == "Data Laboratory":
    st.divider()
    section_label("DATA LABORATORY")
    st.subheader("Selected Engine Dataset")
    st.caption(
        f"Engine {int(selected_engine):03d} · "
        f"{len(history):,} recorded observations"
    )

    st.dataframe(history, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Selected Engine History",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name=f"aerotwin_engine_{int(selected_engine)}_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# ENGINEERING REPORT EXPORT
# ============================================================

st.divider()
section_label("ENGINEERING REPORTS")
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
    use_container_width=True,
)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;font-size:11px;
                line-height:1.8;padding:12px 0;">
        <strong style="color:#cbd5e1;letter-spacing:1px;">AEROTWIN AI</strong>
        <br>
        AI-Powered Engine Health Research · NASA C-MAPSS FD001 · v1.1
        <br>
        Historical research prototype — not approved for flight-safety
        or operational maintenance decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
