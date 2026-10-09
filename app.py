
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroTwin AI | Mission Control",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM DASHBOARD THEME
# ============================================================

PREMIUM_CSS = """
<style>
:root {
    --aero-bg: #0b0f19;
    --aero-card: #111827;
    --aero-border: rgba(148, 163, 184, 0.17);
    --aero-cyan: #22d3ee;
    --aero-text: #f1f5f9;
    --aero-muted: #94a3b8;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(8, 145, 178, 0.09),
            transparent 28%
        ),
        var(--aero-bg);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

[data-testid="stSidebar"] {
    background: #0d1320;
    border-right: 1px solid var(--aero-border);
}

[data-testid="stSidebar"] h2 {
    letter-spacing: 1.5px;
}

.aero-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    flex-wrap: wrap;
    padding: 18px 22px;
    margin-bottom: 20px;
    background: linear-gradient(
        120deg,
        rgba(17, 24, 39, 0.98),
        rgba(15, 31, 48, 0.94)
    );
    border: 1px solid var(--aero-border);
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
    background: rgba(34, 211, 238, 0.12);
    border: 1px solid rgba(34, 211, 238, 0.3);
    color: var(--aero-cyan);
    font-size: 25px;
}

.aero-brand-name {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 850;
    letter-spacing: 2px;
}

.aero-brand-sub {
    color: var(--aero-muted);
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
    background: rgba(34, 211, 238, 0.07);
    border: 1px solid rgba(34, 211, 238, 0.19);
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
    color: var(--aero-muted);
    font-size: 10px;
}

.aero-eyebrow,
.aero-section-label {
    color: var(--aero-cyan);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin: 8px 0 12px 0;
}

.aero-subtitle {
    color: #a8b7ca;
    font-size: 15px;
    margin-top: -8px;
    margin-bottom: 18px;
}

.aero-metric {
    height: 142px;
    padding: 18px;
    border: 1px solid var(--aero-border);
    border-radius: 14px;
    background: linear-gradient(
        145deg,
        rgba(22, 32, 48, 0.98),
        rgba(14, 22, 35, 0.98)
    );
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
    background: var(--metric-accent, #22d3ee);
}

.aero-metric-label {
    color: #9caec2;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 0.9px;
    text-transform: uppercase;
}

.aero-metric-value {
    color: #f8fafc;
    font-size: clamp(23px, 2.4vw, 31px);
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
    border: 1px solid var(--aero-border);
    border-radius: 14px;
    background: rgba(17, 24, 39, 0.7);
}

[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid var(--aero-border);
    border-radius: 12px;
    padding: 14px 16px;
}

[data-testid="stMetricLabel"] {
    color: #a7b7ca;
}

[data-testid="stMetricValue"] {
    color: #f1f5f9;
}

.stButton > button[kind="primary"] {
    min-height: 48px;
    border-radius: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    padding: 5px;
    background: rgba(17, 24, 39, 0.32);
}

hr {
    border-color: rgba(148, 163, 184, 0.15);
}

@media (max-width: 700px) {
    .aero-topbar {
        padding: 14px;
    }

    .aero-top-right {
        text-align: left;
    }

    .aero-metric {
        height: 130px;
    }
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# Keep any existing project-specific CSS.
css_path = BASE_DIR / "assets" / "aerotwin.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# REUSABLE UI HELPERS
# ============================================================

def metric_card(label, value, note, accent="#22d3ee"):
    """Render a premium dashboard metric card."""
    st.markdown(
        f"""
        <div class="aero-metric"
             style="--metric-accent: {accent};">
            <div class="aero-metric-label">{label}</div>
            <div class="aero-metric-value">{value}</div>
            <div class="aero-metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig):
    """Apply a consistent dark aerospace theme to Plotly charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=12, r=12, t=58, b=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.15)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.15)",
    )
    return fig


def section_label(text):
    st.markdown(
        f'<div class="aero-section-label">{text}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# GLOBAL HEADER
# ============================================================

st.markdown(
    """
    <div class="aero-topbar">
        <div class="aero-brand">
            <div class="aero-brand-icon">✈</div>
            <div>
                <div class="aero-brand-name">AEROTWIN</div>
                <div class="aero-brand-sub">
                    ENGINE INTELLIGENCE PLATFORM
                </div>
            </div>
        </div>

        <div class="aero-top-status">
            <span class="aero-status-dot"></span>
            HISTORICAL DATA MODE
        </div>

        <div class="aero-top-right">
            <span class="aero-top-label">AI PROGNOSTICS</span>
            <span class="aero-version">
                RESEARCH PROTOTYPE · v1.1
            </span>
        </div>
    </div>
    <div class="aero-eyebrow">
        AEROSPACE INTELLIGENCE SYSTEM
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Mission Control")

st.markdown(
    """
    <div class="aero-subtitle">
        Turbofan engine monitoring · Sensor intelligence ·
        Remaining useful life estimation
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "RESEARCH PROTOTYPE — Uses historical NASA C-MAPSS FD001 data. "
    "No live aircraft telemetry is connected. Predictions and sensor "
    "interpretations are not certified for maintenance or flight-safety decisions."
)


# ============================================================
# LOAD MODEL, FEATURES AND DATA
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
    with st.spinner("Initializing AeroTwin intelligence systems..."):
        model = get_model()
        feature_names = get_features()
        data = get_data()

except Exception as exc:
    st.error(f"Could not initialize AeroTwin: {exc}")
    st.stop()


if len(feature_names) != 44:
    st.error(
        f"Expected 44 model features; found {len(feature_names)}. "
        "Check the model feature file."
    )
    st.stop()


sensor_columns = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_14",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

required_columns = {
    "engine_id",
    "cycle",
    *feature_names,
    *sensor_columns,
}

missing_columns = sorted(required_columns - set(data.columns))

if missing_columns:
    st.error(f"Dataset is missing required columns: {missing_columns}")
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION AND ENGINE SELECTION
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


# ============================================================
# ENGINE WORKSPACE
# ============================================================

section_label("ENGINE WORKSPACE")

st.subheader(f"Engine {int(selected_engine):03d} — Analysis Console")

st.caption(
    f"{len(history):,} recorded observations · "
    f"Latest recorded cycle: {latest_cycle}"
)

left_action, right_status = st.columns([1.4, 1])

with left_action:
    analyze_clicked = st.button(
        "▶  ANALYZE ENGINE",
        type="primary",
        width="stretch",
    )

with right_status:
    st.markdown(
        """
        <div class="aero-panel">
            <div class="aero-metric-label">DATA MODE</div>
            <div style="font-size:17px;font-weight:750;
                        color:#67e8f9;margin-top:8px;">
                Historical Analysis
            </div>
            <div class="aero-metric-note">
                Recorded dataset · Not live telemetry
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN ANALYSIS AND STORE RESULTS
# ============================================================

if analyze_clicked:
    try:
        with st.spinner(
            "Running model prediction and sensor analysis..."
        ):
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


# Remove results belonging to a previously selected engine.
if st.session_state.get("aerotwin_engine") != int(selected_engine):
    st.session_state.pop("aerotwin_result", None)
    st.session_state.pop("aerotwin_report", None)
    st.session_state.pop("aerotwin_engine", None)


has_result = (
    "aerotwin_result" in st.session_state
    and "aerotwin_report" in st.session_state
)


# ============================================================
# PREMIUM DATASET OVERVIEW
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
# PRE-ANALYSIS STATE
# ============================================================

if not has_result:
    st.info(
        "Select an engine and press ANALYZE ENGINE to run the trained "
        "model and inspect its recorded sensor history."
    )

    if page == "Sensor Diagnostics":
        section_label("SENSOR DIAGNOSTICS")
        st.subheader("Recorded Sensor Trend")

        fig = px.line(
            history,
            x="cycle",
            y=sensor_choice,
            title=(
                f"{sensor_choice} — "
                f"Engine {int(selected_engine):03d}"
            ),
            labels={
                "cycle": "Recorded operating cycle",
                sensor_choice: "Sensor value",
            },
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch",
        )

        st.caption(
            "This chart shows recorded sensor values, not a live sensor feed."
        )

    elif page == "Prognostics":
        section_label("PROGNOSTICS")
        st.subheader("Remaining Useful Life")

        st.write(
            "Run engine analysis to display the trained model's RUL estimate. "
            "An estimate is not a verified measurement of actual remaining life."
        )

    elif page == "Data Laboratory":
        section_label("DATA LABORATORY")
        st.subheader("Dataset Explorer")

        st.dataframe(
            data.head(100),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Showing the first 100 rows of the historical dataset."
        )

    st.stop()


# ============================================================
# ANALYSIS RESULTS
# ============================================================

result = st.session_state["aerotwin_result"]
report = st.session_state["aerotwin_report"]

prediction = float(report["predicted_rul"])
reported_latest_cycle = int(report["latest_cycle"])


# ============================================================
# PREMIUM MODEL OUTPUT CARDS
# ============================================================

st.divider()
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
        f"{reported_latest_cycle}",
        "Last observed cycle for selected engine",
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
        "The model returned a moderate RUL estimate. Validate performance "
        "against appropriate held-out test data before drawing engineering conclusions."
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

    st.plotly_chart(
        style_chart(fig_sensor),
        width="stretch",
    )

    st.caption(
        "Historical readings only. A change in sensor values does not, "
        "by itself, establish a component fault."
    )

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
        col for col in sensor_columns
        if col in latest.columns
    ]

    st.dataframe(
        latest[["cycle"] + available_sensors].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# PROGNOSTICS — HISTORICAL MODEL OUTPUTS
# ============================================================

if page in ("Mission Control", "Prognostics"):
    st.divider()
    section_label("RUL HISTORY")

    st.subheader("Model Prediction Across Recorded History")

    # These are predictions for recorded observations.
    # They are not actual RUL labels or future forecasts.
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
        title="Model Estimates at Each Recorded Cycle",
    )

    st.plotly_chart(
        style_chart(fig_rul),
        width="stretch",
    )

    st.caption(
        "These are model outputs for recorded observations, not ground-truth "
        "remaining life and not a validated forecast of future engine behaviour."
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

    st.dataframe(
        history,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download Selected Engine History",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"aerotwin_engine_{int(selected_engine)}_history.csv"
        ),
        mime="text/csv",
        width="stretch",
    )


# ============================================================
# CSV ANALYSIS EXPORT
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
    file_name=(
        f"aerotwin_engine_{int(selected_engine)}_analysis.csv"
    ),
    mime="text/csv",
    width="stretch",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:11px;
        line-height:1.8;
        padding:12px 0;
    ">
        <strong style="color:#cbd5e1;letter-spacing:1px;">
            AEROTWIN AI
        </strong>
        <br>
        AI-Powered Engine Health Research · NASA C-MAPSS FD001 · v1.1
        <br>
        Historical research prototype — not approved for flight-safety
        or operational maintenance decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
