
import sys
import html
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. CONFIGURATION
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
    page_title="AeroTwin AI | Engine Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CSS
# ============================================================

CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 0%,
        rgba(8,145,178,.10), transparent 30%),
        #0b0f19;
}
.block-container {
    max-width: 1600px;
    padding-top: 1.3rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #0d1320;
    border-right: 1px solid rgba(148,163,184,.18);
}
.aero-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:16px;
    flex-wrap:wrap;
    padding:20px;
    margin-bottom:22px;
    border:1px solid rgba(148,163,184,.18);
    border-radius:16px;
    background:linear-gradient(120deg,#111827,#102238);
}
.aero-brand {
    display:flex;
    align-items:center;
    gap:13px;
}
.aero-logo {
    width:48px;
    height:48px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:12px;
    background:rgba(34,211,238,.12);
    border:1px solid rgba(34,211,238,.30);
    color:#67e8f9;
    font-size:26px;
}
.aero-name {
    color:#f8fafc;
    font-size:24px;
    font-weight:850;
    letter-spacing:2px;
}
.aero-sub {
    color:#94a3b8;
    font-size:10px;
    letter-spacing:1.4px;
    margin-top:4px;
}
.aero-pill {
    border:1px solid rgba(34,211,238,.25);
    background:rgba(34,211,238,.07);
    color:#a5f3fc;
    border-radius:9px;
    padding:10px 12px;
    font-size:10px;
    font-weight:800;
    letter-spacing:1px;
}
.aero-eyebrow,.section-label {
    color:#22d3ee;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.7px;
    text-transform:uppercase;
    margin:10px 0;
}
.aero-metric {
    min-height:130px;
    padding:17px;
    border:1px solid rgba(148,163,184,.18);
    border-radius:14px;
    background:linear-gradient(145deg,#162030,#0e1623);
    border-top:3px solid var(--accent,#22d3ee);
}
.metric-label {
    color:#9caec2;
    font-size:11px;
    font-weight:750;
    letter-spacing:.8px;
}
.metric-value {
    color:#f8fafc;
    font-size:28px;
    font-weight:850;
    margin-top:15px;
    overflow-wrap:anywhere;
}
.metric-note {
    color:#94a3b8;
    font-size:11px;
    margin-top:8px;
}
.component-panel {
    background:rgba(17,24,39,.75);
    border:1px solid rgba(148,163,184,.18);
    border-radius:14px;
    padding:18px;
}
.component-name {
    color:#f8fafc;
    font-size:20px;
    font-weight:800;
}
.component-desc {
    color:#a8b7ca;
    font-size:13px;
    line-height:1.7;
}
[data-testid="stMetric"] {
    background:rgba(17,24,39,.75);
    border:1px solid rgba(148,163,184,.18);
    border-radius:12px;
    padding:13px;
}
[data-testid="stMetricLabel"] {color:#a7b7ca;}
[data-testid="stMetricValue"] {color:#f1f5f9;}
.stButton > button {
    border-radius:9px;
    font-weight:750;
    min-height:42px;
}
hr {border-color:rgba(148,163,184,.15);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Load existing project CSS if present.
css_path = BASE_DIR / "assets" / "aerotwin.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ============================================================
# 3. HEADER
# ============================================================

st.html("""
<div class="aero-header">
  <div class="aero-brand">
    <div class="aero-logo">✈</div>
    <div>
      <div class="aero-name">AEROTWIN</div>
      <div class="aero-sub">ENGINE INTELLIGENCE PLATFORM</div>
    </div>
  </div>
  <div class="aero-pill">HISTORICAL DATA MODE</div>
  <div>
    <div style="color:#e2e8f0;font-size:11px;font-weight:800;
                letter-spacing:1px;">AI PROGNOSTICS</div>
    <div style="color:#94a3b8;font-size:10px;margin-top:5px;">
      RESEARCH PROTOTYPE · v1.2
    </div>
  </div>
</div>
<div class="aero-eyebrow">AEROSPACE INTELLIGENCE SYSTEM</div>
""")

st.title("Mission Control")
st.caption(
    "Turbofan engine explorer · Historical sensor intelligence · RUL estimation"
)

st.warning(
    "Research prototype using historical NASA C-MAPSS FD001 data. "
    "No live aircraft telemetry is connected. Model estimates and sensor "
    "interpretations are not certified for maintenance or flight-safety decisions."
)

# ============================================================
# 4. HELPERS
# ============================================================

def section_label(text):
    st.markdown(
        f'<div class="section-label">{html.escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, accent="#22d3ee"):
    st.markdown(
        f"""
        <div class="aero-metric" style="--accent:{accent}">
          <div class="metric-label">{html.escape(str(label))}</div>
          <div class="metric-value">{html.escape(str(value))}</div>
          <div class="metric-note">{html.escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_style(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=15, r=15, t=55, b=15),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    return fig


# ============================================================
# 5. COMPONENT KNOWLEDGE BASE
# ============================================================

COMPONENTS = {
    "Fan": {
        "position": 0,
        "purpose": (
            "The fan accelerates a large amount of air. In a high-bypass "
            "turbofan, much of the thrust comes from the bypass airflow."
        ),
        "engineering": (
            "Engineering topics: airflow, blade loading, rotational speed, "
            "vibration, and aerodynamic efficiency."
        ),
        "sensors": ["sensor_2", "sensor_3", "sensor_4"],
    },
    "Compressor": {
        "position": 1,
        "purpose": (
            "The compressor raises the pressure of incoming air before "
            "it enters the combustor."
        ),
        "engineering": (
            "Engineering topics: pressure rise, compressor efficiency, "
            "temperature rise, and flow stability."
        ),
        "sensors": ["sensor_2", "sensor_7", "sensor_11", "sensor_12"],
    },
    "Combustor": {
        "position": 2,
        "purpose": (
            "The combustor mixes compressed air with fuel and releases "
            "heat to produce high-energy gas."
        ),
        "engineering": (
            "Engineering topics: heat addition, combustion stability, "
            "fuel-air mixing, and gas temperature."
        ),
        "sensors": ["sensor_3", "sensor_4", "sensor_11"],
    },
    "Turbine": {
        "position": 3,
        "purpose": (
            "The turbine extracts energy from hot gases to drive the "
            "compressor and fan through rotating shafts."
        ),
        "engineering": (
            "Engineering topics: turbine work, thermal loading, shaft "
            "power, efficiency, and material temperature limits."
        ),
        "sensors": ["sensor_4", "sensor_11", "sensor_14", "sensor_17"],
    },
    "Exhaust": {
        "position": 4,
        "purpose": (
            "The exhaust system guides gas out of the engine. The final "
            "gas momentum contributes to engine thrust."
        ),
        "engineering": (
            "Engineering topics: exhaust velocity, pressure, temperature, "
            "and nozzle flow."
        ),
        "sensors": ["sensor_7", "sensor_12", "sensor_20", "sensor_21"],
    },
}

# ============================================================
# 6. INTERACTIVE ENGINE DIAGRAM
# ============================================================

def make_engine_diagram(selected_component):
    names = list(COMPONENTS.keys())

    colors = [
        "#22d3ee" if name == selected_component else "#24364d"
        for name in names
    ]

    fig = go.Figure()

    # A simplified schematic. It is not a true 3D CAD model.
    fig.add_trace(
        go.Bar(
            x=names,
            y=[1] * len(names),
            marker=dict(
                color=colors,
                line=dict(color="#67e8f9", width=1.2),
            ),
            text=names,
            textposition="inside",
            insidetextfont=dict(color="#ffffff", size=12),
            hovertemplate="<b>%{x}</b><extra>Engine component</extra>",
        )
    )

    fig.update_layout(
        title="Simplified Turbofan Flow Path",
        showlegend=False,
        height=235,
        bargap=0.18,
        yaxis=dict(visible=False, range=[0, 1.5]),
        xaxis=dict(
            tickfont=dict(color="#cbd5e1", size=10),
            fixedrange=True,
        ),
        annotations=[
            dict(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text="AIRFLOW  →",
                showarrow=False,
                font=dict(color="#67e8f9", size=11),
            )
        ],
        margin=dict(l=10, r=10, t=65, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
    )

    return fig


# ============================================================
# 7. LOAD MODEL AND DATA
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
    st.error(f"Could not initialize model or dataset: {exc}")
    st.stop()

if len(feature_names) != 44:
    st.error(
        f"Expected 44 engineered features, found {len(feature_names)}. "
        "Check the model and feature-name file."
    )
    st.stop()

sensor_columns = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_11", "sensor_12", "sensor_14", "sensor_17",
    "sensor_20", "sensor_21",
]

required = {"engine_id", "cycle", *feature_names, *sensor_columns}
missing = sorted(required - set(data.columns))

if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.markdown("## ✈️ AEROTWIN")
st.sidebar.caption("ENGINE INTELLIGENCE PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Mission Control",
        "Engine Explorer",
        "Sensor Diagnostics",
        "Prognostics",
        "Data Laboratory",
    ],
)

engine_ids = sorted(data["engine_id"].unique().tolist())
default_index = engine_ids.index(97) if 97 in engine_ids else 0

selected_engine = st.sidebar.selectbox(
    "SELECT HISTORICAL ENGINE",
    engine_ids,
    index=default_index,
)

st.sidebar.divider()
st.sidebar.caption("DATA SOURCE")
st.sidebar.write("NASA C-MAPSS · FD001")
st.sidebar.caption("Historical run-to-failure dataset")
st.sidebar.caption("44 engineered model inputs")
st.sidebar.caption("No live telemetry")

# ============================================================
# 9. ENGINE HISTORY AND SELECTED COMPONENT
# ============================================================

history = (
    data[data["engine_id"] == selected_engine]
    .sort_values("cycle")
    .copy()
)

if history.empty:
    st.error("No recorded data found for this engine.")
    st.stop()

latest_cycle = int(history["cycle"].max())

component_names = list(COMPONENTS.keys())

if "selected_component" not in st.session_state:
    st.session_state["selected_component"] = "Compressor"

selected_component = st.session_state["selected_component"]

# ============================================================
# 10. ENGINE WORKSPACE
# ============================================================

section_label("ENGINE WORKSPACE")
st.subheader(f"Engine {int(selected_engine):03d} — Analysis Console")
st.caption(
    f"{len(history):,} recorded observations · "
    f"Latest recorded cycle: {latest_cycle}"
)

if page in ("Mission Control", "Engine Explorer"):
    section_label("INTERACTIVE TURBOFAN EXPLORER")

    st.markdown(
        "Select an engine section to explore its purpose, engineering "
        "concepts and available historical sensor trends."
    )

    st.plotly_chart(
        make_engine_diagram(selected_component),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    component_cols = st.columns(5)

    for index, name in enumerate(component_names):
        with component_cols[index]:
            if st.button(
                name,
                key=f"component_button_{name}",
                type=(
                    "primary"
                    if name == selected_component
                    else "secondary"
                ),
                use_container_width=True,
            ):
                st.session_state["selected_component"] = name
                st.rerun()

    selected_component = st.session_state["selected_component"]
    component_info = COMPONENTS[selected_component]

    st.markdown(
        f"""
        <div class="component-panel">
          <div class="component-name">
            {html.escape(selected_component)}
          </div>
          <p class="component-desc">
            {html.escape(component_info["purpose"])}
          </p>
          <p class="component-desc">
            {html.escape(component_info["engineering"])}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Diagram is a simplified conceptual flow path, not a dimensionally "
        "accurate engine model. Component-to-sensor associations are "
        "exploratory topics, not verified direct measurements."
    )

    section_label("COMPONENT SENSOR EXPLORER")
    st.subheader(f"Historical readings — {selected_component}")

    relevant_sensors = [
        sensor
        for sensor in component_info["sensors"]
        if sensor in history.columns
    ]

    chosen_sensor = st.selectbox(
        "Choose a historical sensor",
        relevant_sensors,
        key=f"sensor_for_{selected_component}",
    )

    fig_component = px.line(
        history,
        x="cycle",
        y=chosen_sensor,
        title=f"{chosen_sensor} · Engine {int(selected_engine):03d}",
        labels={
            "cycle": "Recorded cycle",
            chosen_sensor: "Recorded sensor value",
        },
    )

    st.plotly_chart(
        chart_style(fig_component),
        use_container_width=True,
    )

    st.caption(
        "This chart displays historical data. It does not prove that the "
        "selected sensor directly measures the selected physical component."
    )

# ============================================================
# 11. DATASET OVERVIEW
# ============================================================

section_label("DATASET OVERVIEW")

overview = st.columns(3)

with overview[0]:
    metric_card(
        "Recorded Engines",
        data["engine_id"].nunique(),
        "Engines in FD001",
        "#22d3ee",
    )

with overview[1]:
    metric_card(
        "Dataset Observations",
        f"{len(data):,}",
        "Historical sensor records",
        "#818cf8",
    )

with overview[2]:
    metric_card(
        "Model Features",
        f"{len(feature_names)}/44",
        "Expected engineered inputs",
        "#34d399",
    )

# ============================================================
# 12. RUN MODEL ANALYSIS
# ============================================================

col_button, col_mode = st.columns([1.4, 1])

with col_button:
    analyze_clicked = st.button(
        "▶ ANALYZE ENGINE",
        type="primary",
        use_container_width=True,
    )

with col_mode:
    st.markdown(
        '<div class="component-panel">'
        '<div class="metric-label">ANALYSIS MODE</div>'
        '<div style="color:#67e8f9;font-size:17px;'
        'font-weight:800;margin-top:8px;">Historical data</div>'
        '<div class="metric-note">Not live aircraft telemetry</div>'
        '</div>',
        unsafe_allow_html=True,
    )

if st.session_state.get("aerotwin_engine") != int(selected_engine):
    st.session_state.pop("aerotwin_result", None)
    st.session_state.pop("aerotwin_report", None)
    st.session_state.pop("aerotwin_engine", None)

if analyze_clicked:
    try:
        with st.spinner("Running trained model and sensor analysis..."):
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

has_result = (
    "aerotwin_result" in st.session_state
    and "aerotwin_report" in st.session_state
)

# ============================================================
# 13. PREDICTIONS AND ANALYSIS
# ============================================================

if has_result:
    result = st.session_state["aerotwin_result"]
    report = st.session_state["aerotwin_report"]

    prediction = float(report["predicted_rul"])
    reported_cycle = int(report["latest_cycle"])

    section_label("MODEL OUTPUTS")
    metrics = st.columns(3)

    with metrics[0]:
        metric_card(
            "Predicted RUL",
            f"{prediction:.2f} cycles",
            "Model estimate; not verified actual life",
            "#22d3ee",
        )

    with metrics[1]:
        metric_card(
            "Latest Recorded Cycle",
            reported_cycle,
            "Last observed cycle",
            "#818cf8",
        )

    with metrics[2]:
        metric_card(
            "Model Features",
            f"{len(feature_names)}/44",
            "Engineered inputs",
            "#34d399",
        )

    st.caption(
        "RUL is a model estimate, not ground truth, a confidence score, "
        "or a guarantee of actual remaining engine life."
    )

    section_label("PROGNOSTICS SUMMARY")
    st.subheader("Model Interpretation")
    st.write(report["rul_category"])

    if prediction <= 15:
        st.warning(
            "The model returned a low RUL estimate. This is a research "
            "indicator, not an operational maintenance instruction."
        )
    elif prediction <= 70:
        st.info(
            "The model returned a moderate RUL estimate. Validate the model "
            "on appropriate held-out data before drawing engineering conclusions."
        )
    else:
        st.success(
            "The model returned a higher RUL estimate. This does not certify "
            "the engine as healthy or safe."
        )

    if page in (
        "Mission Control",
        "Engine Explorer",
        "Sensor Diagnostics",
    ):
        section_label("SENSOR INTELLIGENCE")
        st.subheader("Sensor Trend Explorer")

        sensor = st.selectbox(
            "Sensor",
            sensor_columns,
            key="main_sensor_choice",
        )

        fig_sensor = px.line(
            history,
            x="cycle",
            y=sensor,
            title=f"{sensor} Across Recorded Cycles",
        )

        st.plotly_chart(
            chart_style(fig_sensor),
            use_container_width=True,
        )

        st.subheader("Recent Sensor Analysis")
        st.dataframe(
            report["sensor_report"],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Largest Relative Changes vs Recent Readings")
        st.dataframe(
            report["largest_recent_changes"],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(report["notice"])

        st.subheader("Latest Recorded Sensor Values")
        latest = result["latest"]
        available_sensors = [
            s for s in sensor_columns if s in latest.columns
        ]

        st.dataframe(
            latest[["cycle"] + available_sensors].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    if page in ("Mission Control", "Prognostics"):
        section_label("RUL HISTORY")
        st.subheader("Model Outputs Across Recorded History")

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
            title="Predicted RUL for Recorded Observations",
        )

        st.plotly_chart(
            chart_style(fig_rul),
            use_container_width=True,
        )

        st.caption(
            "Historical model outputs, not ground-truth remaining life "
            "and not validated forecasts of future engine behaviour."
        )

    # ========================================================
    # 14. EXPORTS
    # ========================================================

    section_label("ENGINEERING REPORTS")
    st.subheader("Export Analysis")

    export_df = report["sensor_report"].copy()
    export_df.insert(0, "Engine ID", int(selected_engine))
    export_df.insert(1, "Latest Recorded Cycle", reported_cycle)
    export_df["Model Predicted RUL (cycles)"] = round(prediction, 3)

    st.download_button(
        "Download Sensor Analysis CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"aerotwin_engine_{int(selected_engine)}_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "Press ANALYZE ENGINE to calculate the RUL estimate and display "
        "sensor-analysis results. The Engine Explorer can be explored independently."
    )

# ============================================================
# 15. DATA LABORATORY
# ============================================================

if page == "Data Laboratory":
    section_label("DATA LABORATORY")
    st.subheader("Selected Engine History")

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Selected Engine History",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name=f"aerotwin_engine_{int(selected_engine)}_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# 16. FOOTER
# ============================================================

st.divider()
st.html("""
<div style="text-align:center;color:#94a3b8;font-size:11px;
            line-height:1.9;padding:12px 0;">
  <strong style="color:#cbd5e1;letter-spacing:1px;">AEROTWIN AI</strong>
  <br>
  Interactive Engine Explorer · NASA C-MAPSS FD001 · v1.2
  <br>
  Historical research prototype — not certified for operational decisions.
</div>
""")
