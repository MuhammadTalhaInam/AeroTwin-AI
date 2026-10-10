
import sys
import html
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px

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
# 2. THEME
# ============================================================

CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 0%,
        rgba(8,145,178,.12), transparent 32%),
        #0b0f19;
}
.block-container {
    max-width: 1650px;
    padding-top: 1.2rem;
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
    margin:12px 0;
}
.aero-metric {
    min-height:125px;
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
    font-size:27px;
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
    background:rgba(17,24,39,.78);
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
      RESEARCH PROTOTYPE · v2.1
    </div>
  </div>
</div>
<div class="aero-eyebrow">AEROSPACE INTELLIGENCE SYSTEM</div>
""")

st.title("Mission Control")

st.caption(
    "Interactive turbofan visualization · Historical sensor intelligence "
    "· Remaining Useful Life estimation"
)

st.warning(
    "Research prototype using historical NASA C-MAPSS FD001 data. "
    "No live aircraft telemetry is connected. Model estimates and sensor "
    "interpretations are not certified for maintenance or flight-safety decisions."
)

# ============================================================
# 4. GENERAL HELPERS
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
        "purpose": (
            "The fan accelerates incoming air. In a high-bypass turbofan, "
            "a substantial portion of thrust comes from bypass airflow."
        ),
        "engineering": (
            "Engineering topics: airflow, blade loading, rotational speed, "
            "vibration and aerodynamic efficiency."
        ),
        "sensors": ["sensor_2", "sensor_3", "sensor_4"],
    },
    "Compressor": {
        "purpose": (
            "The compressor raises the pressure of incoming air before "
            "the air enters the combustor."
        ),
        "engineering": (
            "Engineering topics: pressure rise, temperature rise, "
            "efficiency and flow stability."
        ),
        "sensors": ["sensor_2", "sensor_7", "sensor_11", "sensor_12"],
    },
    "Combustor": {
        "purpose": (
            "The combustor adds heat by burning fuel in compressed air, "
            "creating high-energy gas."
        ),
        "engineering": (
            "Engineering topics: heat addition, combustion stability, "
            "fuel-air mixing and gas temperature."
        ),
        "sensors": ["sensor_3", "sensor_4", "sensor_11"],
    },
    "Turbine": {
        "purpose": (
            "The turbine extracts energy from hot gas to drive the "
            "compressor and fan through rotating shafts."
        ),
        "engineering": (
            "Engineering topics: turbine work, thermal loading, shaft "
            "power, efficiency and material temperature limits."
        ),
        "sensors": ["sensor_4", "sensor_11", "sensor_14", "sensor_17"],
    },
    "Exhaust": {
        "purpose": (
            "The exhaust guides gas out of the engine. The exiting gas "
            "momentum contributes to engine thrust."
        ),
        "engineering": (
            "Engineering topics: exhaust velocity, pressure, temperature "
            "and nozzle flow."
        ),
        "sensors": ["sensor_7", "sensor_12", "sensor_20", "sensor_21"],
    },
}

COMPONENT_COLORS = {
    "Fan": "#22d3ee",
    "Compressor": "#60a5fa",
    "Combustor": "#fb923c",
    "Turbine": "#c084fc",
    "Exhaust": "#34d399",
}

# ============================================================
# 6. REAL 3D TURBOFAN GLB VIEWER
# ============================================================

def render_glb_engine():
    """
    Load the real turbofan GLB asset and display it in an
    interactive browser-based 3D viewer.
    """

    model_path = BASE_DIR / "assets" / "turbofan-cutaway.glb"

    if not model_path.is_file():
        st.error("The turbofan 3D model could not be found.")
        st.markdown(
            "Upload the file below into the repository's `assets` folder:"
        )
        st.code("assets/turbofan-cutaway.glb")
        return

    try:
        model_bytes = model_path.read_bytes()

        if not model_bytes:
            st.error("The turbofan GLB file is empty.")
            return

        model_data = base64.b64encode(model_bytes).decode("ascii")

    except OSError as exc:
        st.error(f"Could not read the turbofan model: {exc}")
        return

    viewer_html = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport"
            content="width=device-width, initial-scale=1.0">

      <script type="module"
        src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js">
      </script>

      <style>
        * { box-sizing: border-box; }

        html, body {
          margin: 0;
          padding: 0;
          width: 100%;
          background: transparent;
          font-family: Arial, sans-serif;
          color: #e2e8f0;
        }

        .engine-shell {
          width: 100%;
          overflow: hidden;
          border: 1px solid rgba(148,163,184,.22);
          border-radius: 15px;
          background:
            radial-gradient(
              ellipse at 50% 42%,
              rgba(8,145,178,.13),
              transparent 65%
            ),
            linear-gradient(145deg,#111827,#0b1220);
        }

        .engine-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 10px;
          padding: 15px 18px;
          border-bottom: 1px solid rgba(148,163,184,.16);
        }

        .engine-title {
          font-size: 12px;
          font-weight: 800;
          letter-spacing: 1.4px;
          color: #e2e8f0;
        }

        .engine-status {
          font-size: 10px;
          letter-spacing: 1px;
          color: #67e8f9;
        }

        model-viewer {
          display: block;
          width: 100%;
          height: 510px;
          background: transparent;
          --poster-color: transparent;
          outline: none;
        }

        .engine-help {
          padding: 12px 16px;
          border-top: 1px solid rgba(148,163,184,.14);
          color: #94a3b8;
          font-size: 11px;
          line-height: 1.8;
        }

        .engine-help strong { color: #a5f3fc; }

        .loading {
          padding: 14px;
          color: #67e8f9;
          font-size: 12px;
          text-align: center;
        }

        .error {
          display: none;
          padding: 22px;
          color: #fca5a5;
          font-size: 13px;
          line-height: 1.7;
        }

        @media (max-width: 600px) {
          model-viewer { height: 370px; }
        }
      </style>
    </head>

    <body>
      <div class="engine-shell">

        <div class="engine-topbar">
          <div class="engine-title">
            TURBOFAN · 3D MODEL VIEWER
          </div>
          <div class="engine-status" id="status">
            INITIALIZING
          </div>
        </div>

        <div class="loading" id="loading">
          Loading 3D turbofan asset...
        </div>

        <model-viewer
          id="turbofan"
          src="data:model/gltf-binary;base64,__MODEL_DATA__"
          alt="Interactive 3D turbofan cutaway model"
          camera-controls
          touch-action="pan-y"
          interaction-prompt="auto"
          shadow-intensity="1"
          exposure="1.1"
          environment-image="neutral"
          camera-orbit="35deg 70deg auto"
          auto-rotate
          auto-rotate-delay="1500"
          rotation-per-second="12deg"
          loading="eager"
          reveal="auto">

          <div slot="progress-bar"></div>
        </model-viewer>

        <div class="error" id="error">
          The model could not be loaded. Check your internet connection,
          browser console and the validity of the GLB file.
        </div>

        <div class="engine-help">
          <strong>ROTATE:</strong> Drag with your mouse.
          &nbsp; <strong>ZOOM:</strong> Scroll.
          &nbsp; <strong>EXPLORE:</strong> Use touch gestures.
        </div>

      </div>

      <script>
        const viewer = document.getElementById("turbofan");
        const status = document.getElementById("status");
        const loading = document.getElementById("loading");
        const error = document.getElementById("error");

        viewer.addEventListener("load", () => {
          status.textContent = "MODEL LOADED";
          loading.style.display = "none";
          error.style.display = "none";
        });

        viewer.addEventListener("error", () => {
          status.textContent = "LOAD FAILED";
          loading.style.display = "none";
          error.style.display = "block";
        });
      </script>
    </body>
    </html>
    """

    viewer_html = viewer_html.replace(
        "__MODEL_DATA__",
        model_data,
    )

    components.html(
        viewer_html,
        height=625,
        scrolling=False,
    )


# ============================================================
# 7. LOAD MODEL AND HISTORICAL DATA
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
    with st.spinner("Initializing AeroTwin engine intelligence..."):
        model = get_model()
        feature_names = get_features()
        data = get_data()

except Exception as exc:
    st.error(f"Could not initialize model or dataset: {exc}")
    st.stop()


if len(feature_names) != 44:
    st.error(
        f"Expected 44 engineered features, found {len(feature_names)}. "
        "Check the trained model and feature-name file."
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

required = {"engine_id", "cycle", *feature_names, *sensor_columns}
missing = sorted(required - set(data.columns))

if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# ============================================================
# 8. SIDEBAR NAVIGATION
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
# 9. SELECTED ENGINE HISTORY
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

if st.session_state["selected_component"] not in component_names:
    st.session_state["selected_component"] = "Compressor"

selected_component = st.session_state["selected_component"]

# Clear results when a different engine is selected.
if st.session_state.get("aerotwin_engine") != int(selected_engine):
    st.session_state.pop("aerotwin_result", None)
    st.session_state.pop("aerotwin_report", None)
    st.session_state.pop("aerotwin_engine", None)

# ============================================================
# 10. ENGINE WORKSPACE
# ============================================================

section_label("ENGINE WORKSPACE")

st.subheader(
    f"Engine {int(selected_engine):03d} — Analysis Console"
)

st.caption(
    f"{len(history):,} recorded observations · "
    f"Latest recorded cycle: {latest_cycle}"
)

if page in ("Mission Control", "Engine Explorer"):

    section_label("INTERACTIVE 3D TURBOFAN")

    st.markdown(
        "Explore the downloaded turbofan model. Drag to rotate, "
        "scroll to zoom, and use the component selector to explore "
        "engineering information."
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

    with st.spinner("Loading the turbofan 3D model..."):
        render_glb_engine()

    st.caption(
        "The displayed GLB is a 3D asset. Its rotation is a visual "
        "interaction, not a physical engine simulation. It is not "
        "connected to live aircraft telemetry or the RUL model."
    )

    component_info = COMPONENTS[selected_component]
    component_color = COMPONENT_COLORS[selected_component]

    st.markdown(
        f"""
        <div class="component-panel"
             style="border-top:3px solid {component_color};">
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

    section_label("COMPONENT SENSOR EXPLORER")
    st.subheader(f"Historical readings — {selected_component}")

    relevant_sensors = [
        sensor
        for sensor in component_info["sensors"]
        if sensor in history.columns
    ]

    if relevant_sensors:
        chosen_sensor = st.selectbox(
            "Choose a historical sensor",
            relevant_sensors,
            key=f"sensor_for_{selected_component}",
        )

        fig_component = px.line(
            history,
            x="cycle",
            y=chosen_sensor,
            title=(
                f"{chosen_sensor} · "
                f"Engine {int(selected_engine):03d}"
            ),
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
        "Historical sensor readings are not verified direct measurements "
        "of the selected physical component."
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

if analyze_clicked:
    try:
        with st.spinner("Running model and sensor analysis..."):
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
    and st.session_state.get("aerotwin_engine") == int(selected_engine)
)

# ============================================================
# 13. PREDICTIONS AND SENSOR ANALYSIS
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
            "Engineered model inputs",
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
            "The model returned a moderate RUL estimate. Validate the "
            "model on appropriate held-out data before drawing conclusions."
        )
    else:
        st.success(
            "The model returned a higher RUL estimate. This does not "
            "certify the engine as healthy or safe."
        )

    # --------------------------------------------------------
    # SENSOR DIAGNOSTICS
    # --------------------------------------------------------

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
            sensor_name
            for sensor_name in sensor_columns
            if sensor_name in latest.columns
        ]

        st.dataframe(
            latest[["cycle"] + available_sensors].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # RUL HISTORY
    # --------------------------------------------------------

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
            "These are model outputs for recorded observations, not "
            "ground-truth remaining life or validated future forecasts."
        )

    # --------------------------------------------------------
    # EXPORT ENGINEERING REPORT
    # --------------------------------------------------------

    section_label("ENGINEERING REPORTS")
    st.subheader("Export Analysis")

    export_df = report["sensor_report"].copy()
    export_df.insert(0, "Engine ID", int(selected_engine))
    export_df.insert(1, "Latest Recorded Cycle", reported_cycle)
    export_df["Model Predicted RUL (cycles)"] = round(prediction, 3)

    st.download_button(
        "Download Sensor Analysis CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"aerotwin_engine_{int(selected_engine)}_analysis.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "Press ANALYZE ENGINE to calculate the RUL estimate and display "
        "sensor-analysis results. The 3D model can be explored independently."
    )

# ============================================================
# 14. DATA LABORATORY
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
# 15. FOOTER
# ============================================================

st.divider()

st.html("""
<div style="text-align:center;color:#94a3b8;font-size:11px;
            line-height:1.9;padding:12px 0;">
  <strong style="color:#cbd5e1;letter-spacing:1px;">
    AEROTWIN AI
  </strong>
  <br>
  Interactive 3D Turbofan · NASA C-MAPSS FD001 · v2.1
  <br>
  Historical research prototype — not certified for operational decisions.
</div>
""")
