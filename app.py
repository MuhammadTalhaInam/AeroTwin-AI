
import sys
import html
import math
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
      RESEARCH PROTOTYPE · v2.0
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
# 6. 3D GEOMETRY HELPERS
# ============================================================

def add_x_cylinder(
    fig,
    x0,
    x1,
    radius,
    color,
    name,
    selected=False,
    opacity=1.0,
    segments=28,
):
    """
    Create a cylindrical surface along the X axis using Plotly Mesh3d.
    The cylinder is illustrative geometry, not CAD-accurate.
    """
    vertices_x = []
    vertices_y = []
    vertices_z = []

    for x in (x0, x1):
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertices_x.append(x)
            vertices_y.append(radius * math.cos(angle))
            vertices_z.append(radius * math.sin(angle))

    faces_i = []
    faces_j = []
    faces_k = []

    for i in range(segments):
        nxt = (i + 1) % segments

        a = i
        b = nxt
        c = segments + i
        d = segments + nxt

        faces_i.extend([a, a, b, b])
        faces_j.extend([b, c, c, d])
        faces_k.extend([c, b, d, c])

    # End caps
    left_center = len(vertices_x)
    right_center = left_center + 1

    vertices_x.extend([x0, x1])
    vertices_y.extend([0, 0])
    vertices_z.extend([0, 0])

    for i in range(segments):
        nxt = (i + 1) % segments

        faces_i.append(left_center)
        faces_j.append(nxt)
        faces_k.append(i)

        faces_i.append(right_center)
        faces_j.append(segments + i)
        faces_k.append(segments + nxt)

    fig.add_trace(
        go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=faces_i,
            j=faces_j,
            k=faces_k,
            color=color,
            opacity=opacity,
            flatshading=False,
            name=name,
            legendgroup=name,
            hovertemplate=f"{name}<extra></extra>",
            showscale=False,
            lighting=dict(
                ambient=0.48,
                diffuse=0.8,
                specular=0.45,
                roughness=0.42,
                fresnel=0.15,
            ),
            lightposition=dict(x=100, y=120, z=150),
        )
    )


def add_fan_blades(
    fig,
    x,
    inner_radius,
    outer_radius,
    count,
    color,
    name,
    selected=False,
):
    """
    Add stylized 3D blade surfaces arranged around the engine axis.
    """
    blade_color = "#67e8f9" if selected else color

    for blade_index in range(count):
        angle = 2 * math.pi * blade_index / count

        # Slight blade sweep creates a more turbine-like appearance.
        angle_inner = angle - 0.12
        angle_outer = angle + 0.12

        x_front = x - 0.055
        x_back = x + 0.055

        points = [
            (x_front, inner_radius * math.cos(angle_inner),
             inner_radius * math.sin(angle_inner)),
            (x_front, outer_radius * math.cos(angle_outer),
             outer_radius * math.sin(angle_outer)),
            (x_back, outer_radius * math.cos(angle_outer),
             outer_radius * math.sin(angle_outer)),
            (x_back, inner_radius * math.cos(angle_inner),
             inner_radius * math.sin(angle_inner)),
        ]

        bx = [p[0] for p in points]
        by = [p[1] for p in points]
        bz = [p[2] for p in points]

        fig.add_trace(
            go.Mesh3d(
                x=bx,
                y=by,
                z=bz,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=blade_color,
                opacity=0.96,
                name=name,
                legendgroup=name,
                hovertemplate=f"{name} blade<extra></extra>",
                showscale=False,
                flatshading=False,
                lighting=dict(
                    ambient=0.55,
                    diffuse=0.85,
                    specular=0.4,
                    roughness=0.4,
                ),
            )
        )


def add_engine_rings(fig, x, radius, color, name):
    """Add thin ring details around a cylindrical engine section."""
    angles = [
        2 * math.pi * i / 80
        for i in range(81)
    ]

    fig.add_trace(
        go.Scatter3d(
            x=[x] * len(angles),
            y=[radius * math.cos(a) for a in angles],
            z=[radius * math.sin(a) for a in angles],
            mode="lines",
            line=dict(color=color, width=4),
            name=name,
            hovertemplate=f"{name}<extra></extra>",
            showlegend=False,
        )
    )


# ============================================================
# 7. BUILD THE INTERACTIVE 3D TURBOFAN
# ============================================================

def make_3d_engine(selected_component):
    fig = go.Figure()

    # Each section occupies a different axial position.
    sections = {
        "Fan": {
            "x0": 0.15, "x1": 0.72,
            "radius": 1.05, "inner": 0.22,
        },
        "Compressor": {
            "x0": 0.85, "x1": 2.05,
            "radius": 0.77, "inner": 0.20,
        },
        "Combustor": {
            "x0": 2.12, "x1": 3.05,
            "radius": 0.69, "inner": 0.26,
        },
        "Turbine": {
            "x0": 3.13, "x1": 4.10,
            "radius": 0.72, "inner": 0.20,
        },
        "Exhaust": {
            "x0": 4.18, "x1": 5.15,
            "radius": 0.54, "inner": 0.18,
        },
    }

    # Main outer casing sections.
    for name, section in sections.items():
        active = name == selected_component
        base_color = COMPONENT_COLORS[name]

        # A highlighted section becomes brighter.
        color = "#67e8f9" if active else base_color
        opacity = 0.82 if active else 0.52

        add_x_cylinder(
            fig,
            section["x0"],
            section["x1"],
            section["radius"],
            color,
            name,
            selected=active,
            opacity=opacity,
            segments=36,
        )

        add_engine_rings(
            fig,
            section["x0"],
            section["radius"],
            "#e2e8f0" if active else "#64748b",
            name,
        )

        add_engine_rings(
            fig,
            section["x1"],
            section["radius"],
            "#e2e8f0" if active else "#64748b",
            name,
        )

    # Central shaft through the engine.
    add_x_cylinder(
        fig,
        0.2,
        5.0,
        0.15,
        "#cbd5e1",
        "Central shaft",
        opacity=0.92,
        segments=24,
    )

    # Fan rotor and blades.
    add_x_cylinder(
        fig, 0.25, 0.40, 0.28,
        "#94a3b8", "Fan hub", opacity=1.0,
    )

    add_fan_blades(
        fig,
        x=0.43,
        inner_radius=0.25,
        outer_radius=0.98,
        count=14,
        color=COMPONENT_COLORS["Fan"],
        name="Fan",
        selected=selected_component == "Fan",
    )

    # Compressor: several stages of small rotor blades.
    for stage_x in [0.98, 1.25, 1.52, 1.79]:
        add_fan_blades(
            fig,
            x=stage_x,
            inner_radius=0.23,
            outer_radius=0.70,
            count=10,
            color=COMPONENT_COLORS["Compressor"],
            name="Compressor",
            selected=selected_component == "Compressor",
        )

    # Combustor: inner chamber and outer combustion casing.
    add_x_cylinder(
        fig,
        2.18,
        2.98,
        0.43,
        "#fb923c",
        "Combustor inner chamber",
        opacity=0.96,
        segments=32,
    )

    # Turbine rotor stages.
    for stage_x in [3.35, 3.72, 4.00]:
        add_fan_blades(
            fig,
            x=stage_x,
            inner_radius=0.22,
            outer_radius=0.66,
            count=12,
            color=COMPONENT_COLORS["Turbine"],
            name="Turbine",
            selected=selected_component == "Turbine",
        )

    # Exhaust nozzle: gradually changing radius.
    add_x_cylinder(
        fig,
        4.25,
        5.02,
        0.43,
        COMPONENT_COLORS["Exhaust"],
        "Exhaust inner nozzle",
        opacity=0.78,
        segments=32,
    )

    # Add axial flow direction.
    fig.add_trace(
        go.Scatter3d(
            x=[0.1, 5.35],
            y=[0, 0],
            z=[1.28, 1.28],
            mode="lines+text",
            line=dict(color="#67e8f9", width=5),
            text=["AIR INLET", "EXHAUST FLOW"],
            textposition="top center",
            textfont=dict(color="#a5f3fc", size=10),
            name="Flow direction",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "INTERACTIVE TURBOFAN · "
                + html.escape(selected_component.upper())
                + " SELECTED"
            ),
            font=dict(color="#f1f5f9", size=16),
            x=0.02,
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Engine axis",
                visible=False,
                range=[-0.1, 5.5],
                showbackground=False,
            ),
            yaxis=dict(
                visible=False,
                range=[-1.45, 1.45],
                showbackground=False,
            ),
            zaxis=dict(
                visible=False,
                range=[-1.45, 1.55],
                showbackground=False,
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.7, y=1.25, z=1.25),
            camera=dict(
                eye=dict(x=1.65, y=1.7, z=1.15),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        uirevision="aerotwin-engine-camera",
    )

    return fig


# ============================================================
# 8. LOAD MODEL AND HISTORICAL DATA
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
# 9. SIDEBAR
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
# 10. SELECTED ENGINE HISTORY
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

# ============================================================
# 11. ENGINE WORKSPACE
# ============================================================

section_label("ENGINE WORKSPACE")
st.subheader(f"Engine {int(selected_engine):03d} — Analysis Console")
st.caption(
    f"{len(history):,} recorded observations · "
    f"Latest recorded cycle: {latest_cycle}"
)

if page in ("Mission Control", "Engine Explorer"):
    section_label("INTERACTIVE 3D TURBOFAN")

    st.markdown(
        "Select a component below. Drag to rotate, scroll to zoom, "
        "and use the interactive 3D controls to explore the model."
    )

    # Component selection controls.
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

    # Display the interactive model.
    with st.spinner("Building interactive 3D engine geometry..."):
        engine_figure = make_3d_engine(selected_component)

    st.plotly_chart(
        engine_figure,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "toImage",
                "lasso2d",
                "select2d",
            ],
        },
        key="aerotwin_3d_engine",
    )

    st.caption(
        "Conceptual 3D visualization generated from simplified geometry. "
        "It is not a dimensionally accurate CAD model, CFD simulation, "
        "or live engine digital twin."
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
        "These are historical sensor readings. The selected sensor is not "
        "a verified direct measurement of the selected physical component."
    )

# ============================================================
# 12. DATASET OVERVIEW
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
# 13. RUN MODEL ANALYSIS
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
# 14. PREDICTIONS AND SENSOR ANALYSIS
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
            sensor_name
            for sensor_name in sensor_columns
            if sensor_name in latest.columns
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
            "These are model outputs for recorded observations, not "
            "ground-truth remaining life or validated future forecasts."
        )

    # ========================================================
    # 15. EXPORT ANALYSIS
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
# 16. DATA LABORATORY
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
# 17. FOOTER
# ============================================================

st.divider()

st.html("""
<div style="text-align:center;color:#94a3b8;font-size:11px;
            line-height:1.9;padding:12px 0;">
  <strong style="color:#cbd5e1;letter-spacing:1px;">AEROTWIN AI</strong>
  <br>
  Interactive 3D Engine Explorer · NASA C-MAPSS FD001 · v2.0
  <br>
  Historical research prototype — not certified for operational decisions.
</div>
""")
