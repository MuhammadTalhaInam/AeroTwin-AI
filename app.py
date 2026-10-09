import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="AeroTwin AI",
    page_icon="✈️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aerotwin_model" / "gradient_boosting_model.pkl"
FEATURE_PATH = BASE_DIR / "aerotwin_model" / "feature_names.json"
DATA_PATH = BASE_DIR / "CMAPSSData" / "train_FD001.txt"

SENSOR_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_11", "sensor_12", "sensor_14",
    "sensor_17", "sensor_20", "sensor_21"
]

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    with open(FEATURE_PATH, "r") as file:
        return json.load(file)

@st.cache_data
def load_dataset():
    data = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    data = data.sort_values(["engine_id", "cycle"]).copy()

    for sensor in SENSOR_COLS:
        data[f"{sensor}_rolling_mean_5"] = (
            data.groupby("engine_id")[sensor]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
        )

        data[f"{sensor}_change"] = (
            data.groupby("engine_id")[sensor]
            .diff()
            .fillna(0)
        )

    return data

st.title("✈️ AeroTwin AI")
st.caption("Aircraft Engine Health & Remaining Useful Life Research Prototype")

st.warning(
    "Research demonstration only. Predictions are not certified for aircraft "
    "maintenance or flight-safety decisions."
)

try:
    model = load_model()
    feature_names = load_features()
    data = load_dataset()
except Exception as error:
    st.error(f"Could not load AeroTwin files: {error}")
    st.stop()

if len(feature_names) != 44:
    st.error(f"Expected 44 model features, found {len(feature_names)}.")
    st.stop()

st.sidebar.header("Engine Selection")
engine_ids = sorted(data["engine_id"].unique().tolist())

selected_engine = st.sidebar.selectbox(
    "Choose a simulated engine",
    engine_ids
)

history = data[data["engine_id"] == selected_engine].copy()
latest = history.iloc[[-1]]
model_input = latest[feature_names]

if model_input.isnull().any().any():
    st.error("Missing feature values found. Prediction stopped.")
    st.stop()

prediction = float(model.predict(model_input)[0])
prediction = max(0.0, prediction)

st.subheader(f"Engine #{selected_engine}")

col1, col2, col3 = st.columns(3)
col1.metric("Latest Cycle", int(latest["cycle"].iloc[0]))
col2.metric("Estimated RUL", f"{prediction:.1f} cycles")
col3.metric("Model Features", f"{len(feature_names)}/44")

st.divider()

st.subheader("Sensor History")
sensor_choice = st.selectbox(
    "Select a sensor to inspect",
    SENSOR_COLS
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history["cycle"], history[sensor_choice], linewidth=2)
ax.set_xlabel("Operating Cycle")
ax.set_ylabel(sensor_choice)
ax.set_title(f"{sensor_choice} — Engine #{selected_engine}")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.subheader("RUL Prediction Across Engine History")

# Demonstration: predict each observed cycle using only its available
# past sensor readings. This creates a trend, not independent ground truth.
history_features = history[feature_names]
history_predictions = model.predict(history_features).clip(min=0)

trend = pd.DataFrame({
    "Cycle": history["cycle"].to_numpy(),
    "Predicted RUL (cycles)": history_predictions
})

st.line_chart(
    trend.set_index("Cycle"),
    y="Predicted RUL (cycles)"
)

st.subheader("Latest Sensor Readings")
st.dataframe(
    latest[["cycle"] + SENSOR_COLS].reset_index(drop=True),
    use_container_width=True
)

st.caption(
    "Dataset: NASA C-MAPSS FD001. RUL estimates are model outputs, "
    "not guaranteed service-life measurements."
)
