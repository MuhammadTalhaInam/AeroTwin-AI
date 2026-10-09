
import json
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

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

def load_features():
    with open(FEATURE_PATH, "r") as f:
        return json.load(f)

def load_model():
    return joblib.load(MODEL_PATH)

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
            data.groupby("engine_id")[sensor].diff().fillna(0)
        )

    return data

def predict_engine(model, data, feature_names, engine_id):
    history = data[data["engine_id"] == engine_id].copy()
    if history.empty:
        raise ValueError(f"Engine {engine_id} was not found.")

    latest = history.iloc[[-1]]
    missing = [c for c in feature_names if c not in latest.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    prediction = float(model.predict(latest[feature_names])[0])

    return {
        "history": history,
        "latest": latest,
        "prediction": max(0.0, prediction),
        "latest_cycle": int(latest["cycle"].iloc[0]),
    }
