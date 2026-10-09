
import numpy as np
import pandas as pd


def analyze_engine(result, sensor_columns):
    """
    Summarize the latest engine readings and recent sensor trends.
    This is a research aid, not a certified fault diagnosis.
    """
    history = result["history"].copy()
    latest = result["latest"].iloc[0]
    rul = float(result["prediction"])

    recent = history.tail(min(10, len(history)))
    previous = history.iloc[-min(10, len(history)):-1]

    sensor_report = []

    for sensor in sensor_columns:
        current_value = float(latest[sensor])

        if len(previous) > 0:
            baseline = float(previous[sensor].mean())
            change_pct = (
                (current_value - baseline) / abs(baseline) * 100
                if abs(baseline) > 1e-9 else 0.0
            )
        else:
            baseline = current_value
            change_pct = 0.0

        sensor_report.append({
            "Sensor": sensor,
            "Latest Reading": round(current_value, 3),
            "Recent Average": round(baseline, 3),
            "Change vs Recent Average (%)": round(change_pct, 2),
        })

    report = pd.DataFrame(sensor_report)

    if rul <= 15:
        rul_category = "Low predicted RUL"
    elif rul <= 70:
        rul_category = "Moderate predicted RUL"
    else:
        rul_category = "Higher predicted RUL"

    largest_change = (
        report.assign(
            _magnitude=report["Change vs Recent Average (%)"].abs()
        )
        .sort_values("_magnitude", ascending=False)
        .drop(columns="_magnitude")
        .head(3)
    )

    return {
        "predicted_rul": rul,
        "rul_category": rul_category,
        "latest_cycle": int(latest["cycle"]),
        "sensor_report": report,
        "largest_recent_changes": largest_change,
        "recent_cycles_analyzed": len(recent),
        "notice": (
            "Sensor changes are descriptive comparisons with recent readings. "
            "They are not validated fault thresholds or confirmed diagnoses."
        ),
    }
