import re

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Machine Health Monitoring",
    page_icon="⚙️",
    layout="wide"
)

# ============================================
# Load Models (cached so they load once, not on every click)
# ============================================

@st.cache_resource
def load_models():
    return {
        "rul_model": joblib.load("xgb_rul_model.pkl"),
        "risk_classifier": joblib.load("xgb_risk_classifier.pkl"),
        "isolation_forest": joblib.load("isolation_forest.pkl"),
        "label_encoder": joblib.load("label_encoder.pkl"),
        "feature_columns": joblib.load("feature_columns.pkl"),
    }

models = load_models()
xgb_rul_model = models["rul_model"]
xgb_risk_classifier = models["risk_classifier"]
isolation_forest = models["isolation_forest"]
label_encoder = models["label_encoder"]
feature_columns = models["feature_columns"]

ROLLING_WINDOWS = [5, 20]  # must match training (see notebook cell 13)


def natural_sort_key(name: str):
    """Sort sensor1, sensor2, ..., sensor21 in numeric order, not alphabetical."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


# Raw sensor names only (no engineered suffix), sorted numerically
sensor_columns = sorted(
    (
        col for col in feature_columns
        if col.startswith("sensor")
        and "_rolling_" not in col
        and "_rate_of_change" not in col
    ),
    key=natural_sort_key,
)

# ============================================
# Title
# ============================================

st.title("⚙️ Machine Health Monitoring System")
st.write("Predictive Maintenance Dashboard using XGBoost and Isolation Forest")
st.caption(
    "Trained on the NASA C-MAPSS turbofan engine degradation dataset — "
    "built for this specific set of 21 engine sensors, not general-purpose machinery."
)

# ============================================
# Feature engineering — mirrors training exactly (notebook cell 13)
# ============================================

def engineer_features(history_df: pd.DataFrame) -> pd.Series:
    """
    Takes a sensor-reading history for ONE engine (sorted by cycle) and
    returns the fully engineered feature row (raw + rolling + rate-of-change)
    for the LAST cycle, in the exact column order the models expect.
    """
    df = history_df.sort_values("cycle").copy()

    for sensor in sensor_columns:
        for w in ROLLING_WINDOWS:
            df[f"{sensor}_rolling_mean_{w}"] = (
                df[sensor].rolling(window=w, min_periods=1).mean()
            )
            df[f"{sensor}_rolling_std_{w}"] = (
                df[sensor].rolling(window=w, min_periods=1).std()
            )
        df[f"{sensor}_rate_of_change"] = df[sensor].diff()

    last_row = df.iloc[[-1]].copy()

    for feature in feature_columns:
        if feature not in last_row.columns:
            last_row[feature] = 0.0

    # First-cycle std/diff can be NaN (nothing to compare against yet)
    last_row[feature_columns] = last_row[feature_columns].fillna(0.0)

    return last_row[feature_columns]


def run_prediction(input_row: pd.DataFrame):
    predicted_rul = max(float(xgb_rul_model.predict(input_row)[0]), 0)

    predicted_risk_encoded = xgb_risk_classifier.predict(input_row)[0]
    predicted_risk = label_encoder.inverse_transform([predicted_risk_encoded])[0]

    anomaly_prediction = isolation_forest.predict(input_row)[0]
    anomaly_score = isolation_forest.decision_function(input_row)[0]
    anomaly_status = "Anomaly" if anomaly_prediction == -1 else "Normal"

    return predicted_rul, predicted_risk, anomaly_status, anomaly_score


def show_results(dataset, engine_id, cycle, predicted_rul, predicted_risk, anomaly_status, anomaly_score):
    st.subheader("Machine Health Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted RUL", f"{predicted_rul:.1f} cycles")
    col2.metric("Risk", predicted_risk)
    col3.metric("Anomaly Status", anomaly_status)
    col4.metric("Anomaly Score", f"{anomaly_score:.4f}")

    if predicted_risk == "High Risk" or anomaly_status == "Anomaly":
        st.error("⚠️ This engine needs attention.")
    elif predicted_risk == "Warning":
        st.warning("This engine should be monitored closely.")
    else:
        st.success("Engine health looks normal.")

    st.subheader("Machine Summary")
    summary = pd.DataFrame({
        "Parameter": ["Dataset", "Engine ID", "Current Cycle", "Predicted RUL", "Predicted Risk", "Anomaly Status"],
        "Value": [dataset, engine_id, cycle, f"{predicted_rul:.2f} cycles", predicted_risk, anomaly_status],
    })
    st.table(summary)


# ============================================
# Sidebar
# ============================================

st.sidebar.header("Engine Information")
dataset = st.sidebar.selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"])
engine_id = st.sidebar.text_input("Engine ID", value=f"{dataset}_1")

# ============================================
# Input mode: CSV upload (recommended) or manual entry
# ============================================

tab_csv, tab_manual = st.tabs(["📄 Upload CSV (recommended)", "⌨️ Manual entry"])

# ---------- CSV tab ----------
with tab_csv:
    st.write(
        "Upload a CSV with this engine's sensor readings over time — one row "
        "per cycle. More history gives more accurate rolling/trend features "
        "(the models were trained on 5- and 20-cycle rolling windows)."
    )

    template = pd.DataFrame({"cycle": [1, 2, 3]})
    for s in sensor_columns:
        template[s] = 0.0
    st.download_button(
        "⬇️ Download CSV template",
        template.to_csv(index=False),
        file_name="sensor_readings_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Sensor readings CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            data = None

        if data is not None:
            missing = [c for c in ["cycle", *sensor_columns] if c not in data.columns]
            if missing:
                st.error(
                    "The CSV is missing these expected column(s): "
                    + ", ".join(missing)
                    + ". Use the template above to check the exact column names."
                )
            else:
                st.success(f"Loaded {len(data)} cycle(s) for this engine.")
                st.dataframe(data.tail(10), use_container_width=True)

                last_cycle = int(data["cycle"].max())

                if st.button("🔍 Analyze Machine Health", type="primary", key="csv_analyze"):
                    input_row = engineer_features(data)
                    predicted_rul, predicted_risk, anomaly_status, anomaly_score = run_prediction(input_row)
                    show_results(
                        dataset, engine_id, last_cycle,
                        predicted_rul, predicted_risk, anomaly_status, anomaly_score,
                    )

# ---------- Manual tab ----------
with tab_manual:
    st.write(
        "Quick single-reading test. Rolling/trend features aren't available "
        "from a single reading, so they're set to 0 here — use the CSV tab "
        "for a full, more accurate analysis."
    )

    cycle = st.number_input("Current Cycle", min_value=1, value=100, key="manual_cycle")

    sensor_values = {}
    cols = st.columns(3)
    for i, sensor in enumerate(sensor_columns):
        with cols[i % 3]:
            sensor_values[sensor] = st.number_input(sensor, value=0.0, format="%.4f", key=f"manual_{sensor}")

    if st.button("🔍 Analyze Machine Health", type="primary", key="manual_analyze"):
        input_row = pd.DataFrame([sensor_values])
        for feature in feature_columns:
            if feature not in input_row.columns:
                input_row[feature] = 0.0
        input_row = input_row[feature_columns]

        predicted_rul, predicted_risk, anomaly_status, anomaly_score = run_prediction(input_row)
        show_results(
            dataset, engine_id, cycle,
            predicted_rul, predicted_risk, anomaly_status, anomaly_score,
        )
