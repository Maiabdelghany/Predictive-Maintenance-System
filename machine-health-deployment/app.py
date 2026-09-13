import streamlit as st
import pandas as pd
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

# ============================================
# Title
# ============================================

st.title("⚙️ Machine Health Monitoring System")
st.write("Predictive Maintenance Dashboard using XGBoost and Isolation Forest")

# ============================================
# Sidebar
# ============================================

st.sidebar.header("Engine Information")

dataset = st.sidebar.selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"])
engine_id = st.sidebar.text_input("Engine ID", value=f"{dataset}_1")
cycle = st.sidebar.number_input("Current Cycle", min_value=1, value=100)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Note: this demo takes a single snapshot of raw sensor readings. "
    "Rolling/trend features (which the models were also trained on) "
    "are set to 0 by default here since they need sensor history over "
    "time, not a single reading."
)

# ============================================
# Sensor Input
# ============================================

st.subheader("Sensor Measurements")

sensor_columns = [
    col for col in feature_columns
    if col.startswith("sensor")
    and "_rolling_" not in col
    and "_rate_of_change" not in col
]

sensor_values = {}
cols = st.columns(3)

for i, sensor in enumerate(sensor_columns):
    with cols[i % 3]:
        sensor_values[sensor] = st.number_input(sensor, value=0.0, format="%.4f")

# ============================================
# Prediction
# ============================================

if st.button("🔍 Analyze Machine Health", type="primary"):

    # Build input row matching the exact training feature set/order
    input_data = pd.DataFrame([sensor_values])

    for feature in feature_columns:
        if feature not in input_data.columns:
            input_data[feature] = 0.0

    input_data = input_data[feature_columns]

    # RUL prediction
    predicted_rul = max(float(xgb_rul_model.predict(input_data)[0]), 0)

    # Risk prediction
    predicted_risk_encoded = xgb_risk_classifier.predict(input_data)[0]
    predicted_risk = label_encoder.inverse_transform([predicted_risk_encoded])[0]

    # Anomaly detection
    anomaly_prediction = isolation_forest.predict(input_data)[0]
    anomaly_score = isolation_forest.decision_function(input_data)[0]
    anomaly_status = "Anomaly" if anomaly_prediction == -1 else "Normal"

    # Dashboard
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
