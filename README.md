# ⚙️ Machine Health Monitoring System

**A predictive maintenance solution for turbofan engines** — turning raw sensor data into remaining-life forecasts, risk levels, and anomaly alerts.

🔗 **Live app:** [predictive-maintenance-system.streamlit.app](https://predictive-maintenance-system-ljioah2est9k9ydkpufqhq.streamlit.app)

---

## 📌 Overview

Industrial equipment — jet engines, turbines, factory machinery — degrades gradually before it fails. Most organizations handle this reactively (fix it after it breaks) or on a fixed schedule (service it every N months regardless of condition). Both waste money, and both ignore the one thing the machine is constantly telling us: **its sensor readings**.

This project builds a complete predictive maintenance pipeline on top of NASA's C-MAPSS turbofan engine dataset, answering three operational questions from the same sensor data:

| Question | Model | Output |
|---|---|---|
| **How long until failure?** | Random Forest / XGBoost Regressor | Remaining Useful Life (RUL), in cycles |
| **How urgent is it?** | Random Forest / XGBoost Classifier | Healthy / Warning / High Risk |
| **How unusual is this behavior?** | Isolation Forest (unsupervised) | Anomaly status + anomaly score |

The final models are deployed in a live, interactive **Streamlit** dashboard with continuous deployment from GitHub.

---

## 🎯 Why This Project

Predictive maintenance sits at the intersection of real business impact and a rich ML problem — it combines regression, classification, and unsupervised anomaly detection on the same real-world data.

- **Cost savings** — maintenance is scheduled only when actually needed, not on a fixed calendar or after failure.
- **Safety** — catching degradation early prevents catastrophic failures in critical equipment like aircraft engines.
- **Uptime** — unplanned downtime in manufacturing or aviation costs far more than the part that failed.
- **Industry relevance** — one of the most widely adopted industrial AI use cases, used by airlines, manufacturers, and energy companies.

---

## 📊 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) — the standard benchmark for predictive maintenance research. It simulates turbofan jet engines run to failure under realistic operating conditions, recording **21 sensor channels** and **3 operational settings** at every cycle.

| Sub-dataset | Train rows | Test rows | Engines | Avg. engine life (cycles) |
|---|---|---|---|---|
| FD001 | 20,631 | 13,096 | 100 | 206.3 |
| FD002 | 53,759 | 33,991 | 259 | 206.8 |
| FD003 | 24,720 | 16,596 | 100 | 247.2 |
| FD004 | 61,249 | 41,214 | 248 | 246.0 |

The four sub-datasets vary in operating conditions and fault modes, letting models be tested under increasingly realistic, complex conditions.

---

## 🔧 Methodology

### 1. Data Preparation
- Merged all four C-MAPSS sub-datasets, tagging each engine with a unique ID.
- Computed ground-truth **Remaining Useful Life (RUL)** per row: `(max cycle for engine) − (current cycle)`.
- Checked for missing values and duplicates — none found in the raw data.

### 2. Feature Engineering
A single snapshot can't show whether a sensor is drifting. For each of the 21 sensors:
- Rolling **mean** and **standard deviation** over 5-cycle and 20-cycle windows (short- and long-term trend).
- **Rate of change** between consecutive cycles.

This expanded the feature set from **24 raw columns → 129 engineered features** used to train every model.

### 3. Models Trained

| Model | Algorithm | Task |
|---|---|---|
| RUL Regression | `RandomForestRegressor`, `XGBRegressor` | Predict cycles remaining |
| Risk Classification | `RandomForestClassifier`, `XGBClassifier` | Classify Healthy (RUL > 30) / Warning (11–30) / High Risk (≤ 10) |
| Anomaly Detection | `IsolationForest` | Flag statistically unusual sensor patterns (unsupervised) |

**XGBoost** was selected as the final production model for RUL and Risk based on validation performance.

### 4. Why These Models
- **Random Forest (baseline)** — robust, low-effort starting point for tabular data; handles non-linear relationships without feature scaling and rarely overfits badly.
- **XGBoost (final model)** — sequential gradient boosting consistently outperforms Random Forest on structured data; matched or beat RF's MAE/RMSE on RUL and gave stronger precision/recall on the High Risk class, while training fast enough to iterate across 129 features and 4 datasets.
- **Isolation Forest (anomaly detection)** — failures are rare, so there are almost no labeled anomaly examples. Isolation Forest is unsupervised: it learns what *normal* looks like and flags points that are unusually easy to isolate — a good fit for plentiful-normal / few-failure data.
- **Three separate models instead of one** — RUL (a number), Risk (a category), and Anomaly (a statistical outlier flag) are conceptually different questions. Keeping them separate makes each simpler to train, easier to evaluate, and more interpretable when outputs disagree.

---

## 📈 Results

### RUL Prediction (XGBoost) — final test set

| Dataset | MAE (cycles) | RMSE (cycles) |
|---|---|---|
| FD001 | 21.10 | 28.88 |
| FD002 | 23.01 | 30.38 |
| FD003 | 32.95 | 46.06 |
| FD004 | 28.56 | 37.33 |

Predictions land within roughly 21–33 cycles of true remaining life — strong given engines run 128 to 500+ cycles.

### Risk Classification (XGBoost) — final test set

| Dataset | Accuracy | High Risk F1 | Warning F1 |
|---|---|---|---|
| FD001 | 91.0% | 0.71 | 0.76 |
| FD002 | 82.2% | 0.67 | 0.51 |
| FD003 | 95.0% | 1.00 | 0.84 |
| FD004 | 89.9% | 0.76 | 0.73 |

Overall accuracy stayed 82–95%. The rarer, operationally critical **High Risk** class was detected reliably.

### Anomaly Detection (Isolation Forest)

- **FD001–FD003:** 0 anomalies flagged in the final test snapshots.
- **FD004** (most complex dataset): **2 / 248 engines (0.81%)** flagged anomalous — both also showed elevated risk, supporting that the anomaly signal picks up genuinely unusual degradation patterns rather than noise.

---

## 🚀 Deployment

The three trained models, label encoder, and 129 feature columns were saved with `joblib` and packaged into a **Streamlit** web dashboard, deployed on **Streamlit Community Cloud** and connected to GitHub for continuous deployment — any push to the repo redeploys the live app automatically within about a minute.

**Trained models → GitHub Repo → Streamlit Community Cloud → Live Dashboard**

### Dashboard Features
- **CSV Upload Mode** — upload a full sensor-reading history for one engine; the app computes the same rolling/rate-of-change features used in training and predicts on the latest cycle.
- **Manual Entry Mode** — a quick single-reading test for demos, with a clear note that trend features default to zero without a history.
- **Live Results** — predicted RUL, Risk level, Anomaly status, and Anomaly score, shown with color-coded alerts.

---

## 🗂️ Project Structure

```
├── Predictive_Maintenance_System.ipynb   # Data prep, feature engineering, training, evaluation
├── app.py                                # Streamlit dashboard (loads saved models, serves predictions)
├── xgb_rul_model.pkl                     # Trained XGBoost RUL regressor
├── xgb_risk_classifier.pkl               # Trained XGBoost risk classifier
├── isolation_forest.pkl                  # Trained Isolation Forest anomaly detector
├── label_encoder.pkl                     # Risk label encoder
├── feature_columns.pkl                   # Ordered list of the 129 engineered feature columns
├── requirements.txt                      # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

- **Data & ML:** Python, pandas, NumPy, scikit-learn, XGBoost
- **Model persistence:** joblib
- **Dashboard / deployment:** Streamlit, Streamlit Community Cloud, GitHub (CI/CD)

