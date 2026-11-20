"""
app.py
------
BattSense AI — Streamlit web application
Lithium-Ion Battery Temperature Prediction

Run with:
    streamlit run app.py

Requires models/ and results/ to exist — run main.py first if they don't.
"""

import os
import numpy as np
import streamlit as st
import joblib
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Feature ranges from the NASA training dataset ─────────────────────────────
# Used for input validation and slider bounds.
RANGES = {
    "voltage_v":      (1.737,    4.233,   3.497),   # min, max, default
    "current_a":      (-2.030,   0.014,  -2.000),
    "elapsed_time_s": (0.0,   3690.0,    900.0),
}

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BattSense AI",
    page_icon="🔋",
    layout="wide",
)

# ── Load models (cached — loaded once per session) ────────────────────────────
@st.cache_resource
def load_models():
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    lr  = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
    sc  = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    return xgb, lr, sc

models_ok = os.path.isdir(MODELS_DIR) and all(
    os.path.isfile(os.path.join(MODELS_DIR, f))
    for f in ["xgb_model.joblib", "lr_model.joblib", "scaler.joblib"]
)

if models_ok:
    xgb_model, lr_model, scaler = load_models()
else:
    xgb_model = lr_model = scaler = None

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔋 BattSense AI")
st.subheader("Lithium-Ion Battery Temperature Prediction")
st.markdown(
    "Predicts battery temperature (°C) from real-time electrical operating "
    "parameters using machine learning trained on NASA experimental data."
)
st.divider()

# ── Sidebar — navigation ──────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Temperature", "📊 Model Performance", "📈 Result Graphs", "ℹ️ About Dataset"],
)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Predict Temperature
# ════════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Temperature":

    st.header("🔮 Predict Battery Temperature")
    st.markdown(
        "Enter the current operating conditions of the battery. "
        "Values must fall within the ranges observed in the NASA training dataset."
    )

    if not models_ok:
        st.error(
            "Model files not found in `models/`. "
            "Run `python main.py` first to train and save the models."
        )
        st.stop()

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        v_min, v_max, v_def = RANGES["voltage_v"]
        voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=float(v_min),
            max_value=float(v_max),
            value=float(v_def),
            step=0.001,
            format="%.3f",
            help=f"Training range: {v_min:.3f} – {v_max:.3f} V",
        )

    with col2:
        c_min, c_max, c_def = RANGES["current_a"]
        current = st.number_input(
            "Battery Current (A)",
            min_value=float(c_min),
            max_value=float(c_max),
            value=float(c_def),
            step=0.001,
            format="%.3f",
            help=f"Training range: {c_min:.3f} – {c_max:.3f} A  "
                 f"(negative = discharging)",
        )

    with col3:
        t_min, t_max, t_def = RANGES["elapsed_time_s"]
        elapsed = st.number_input(
            "Elapsed Discharge Time (s)",
            min_value=float(t_min),
            max_value=float(t_max),
            value=float(t_def),
            step=1.0,
            format="%.1f",
            help=f"Training range: {t_min:.0f} – {t_max:.0f} s",
        )

    # ── Validation ────────────────────────────────────────────────────────────
    warnings_list = []
    v_min, v_max, _ = RANGES["voltage_v"]
    c_min, c_max, _ = RANGES["current_a"]
    t_min, t_max, _ = RANGES["elapsed_time_s"]

    if not (v_min <= voltage <= v_max):
        warnings_list.append(
            f"Voltage {voltage:.3f} V is outside the training range "
            f"({v_min:.3f} – {v_max:.3f} V). Prediction may be unreliable."
        )
    if not (c_min <= current <= c_max):
        warnings_list.append(
            f"Current {current:.3f} A is outside the training range "
            f"({c_min:.3f} – {c_max:.3f} A). Prediction may be unreliable."
        )
    if not (t_min <= elapsed <= t_max):
        warnings_list.append(
            f"Elapsed time {elapsed:.1f} s is outside the training range "
            f"({t_min:.0f} – {t_max:.0f} s). Prediction may be unreliable."
        )

    for w in warnings_list:
        st.warning(w)

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("⚡ Predict Temperature", type="primary", use_container_width=True):

        X = np.array([[voltage, current, elapsed]])

        # XGBoost — raw features
        xgb_pred = float(xgb_model.predict(X)[0])

        # Linear Regression — scaled features
        X_scaled  = scaler.transform(X)
        lr_pred   = float(np.dot(X_scaled, lr_model.coef_) + lr_model.intercept_)

        st.markdown("---")
        st.subheader("Prediction Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric(
                label="🏆 XGBoost Predicted Temperature",
                value=f"{xgb_pred:.2f} °C",
                help="Primary model — lower MAE on held-out test cycles.",
            )

        with res_col2:
            st.metric(
                label="📐 Linear Regression Predicted Temperature",
                value=f"{lr_pred:.2f} °C",
                help="Baseline model.",
            )

        # Context note
        st.info(
            "These are estimates of the battery cell temperature at the entered "
            "operating state. They do not predict future overheating or control "
            "any safety system."
        )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.header("📊 Model Performance")
    st.markdown(
        "Metrics evaluated on **128 held-out test cycles (38,143 rows)** "
        "that were not used during training."
    )

    # Hard-coded from results/metrics.txt — these are the real values
    metrics = {
        "Linear Regression": {"MAE": 0.9104, "RMSE": 1.2079, "R²": 0.9075},
        "XGBoost":           {"MAE": 0.5045, "RMSE": 0.6795, "R²": 0.9707},
    }

    # Fallback: parse metrics.txt if it exists (keeps app in sync if main.py is re-run)
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path) as f:
                lines = f.readlines()
            parsed, current_model = {}, None
            for line in lines:
                line = line.strip()
                if line in ("Linear Regression", "XGBoost"):
                    current_model = line
                    parsed[current_model] = {}
                elif current_model and line.startswith("MAE"):
                    parsed[current_model]["MAE"] = float(line.split(":")[1].strip().split()[0])
                elif current_model and line.startswith("RMSE"):
                    parsed[current_model]["RMSE"] = float(line.split(":")[1].strip().split()[0])
                elif current_model and line.startswith("R2"):
                    parsed[current_model]["R²"] = float(line.split(":")[1].strip())
            if parsed:
                metrics = parsed
        except Exception:
            pass  # fall back to hard-coded values

    col1, col2 = st.columns(2)

    for col, (model_name, m) in zip([col1, col2], metrics.items()):
        with col:
            st.subheader(model_name)
            st.metric("MAE",  f"{m['MAE']:.4f} °C",  help="Mean Absolute Error")
            st.metric("RMSE", f"{m['RMSE']:.4f} °C", help="Root Mean Squared Error")
            st.metric("R²",   f"{m['R²']:.4f}",       help="Coefficient of Determination")

    st.divider()

    # Verdict
    lr_mae  = metrics.get("Linear Regression", {}).get("MAE", 1.0)
    xgb_mae = metrics.get("XGBoost",           {}).get("MAE", 0.5)
    better  = "XGBoost" if xgb_mae < lr_mae else "Linear Regression"
    pct     = abs(lr_mae - xgb_mae) / lr_mae * 100

    st.success(
        f"**{better}** achieves lower MAE on the test cycles "
        f"({pct:.1f} % improvement over the baseline)."
    )

    st.markdown("### What these metrics mean")
    st.markdown(
        "- **MAE** — average absolute difference between predicted and actual temperature. "
        "An MAE of 0.50 °C means predictions are off by about half a degree on average.\n"
        "- **RMSE** — similar to MAE but penalises large errors more. "
        "Lower is better.\n"
        "- **R²** — proportion of temperature variance explained by the model. "
        "1.0 is perfect; 0.97 means the model explains 97 % of the variation."
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Result Graphs
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Result Graphs":

    st.header("📈 Result Graphs")

    graphs = [
        (
            "actual_vs_predicted_xgboost.png",
            "Actual vs Predicted Temperature — XGBoost",
            "Each point is one test-set time step. Points lying on the red dashed "
            "line represent a perfect prediction. The tight cluster along the diagonal "
            "shows the XGBoost model closely tracks the measured temperature across "
            "the full 22 – 42 °C range.",
        ),
        (
            "mae_comparison.png",
            "MAE Comparison — Linear Regression vs XGBoost",
            "XGBoost reduces mean absolute error by 44.6 % compared to the Linear "
            "Regression baseline, confirming that the relationship between electrical "
            "parameters and battery temperature is non-linear.",
        ),
        (
            "feature_importance_xgboost.png",
            "XGBoost Feature Importance",
            "Elapsed discharge time is the most influential feature — temperature "
            "accumulates continuously over a cycle, so the time offset within a cycle "
            "carries the most predictive information. Voltage is second because "
            "terminal-voltage drop is closely linked to internal resistance and heat "
            "generation. Current is third; it is relatively constant during standard "
            "discharge so it varies less across the dataset.",
        ),
    ]

    results_missing = not os.path.isdir(RESULTS_DIR)

    for filename, title, interpretation in graphs:
        path = os.path.join(RESULTS_DIR, filename)
        st.subheader(title)
        if os.path.isfile(path):
            img = Image.open(path)
            st.image(img, use_container_width=False)
        else:
            st.warning(f"`{filename}` not found. Run `python main.py` to generate it.")
        st.caption(interpretation)
        st.markdown("")

    if results_missing:
        st.error("The `results/` folder is missing. Run `python main.py` to generate all graphs.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About Dataset
# ════════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Dataset":

    st.header("ℹ️ About the Dataset")

    st.markdown(
        """
### NASA Li-ion Battery Aging Dataset

This project uses real experimental measurements from the
**NASA Prognostics Center of Excellence (PCoE)** battery aging dataset.

Four 18650 lithium-ion cells were repeatedly charged and discharged under
controlled laboratory conditions until they reached end of life. This project
uses only the **discharge cycles** from batteries **B0005, B0006, B0007, and B0018**.

| Property | Value |
|---|---|
| Total rows | 185,721 |
| Discharge cycles | 636 |
| Batteries | B0005, B0006, B0007, B0018 |
| Ambient temperature | 24 °C (constant laboratory chamber) |
| Target variable | Measured battery temperature (°C) |

### Input features

| Feature | Unit | Description |
|---|---|---|
| Voltage | V | Terminal voltage of the cell; falls as discharge progresses |
| Current | A | Discharge current (negative convention); proportional to heat generation via I²R |
| Elapsed time | s | Time elapsed since the start of the discharge cycle |

### Why ambient temperature is excluded

The NASA laboratory held the environmental chamber at exactly 24.0 °C throughout
all experiments. Because it never varies, it carries no predictive information and
was excluded from the model features.

### Limitations

- **Laboratory data only.** These measurements come from individual cells under
  controlled, constant-current discharge. A real EV battery pack experiences
  variable loads, cell-to-cell thermal gradients, and active cooling — none of
  which are captured here.
- **Fixed ambient temperature.** The model cannot generalise to environments
  significantly warmer or cooler than 24 °C.
- **Operating range.** Predictions outside the training voltage (1.74 – 4.23 V),
  current (−2.03 – 0.01 A), and time (0 – 3690 s) ranges are unreliable.
- **Point-in-time estimate.** The model estimates temperature at the current
  operating state. It does not predict future overheating or control any safety
  system.

### Dataset source

NASA Prognostics Center of Excellence (PCoE) Data Repository —
*Battery Data Set*, K. A. Goebel et al.
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
"""
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "BattSense AI · ME Third-Year Assignment · "
    "Data: NASA PCoE Li-ion Battery Aging Dataset · "
    "Models: Linear Regression & XGBoost"
)
