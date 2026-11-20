"""
main.py
-------
Lithium-Ion Battery Temperature Prediction Using XGBoost
ME Third-Year Assignment — 15 Marks

Dataset : battery_temperature_data.csv
         185,721 discharge-cycle measurements from NASA Li-ion batteries
         B0005, B0006, B0007, B0018.

Pipeline
--------
1. Load and validate the CSV.
2. Split by complete discharge cycles (80 / 20) — no row-level random split.
3. Train  Linear Regression  (baseline)
          XGBoost Regressor   (main model)
4. Evaluate on held-out test cycles: MAE, RMSE, R².
5. Save three plots and a metrics summary to results/.
6. Print an engineering interpretation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from xgboost               import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "battery_temperature_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RANDOM_SEED = 42
FEATURES    = ["voltage_v", "current_a", "elapsed_time_s"]
TARGET      = "battery_temp_c"

# ── 1. Load and validate ───────────────────────────────────────────────────────
print("=" * 60)
print("Lithium-Ion Battery Temperature Prediction — XGBoost")
print("=" * 60)

if not os.path.isfile(CSV_PATH):
    print(f"\nERROR: {CSV_PATH} not found.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"\nLoaded: {len(df):,} rows × {df.shape[1]} columns")

# Check required columns
required = FEATURES + [TARGET, "battery_id", "cycle_id"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    print(f"ERROR: Missing columns: {missing_cols}")
    sys.exit(1)

# No nulls, no duplicates — confirm
assert df[required].isnull().sum().sum() == 0, "Unexpected NaNs found."
assert df.duplicated().sum() == 0,             "Unexpected duplicate rows found."

print(f"Features : {FEATURES}")
print(f"Target   : {TARGET}  "
      f"(range {df[TARGET].min():.2f} – {df[TARGET].max():.2f} °C, "
      f"mean {df[TARGET].mean():.2f} °C)")

# Confirm ambient temperature is constant (excluded by design)
assert df["ambient_temp_c"].nunique() == 1, "ambient_temp_c is not constant — review exclusion."
print(f"\nambient_temp_c = {df['ambient_temp_c'].iloc[0]:.1f} °C throughout "
      f"(constant lab condition — excluded from features).")

# ── 2. Cycle-aware train / test split ─────────────────────────────────────────
# Unit of splitting: unique (battery_id, cycle_id) pair.
# This ensures no discharge cycle is shared between train and test sets,
# preventing leakage of correlated time-series measurements.

cycles = (df[["battery_id", "cycle_id"]]
          .drop_duplicates()
          .sample(frac=1, random_state=RANDOM_SEED)
          .reset_index(drop=True))

n_train = int(len(cycles) * 0.80)
train_cycles = cycles.iloc[:n_train]
test_cycles  = cycles.iloc[n_train:]

train_df = df.merge(train_cycles, on=["battery_id", "cycle_id"])
test_df  = df.merge(test_cycles,  on=["battery_id", "cycle_id"])

# Assert zero overlap
overlap = (set(zip(train_df["battery_id"], train_df["cycle_id"])) &
           set(zip(test_df["battery_id"],  test_df["cycle_id"])))
assert len(overlap) == 0, f"Cycle overlap detected: {overlap}"

print(f"\nTotal cycles : {len(cycles)}")
print(f"Train cycles : {len(train_cycles)}  ({len(train_cycles)/len(cycles)*100:.1f} %)"
      f"  → {len(train_df):,} rows")
print(f"Test  cycles : {len(test_cycles)}   ({len(test_cycles)/len(cycles)*100:.1f} %)"
      f"  → {len(test_df):,} rows")

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURES].values
y_test  = test_df[TARGET].values

# ── 3. Preprocessing — fit on training data only ──────────────────────────────
# StandardScaler required for Linear Regression; fitted only on train set
# so test-set statistics never influence the transformation.
scaler = StandardScaler()
# np.ascontiguousarray avoids a numpy 2.0 / sklearn matmul warning on macOS
X_train_scaled = np.ascontiguousarray(scaler.fit_transform(X_train))
X_test_scaled  = np.ascontiguousarray(scaler.transform(X_test))

# ── 4. Train models ───────────────────────────────────────────────────────────
print("\n--- Training Linear Regression (baseline) ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

print("--- Training XGBoost Regressor ---")
# Conservative hyperparameters — no exhaustive tuning
xgb_model = XGBRegressor(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = RANDOM_SEED,
    verbosity        = 0,
)
# Tree models are invariant to feature scaling — use raw values
xgb_model.fit(X_train, y_train)

# ── Save models and scaler for use by app.py ──────────────────────────────────
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
joblib.dump(lr_model,  os.path.join(MODELS_DIR, "lr_model.joblib"))
joblib.dump(scaler,    os.path.join(MODELS_DIR, "scaler.joblib"))
print("Models saved to models/")

# ── 5. Evaluate on held-out test cycles ───────────────────────────────────────
# np.dot avoids a numpy 2.0 / Apple Silicon bug where the @ operator emits
# spurious RuntimeWarnings on certain array shapes (numpy gh-26842).
# Results are numerically identical to lr_model.predict().
y_pred_lr  = np.dot(X_test_scaled, lr_model.coef_) + lr_model.intercept_
y_pred_xgb = xgb_model.predict(X_test)

def compute_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

res_lr  = compute_metrics("Linear Regression", y_test, y_pred_lr)
res_xgb = compute_metrics("XGBoost",           y_test, y_pred_xgb)

print("\n" + "=" * 60)
print("Evaluation — Held-Out Test Cycles")
print("=" * 60)
for r in [res_lr, res_xgb]:
    print(f"\n{r['model']}")
    print(f"  MAE  : {r['MAE']:.4f} °C")
    print(f"  RMSE : {r['RMSE']:.4f} °C")
    print(f"  R²   : {r['R2']:.4f}")

# Verdict
better = "XGBoost" if res_xgb["MAE"] < res_lr["MAE"] else "Linear Regression"
mae_diff = abs(res_lr["MAE"] - res_xgb["MAE"])
pct_diff = mae_diff / res_lr["MAE"] * 100
print(f"\nVerdict: {better} achieves lower MAE "
      f"(difference: {mae_diff:.4f} °C, {pct_diff:.1f} %).")

# Save metrics
metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Battery Temperature Prediction — Model Evaluation\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset : 185,721 rows, 636 discharge cycles\n")
    f.write(f"Features: {', '.join(FEATURES)}\n")
    f.write(f"Target  : {TARGET}\n\n")
    for r in [res_lr, res_xgb]:
        f.write(f"{r['model']}\n")
        f.write(f"  MAE  : {r['MAE']:.4f} °C\n")
        f.write(f"  RMSE : {r['RMSE']:.4f} °C\n")
        f.write(f"  R2   : {r['R2']:.4f}\n\n")
    f.write(f"Verdict\n-------\n{better} achieves lower MAE.\n")
    f.write(f"MAE difference: {mae_diff:.4f} °C ({pct_diff:.1f} % reduction).\n")
print(f"\nMetrics saved: {metrics_path}")

# ── 6. Visualisations ─────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# -- Plot 1: Actual vs Predicted — XGBoost ------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred_xgb, alpha=0.15, s=3, color="steelblue", label="Test points")
lo = min(y_test.min(), y_pred_xgb.min()) - 0.5
hi = max(y_test.max(), y_pred_xgb.max()) + 0.5
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Battery Temperature (°C)")
ax.set_ylabel("Predicted Battery Temperature (°C)")
ax.set_title(
    f"XGBoost — Actual vs Predicted Temperature\n"
    f"MAE = {res_xgb['MAE']:.3f} °C   R² = {res_xgb['R2']:.4f}"
)
ax.legend(markerscale=5, fontsize=9)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
fig.tight_layout()
p1 = os.path.join(RESULTS_DIR, "actual_vs_predicted_xgboost.png")
fig.savefig(p1)
plt.close(fig)
print(f"Saved: {p1}")

# -- Plot 2: MAE comparison ---------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
model_names = ["Linear\nRegression", "XGBoost"]
maes        = [res_lr["MAE"], res_xgb["MAE"]]
colors      = ["#4C72B0", "#DD8452"]
bars = ax.bar(model_names, maes, color=colors, width=0.45,
              edgecolor="black", linewidth=0.6)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f} °C", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Mean Absolute Error (°C)")
ax.set_title("MAE Comparison — Linear Regression vs XGBoost")
ax.set_ylim(0, max(maes) * 1.3)
fig.tight_layout()
p2 = os.path.join(RESULTS_DIR, "mae_comparison.png")
fig.savefig(p2)
plt.close(fig)
print(f"Saved: {p2}")

# -- Plot 3: XGBoost feature importance ---------------------------------------
importances  = xgb_model.feature_importances_
sorted_idx   = np.argsort(importances)          # ascending for horizontal bar
sorted_feats = [FEATURES[i] for i in sorted_idx]
sorted_imps  = importances[sorted_idx]

label_map = {
    "voltage_v":      "Voltage (V)",
    "current_a":      "Current (A)",
    "elapsed_time_s": "Elapsed Time (s)",
}
labels = [label_map[f] for f in sorted_feats]

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.barh(labels, sorted_imps, color="#2ca02c",
               edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, sorted_imps):
    ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance (XGBoost gain)")
ax.set_title("XGBoost Feature Importance")
ax.set_xlim(0, max(sorted_imps) * 1.25)
fig.tight_layout()
p3 = os.path.join(RESULTS_DIR, "feature_importance_xgboost.png")
fig.savefig(p3)
plt.close(fig)
print(f"Saved: {p3}")

# ── 7. Engineering summary ────────────────────────────────────────────────────
top_feat  = label_map[FEATURES[int(np.argmax(importances))]]
feat_rank = " > ".join(label_map[FEATURES[i]] for i in np.argsort(importances)[::-1])

summary = f"""
{'='*60}
ENGINEERING SUMMARY
{'='*60}

1. WHY BATTERY TEMPERATURE PREDICTION MATTERS
   Lithium-ion cells degrade faster and risk thermal runaway when
   operated above safe temperatures (~60 °C for 18650 cells).
   Estimating cell temperature from electrical measurements lets
   a battery management system detect abnormal heating without
   adding a physical sensor to every cell in a pack.

2. WHAT THE INPUT FEATURES REPRESENT
   Voltage (V)       — terminal voltage falls as the cell discharges
                       and as internal resistance rises with age;
                       sharp drops signal high heat generation.
   Current (A)       — Ohmic heating is proportional to I²R; a larger
                       discharge current produces more heat per second.
   Elapsed Time (s)  — captures position within the discharge cycle;
                       temperature accumulates continuously so a
                       measurement taken late in a cycle will be warmer
                       than one taken at the start, even at the same
                       voltage and current.

   ambient_temp_c was excluded: it equals exactly 24.0 °C for every
   row (NASA laboratory condition), so it carries zero predictive
   information.

3. HOW XGBOOST LEARNS
   XGBoost builds decision trees sequentially. Each tree fits the
   residual error left by all previous trees (gradient boosting).
   Unlike Linear Regression, it captures non-linear interactions —
   for example, the fact that temperature rises more sharply at high
   current late in a discharge cycle — without explicit feature
   engineering.

4. MODEL PERFORMANCE (held-out test cycles — {len(test_cycles)} cycles, {len(test_df):,} rows)
   Linear Regression — MAE: {res_lr['MAE']:.4f} °C  RMSE: {res_lr['RMSE']:.4f} °C  R²: {res_lr['R2']:.4f}
   XGBoost           — MAE: {res_xgb['MAE']:.4f} °C  RMSE: {res_xgb['RMSE']:.4f} °C  R²: {res_xgb['R2']:.4f}

   {better} achieves lower MAE on the test cycles
   (difference: {mae_diff:.4f} °C, {pct_diff:.1f} %).

5. MOST INFLUENTIAL FEATURES
   Ranking by XGBoost gain: {feat_rank}
   Most important: {top_feat}

6. LIMITATIONS
   a) Constant ambient temperature: the NASA lab held the chamber at
      24 °C throughout; this model cannot generalise to environments
      with varying ambient conditions.
   b) Single-cell laboratory data: measurements come from individual
      18650 cells under controlled discharge. A real EV pack behaves
      differently — cells interact thermally, cooling systems actively
      remove heat, and gradients form across modules.
   c) Operating range: predictions outside the training voltage and
      current ranges are unreliable.
   d) This model estimates temperature at the current operating state.
      It does not predict future overheating, control any cooling
      system, or guarantee battery safety.
{'='*60}
"""
print(summary)

summary_path = os.path.join(RESULTS_DIR, "engineering_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary.strip())
print(f"Summary saved : {summary_path}")
print("All results written to results/")
