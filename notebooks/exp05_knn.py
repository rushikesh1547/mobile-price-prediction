"""
==============================================================================
 Experiment 5 - k-Nearest Neighbors (k-NN)
 Used Mobile Phone Price Prediction
==============================================================================

AIM:
  (a) Classify price tier with KNeighborsClassifier.
  (b) Predict price with KNeighborsRegressor.
  (c) Evaluate K from 1-50, distance metrics, weighting, and scaling effects.

THEORY:
  k-NN is a non-parametric, instance-based method.
  - Classification: majority vote of K nearest neighbors.
  - Regression: average (or weighted average) of K nearest neighbors.
  - Distance metrics: Euclidean, Manhattan, Minkowski.
  - Small K = complex boundary (overfit), large K = smooth (underfit).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.utils import setup_project, print_section, print_subsection, save_fig, RANDOM_STATE
from src.preprocessing import prepare_classification_data, prepare_regression_data
from src.evaluate import classification_metrics, regression_metrics, print_confusion_matrix

setup_project()
print_section("Experiment 5: k-Nearest Neighbors")

# ── Data ─────────────────────────────────────────────────────────────────────
# Classification data (scaled)
(Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test,
 c_feats, c_scaler, le) = prepare_classification_data()

# Regression data (scaled)
(Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test,
 r_feats, r_scaler) = prepare_regression_data()

# Unscaled classification data for comparison
(Xc_train_u, Xc_val_u, Xc_test_u, yc_train_u, yc_val_u, yc_test_u,
 _, _, _) = prepare_classification_data(scale=False)

print(f"Classification - Train: {Xc_train.shape}, Val: {Xc_val.shape}")
print(f"Regression - Train: {Xr_train.shape}, Val: {Xr_val.shape}")

# ======================================================================
# 5A: K SWEEP FOR CLASSIFICATION
# ======================================================================
print_section("5A: K Sweep - Classification (Accuracy vs K)")

K_range = range(1, 51)
train_accs, val_accs = [], []

for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xc_train, yc_train)
    train_accs.append(knn.score(Xc_train, yc_train))
    val_accs.append(knn.score(Xc_val, yc_val))

best_k_cls = list(K_range)[np.argmax(val_accs)]
print(f"  Best K (classification): {best_k_cls} (Val Accuracy: {max(val_accs):.4f})")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(list(K_range), train_accs, 'b-', label='Train Accuracy', linewidth=1.5, alpha=0.7)
ax.plot(list(K_range), val_accs, 'r-', label='Validation Accuracy', linewidth=1.5)
ax.axvline(best_k_cls, color='green', linestyle='--', label=f'Best K={best_k_cls}')
ax.set_xlabel('K (Number of Neighbors)')
ax.set_ylabel('Accuracy')
ax.set_title('k-NN Classification: Accuracy vs K')
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(fig, 'exp05_k_sweep_classification.png')

# ======================================================================
# 5B: K SWEEP FOR REGRESSION
# ======================================================================
print_section("5B: K Sweep - Regression (RMSE vs K)")

train_rmses, val_rmses = [], []
for k in K_range:
    knn_r = KNeighborsRegressor(n_neighbors=k)
    knn_r.fit(Xr_train, yr_train)
    tr_pred = knn_r.predict(Xr_train)
    vl_pred = knn_r.predict(Xr_val)
    train_rmses.append(np.sqrt(np.mean((yr_train - tr_pred)**2)))
    val_rmses.append(np.sqrt(np.mean((yr_val - vl_pred)**2)))

best_k_reg = list(K_range)[np.argmin(val_rmses)]
print(f"  Best K (regression): {best_k_reg} (Val RMSE: {min(val_rmses):.4f})")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(list(K_range), train_rmses, 'b-', label='Train RMSE', linewidth=1.5, alpha=0.7)
ax.plot(list(K_range), val_rmses, 'r-', label='Validation RMSE', linewidth=1.5)
ax.axvline(best_k_reg, color='green', linestyle='--', label=f'Best K={best_k_reg}')
ax.set_xlabel('K (Number of Neighbors)')
ax.set_ylabel('RMSE')
ax.set_title('k-NN Regression: RMSE vs K')
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(fig, 'exp05_k_sweep_regression.png')

# ======================================================================
# 5C: DISTANCE METRICS COMPARISON
# ======================================================================
print_section("5C: Distance Metrics Comparison")

metrics_list = ['euclidean', 'manhattan', 'minkowski']
dist_results = []

for metric in metrics_list:
    knn = KNeighborsClassifier(n_neighbors=best_k_cls, metric=metric)
    knn.fit(Xc_train, yc_train)
    y_pred = knn.predict(Xc_val)
    m = classification_metrics(yc_val, y_pred, label=f"K={best_k_cls}, {metric}")
    dist_results.append({'Metric': metric, **m})

print("\n  Distance metric comparison:")
print(pd.DataFrame(dist_results).to_string(index=False))

# ======================================================================
# 5D: UNIFORM vs DISTANCE WEIGHTING
# ======================================================================
print_section("5D: Uniform vs Distance Weighting")

weight_results = []
for w in ['uniform', 'distance']:
    knn = KNeighborsClassifier(n_neighbors=best_k_cls, weights=w)
    knn.fit(Xc_train, yc_train)
    y_pred = knn.predict(Xc_val)
    m = classification_metrics(yc_val, y_pred, label=f"K={best_k_cls}, {w}")
    weight_results.append({'Weighting': w, **m})

print("\n  Weighting comparison:")
print(pd.DataFrame(weight_results).to_string(index=False))

# ======================================================================
# 5E: SCALING EFFECT
# ======================================================================
print_section("5E: Effect of Feature Scaling")

# Unscaled
knn_u = KNeighborsClassifier(n_neighbors=best_k_cls)
knn_u.fit(Xc_train_u, yc_train_u)
acc_unscaled = knn_u.score(Xc_val_u, yc_val_u)

# Scaled
knn_s = KNeighborsClassifier(n_neighbors=best_k_cls)
knn_s.fit(Xc_train, yc_train)
acc_scaled = knn_s.score(Xc_val, yc_val)

print(f"  Unscaled accuracy: {acc_unscaled:.4f}")
print(f"  Scaled accuracy:   {acc_scaled:.4f}")
print(f"  Improvement:       {(acc_scaled - acc_unscaled)*100:.2f} percentage points")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Unscaled', 'Scaled'], [acc_unscaled, acc_scaled],
              color=['salmon', 'steelblue'])
ax.set_ylabel('Validation Accuracy')
ax.set_title(f'k-NN (K={best_k_cls}): Effect of Feature Scaling')
ax.set_ylim(min(acc_unscaled, acc_scaled) - 0.05, 1.0)
for bar, val in zip(bars, [acc_unscaled, acc_scaled]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontsize=11)
save_fig(fig, 'exp05_scaling_effect.png')

# ======================================================================
# 5F: BEST MODEL - TEST SET EVALUATION
# ======================================================================
print_section("5F: Best Model - Test Set Evaluation")

best_knn_cls = KNeighborsClassifier(n_neighbors=best_k_cls, weights='distance')
best_knn_cls.fit(Xc_train, yc_train)
y_pred_test = best_knn_cls.predict(Xc_test)
classification_metrics(yc_test, y_pred_test, label="k-NN Cls Test")
print_confusion_matrix(yc_test, y_pred_test, labels=[0, 1, 2],
                       title=f'k-NN (K={best_k_cls}) Confusion Matrix',
                       save_name='exp05_confusion.png')

best_knn_reg = KNeighborsRegressor(n_neighbors=best_k_reg, weights='distance')
best_knn_reg.fit(Xr_train, yr_train)
yr_pred_test = best_knn_reg.predict(Xr_test)
regression_metrics(yr_test, yr_pred_test, label="k-NN Reg Test")

# ======================================================================
# OBSERVATIONS & CONCLUSION
# ======================================================================
print_section("Observations & Conclusion")
print(f"""
OBSERVATIONS:
1. K=1 overfits (train accuracy 100%, lower validation accuracy).
2. Best K for classification: {best_k_cls}; for regression: {best_k_reg}.
3. As K increases, the decision boundary smoothens (less overfitting).
4. Distance weighting generally improves or matches uniform weighting.
5. Feature SCALING is crucial for k-NN since it relies on distance
   calculations - without scaling, features with larger ranges dominate.
6. Euclidean and Manhattan metrics perform comparably on this data.

UNDER/OVERFITTING:
- K=1: severe overfitting (memorizes training data)
- K=50: underfitting (over-smoothed predictions)
- The optimal K balances these extremes.

CONCLUSION:
k-NN is a simple yet effective non-parametric method. With proper scaling
and K selection, it achieves competitive results for both classification
and regression tasks on the phone price dataset.
""")

print("\n[OK] Experiment 5 complete.\n")
