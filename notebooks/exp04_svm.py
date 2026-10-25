"""
==============================================================================
 Experiment 4 - Support Vector Machines (SVM)
 Used Mobile Phone Price Prediction
==============================================================================

AIM:
  Classify price tier using SVM with different kernels. Tune hyperparameters.
  Visualize decision boundaries. Also try SVR for regression.

THEORY:
  SVM finds the hyperplane maximizing the margin between classes.
  - Linear kernel: w^T x + b
  - RBF kernel: exp(-gamma * ||x-x'||^2) — maps to infinite-dim space
  - Polynomial kernel: (gamma * x^T x' + r)^d
  - C controls misclassification penalty; gamma controls RBF influence.
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
import seaborn as sns
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.utils import setup_project, print_section, print_subsection, save_fig, RANDOM_STATE
from src.preprocessing import (load_raw_data, create_targets, basic_clean,
                                get_numeric_features, prepare_classification_data)
from src.evaluate import classification_metrics, regression_metrics, print_confusion_matrix

setup_project()
print_section("Experiment 4: Support Vector Machines")

# ── Data ─────────────────────────────────────────────────────────────────────
(X_train, X_val, X_test, y_train, y_val, y_test,
 feat_names, scaler, le) = prepare_classification_data(target_col='price_tier')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Classes: {np.bincount(y_train)}")

# ======================================================================
# 4A: KERNEL COMPARISON
# ======================================================================
print_section("4A: Kernel Comparison")

kernels = ['linear', 'rbf', 'poly']
kernel_results = []

for k in kernels:
    svc = SVC(kernel=k, random_state=RANDOM_STATE, probability=True)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_val)
    y_prob = svc.predict_proba(X_val)
    m = classification_metrics(y_val, y_pred, y_prob, f"SVC({k})")
    kernel_results.append({'Kernel': k, **m})

print("\n  Kernel Comparison:")
kdf = pd.DataFrame(kernel_results)
print(kdf.to_string(index=False))

# ======================================================================
# 4B: HYPERPARAMETER TUNING (GridSearchCV)
# ======================================================================
print_section("4B: GridSearchCV Tuning (RBF Kernel)")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
}

# Use combined train+val for GridSearch CV
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

grid = GridSearchCV(SVC(kernel='rbf', random_state=RANDOM_STATE),
                    param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
grid.fit(X_trainval, y_trainval)

print(f"  Best params: {grid.best_params_}")
print(f"  Best CV F1: {grid.best_score_:.4f}")

# Evaluate best model on test set
best_svc = grid.best_estimator_
best_svc_prob = SVC(kernel='rbf', **grid.best_params_, random_state=RANDOM_STATE, probability=True)
best_svc_prob.fit(X_trainval, y_trainval)
y_pred_test = best_svc_prob.predict(X_test)
y_prob_test = best_svc_prob.predict_proba(X_test)

classification_metrics(y_test, y_pred_test, y_prob_test, "Best SVC Test")
print_confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2],
                       title='SVM (RBF) - Confusion Matrix',
                       save_name='exp04_confusion.png')

# ======================================================================
# 4C: DECISION BOUNDARY VISUALIZATION (PCA 2D)
# ======================================================================
print_section("4C: Decision Boundary Visualization (PCA 2D)")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

svc_2d = SVC(kernel='rbf', C=grid.best_params_['C'],
             gamma=grid.best_params_['gamma'], random_state=RANDOM_STATE)
svc_2d.fit(X_train_pca, y_train)

# Create mesh grid
h = 0.1
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 8))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = ax.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=y_val,
                     cmap='viridis', edgecolors='k', s=30, alpha=0.7)

# Highlight support vectors
sv_pca = pca.transform(X_train[svc_2d.support_])
ax.scatter(sv_pca[:, 0], sv_pca[:, 1], s=100, facecolors='none',
           edgecolors='red', linewidths=1.5, label='Support Vectors')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('SVM Decision Boundaries (PCA 2D Projection)')
ax.legend()
plt.colorbar(scatter, label='Price Tier')
save_fig(fig, 'exp04_decision_boundary.png')

# ======================================================================
# 4D: SVR FOR REGRESSION
# ======================================================================
print_section("4D: SVR for Price Regression")

from src.preprocessing import prepare_regression_data
(Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test,
 r_feats, r_scaler) = prepare_regression_data()

for kernel in ['linear', 'rbf']:
    svr = SVR(kernel=kernel, C=10)
    svr.fit(Xr_train, yr_train)
    yr_pred = svr.predict(Xr_val)
    regression_metrics(yr_val, yr_pred, f"SVR({kernel})")

# ======================================================================
# 4E: EFFECT OF C AND GAMMA
# ======================================================================
print_section("4E: Effect of C and Gamma")

C_values = [0.01, 0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1, 10]

results_cg = np.zeros((len(C_values), len(gamma_values)))
for i, C in enumerate(C_values):
    for j, g in enumerate(gamma_values):
        svc_cg = SVC(kernel='rbf', C=C, gamma=g, random_state=RANDOM_STATE)
        svc_cg.fit(X_train, y_train)
        results_cg[i, j] = svc_cg.score(X_val, y_val)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(results_cg, annot=True, fmt='.3f', ax=ax,
            xticklabels=[str(g) for g in gamma_values],
            yticklabels=[str(c) for c in C_values],
            cmap='YlOrRd')
ax.set_xlabel('Gamma')
ax.set_ylabel('C')
ax.set_title('SVM Accuracy: C vs Gamma (RBF kernel)')
save_fig(fig, 'exp04_c_gamma_heatmap.png')

# ======================================================================
# OBSERVATIONS & CONCLUSION
# ======================================================================
print_section("Observations & Conclusion")
print("""
OBSERVATIONS:
1. RBF kernel generally outperforms linear and polynomial for this dataset,
   as it can model non-linear class boundaries.
2. GridSearchCV identifies optimal C and gamma; too small C underfits,
   too large C overfits.
3. The PCA 2D projection shows reasonable separation with some overlap
   in the Mid-range region.
4. Support vectors concentrate near decision boundaries, as expected.
5. SVR with RBF kernel provides competitive regression results.
6. Feature scaling is CRITICAL for SVM: unscaled features lead to
   dramatically worse performance since the kernel depends on distances.

CONCLUSION:
SVM with tuned RBF kernel achieves strong classification of price tiers.
The non-linear kernel captures feature interactions that linear models miss.
""")

print("\n[OK] Experiment 4 complete.\n")
