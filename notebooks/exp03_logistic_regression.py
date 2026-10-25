"""
==============================================================================
 Experiment 3 - Logistic Regression
 Used Mobile Phone Price Prediction
==============================================================================

AIM:
  Binary classification (Premium vs Not Premium) and multinomial (3 tiers).
  Implement sigmoid + cross-entropy + gradient descent from scratch, then
  compare with sklearn. Report confusion matrix, ROC-AUC, PR curve, etc.

THEORY:
  Logistic Regression models P(y=1|x) = sigma(w^T x + b) where sigma(z) = 1/(1+e^{-z}).
  Loss: Binary Cross-Entropy = -[y*log(p) + (1-y)*log(1-p)]
  Optimized with gradient descent: w <- w - lr * X^T (p - y) / n
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, classification_report)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split

from src.utils import setup_project, print_section, print_subsection, save_fig, RANDOM_STATE
from src.preprocessing import load_raw_data, create_targets, basic_clean, get_numeric_features
from src.evaluate import classification_metrics, print_confusion_matrix

setup_project()
print_section("Experiment 3: Logistic Regression")

# ── Data ─────────────────────────────────────────────────────────────────────
df = load_raw_data()
df = create_targets(df)
df = basic_clean(df)

NUM_FEATS = get_numeric_features()
X = df[NUM_FEATS].values.astype(float)
y_bin = df['is_premium'].values
y_multi = df['price_tier'].cat.codes.values  # 0=Budget, 1=Mid, 2=Premium

X_train, X_temp, yb_train, yb_temp = train_test_split(X, y_bin, test_size=0.30,
                                                        random_state=RANDOM_STATE, stratify=y_bin)
X_val, X_test, yb_val, yb_test = train_test_split(X_temp, yb_temp, test_size=0.50,
                                                    random_state=RANDOM_STATE, stratify=yb_temp)

_, _, ym_train, ym_temp = train_test_split(X, y_multi, test_size=0.30,
                                            random_state=RANDOM_STATE, stratify=y_multi)
_, _, ym_val, ym_test = train_test_split(X_temp[:, :1], ym_temp, test_size=0.50,
                                          random_state=RANDOM_STATE, stratify=ym_temp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train_s.shape}, Val: {X_val_s.shape}, Test: {X_test_s.shape}")
print(f"Binary class distribution (train): {np.bincount(yb_train)}")

# ======================================================================
# 3A: FROM SCRATCH - Binary Logistic Regression
# ======================================================================
print_section("3A: Logistic Regression from Scratch")

def sigmoid(z):
    """Numerically stable sigmoid function."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def binary_cross_entropy(y, p):
    """Binary cross-entropy loss."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def logistic_regression_gd(X, y, lr=0.1, epochs=1000, verbose=True):
    """Train binary logistic regression with gradient descent."""
    n, d = X.shape
    X_b = np.c_[np.ones(n), X]
    w = np.zeros(d + 1)
    losses = []
    for ep in range(epochs):
        z = X_b @ w
        p = sigmoid(z)
        loss = binary_cross_entropy(y, p)
        losses.append(loss)
        grad = (1 / n) * X_b.T @ (p - y)
        w -= lr * grad
        if verbose and (ep + 1) % 200 == 0:
            print(f"    Epoch {ep+1:4d}: Loss = {loss:.6f}")
    return w, losses

def predict_logistic(X, w, threshold=0.5):
    """Predict binary labels."""
    X_b = np.c_[np.ones(X.shape[0]), X]
    probs = sigmoid(X_b @ w)
    return (probs >= threshold).astype(int), probs

w_scratch, losses = logistic_regression_gd(X_train_s, yb_train, lr=0.1, epochs=1000)
y_pred_scratch, y_prob_scratch = predict_logistic(X_val_s, w_scratch)

print_subsection("Scratch - Validation Metrics")
m_scratch = classification_metrics(yb_val, y_pred_scratch, y_prob_scratch, "Scratch")

# Loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='steelblue', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Logistic Regression (scratch) - Training Loss')
save_fig(fig, 'exp03_scratch_loss.png')

# ======================================================================
# 3B: SKLEARN - Binary Logistic Regression
# ======================================================================
print_section("3B: sklearn Binary Logistic Regression")

lr_sk = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_sk.fit(X_train_s, yb_train)
y_pred_sk = lr_sk.predict(X_val_s)
y_prob_sk = lr_sk.predict_proba(X_val_s)[:, 1]

m_sk = classification_metrics(yb_val, y_pred_sk, y_prob_sk, "sklearn")

# ── Comparison table ─────────────────────────────────────────────────────────
print_subsection("Scratch vs sklearn Comparison")
comp = pd.DataFrame([
    {'Model': 'Scratch', **m_scratch},
    {'Model': 'sklearn', **m_sk}
])
print(comp.to_string(index=False))

# ── Confusion Matrix ─────────────────────────────────────────────────────────
print_confusion_matrix(yb_val, y_pred_sk, labels=[0, 1],
                       title='Binary Logistic - Confusion Matrix',
                       save_name='exp03_confusion_binary.png')

# ======================================================================
# 3C: ROC-AUC & PR Curve
# ======================================================================
print_section("3C: ROC-AUC & PR Curve")

fpr, tpr, thresholds_roc = roc_curve(yb_val, y_prob_sk)
roc_auc = auc(fpr, tpr)

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(yb_val, y_prob_sk)
ap = average_precision_score(yb_val, y_prob_sk)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=1)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(recall_vals, precision_vals, 'g-', linewidth=2, label=f'PR (AP = {ap:.3f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, 'exp03_roc_pr_curves.png')

print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Average Precision: {ap:.4f}")

# ======================================================================
# 3D: DECISION THRESHOLD TUNING
# ======================================================================
print_section("3D: Decision Threshold Tuning")

thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []
for t in thresholds:
    preds = (y_prob_sk >= t).astype(int)
    from sklearn.metrics import f1_score
    f1_scores.append(f1_score(yb_val, preds, zero_division=0))

best_threshold = thresholds[np.argmax(f1_scores)]
print(f"  Best threshold: {best_threshold:.2f} (F1 = {max(f1_scores):.4f})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, f1_scores, 'o-', color='darkorange', linewidth=2)
ax.axvline(best_threshold, color='green', linestyle='--', label=f'Best = {best_threshold:.2f}')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs Decision Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(fig, 'exp03_threshold_tuning.png')

# ======================================================================
# 3E: MULTINOMIAL LOGISTIC REGRESSION (3 tiers)
# ======================================================================
print_section("3E: Multinomial Logistic Regression (3 Tiers)")

# Re-split for multiclass
X_train2, X_temp2, ym_train2, ym_temp2 = train_test_split(
    X, y_multi, test_size=0.30, random_state=RANDOM_STATE, stratify=y_multi)
X_val2, X_test2, ym_val2, ym_test2 = train_test_split(
    X_temp2, ym_temp2, test_size=0.50, random_state=RANDOM_STATE, stratify=ym_temp2)

scaler2 = StandardScaler()
X_train2_s = scaler2.fit_transform(X_train2)
X_val2_s = scaler2.transform(X_val2)
X_test2_s = scaler2.transform(X_test2)

lr_multi = LogisticRegression(solver='lbfgs',
                               max_iter=1000, random_state=RANDOM_STATE)
lr_multi.fit(X_train2_s, ym_train2)
ym_pred = lr_multi.predict(X_val2_s)
ym_prob = lr_multi.predict_proba(X_val2_s)

labels = ['Budget', 'Mid-range', 'Premium']
m_multi = classification_metrics(ym_val2, ym_pred, ym_prob, "Multinomial")

print("\n  Classification Report:")
print(classification_report(ym_val2, ym_pred, target_names=labels))

print_confusion_matrix(ym_val2, ym_pred, labels=[0, 1, 2],
                       title='Multinomial Logistic - Confusion Matrix',
                       save_name='exp03_confusion_multi.png')

# ======================================================================
# 3F: TEST SET EVALUATION
# ======================================================================
print_section("3F: Test Set Evaluation")

y_pred_test = lr_sk.predict(X_test_s)
y_prob_test = lr_sk.predict_proba(X_test_s)[:, 1]
classification_metrics(yb_test, y_pred_test, y_prob_test, "Binary Test")

ym_pred_test = lr_multi.predict(X_test2_s)
ym_prob_test = lr_multi.predict_proba(X_test2_s)
classification_metrics(ym_test2, ym_pred_test, ym_prob_test, "Multi Test")

# ======================================================================
# OBSERVATIONS & CONCLUSION
# ======================================================================
print_section("Observations & Conclusion")
print("""
OBSERVATIONS:
1. Scratch and sklearn implementations produce comparable metrics,
   validating the from-scratch gradient descent approach.
2. Binary classification (Premium vs Not) achieves strong ROC-AUC,
   since Premium phones have distinctive feature patterns (higher
   new price, more RAM, newer release year).
3. Threshold tuning improves F1 when classes are imbalanced.
4. Multinomial logistic regression handles the 3-tier classification
   reasonably well, with Budget and Premium better separated than Mid-range.

CONCLUSION:
Logistic Regression provides interpretable classification for price tiers.
The decision boundary is linear in feature space, which works well for
well-separated classes. Non-linear classifiers (SVM, k-NN) may improve
mid-range classification.
""")

print("\n[OK] Experiment 3 complete.\n")
