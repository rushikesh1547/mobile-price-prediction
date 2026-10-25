"""
evaluate.py — Evaluation helpers for regression and classification.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    mean_absolute_percentage_error
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def regression_metrics(y_true, y_pred, label: str = '') -> dict:
    """Compute standard regression metrics and print them."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE(%)': mape}
    if label:
        print(f"  [{label}] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  MAPE={mape:.1f}%")
    return metrics


def classification_metrics(y_true, y_pred, y_prob=None, label: str = '',
                           average: str = 'weighted') -> dict:
    """Compute classification metrics and print them."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    if y_prob is not None:
        try:
            if y_prob.ndim == 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
            metrics['ROC-AUC'] = auc
        except Exception:
            pass
    if label:
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        print(f"  [{label}] {', '.join(parts)}")
    return metrics


def print_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix',
                           save_name=None):
    """Plot and optionally save a confusion matrix heatmap."""
    from src.utils import save_fig
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    if save_name:
        save_fig(fig, save_name, close=True)
    else:
        plt.close(fig)
    return cm


def clustering_metrics(X, labels, label: str = '') -> dict:
    """Compute clustering quality metrics."""
    n_clusters = len(set(labels) - {-1})
    if n_clusters < 2:
        print(f"  [{label}] Only {n_clusters} cluster(s); metrics not applicable.")
        return {}
    mask = labels != -1
    if mask.sum() < 2:
        return {}
    sil = silhouette_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])
    metrics = {'Silhouette': sil, 'Davies-Bouldin': db, 'Calinski-Harabasz': ch}
    if label:
        print(f"  [{label}] Silhouette={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")
    return metrics


def build_leaderboard(results: list) -> pd.DataFrame:
    """Build a leaderboard DataFrame from a list of result dicts.

    Each dict should have at minimum: 'Model', and one or more metric keys.
    """
    df = pd.DataFrame(results)
    if 'RMSE' in df.columns:
        df = df.sort_values('RMSE')
    elif 'F1' in df.columns:
        df = df.sort_values('F1', ascending=False)
    return df.reset_index(drop=True)
