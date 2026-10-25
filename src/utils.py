"""
Utility functions for the Used Mobile Phone Price Prediction project.
Common constants, seed management, plotting helpers, and path management.
"""

import os
import sys
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

# Target column names
TARGET_REG = 'normalized_used_price'
TARGET_BIN = 'is_premium'
TARGET_MULTI = 'price_tier'

# Tier labels
TIER_LABELS = ['Budget', 'Mid-range', 'Premium']

# ── Seed management ─────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_STATE):
    """Set random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_project():
    """Set seeds, suppress warnings, configure plotting defaults."""
    set_seed()
    warnings.filterwarnings('ignore')
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 120,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })
    # Ensure directories exist
    for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


# ── Path helpers ─────────────────────────────────────────────────────────────

def fig_path(name: str) -> str:
    """Return full path for a figure file in reports/figures/."""
    return os.path.join(FIGURES_DIR, name)


def model_path(name: str) -> str:
    """Return full path for a model file in models/."""
    return os.path.join(MODELS_DIR, name)


# ── Plotting helpers ─────────────────────────────────────────────────────────

def save_fig(fig, name: str, close: bool = True):
    """Save a matplotlib figure to reports/figures/."""
    path = fig_path(name)
    fig.savefig(path)
    print(f"  [saved] {path}")
    if close:
        plt.close(fig)


def plot_residuals(y_true, y_pred, title: str = 'Residual Analysis', save_name: str = None):
    """Create a 2-panel residual analysis plot."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted vs Residuals
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residual')
    axes[0].set_title(f'{title} — Residuals vs Predicted')

    # Histogram of residuals
    axes[1].hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{title} — Residual Distribution')

    plt.tight_layout()
    if save_name:
        save_fig(fig, save_name, close=True)
    else:
        plt.close(fig)
    return fig


def plot_predicted_vs_actual(y_true, y_pred, title: str = 'Predicted vs Actual',
                              save_name: str = None):
    """Scatter plot of predicted vs actual values with perfect-fit line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color='steelblue')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_name:
        save_fig(fig, save_name, close=True)
    else:
        plt.close(fig)
    return fig


def print_section(title: str, char: str = '=', width: int = 80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")
