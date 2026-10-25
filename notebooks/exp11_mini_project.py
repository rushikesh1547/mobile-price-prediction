"""
==============================================================================
 Experiment 11 - Integrated Capstone Mini-Project: Used Phone Price Prediction
 & Market Intelligence System
==============================================================================

AIM:
  Integrate all modules and machine learning paradigms developed across Experiments 1-10
  into an end-to-end, reproducible analytics and prediction system:
  1. Exploratory Data Analysis (distributions, correlations, feature interactions).
  2. Data Preprocessing & Advanced Imputation Pipeline.
  3. Feature Engineering (phone age, brand tier, spec score, retention ratio).
  4. Multi-Paradigm Model Zoo (OLS, Ridge, Lasso, Poly, SVR, KNN, AdaBoost, GBM, XGBoost, LightGBM).
  5. Multi-Class Classification & Market Tier Segmentation (Logistic, SVM, KNN, AdaBoost).
  6. Model Explainability with SHAP (global feature importance, local waterfall).
  7. Reinforcement Learning Multi-Armed Bandit for Dynamic Pricing Strategy.
  8. Model Serialization for Web Application Deployment.
"""

import os, sys, time, json
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
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.utils import setup_project, print_section, print_subsection, save_fig, RANDOM_STATE, MODELS_DIR
from src.preprocessing import load_raw_data, create_targets
from src.features import engineer_all_features

setup_project()
print_section("Experiment 11: Integrated Capstone Mini-Project")

# ======================================================================
# PART 1: DATA INGESTION & EXPLORATORY DATA ANALYSIS (EDA)
# ======================================================================
print_section("Part 1: Ingestion & Exploratory Data Analysis")

df_raw = load_raw_data()
df = create_targets(df_raw)
print(f"  Loaded raw dataset: {df.shape[0]} records x {df.shape[1]} columns")
print(f"  Class distribution for price tiers:\n{df['price_tier'].value_counts(normalize=True).round(3).to_string()}")

# Correlation Matrix
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax, square=True)
ax.set_title("Exploratory Data Analysis: Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
save_fig(fig, "exp11_correlation_matrix.png")
plt.close()

# ======================================================================
# PART 2 & 3: FEATURE ENGINEERING & PREPROCESSING PIPELINE
# ======================================================================
print_section("Parts 2 & 3: Feature Engineering & Preprocessing Pipeline")

df_feat = engineer_all_features(df)
print(f"  After feature engineering: {df_feat.shape[1]} total features created")

# Define target and predictors (retention-based modeling to eliminate target leakage and guarantee resale <= new price)
df_feat['retention_ratio'] = np.clip(
    df_feat['normalized_used_price'] / np.maximum(df_feat['normalized_new_price'], 0.01),
    0.05,
    1.0
)

target_reg = 'retention_ratio'
target_cls = 'price_tier'
drop_cols = ['normalized_used_price', 'retention_ratio', 'price_retention_ratio', 'price_tier', 'is_premium']

feature_cols = [c for c in df_feat.columns if c not in drop_cols]
categorical_features = [c for c in ['brand', 'os', '4g', '5g'] if c in feature_cols]
numeric_features = [c for c in feature_cols if c not in categorical_features]

print(f"  Numeric predictors ({len(numeric_features)}): {numeric_features}")
print(f"  Categorical predictors ({len(categorical_features)}): {categorical_features}")

# Train/Val/Test Split (70/15/15)
X = df_feat[feature_cols]
y_r = df_feat[target_reg].values
y_used = df_feat['normalized_used_price'].values
y_c_raw = df_feat[target_cls].values

le = LabelEncoder()
y_c = le.fit_transform(y_c_raw)

X_train, X_temp, y_r_train, y_r_temp, y_c_train, y_c_temp, y_u_train, y_u_temp = train_test_split(
    X, y_r, y_c, y_used, test_size=0.30, random_state=RANDOM_STATE, stratify=y_c
)
X_val, X_test, y_r_val, y_r_test, y_c_val, y_c_test, y_u_val, y_u_test = train_test_split(
    X_temp, y_r_temp, y_c_temp, y_u_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_c_temp
)

print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Build Sklearn Preprocessor Pipeline
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# Fit preprocessor strictly on training data
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# Extract encoded feature names
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
all_processed_feature_names = numeric_features + cat_feature_names

print(f"  Processed matrix: {X_train_proc.shape[1]} features after encoding & scaling")

# ======================================================================
# PART 4: COMPREHENSIVE MODEL ZOO & LEADERBOARD (REGRESSION)
# ======================================================================
print_section("Part 4A: Comprehensive Regression Benchmark Zoo (Retention Modeling)")

reg_models = {
    'Linear Regression (OLS)': LinearRegression(),
    'Ridge Regression (L2)': Ridge(alpha=10.0),
    'Lasso Regression (L1)': Lasso(alpha=0.001, random_state=RANDOM_STATE),
    'Support Vector Regression (RBF)': SVR(C=2.0, epsilon=0.01),
    'KNN Regressor (k=7)': KNeighborsRegressor(n_neighbors=7, weights='distance'),
    'AdaBoost Regressor': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
    'Gradient Boosting (GBDT)': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE),
    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE)
}

if HAS_XGB:
    reg_models['XGBoost Regressor'] = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE, n_jobs=-1
    )
if HAS_LGB:
    reg_models['LightGBM Regressor'] = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )

reg_results = []
best_reg_model = None
best_reg_name = ""
lowest_rmse = float('inf')

for name, model in reg_models.items():
    t0 = time.time()
    model.fit(X_train_proc, y_r_train)
    fit_time = time.time() - t0
    
    y_test_pred = model.predict(X_test_proc)
    y_test_pred = np.clip(y_test_pred, 0.05, 1.0)
    
    # Retention metrics
    r2 = r2_score(y_r_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_r_test, y_test_pred))
    mae = mean_absolute_error(y_r_test, y_test_pred)
    mape = np.mean(np.abs((y_r_test - y_test_pred) / (y_r_test + 1e-8))) * 100
    
    # Implied resale price metrics
    implied_pred = y_test_pred * X_test['normalized_new_price'].values
    resale_r2 = r2_score(y_u_test, implied_pred)
    resale_rmse = np.sqrt(mean_squared_error(y_u_test, implied_pred))
    resale_mae = mean_absolute_error(y_u_test, implied_pred)
    
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_reg_model = model
        best_reg_name = name
        
    reg_results.append({
        'Model': name,
        'Retention R²': r2,
        'Retention RMSE': rmse,
        'Retention MAE': mae,
        'Implied Resale R²': resale_r2,
        'Implied Resale RMSE': resale_rmse,
        'Implied Resale MAE': resale_mae,
        'Train Time (s)': fit_time
    })

df_reg_leaderboard = pd.DataFrame(reg_results).sort_values(by='Retention R²', ascending=False)
print("\n=== RETENTION-BASED REGRESSION LEADERBOARD ===")
print(df_reg_leaderboard.round(4).to_string(index=False))

# Baseline direct unconstrained model for comparison
baseline_gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE)
baseline_gb.fit(X_train_proc, y_u_train)
baseline_pred = baseline_gb.predict(X_test_proc)
baseline_r2 = r2_score(y_u_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_u_test, baseline_pred))
baseline_mae = mean_absolute_error(y_u_test, baseline_pred)
print(f"\n[Old Model Baseline - Direct Price GBDT]: Test R²={baseline_r2:.4f}, RMSE={baseline_rmse:.4f}, MAE={baseline_mae:.4f}")

# Train 95% Quantile Regressors on retention
print("  Fitting Quantile Regressors for 95% Confidence Intervals (alpha=0.025, 0.975)...")
q_low = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=80, max_depth=3, random_state=RANDOM_STATE)
q_high = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=80, max_depth=3, random_state=RANDOM_STATE)
q_low.fit(X_train_proc, y_r_train)
q_high.fit(X_train_proc, y_r_train)

# Validation residuals on retention
y_val_pred = np.clip(best_reg_model.predict(X_val_proc), 0.05, 1.0)
val_resids = y_r_val - y_val_pred
val_ret_rmse = float(np.sqrt(mean_squared_error(y_r_val, y_val_pred)))
resid_q025 = float(np.percentile(val_resids, 2.5))
resid_q975 = float(np.percentile(val_resids, 97.5))

# Plot Regression Leaderboard
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_reg_leaderboard, y='Model', x='Retention R²', palette='Blues_r', ax=ax, edgecolor='black')
ax.set_title("Retention Regression Leaderboard (Test R²)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Retention R² Score (Higher = Better)", fontsize=11)
ax.set_xlim([max(0.0, df_reg_leaderboard['Retention R²'].min() - 0.05), 1.0])
ax.grid(axis='x', alpha=0.3)
for p in ax.patches:
    ax.annotate(f"{p.get_width():.4f}", (p.get_width() + 0.005, p.get_y() + p.get_height()/2.),
                va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
save_fig(fig, "exp11_regression_leaderboard.png")
plt.close()

# ======================================================================
# PART 4B: CLASSIFICATION ZOO (PRICE TIER PREDICTION)
# ======================================================================
print_section("Part 4B: Multi-Class Price Tier Classification Leaderboard")

cls_models = {
    'Multinomial Logistic Regression': LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    'Support Vector Classifier (RBF)': SVC(C=2.0, kernel='rbf', probability=True, random_state=RANDOM_STATE),
    'KNN Classifier (k=7)': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'AdaBoost Classifier': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
}

cls_results = []
best_cls_model = None
best_cls_name = ""
highest_acc = 0.0

for name, model in cls_models.items():
    model.fit(X_train_proc, y_c_train)
    y_test_pred = model.predict(X_test_proc)
    
    acc = accuracy_score(y_c_test, y_test_pred)
    f1 = f1_score(y_c_test, y_test_pred, average='weighted')
    
    if acc > highest_acc:
        highest_acc = acc
        best_cls_model = model
        best_cls_name = name
        
    cls_results.append({
        'Model': name,
        'Accuracy (%)': acc * 100,
        'Weighted F1': f1
    })

df_cls_leaderboard = pd.DataFrame(cls_results).sort_values(by='Accuracy (%)', ascending=False)
print("\n=== CLASSIFICATION LEADERBOARD ===")
print(df_cls_leaderboard.round(2).to_string(index=False))

# ======================================================================
# PART 5: EXPLAINABLE AI WITH SHAP
# ======================================================================
print_section("Part 5: Explainable AI with SHAP (Feature Attribution)")

if HAS_SHAP and hasattr(best_reg_model, 'predict'):
    print(f"  Computing SHAP values on test set using {best_reg_name}...")
    try:
        # Use TreeExplainer if tree-based, else KernelExplainer
        if any(tree_kw in best_reg_name for tree_kw in ['Gradient', 'GBDT', 'XGBoost', 'LightGBM', 'Hist']):
            explainer = shap.TreeExplainer(best_reg_model)
            shap_values = explainer.shap_values(X_test_proc)
        else:
            explainer = shap.Explainer(best_reg_model, X_train_proc[:100])
            shap_values = explainer(X_test_proc[:200]).values
            
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_proc, feature_names=all_processed_feature_names,
                          max_display=12, show=False)
        plt.title("SHAP Global Feature Attributions (Price Valuation)", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        save_fig(fig, "exp11_shap_summary.png")
        plt.close()
        print("  SHAP summary plot generated.")
    except Exception as e:
        print(f"  SHAP explanation note: {e}")
        # Generate feature importance fallback
        fig, ax = plt.subplots(figsize=(10, 5))
        if hasattr(best_reg_model, 'feature_importances_'):
            imps = best_reg_model.feature_importances_
            top_i = np.argsort(imps)[::-1][:12]
            sns.barplot(x=imps[top_i], y=[all_processed_feature_names[i] for i in top_i], ax=ax, palette='Blues_r')
            ax.set_title("Top 12 Predictive Features for Resale Price", fontsize=13, fontweight='bold')
            plt.tight_layout()
            save_fig(fig, "exp11_shap_summary.png")
            plt.close()
else:
    print("  Generating feature importance fallback plot...")
    fig, ax = plt.subplots(figsize=(10, 5))
    if hasattr(best_reg_model, 'feature_importances_'):
        imps = best_reg_model.feature_importances_
        top_i = np.argsort(imps)[::-1][:12]
        sns.barplot(x=imps[top_i], y=[all_processed_feature_names[i] for i in top_i], ax=ax, palette='Blues_r')
    plt.tight_layout()
    save_fig(fig, "exp11_shap_summary.png")
    plt.close()

# ======================================================================
# PART 6: MULTI-ARMED BANDIT FOR DYNAMIC PRICING STRATEGY
# ======================================================================
print_section("Part 6: Reinforcement Learning Multi-Armed Bandit for Dynamic Pricing")

class DynamicPricingBanditEnv:
    """
    Simulates consumer purchasing response across 4 reseller markup strategies:
    Arm 0: 'Premium' (+15% markup, 35% conversion, high revenue if converted)
    Arm 1: 'Market'   (0% markup, 65% conversion)
    Arm 2: 'Discount' (-10% markup, 85% conversion)
    Arm 3: 'Clearance'(-25% markup, 95% conversion, fast turnover)
    """
    def __init__(self, base_price=100.0):
        self.base_price = base_price
        self.multipliers = [1.15, 1.00, 0.90, 0.75]
        self.conv_probs = [0.35, 0.65, 0.85, 0.95]
        # Expected reward = multiplier * conv_prob
        self.expected_rewards = [m * p for m, p in zip(self.multipliers, self.conv_probs)]
        self.optimal_arm = np.argmax(self.expected_rewards)
        self.optimal_reward = self.expected_rewards[self.optimal_arm]

    def pull(self, arm):
        converted = np.random.rand() < self.conv_probs[arm]
        revenue = (self.base_price * self.multipliers[arm]) if converted else 0.0
        return revenue, self.optimal_reward * self.base_price - self.expected_rewards[arm] * self.base_price

# Simulate 3 Bandit Algorithms over T=1500 rounds
T = 1500
np.random.seed(RANDOM_STATE)
env = DynamicPricingBanditEnv()
n_arms = 4

# 1. Random Exploration
regret_random = []
cum_regret = 0.0
for _ in range(T):
    arm = np.random.choice(n_arms)
    _, reg = env.pull(arm)
    cum_regret += reg
    regret_random.append(cum_regret)

# 2. Epsilon-Greedy (eps=0.10)
regret_eps = []
cum_regret = 0.0
counts_eps = np.zeros(n_arms)
values_eps = np.zeros(n_arms)
eps = 0.10

for t in range(T):
    if np.random.rand() < eps or t < n_arms:
        arm = np.random.choice(n_arms)
    else:
        arm = np.argmax(values_eps)
    rev, reg = env.pull(arm)
    counts_eps[arm] += 1
    values_eps[arm] += (rev - values_eps[arm]) / counts_eps[arm]
    cum_regret += reg
    regret_eps.append(cum_regret)

# 3. UCB1 (Upper Confidence Bound)
regret_ucb = []
cum_regret = 0.0
counts_ucb = np.zeros(n_arms)
values_ucb = np.zeros(n_arms)

for t in range(T):
    if t < n_arms:
        arm = t
    else:
        ucb_scores = values_ucb + 120.0 * np.sqrt(2.0 * np.log(t + 1) / counts_ucb)
        arm = np.argmax(ucb_scores)
    rev, reg = env.pull(arm)
    counts_ucb[arm] += 1
    values_ucb[arm] += (rev - values_ucb[arm]) / counts_ucb[arm]
    cum_regret += reg
    regret_ucb.append(cum_regret)

print(f"  Bandit Expected Rewards per Arm: {[round(r, 2) for r in env.expected_rewards]}")
print(f"  Optimal Pricing Strategy: Arm {env.optimal_arm} ({['+15%', '0%', '-10%', '-25%'][env.optimal_arm]} Markup)")
print(f"  Final Regret after {T} rounds:")
print(f"    Random Exploration: ${regret_random[-1]:,.2f}")
print(f"    Epsilon-Greedy:     ${regret_eps[-1]:,.2f}")
print(f"    UCB1 Algorithm:     ${regret_ucb[-1]:,.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(regret_random, label='Random Strategy', color='#94a3b8', lw=1.8, linestyle=':')
ax.plot(regret_eps, label='ε-Greedy (ε=0.10)', color='#d97706', lw=2)
ax.plot(regret_ucb, label='UCB1 (Exploration-Exploitation)', color='#2563eb', lw=2.5)
ax.set_title("Multi-Armed Bandit Cumulative Pricing Regret", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Rounds (Inventory Customer Visits)", fontsize=11)
ax.set_ylabel("Cumulative Opportunity Loss ($)", fontsize=11)
ax.legend(frameon=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, "exp11_bandit_regret.png")
plt.close()

# ======================================================================
# PART 7: MODEL EXPORT & DEPLOYMENT SERIALIZATION
# ======================================================================
print_section("Part 7: Model Serialization for Web Application")

os.makedirs(MODELS_DIR, exist_ok=True)

# Train a K-Means model on the 6 key phone features for the web app persona
cluster_cols = ['normalized_new_price', 'normalized_used_price', 'ram', 'internal_memory', 'battery', 'screen_size']
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df[cluster_cols].dropna().values)
kmeans_app = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit(X_cluster_scaled)

# Save pipeline artifacts
joblib.dump(best_reg_model, os.path.join(MODELS_DIR, 'best_price_regressor.pkl'))
joblib.dump(best_cls_model, os.path.join(MODELS_DIR, 'best_tier_classifier.pkl'))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'tier_label_encoder.pkl'))
joblib.dump(kmeans_app, os.path.join(MODELS_DIR, 'kmeans_market_segments.pkl'))
joblib.dump(scaler_cluster, os.path.join(MODELS_DIR, 'cluster_scaler.pkl'))
joblib.dump(q_low, os.path.join(MODELS_DIR, 'quantile_regressor_low.pkl'))
joblib.dump(q_high, os.path.join(MODELS_DIR, 'quantile_regressor_high.pkl'))

metadata = {
    'best_regressor': best_reg_name,
    'best_reg_r2': float(df_reg_leaderboard.iloc[0]['Retention R²']),
    'best_reg_rmse': float(df_reg_leaderboard.iloc[0]['Retention RMSE']),
    'best_reg_mae': float(df_reg_leaderboard.iloc[0]['Retention MAE']),
    'implied_resale_r2': float(df_reg_leaderboard.iloc[0]['Implied Resale R²']),
    'implied_resale_rmse': float(df_reg_leaderboard.iloc[0]['Implied Resale RMSE']),
    'implied_resale_mae': float(df_reg_leaderboard.iloc[0]['Implied Resale MAE']),
    'val_retention_rmse': float(val_ret_rmse),
    'retention_resid_q025': float(resid_q025),
    'retention_resid_q975': float(resid_q975),
    'old_baseline_direct_price': {
        'model': 'Direct Price GBDT',
        'test_r2': float(baseline_r2),
        'rmse': float(baseline_rmse),
        'mae': float(baseline_mae)
    },
    'leaderboard_retention': df_reg_leaderboard.to_dict(orient='records'),
    'best_classifier': best_cls_name,
    'best_cls_accuracy': float(df_cls_leaderboard.iloc[0]['Accuracy (%)']),
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'all_processed_features': all_processed_feature_names,
    'tier_classes': list(le.classes_),
    'optimal_bandit_arm': int(env.optimal_arm),
    'price_scale_factor': 1500.0,
    'currency': 'USD',
    'export_time': time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:

    json.dump(metadata, f, indent=2)

print(f"  Exported models and preprocessor pipeline to {MODELS_DIR}")
print(f"  Best Regressor: {best_reg_name} (R² = {metadata['best_reg_r2']:.4f})")
print(f"  Best Classifier: {best_cls_name} (Acc = {metadata['best_cls_accuracy']:.2f}%)")

print_section("Integrated Mini-Project Complete")
print("[OK] Experiment 11 execution successful. Ready for Streamlit UI and final report.")
