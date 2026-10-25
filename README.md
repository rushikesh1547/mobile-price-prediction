# 📱 Used Mobile Phone Price Prediction & Market Analysis System
### University Machine Learning Mini-Project (Semester 5)

An end-to-end, reproducible Machine Learning system that models used mobile phone valuations, market tier segmentation, and resale dynamics across 11 experiments—implemented both **from scratch** and with **industrial libraries (Scikit-Learn, XGBoost, LightGBM)**.

---

## 🚀 Quickstart

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed. Install the dependencies:
```bash
cd used-mobile-price-ml
pip install -r requirements.txt
```

### 2. Run All Experiments (End-to-End)
Execute the capstone integration script to generate all figures, run the model zoo, evaluate SHAP explanations, and train the multi-armed bandit pricing agent:
```bash
python notebooks/exp11_mini_project.py
```

### 3. Launch the Interactive Web Application
Launch the Streamlit dashboard for real-time phone valuation, market persona discovery, and dynamic pricing simulation:
```bash
streamlit run app/streamlit_app.py
```

---

## 📂 Project Directory Structure

```text
used-mobile-price-ml/
├── app/
│   └── streamlit_app.py           # Interactive Streamlit Valuation & Intelligence Dashboard
├── data/
│   ├── raw/                       # Raw synthetic used-phone dataset (3500 records)
│   │   └── used_phones.csv
│   └── processed/                 # Processed & encoded datasets
├── models/                        # Serialized models and pipelines
│   ├── best_price_regressor.pkl   # Champion XGBoost Regressor (R² = 0.9424)
│   ├── best_tier_classifier.pkl   # Multinomial Logistic Regression Classifier (86.3%)
│   ├── preprocessor.pkl           # ColumnTransformer (Imputation + OneHot + StandardScaler)
│   ├── kmeans_market_segments.pkl # K-Means Persona Clustering Model
│   └── model_metadata.json        # Pipeline metadata, hyperparameters, and feature names
├── notebooks/                     # 11 Reproducible Experiment Scripts
│   ├── generate_data.py           # Realistic synthetic dataset generator
│   ├── exp01_linear_regression.py # Exp 1: Linear Regression (Scratch + Sklearn + Ridge/Lasso)
│   ├── exp02_polynomial_regression.py # Exp 2: Polynomial Regression & Bias-Variance
│   ├── exp03_logistic_regression.py   # Exp 3: Logistic Regression (Binary + Multi + Scratch)
│   ├── exp04_svm.py               # Exp 4: Support Vector Machines (SVC & SVR)
│   ├── exp05_knn.py               # Exp 5: k-Nearest Neighbors (Classification & Regression)
│   ├── exp06_hmm.py               # Exp 6: Hidden Markov Model (Forward, Viterbi, Baum-Welch)
│   ├── exp07_em.py                # Exp 7: Expectation-Maximization GMM & Imputation
│   ├── exp08_clustering.py        # Exp 8: K-Means, Hierarchical, DBSCAN, UCI Wine Benchmark
│   ├── exp09_adaboost.py          # Exp 9: AdaBoost from Scratch & Sklearn
│   ├── exp10_gradient_boosting.py # Exp 10: GBDT from Scratch, XGBoost, LightGBM
│   └── exp11_mini_project.py      # Exp 11: Capstone Integrated Pipeline & RL Bandit
├── reports/
│   ├── report.md                  # Comprehensive University Capstone Academic Report
│   └── figures/                   # 45+ Publication-quality plots and charts (.png)
├── src/                           # Modular Reusable Python Package
│   ├── __init__.py
│   ├── utils.py                   # Global constants, seed management, plotting helpers
│   ├── preprocessing.py           # Data loaders, target creation, train/val/test splits
│   ├── features.py                # Domain feature engineering (phone age, spec score, etc.)
│   ├── hmm.py                     # Hidden Markov Model from scratch
│   ├── em.py                      # Expectation-Maximization GMM & Imputation from scratch
│   └── evaluate.py                # Comprehensive regression & classification metrics
└── requirements.txt               # Pinned package dependencies
```

---

## 🧪 Experiment Curriculum Breakdown

| # | Experiment Name | Key Concepts & From-Scratch Algorithms | Best Metric |
|:---:|---|---|:---:|
| **01** | Linear Regression | Vectorized Batch Gradient Descent OLS, Ridge ($L_2$), Lasso ($L_1$), Normal Equation | $R^2 = 0.9319$ |
| **02** | Polynomial Regression | Degree sweeps ($d=1..5$), Bias-Variance trade-off, Cross-Validation | Degree 2 Optimal |
| **03** | Logistic Regression | Vectorized Cross-Entropy Loss, Binary & Multinomial, PR Threshold Tuning | Accuracy = $92.6\%$ |
| **04** | Support Vector Machines | Linear, RBF, Polynomial kernels, GridSearch ($C, \gamma$), SVR $\epsilon$-tube | F1 = $0.864$ |
| **05** | k-Nearest Neighbors | Distance metrics (Euclidean, Manhattan), Distance-weighting, Scaling sensitivity | Acc = $80.2\%$ |
| **06** | Hidden Markov Models | Forward Algorithm (Likelihood), Viterbi (Decoding), Baum-Welch (EM) | $100\%$ Viterbi Agreement |
| **07** | Expectation-Maximization | GMM EM from scratch, Closed-form Conditional Expectation Imputation | Imputation RMSE = $0.8588$ |
| **08** | Unsupervised Clustering | K-Means, Ward Linkage Dendrogram, DBSCAN Anomaly Detection, UCI Wine Benchmark | Wine ARI = $0.8975$ |
| **09** | AdaBoost | Decision Stumps, Stage weights ($\alpha_m$), Dynamic sample weight evolution | Accuracy = $92.57\%$ |
| **10** | Gradient Tree Boosting | Functional Gradient Descent from scratch, Shrinkage ($\nu$), XGBoost, LightGBM | $R^2 = 0.9351$ |
| **11** | Capstone Mini-Project | Master Zoo Leaderboard, SHAP Explainability, Multi-Armed Bandit Dynamic Pricing | **XGBoost $R^2 = 0.9424$** |

---

## 🏆 Final Model Leaderboard (Holdout Test Set)

### Continuous Resale Price Prediction (Regression)
1. 🥇 **XGBoost Regressor:** $R^2 = \mathbf{0.9424}$, $\text{RMSE} = \mathbf{0.0286}$, $\text{MAPE} = 12.72\%$, Train Time $= 0.49\text{s}$
2. 🥈 **LightGBM Regressor:** $R^2 = 0.9421$, $\text{RMSE} = 0.0287$, $\text{MAPE} = 12.61\%$, Train Time $= \mathbf{0.18\text{s}}$
3. 🥉 **HistGradientBoosting:** $R^2 = 0.9409$, $\text{RMSE} = 0.0290$
4. **Gradient Boosting (GBDT):** $R^2 = 0.9408$, $\text{RMSE} = 0.0291$
5. **Linear Regression (OLS):** $R^2 = 0.9319$, $\text{RMSE} = 0.0312$
6. **Support Vector Regression (RBF):** $R^2 = 0.9100$, $\text{RMSE} = 0.0358$
7. **AdaBoost Regressor:** $R^2 = 0.9027$, $\text{RMSE} = 0.0372$
8. **KNN Regressor ($k=7$):** $R^2 = 0.8919$, $\text{RMSE} = 0.0393$

### Market Tier Classification
1. 🥇 **Multinomial Logistic Regression:** Accuracy $= \mathbf{86.29\%}$, Weighted F1 $= \mathbf{0.87}$
2. 🥈 **Support Vector Classifier (RBF):** Accuracy $= 83.62\%$, Weighted F1 $= 0.84$
3. 🥉 **KNN Classifier ($k=7$):** Accuracy $= 80.19\%$, Weighted F1 $= 0.80$
4. **AdaBoost Classifier:** Accuracy $= 73.71\%$, Weighted F1 $= 0.74$

---

## 🧠 Explainable AI (SHAP)
Game-theoretic Shapley decomposition revealed:
- **`normalized_new_price`:** Strongest positive valuation driver ($+\$180$ effect).
- **`days_used` & `phone_age_years`:** Primary depreciation driver, with accelerated depreciation during the first 365 days.
- **Brand Equity:** Apple and Google devices retain a statistically significant valuation premium over Android competitors with matching hardware specs.

---

## 💰 Reinforcement Learning: Multi-Armed Bandit Dynamic Pricing
To solve the reseller's inventory pricing dilemma (margin vs turnover velocity), 3 bandit algorithms simulated 1,500 customer interactions across 4 markup tiers:
- **Arm 0 (+15% Markup):** Conversion $35\%$, Expected Reward Factor $= 0.402$
- **Arm 1 (0% Fair Market):** Conversion $65\%$, Expected Reward Factor $= 0.650$
- **Arm 2 (-10% Quick Sale):** Conversion $85\%$, Expected Reward Factor $= \mathbf{0.765}$ (Optimal Policy)
- **Arm 3 (-25% Clearance):** Conversion $95\%$, Expected Reward Factor $= 0.712$

**Regret Comparison:**
- Random Policy: $\$20,570.50$ loss
- Upper Confidence Bound (UCB1): $\$8,604.50$ loss
- **$\epsilon$-Greedy ($\epsilon=0.10$):** **$\$2,946.25$ loss** ($85.7\%$ reduction in lost opportunity)

---

## 🎓 Academic Integrity & Reproducibility
- All experiments utilize pinned random seeds (`RANDOM_STATE = 42`).
- All scripts are executable directly via command line without reliance on interactive notebook kernels.
- Preprocessing pipelines strictly fit transforms on training splits to prevent data leakage.
