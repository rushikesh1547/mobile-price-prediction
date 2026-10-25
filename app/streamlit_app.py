"""
Used Mobile Phone Price Prediction & Market Intelligence System
Interactive University Mini-Project Web Application
Built with Streamlit & Scikit-Learn / XGBoost / LightGBM
"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

# App Base Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    preprocess_input,
    validate_phone_inputs,
    predict_valuation,
    compute_depreciation_horizon,
    PRICE_SCALE_FACTOR
)

# Set Page Configuration
st.set_page_config(
    page_title="PhoneValuate AI — Used Phone Valuation & Intelligence",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-premium { background: rgba(217, 119, 6, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .badge-mid { background: rgba(37, 99, 235, 0.2); color: #60a5fa; border: 1px solid #2563eb; }
    .badge-budget { background: rgba(22, 163, 74, 0.2); color: #4ade80; border: 1px solid #16a34a; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Load Trained Artifacts
@st.cache_resource
def load_artifacts():
    try:
        regressor = joblib.load(os.path.join(MODELS_DIR, 'best_price_regressor.pkl'))
        classifier = joblib.load(os.path.join(MODELS_DIR, 'best_tier_classifier.pkl'))
        preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
        label_enc = joblib.load(os.path.join(MODELS_DIR, 'tier_label_encoder.pkl'))
        kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans_market_segments.pkl'))
        cluster_scaler = joblib.load(os.path.join(MODELS_DIR, 'cluster_scaler.pkl'))
        
        q_low = None
        q_high = None
        q_low_p = os.path.join(MODELS_DIR, 'quantile_regressor_low.pkl')
        q_high_p = os.path.join(MODELS_DIR, 'quantile_regressor_high.pkl')
        if os.path.exists(q_low_p) and os.path.exists(q_high_p):
            q_low = joblib.load(q_low_p)
            q_high = joblib.load(q_high_p)
            
        with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        return regressor, classifier, preprocessor, label_enc, kmeans, cluster_scaler, metadata, q_low, q_high, True
    except Exception as e:
        return None, None, None, None, None, None, None, None, None, False

regressor, classifier, preprocessor, label_enc, kmeans, cluster_scaler, metadata, q_low, q_high, loaded = load_artifacts()

# App Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="color: #f8fafc; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.025em;">
                📱 PhoneValuate AI
            </h1>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                Used Mobile Phone Price Prediction & Market Intelligence System
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 0.4rem 0.8rem; border-radius: 8px; font-size: 0.85rem; font-weight: 600;">
                Production Model: Retention XGBoost
            </span>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.4rem;">University ML Capstone Mini-Project</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.error("⚠️ Pre-trained model artifacts not found in `models/`. Please execute `python notebooks/exp11_mini_project.py` first to generate serialization files.")
    st.stop()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Valuation",
    "📊 Model Leaderboard & Zoo",
    "🧠 Explainable AI (SHAP)",
    "💰 Reseller Dynamic Pricing (RL Bandit)"
])

# ==============================================================================
# TAB 1: REAL-TIME VALUATION & MARKET INTELLIGENCE
# ==============================================================================
with tab1:
    col_input, col_output = st.columns([1.1, 1.4], gap="large")
    
    with col_input:
        st.markdown("### 📋 Phone Specifications")
        with st.container():
            c1, c2 = st.columns(2)
            brand = c1.selectbox("Brand", ["Apple", "Samsung", "OnePlus", "Google", "Xiaomi", "Huawei", "Oppo", "Vivo", "Motorola", "Realme", "Sony", "Nokia", "LG"], index=6)
            os_type = c2.selectbox("Operating System", ["iOS" if brand == "Apple" else "Android", "iOS", "Android", "Other"], index=0)
            
            c3, c4 = st.columns(2)
            release_year = c3.selectbox("Release Year", list(range(2024, 2017, -1)), index=0)
            days_used = c4.slider("Days in Use", min_value=30, max_value=1800, value=365, step=15)
            
            c5, c6 = st.columns(2)
            ram = c5.select_slider("RAM (GB)", options=[2, 3, 4, 6, 8, 12, 16, 24], value=6)
            internal_memory = c6.select_slider("Internal Storage (GB)", options=[16, 32, 64, 128, 256, 512, 1024], value=128)
            
            c7, c8 = st.columns(2)
            battery = c7.slider("Battery Capacity (mAh)", min_value=1500, max_value=7000, value=4500, step=100)
            screen_size = c8.slider("Screen Size (inches)", min_value=4.5, max_value=7.2, value=6.4, step=0.1)
            
            c9, c10 = st.columns(2)
            main_cam = c9.slider("Main Camera (MP)", min_value=8, max_value=200, value=50, step=2)
            selfie_cam = c10.slider("Selfie Camera (MP)", min_value=5, max_value=64, value=16, step=1)
            
            c11, c12, c13 = st.columns(3)
            weight = c11.number_input("Weight (g)", value=185, min_value=100, max_value=350)
            has_4g = c12.selectbox("4G LTE", ["yes", "no"], index=0)
            has_5g = c13.selectbox("5G Ready", ["yes", "no"], index=0 if release_year >= 2021 else 1)
            
            st.markdown("---")
            orig_price = st.number_input("Original New Retail Price ($ USD)", min_value=50.0, max_value=2500.0, value=150.0, step=25.0, help="The MSRP or retail price of this phone when it was brand new.")
            
            # Input validation warnings
            input_specs = {
                'brand': brand,
                'os': os_type,
                'release_year': release_year,
                'days_used': days_used,
                'orig_price': orig_price,
                'ram': ram,
                'internal_memory': internal_memory,
                'battery': battery,
                'screen_size': screen_size,
                'main_camera_mp': main_cam,
                'selfie_camera_mp': selfie_cam,
                'weight': weight,
                '4g': has_4g,
                '5g': has_5g
            }
            advisory_warnings = validate_phone_inputs(input_specs)
            if advisory_warnings:
                for warn_msg in advisory_warnings:
                    st.warning(f"⚠️ {warn_msg}")
            
            st.write("")
            calc_btn = st.button("🚀 Calculate Fair Market Valuation", type="primary", use_container_width=True)
            live_mode = st.checkbox("⚡ Live Mode (automatically recalculate when inputs change)", value=True)
            
    with col_output:
        st.markdown("### 🏷️ Estimated Fair Market Valuation")
        
        # Determine if we should compute prediction
        should_compute = calc_btn or live_mode or ('has_evaluated' in st.session_state)
        
        if not should_compute:
            st.info("""
            👈 **How to use this tool:**
            1. Configure the phone specifications on the left (Brand, RAM, Storage, Age, Original Price).
            2. Click the **'🚀 Calculate Fair Market Valuation'** button above to generate the predicted resale price.
            3. Or check **'Live Mode'** to see the valuation update dynamically as you adjust sliders.
            """)
        else:
            st.session_state['has_evaluated'] = True
            
            try:
                # 1. Compute unified valuation via src.preprocessing.predict_valuation
                val_res = predict_valuation(
                    specs=input_specs,
                    regressor=regressor,
                    preprocessor=preprocessor,
                    orig_price_usd=orig_price,
                    val_retention_rmse=metadata.get('val_retention_rmse', 0.035),
                    q_low_model=q_low,
                    q_high_model=q_high
                )
                
                pred_used_price_usd = val_res['resale_usd']
                lower_usd = val_res['lower_usd']
                upper_usd = val_res['upper_usd']
                retention_rate = val_res['retention_rate']
                capped_at_new = val_res['capped_at_new']
                
                # 2. Market Tier Classification & Cluster Persona
                df_feat, X_proc = preprocess_input(input_specs, preprocessor=preprocessor)
                tier_idx = int(classifier.predict(X_proc)[0])
                tier_name = label_enc.inverse_transform([tier_idx])[0]
                
                norm_new_price = orig_price / PRICE_SCALE_FACTOR
                pred_norm_price = val_res['pred_retention'] * norm_new_price
                cluster_input = np.array([[norm_new_price, pred_norm_price, ram, internal_memory, battery, screen_size]])
                cluster_id = int(kmeans.predict(cluster_scaler.transform(cluster_input))[0])
                cluster_personas = {0: "Budget Essential", 1: "Power Mid-Range", 2: "Flagship Luxury"}
                persona_name = cluster_personas.get(cluster_id, "Standard")
                
                phone_age_years = days_used / 365.0
                specs_score = float(df_feat['specs_per_price'].iloc[0])
                
                if calc_btn:
                    st.success(f"✅ Valuation generated successfully for **{brand} ({release_year})**!")
                    
                # Metrics display cards
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Estimated Resale Price</div>
                        <div style="color: #38bdf8; font-size: 2.2rem; font-weight: 800; margin: 0.25rem 0;">${pred_used_price_usd:,.2f}</div>
                        <div style="color: #64748b; font-size: 0.85rem;">95% Range: <b>${lower_usd:,.0f} — ${upper_usd:,.0f}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    if capped_at_new:
                        st.caption("ℹ️ Resale valuation was capped at the original retail price ($" + f"{orig_price:,.2f}).")
                    
                with m2:
                    badge_class = 'badge-premium' if tier_name == 'Premium' else ('badge-mid' if tier_name == 'Mid-range' else 'badge-budget')
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Market Tier & Retention</div>
                        <div style="margin: 0.4rem 0;">
                            <span class="badge {badge_class}" style="font-size: 1.1rem; padding: 0.35rem 0.9rem;">{tier_name} Tier</span>
                        </div>
                        <div style="color: #64748b; font-size: 0.85rem;">Value Retained: <b style="color: {'#4ade80' if retention_rate > 50 else '#fbbf24'};">{retention_rate:.1f}%</b> of original</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Persona and Spec Rating
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem 1.25rem; border-radius: 10px; border-left: 4px solid #38bdf8; margin-bottom: 1.5rem;">
                    <b style="color: #f8fafc;">Cluster Persona:</b> <span style="color: #38bdf8; font-weight: 600;">{persona_name}</span> &nbsp;|&nbsp; 
                    <b style="color: #f8fafc;">Spec Score:</b> <span style="color: #4ade80; font-weight: 600;">{specs_score:.2f} pts</span> &nbsp;|&nbsp;
                    <b style="color: #f8fafc;">Days Used:</b> {days_used} days ({phone_age_years:.1f} yrs)
                </div>
                """, unsafe_allow_html=True)
                
                # Interactive Monotonic Depreciation Simulator
                st.markdown("#### 📉 Projected Depreciation Horizon")
                depr_res = compute_depreciation_horizon(
                    specs=input_specs,
                    regressor=regressor,
                    preprocessor=preprocessor,
                    orig_price_usd=orig_price,
                    months_eval=[0, 3, 6, 9, 12, 18, 24, 36]
                )
                
                current_months = days_used / 30.4375
                
                fig_depr = go.Figure()
                fig_depr.add_trace(go.Scatter(
                    x=depr_res['months'],
                    y=depr_res['prices'],
                    mode='lines+markers',
                    name='Fair Market Valuation',
                    line=dict(color='#38bdf8', width=3),
                    marker=dict(size=7, color='#0284c7'),
                    hovertemplate='Age: %{x} months<br>Valuation: $%{y:,.2f}<extra></extra>'
                ))
                
                # Mark Today with vertical dashed line
                fig_depr.add_vline(
                    x=current_months,
                    line_dash="dash",
                    line_color="#f59e0b",
                    line_width=2,
                    annotation_text=f"Today ({current_months:.1f}m / {days_used}d)",
                    annotation_position="top right",
                    annotation_font=dict(color="#f59e0b", size=11)
                )
                
                fig_depr.update_layout(
                    xaxis_title="Months of Use",
                    yaxis_title="Estimated Resale Value ($ USD)",
                    margin=dict(l=15, r=15, t=25, b=15),
                    height=240,
                    template="plotly_dark",
                    paper_bgcolor="rgba(15, 23, 42, 0.4)",
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 3, 6, 9, 12, 18, 24, 36],
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.08)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.08)',
                        tickprefix="$"
                    )
                )
                st.plotly_chart(fig_depr, use_container_width=True)
                
            except Exception as e:
                st.error(f"Inference error: {str(e)}")

# ==============================================================================
# TAB 2: MODEL LEADERBOARD & EXPERIMENT ZOO
# ==============================================================================
with tab2:
    st.markdown("### 🏆 Comprehensive Model Zoo & Experiment Leaderboards")
    st.markdown("Benchmarking 10 machine learning paradigms on the Used Mobile Phone Dataset.")
    
    col_lb1, col_lb2 = st.columns(2)
    
    with col_lb1:
        st.markdown("#### Retention-Based Continuous Resale Modeling (Production Architecture)")
        st.caption("Target: retention ratio = used_price / new_price. Resale = predicted_retention * new_price. Resale <= new price is guaranteed by construction.")
        leaderboard_ret = metadata.get('leaderboard_retention', [])
        if leaderboard_ret:
            st.dataframe(pd.DataFrame(leaderboard_ret).round(4), use_container_width=True)
            
        old_bl = metadata.get('old_baseline_direct_price', {})
        if old_bl:
            st.info(f"**Baseline Direct Price Model (Old Unconstrained GBDT):** Test R² = {old_bl.get('test_r2', 0.9417):.4f}, RMSE = {old_bl.get('rmse', 0.0288):.4f}, MAE = {old_bl.get('mae', 0.0238):.4f}.  \n*Retention modeling achieves superior resale accuracy (Implied Resale R² = {metadata.get('implied_resale_r2', 0.9447):.4f}, RMSE = {metadata.get('implied_resale_rmse', 0.0281):.4f}) while mathematically eliminating the >100% retention bug.*")
        
    with col_lb2:
        st.markdown("#### Price Tier Market Classification")
        cls_data = [
            {"Model": "Multinomial Logistic Regression", "Accuracy": f"{metadata.get('best_cls_accuracy', 86.48):.2f}%", "Macro F1": "0.87", "Type": "Probabilistic Linear"},
            {"Model": "Support Vector Classifier (RBF)", "Accuracy": "84.38%", "Macro F1": "0.85", "Type": "Max-Margin Kernel"},
            {"Model": "KNN Classifier (k=7)", "Accuracy": "78.10%", "Macro F1": "0.78", "Type": "Non-Parametric"},
            {"Model": "AdaBoost Classifier", "Accuracy": "73.71%", "Macro F1": "0.74", "Type": "Ensemble Stumps"}
        ]
        st.dataframe(pd.DataFrame(cls_data), use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 🖼️ Model Performance Artifacts")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        reg_lb_img = os.path.join(REPORTS_DIR, 'exp11_regression_leaderboard.png')
        if os.path.exists(reg_lb_img):
            st.image(reg_lb_img, caption="Regression Model Leaderboard across 10 Algorithms", use_container_width=True)
    with fig_col2:
        res_diag_img = os.path.join(REPORTS_DIR, 'exp10_residuals_diagnostics.png')
        if os.path.exists(res_diag_img):
            st.image(res_diag_img, caption="Normal Q-Q Plot and Residual Distribution of Best Booster", use_container_width=True)

# ==============================================================================
# TAB 3: EXPLAINABLE AI (SHAP)
# ==============================================================================
with tab3:
    st.markdown("### 🧠 Explainable AI: Feature Attribution via SHAP")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) computes cooperative game-theoretic Shapley values
    to explain the exact contribution of each phone attribute to the final predicted valuation.
    """)
    
    shap_img = os.path.join(REPORTS_DIR, 'exp11_shap_summary.png')
    if os.path.exists(shap_img):
        st.image(shap_img, caption="Global SHAP Beeswarm Summary Plot: Feature Impact on Resale Valuation", use_container_width=True)
    else:
        st.info("SHAP visualization artifact available in reports/figures/.")
        
    st.markdown("""
    #### 🔑 Key Valuation Insights from SHAP Analysis:
    1. **`normalized_new_price` (Primary Anchor):** Original retail price exerts the single strongest positive force. High new prices shift valuation rightwards by up to +$180.
    2. **`days_used` & `phone_age_years` (Depreciation Velocity):** Rapidly pulls value down in the first 500 days.
    3. **`brand_tier` & `brand_Apple`:** Apple and Google devices retain a substantial valuation premium (up to +$65) compared to equivalent Android hardware.
    4. **`ram` and `internal_memory`:** Memory configurations provide linear price lift, especially between 6GB and 12GB RAM.
    """)

# ==============================================================================
# TAB 4: MULTI-ARMED BANDIT DYNAMIC PRICING
# ==============================================================================
with tab4:
    st.markdown("### 💰 Reinforcement Learning: Reseller Multi-Armed Bandit")
    st.markdown("""
    Second-hand retailers face a classic **Exploration vs Exploitation** tradeoff:
    - Markup too high $\\rightarrow$ High margin per unit, but conversion rate drops.
    - Discount too low $\\rightarrow$ High sales velocity, but leaving money on the table.
    """)
    
    col_b1, col_b2 = st.columns([1, 1.2], gap="large")
    with col_b1:
        st.markdown("#### Pricing Strategies (Arms)")
        bandit_arms_df = pd.DataFrame([
            {"Arm": "Arm 0: Premium", "Markup": "+15%", "Conv. Rate": "35%", "Expected Reward Factor": "0.402"},
            {"Arm": "Arm 1: Fair Market", "Markup": "0%", "Conv. Rate": "65%", "Expected Reward Factor": "0.650"},
            {"Arm": "Arm 2: Quick Sale", "Markup": "-10%", "Conv. Rate": "85%", "Expected Reward Factor": "0.765 (Optimal)"},
            {"Arm": "Arm 3: Clearance", "Markup": "-25%", "Conv. Rate": "95%", "Expected Reward Factor": "0.712"}
        ])
        st.dataframe(bandit_arms_df, use_container_width=True)
        st.success("🎯 **Algorithm Recommendation:** Arm 2 (Competitive Quick Sale at 10% discount) yields the highest expected profit by optimizing the conversion-margin frontier!")
        
    with col_b2:
        bandit_img = os.path.join(REPORTS_DIR, 'exp11_bandit_regret.png')
        if os.path.exists(bandit_img):
            st.image(bandit_img, caption="Cumulative Regret Comparison: Epsilon-Greedy vs UCB1 vs Random", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
    Used Mobile Phone Price Prediction & Market Analysis System • Machine Learning Mini-Project • Reproducible Pipeline
</div>
""", unsafe_allow_html=True)
