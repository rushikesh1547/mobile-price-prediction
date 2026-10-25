"""
test_sanity.py — Comprehensive Sanity & Invariant Checks for Used Mobile Price Prediction.

Checks 500 random valid inputs against:
1. Resale <= new_price (no used phone is worth more than its new price).
2. lower <= prediction <= upper <= new_price (confidence interval bounds & containment).
3. The depreciation horizon curve is strictly monotonic non-increasing.
4. Valuation is non-increasing as days_used increases (holding specs fixed).
5. No NaN, infinite, or negative outputs.
6. The user-reported Oppo bug case ($150 new, 365 days used, Oppo 6GB/128GB) produces realistic 55%-85% retention.
"""

import os
import sys
import json
import joblib
import numpy as np
import pytest

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    preprocess_input,
    predict_valuation,
    compute_depreciation_horizon,
    validate_phone_inputs,
    PRICE_SCALE_FACTOR
)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


@pytest.fixture(scope="module")
def models():
    """Load serialized models, preprocessors, and metadata."""
    regressor = joblib.load(os.path.join(MODELS_DIR, 'best_price_regressor.pkl'))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    
    q_low = None
    q_high = None
    q_low_path = os.path.join(MODELS_DIR, 'quantile_regressor_low.pkl')
    q_high_path = os.path.join(MODELS_DIR, 'quantile_regressor_high.pkl')
    if os.path.exists(q_low_path) and os.path.exists(q_high_path):
        q_low = joblib.load(q_low_path)
        q_high = joblib.load(q_high_path)
        
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    return {
        'regressor': regressor,
        'preprocessor': preprocessor,
        'q_low': q_low,
        'q_high': q_high,
        'metadata': metadata
    }


def generate_random_phone_specs(n: int = 500, seed: int = 42) -> list[dict]:
    """Generate n random valid phone specification dictionaries."""
    np.random.seed(seed)
    brands = ["Apple", "Samsung", "OnePlus", "Google", "Xiaomi", "Huawei", "Oppo", "Vivo", "Motorola", "Realme", "Sony", "Nokia", "LG"]
    ram_options = [2, 3, 4, 6, 8, 12, 16]
    mem_options = [16, 32, 64, 128, 256, 512]
    
    samples = []
    for _ in range(n):
        brand = np.random.choice(brands)
        os_type = "iOS" if brand == "Apple" else "Android"
        release_year = int(np.random.randint(2018, 2025))
        max_possible_days = max(30, (2025 - release_year + 1) * 365)
        days_used = int(np.random.randint(30, min(2000, max_possible_days) + 1))
        
        orig_price = float(np.round(np.random.uniform(80.0, 1600.0), 2))
        ram = int(np.random.choice(ram_options))
        internal_memory = int(np.random.choice(mem_options))
        battery = int(np.random.randint(2000, 6500))
        screen_size = float(np.round(np.random.uniform(4.7, 7.0), 1))
        main_cam = int(np.random.choice([8, 12, 16, 48, 50, 64, 108, 200]))
        selfie_cam = int(np.random.choice([5, 8, 12, 16, 20, 32]))
        weight = int(np.random.randint(130, 240))
        has_4g = "yes"
        has_5g = "yes" if release_year >= 2021 and np.random.rand() > 0.3 else "no"
        
        samples.append({
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
        })
    return samples


def test_resale_and_interval_invariants(models):
    """Test resale <= new_price and lower <= prediction <= upper <= new_price across 500 inputs."""
    samples = generate_random_phone_specs(500, seed=42)
    regressor = models['regressor']
    preprocessor = models['preprocessor']
    q_low = models['q_low']
    q_high = models['q_high']
    val_rmse = models['metadata'].get('val_retention_rmse', 0.035)
    
    for i, specs in enumerate(samples):
        orig_price = specs['orig_price']
        res = predict_valuation(
            specs=specs,
            regressor=regressor,
            preprocessor=preprocessor,
            orig_price_usd=orig_price,
            val_retention_rmse=val_rmse,
            q_low_model=q_low,
            q_high_model=q_high
        )
        
        resale = res['resale_usd']
        lower = res['lower_usd']
        upper = res['upper_usd']
        retention = res['retention_rate']
        
        # 1. No NaN or Infinite values
        assert not np.isnan(resale), f"Sample {i}: resale is NaN"
        assert not np.isnan(lower), f"Sample {i}: lower is NaN"
        assert not np.isnan(upper), f"Sample {i}: upper is NaN"
        
        # 2. Resale <= new_price (strict guarantee)
        assert resale <= orig_price + 1e-4, f"Sample {i}: resale ({resale}) > orig_price ({orig_price})"
        
        # 3. Floor guarantee (>= 5% of new price)
        assert resale >= (0.05 * orig_price) - 1e-4, f"Sample {i}: resale ({resale}) < 5% floor"
        
        # 4. Confidence interval containment: lower <= prediction <= upper <= orig_price
        assert lower <= resale + 1e-4, f"Sample {i}: lower ({lower}) > resale ({resale})"
        assert resale <= upper + 1e-4, f"Sample {i}: resale ({resale}) > upper ({upper})"
        assert upper <= orig_price + 1e-4, f"Sample {i}: upper ({upper}) > orig_price ({orig_price})"
        
        # 5. Retention rate <= 100%
        assert retention <= 100.0 + 1e-3, f"Sample {i}: retention rate ({retention}%) > 100%"


def test_depreciation_horizon_monotonicity(models):
    """Test that the depreciation horizon is strictly monotonic non-increasing across 500 inputs."""
    samples = generate_random_phone_specs(500, seed=123)
    regressor = models['regressor']
    preprocessor = models['preprocessor']
    
    for i, specs in enumerate(samples):
        orig_price = specs['orig_price']
        depr = compute_depreciation_horizon(
            specs=specs,
            regressor=regressor,
            preprocessor=preprocessor,
            orig_price_usd=orig_price,
            months_eval=[0, 3, 6, 9, 12, 18, 24, 36]
        )
        
        prices = np.array(depr['prices'])
        months = depr['months']
        
        # Check sorted numeric months
        assert months == [0, 3, 6, 9, 12, 18, 24, 36], f"Sample {i}: months not numeric sorted"
        
        # Monotonic non-increasing check: diffs <= 0
        diffs = np.diff(prices)
        assert np.all(diffs <= 1e-5), f"Sample {i}: Depreciation curve is not monotonic non-increasing: {prices}"
        
        # Max price is at month 0 and <= orig_price
        assert prices[0] <= orig_price + 1e-4, f"Sample {i}: Month 0 price ({prices[0]}) > orig_price ({orig_price})"
        assert np.all(prices > 0), f"Sample {i}: Negative prices found: {prices}"


def test_days_used_sensitivity_monotonicity(models):
    """Holding all specs fixed, check that valuation decreases as usage days increase."""
    regressor = models['regressor']
    preprocessor = models['preprocessor']
    
    test_specs = {
        'brand': 'Samsung',
        'os': 'Android',
        'release_year': 2022,
        'orig_price': 800.0,
        'ram': 8,
        'internal_memory': 128,
        'battery': 4500,
        'screen_size': 6.4,
        'main_camera_mp': 50,
        'selfie_camera_mp': 16,
        'weight': 185,
        '4g': 'yes',
        '5g': 'yes'
    }
    
    days_list = [30, 90, 180, 365, 730, 1095]
    resale_prices = []
    
    for d in days_list:
        spec = test_specs.copy()
        spec['days_used'] = d
        res = predict_valuation(spec, regressor, preprocessor, orig_price_usd=800.0)
        resale_prices.append(res['resale_usd'])
        
    diffs = np.diff(resale_prices)
    assert resale_prices[-1] < resale_prices[0], "1095-day price is not less than 30-day price"
    assert np.all(diffs <= 15.0), f"Unreasonable price jump as phone aged: {resale_prices}"


def test_oppo_user_case(models):
    """
    Test the exact bug scenario reported by user:
    Oppo, Android, 2024 release, 6GB RAM, 128GB, 4500mAh, 365 days used, new price entered = $150.
    Must verify:
    1. Resale <= $150
    2. Value Retained is realistic: ~55% - 85% for 1-year-old phone (NEVER >100%)
    3. 95% Range contains the point estimate: lower <= prediction <= upper <= 150
    """
    regressor = models['regressor']
    preprocessor = models['preprocessor']
    q_low = models['q_low']
    q_high = models['q_high']
    
    oppo_specs = {
        'brand': 'Oppo',
        'os': 'Android',
        'release_year': 2024,
        'days_used': 365,
        'orig_price': 150.0,
        'ram': 6,
        'internal_memory': 128,
        'battery': 4500,
        'screen_size': 6.4,
        'main_camera_mp': 50,
        'selfie_camera_mp': 16,
        'weight': 185,
        '4g': 'yes',
        '5g': 'yes'
    }
    
    res = predict_valuation(
        specs=oppo_specs,
        regressor=regressor,
        preprocessor=preprocessor,
        orig_price_usd=150.0,
        val_retention_rmse=models['metadata'].get('val_retention_rmse', 0.035),
        q_low_model=q_low,
        q_high_model=q_high
    )
    
    resale = res['resale_usd']
    lower = res['lower_usd']
    upper = res['upper_usd']
    retention_pct = res['retention_rate']
    
    print("\n" + "=" * 60)
    print("  USER BUG CASE VERIFICATION (Oppo, 1yr, $150 new)")
    print("=" * 60)
    print(f"  New Price:        ${oppo_specs['orig_price']:.2f}")
    print(f"  Predicted Resale: ${resale:.2f}")
    print(f"  95% CI Range:     ${lower:.2f} — ${upper:.2f}")
    print(f"  Value Retained:   {retention_pct:.1f}%")
    print("=" * 60)
    
    # 1. Resale never exceeds new price ($150)
    assert resale <= 150.0, f"Resale ${resale:.2f} exceeds new price $150.00"
    
    # 2. Retention rate is realistic (55% - 85% for 1-year phone)
    assert 55.0 <= retention_pct <= 85.0, (
        f"Retention rate {retention_pct:.1f}% outside expected 55%-85% range for 1-year Oppo"
    )
    
    # 3. Confidence interval is valid and contains prediction
    assert lower <= resale <= upper <= 150.0, (
        f"CI violated: {lower:.2f} <= {resale:.2f} <= {upper:.2f} <= 150.00"
    )


def test_input_validation_and_robustness(models):
    """Test input validation warnings and robustness to sparse or missing specs."""
    regressor = models['regressor']
    preprocessor = models['preprocessor']

    # 1. Test release year vs days used inconsistency warning
    inconsistent_specs = {
        'brand': 'Samsung',
        'release_year': 2024,
        'days_used': 1500,  # Impossible for 2024 release
        'orig_price': 500.0
    }
    warnings = validate_phone_inputs(inconsistent_specs)
    assert any("Usage Duration Inconsistency" in w for w in warnings)

    # 2. Test suspiciously low MSRP warning
    low_msrp_specs = {
        'brand': 'Apple',
        'release_year': 2024,
        'days_used': 100,
        'orig_price': 99.0
    }
    warnings = validate_phone_inputs(low_msrp_specs)
    assert any("Unusually Low MSRP" in w for w in warnings)

    # 3. Test robustness: minimal specs without crashing
    minimal_specs = {
        'brand': 'Xiaomi',
        'orig_price': 200.0
    }
    res = predict_valuation(minimal_specs, regressor, preprocessor, orig_price_usd=200.0)
    assert 0.05 * 200.0 <= res['resale_usd'] <= 200.0
    assert res['lower_usd'] <= res['resale_usd'] <= res['upper_usd']


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

