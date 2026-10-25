"""
preprocessing.py — Data loading, cleaning, target creation, train/test splits,
and unified preprocessing & inference logic.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import DATA_RAW, DATA_PROCESSED, RANDOM_STATE, TIER_LABELS
from src.features import engineer_all_features, BRAND_TIER

# ── Currency & Scale Constants ───────────────────────────────────────────────
# Currency is USD ($).
# The normalized price scale maps $1,500.00 USD MSRP to normalized_new_price = 1.0.
PRICE_SCALE_FACTOR = 1500.0
CURRENCY_SYMBOL = "$"
CURRENCY_CODE = "USD"

NUMERIC_FEATURES = [
    'screen_size', 'main_camera_mp', 'selfie_camera_mp',
    'internal_memory', 'ram', 'battery', 'weight',
    'release_year', 'days_used', 'normalized_new_price',
    'phone_age_years', 'brand_tier', 'total_camera_mp',
    'specs_per_price', 'log_days_used', 'log_new_price',
    'ram_x_memory', 'age_x_brand_tier'
]

CATEGORICAL_FEATURES = ['brand', 'os', '4g', '5g']
ALL_MODEL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data() -> pd.DataFrame:
    """Load the raw synthetic used-phone dataset."""
    path = os.path.join(DATA_RAW, 'used_phones.csv')
    df = pd.read_csv(path)
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add classification and retention targets to the dataframe.

    - retention_ratio: used_price / new_price clipped to [0.05, 1.0]
    - is_premium: binary (1 if top-33% of used price, 0 otherwise)
    - price_tier: 'Budget' / 'Mid-range' / 'Premium' via quantiles
    """
    df = df.copy()
    if 'normalized_used_price' in df.columns and 'normalized_new_price' in df.columns:
        norm_new = np.maximum(df['normalized_new_price'], 0.01)
        df['retention_ratio'] = np.clip(df['normalized_used_price'] / norm_new, 0.05, 1.0)

    q33 = df['normalized_used_price'].quantile(0.33)
    q66 = df['normalized_used_price'].quantile(0.66)

    df['price_tier'] = pd.cut(
        df['normalized_used_price'],
        bins=[-np.inf, q33, q66, np.inf],
        labels=TIER_LABELS
    )
    df['is_premium'] = (df['price_tier'] == 'Premium').astype(int)
    return df


def get_numeric_features() -> list:
    """Return list of numeric feature column names."""
    return list(NUMERIC_FEATURES)


def get_categorical_features() -> list:
    """Return list of categorical feature column names."""
    return list(CATEGORICAL_FEATURES)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fill missing numerics with median for quick experiments."""
    df = df.copy()
    num_cols = [c for c in get_numeric_features() if c in df.columns]
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""
    df = df.copy()
    cat_cols = [c for c in get_categorical_features() if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def preprocess_input(
    data: pd.DataFrame | dict | list[dict],
    preprocessor=None,
    is_training: bool = False
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Unified preprocessing function shared by training and inference.
    
    Currency & Scale Specification:
    -------------------------------
    - Currency: USD ($).
    - Retail Price Scale:
      The training dataset normalizes MSRP using a reference scale where $1,500.00 USD = 1.0.
      If input specifies 'orig_price', 'new_price', or 'new_price_usd', it is converted via:
          normalized_new_price = orig_price / PRICE_SCALE_FACTOR (1500.0)
      If 'normalized_new_price' is already present, it is preserved and clipped to [0.01, 2.5].
    - Target Modeling:
      The regression model predicts 'retention_ratio' = used_price / new_price in [0.05, 1.0].
      At inference, resale valuation in USD is computed via:
          resale_usd = predicted_retention * orig_price_usd
      This guarantees resale <= orig_price by construction.
    
    Parameters:
    -----------
    data : pd.DataFrame, dict, or list of dicts
        Input phone specifications.
    preprocessor : sklearn ColumnTransformer, optional
        Fitted ColumnTransformer pipeline. If provided, returns the transformed numpy array.
    is_training : bool, default False
        Whether called during training.
        
    Returns:
    --------
    tuple[pd.DataFrame, np.ndarray | None]:
        df_features: DataFrame containing the 22 required features in exact column order.
        X_proc: Transformed array if preprocessor is passed, else None.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Map raw price to normalized_new_price if needed
    if 'normalized_new_price' not in df.columns:
        if 'orig_price' in df.columns:
            df['normalized_new_price'] = df['orig_price'] / PRICE_SCALE_FACTOR
        elif 'new_price' in df.columns:
            df['normalized_new_price'] = df['new_price'] / PRICE_SCALE_FACTOR
        elif 'new_price_usd' in df.columns:
            df['normalized_new_price'] = df['new_price_usd'] / PRICE_SCALE_FACTOR
        else:
            df['normalized_new_price'] = 0.5  # Neutral default

    df['normalized_new_price'] = df['normalized_new_price'].clip(lower=0.01)

    # Defaults for optional / missing columns
    defaults = {
        'brand': 'Samsung',
        'os': 'Android',
        'screen_size': 6.2,
        'main_camera_mp': 50.0,
        'selfie_camera_mp': 16.0,
        'internal_memory': 128.0,
        'ram': 8.0,
        'battery': 4200.0,
        'weight': 180.0,
        'release_year': 2024,
        'days_used': 365,
        '4g': 'yes',
        '5g': 'yes'
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    # Correct OS for Apple if mismatched
    df.loc[df['brand'] == 'Apple', 'os'] = 'iOS'

    # Apply identical feature engineering
    df = engineer_all_features(df)

    # Reorder columns to match model expectations
    df_features = df[ALL_MODEL_INPUT_FEATURES].copy()

    X_proc = None
    if preprocessor is not None:
        X_proc = preprocessor.transform(df_features)

    return df_features, X_proc


def validate_phone_inputs(specs: dict) -> list[str]:
    """Validate user inputs and return advisory warning messages."""
    warnings = []
    release_year = int(specs.get('release_year', 2024))
    days_used = int(specs.get('days_used', 365))

    # Reference year: 2025 (or 2026)
    max_days = max(30, (2025 - release_year + 1) * 365)
    if days_used > max_days:
        warnings.append(
            f"Usage Duration Inconsistency: A phone released in {release_year} cannot have been in use for {days_used} days (exceeds plausible maximum of ~{max_days} days)."
        )

    orig_price = float(specs.get('orig_price', specs.get('new_price', specs.get('new_price_usd', 899.0))))
    ram = float(specs.get('ram', 8))
    internal_memory = float(specs.get('internal_memory', 128))
    brand = str(specs.get('brand', ''))

    if brand in ['Apple', 'Google'] and orig_price < 250.0:
        warnings.append(
            f"Unusually Low MSRP: ${orig_price:.0f} is unusually low for a new {brand} device (typical retail MSRP is $600–$1,200). Valuation may be distorted."
        )
    elif (ram >= 12 or internal_memory >= 512) and orig_price < 200.0:
        warnings.append(
            f"Unusually Low MSRP: ${orig_price:.0f} is unusually low for flagship specifications ({int(ram)}GB RAM, {int(internal_memory)}GB storage)."
        )
    elif orig_price > 1200.0 and ram <= 4 and internal_memory <= 64 and brand != 'Apple':
        warnings.append(
            f"Unusually High MSRP: ${orig_price:.0f} is unusually high for entry-level specifications ({int(ram)}GB RAM, {int(internal_memory)}GB storage)."
        )

    return warnings


def predict_valuation(
    specs: dict,
    regressor,
    preprocessor,
    orig_price_usd: float,
    val_retention_rmse: float = 0.035,
    q_low_model=None,
    q_high_model=None
) -> dict:
    """Compute point prediction, 95% confidence interval, and sanity guards in retention space."""
    # Ensure specs contains orig_price
    specs_copy = specs.copy()
    specs_copy['orig_price'] = orig_price_usd
    specs_copy['normalized_new_price'] = orig_price_usd / PRICE_SCALE_FACTOR

    df_feat, X_proc = preprocess_input(specs_copy, preprocessor=preprocessor)

    # 1. Point prediction in retention space [0.05, 1.0]
    raw_pred_retention = float(regressor.predict(X_proc)[0])
    pred_retention = float(np.clip(raw_pred_retention, 0.05, 1.0))

    # 2. 95% Prediction Interval in retention space
    if q_low_model is not None and q_high_model is not None:
        low_ret = float(q_low_model.predict(X_proc)[0])
        high_ret = float(q_high_model.predict(X_proc)[0])
    else:
        margin = 1.96 * val_retention_rmse
        low_ret = pred_retention - margin
        high_ret = pred_retention + margin

    # Guarantee lower <= pred <= upper within [0.05, 1.0]
    low_ret = float(np.clip(min(low_ret, pred_retention), 0.05, 1.0))
    high_ret = float(np.clip(max(high_ret, pred_retention), 0.05, 1.0))

    # 3. Convert to USD with hard sanity guards
    floor_price = max(10.0, 0.05 * orig_price_usd)
    raw_resale_usd = pred_retention * orig_price_usd
    capped_at_new = raw_resale_usd >= orig_price_usd

    resale_usd = float(np.clip(raw_resale_usd, floor_price, orig_price_usd))
    lower_usd = float(np.clip(low_ret * orig_price_usd, floor_price, resale_usd))
    upper_usd = float(np.clip(high_ret * orig_price_usd, resale_usd, orig_price_usd))

    # Hard mathematical invariant assertion
    assert lower_usd <= resale_usd <= upper_usd <= orig_price_usd + 1e-4, (
        f"Sanity violation: {lower_usd:.2f} <= {resale_usd:.2f} <= {upper_usd:.2f} <= {orig_price_usd:.2f}"
    )

    return {
        'resale_usd': resale_usd,
        'lower_usd': lower_usd,
        'upper_usd': upper_usd,
        'retention_rate': (resale_usd / orig_price_usd) * 100.0,
        'pred_retention': pred_retention,
        'capped_at_new': capped_at_new,
        'floor_applied': raw_resale_usd < floor_price
    }


def compute_depreciation_horizon(
    specs: dict,
    regressor,
    preprocessor,
    orig_price_usd: float,
    months_eval: list[int] = None
) -> dict:
    """Predict resale price at ages [0, 3, 6, 9, 12, 18, 24, 36] months.
    
    Recomputes all age-dependent features at each evaluation point and enforces
    monotonic non-increasing prices via np.minimum.accumulate.
    """
    if months_eval is None:
        months_eval = [0, 3, 6, 9, 12, 18, 24, 36]

    months_eval = sorted(months_eval)

    base_specs = specs.copy()
    base_specs['orig_price'] = orig_price_usd
    base_specs['normalized_new_price'] = orig_price_usd / PRICE_SCALE_FACTOR

    # Batch process all evaluation months in a single call for high performance
    batch_specs = []
    for m in months_eval:
        m_days = max(1, int(round(m * 30.4375)))
        row_m = base_specs.copy()
        row_m['days_used'] = m_days
        batch_specs.append(row_m)

    _, X_batch = preprocess_input(batch_specs, preprocessor=preprocessor)
    ret_batch = regressor.predict(X_batch)
    ret_batch = np.clip(ret_batch, 0.05, 1.0)
    eval_prices = [float(p) for p in (ret_batch * orig_price_usd)]

    # Enforce monotonic non-increasing depreciation
    monotonic_prices = np.minimum.accumulate(eval_prices)
    floor_price = max(10.0, 0.05 * orig_price_usd)
    monotonic_prices = np.clip(monotonic_prices, floor_price, orig_price_usd)

    return {
        'months': months_eval,
        'prices': [float(p) for p in monotonic_prices],
        'raw_prices': eval_prices
    }


def prepare_regression_data(df: pd.DataFrame = None, target: str = 'normalized_used_price',
                             test_size: float = 0.15, val_size: float = 0.15, scale: bool = True):
    """Full preprocessing pipeline for regression experiments.

    Supports target='normalized_used_price' or target='retention' / 'retention_ratio'.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    if df is None:
        df = load_raw_data()
    df = create_targets(df)
    df = basic_clean(df)
    df = encode_categoricals(df)

    actual_target = 'retention_ratio' if target in ['retention', 'retention_ratio'] else 'normalized_used_price'
    drop_cols = ['normalized_used_price', 'retention_ratio', 'price_retention_ratio', 'price_tier', 'is_premium']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(float)
    y = df[actual_target].values

    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=RANDOM_STATE
    )
    # Val / test split
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=RANDOM_STATE
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler


def prepare_classification_data(target_col: str = 'price_tier', df: pd.DataFrame = None,
                                 test_size: float = 0.15, val_size: float = 0.15,
                                 scale: bool = True):
    """Full preprocessing pipeline for classification experiments.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler, le
    """
    if df is None:
        df = load_raw_data()
    df = create_targets(df)
    df = basic_clean(df)
    df = encode_categoricals(df)

    drop_cols = ['normalized_used_price', 'price_tier', 'is_premium']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(float)

    le = LabelEncoder()
    if target_col == 'price_tier':
        y = le.fit_transform(df['price_tier'])
    else:
        y = df[target_col].values
        le = None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=RANDOM_STATE,
        stratify=y
    )
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=RANDOM_STATE,
        stratify=y_temp
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler, le


def save_processed(df: pd.DataFrame, name: str = 'processed_phones.csv'):
    """Save a processed dataframe to data/processed/."""
    path = os.path.join(DATA_PROCESSED, name)
    df.to_csv(path, index=False)
    print(f"  Saved processed data: {path}")
    return path
