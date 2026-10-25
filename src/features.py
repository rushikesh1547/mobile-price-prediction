"""
features.py — Feature engineering functions for the phone price project.
"""
import numpy as np
import pandas as pd


# Brand tier mapping (derived from market positioning)
BRAND_TIER = {
    'Apple': 3, 'Google': 3, 'OnePlus': 2, 'Samsung': 2,
    'Sony': 2, 'Huawei': 2, 'Oppo': 1, 'Vivo': 1,
    'Xiaomi': 1, 'Motorola': 1, 'Realme': 1, 'Nokia': 1, 'LG': 1
}


def add_phone_age(df: pd.DataFrame) -> pd.DataFrame:
    """Add phone_age_years = days_used / 365."""
    df = df.copy()
    df['phone_age_years'] = df['days_used'] / 365.0
    return df


def add_price_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_retention_ratio = used_price / new_price (only if used price is in df)."""
    df = df.copy()
    if 'normalized_used_price' in df.columns and 'normalized_new_price' in df.columns:
        df['price_retention_ratio'] = np.where(
            df['normalized_new_price'] > 0,
            np.clip(df['normalized_used_price'] / df['normalized_new_price'], 0.05, 1.0),
            0.0
        )
    return df


def add_brand_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Add brand_tier (1=Budget, 2=Mid, 3=Premium)."""
    df = df.copy()
    df['brand_tier'] = df['brand'].map(BRAND_TIER).fillna(1).astype(int)
    return df


def add_camera_total(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_camera_mp = main + selfie camera."""
    df = df.copy()
    main = df['main_camera_mp'].fillna(50.0)
    selfie = df['selfie_camera_mp'].fillna(16.0)
    df['total_camera_mp'] = main + selfie
    return df


def add_specs_per_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add specs_per_price score (ram + memory + camera normalized by new price)."""
    df = df.copy()
    ram_norm = df['ram'].fillna(8.0) / 16.0
    mem_norm = df['internal_memory'].fillna(128.0) / 512.0
    cam_norm = df['main_camera_mp'].fillna(50.0) / 200.0
    spec_score = ram_norm + mem_norm + cam_norm
    norm_new = df['normalized_new_price'].clip(lower=0.01) if 'normalized_new_price' in df.columns else 0.5
    df['specs_per_price'] = spec_score / (norm_new + 1e-5)
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed versions of skewed features."""
    df = df.copy()
    df['log_days_used'] = np.log1p(df['days_used'].clip(lower=0))
    df['log_new_price'] = np.log1p(df['normalized_new_price'].clip(lower=0))
    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features."""
    df = df.copy()
    ram_val = df['ram'].fillna(8.0)
    mem_val = df['internal_memory'].fillna(128.0)
    df['ram_x_memory'] = (ram_val * mem_val) / 1000.0
    
    age = df['phone_age_years'] if 'phone_age_years' in df.columns else (df['days_used'] / 365.0)
    tier = df['brand_tier'] if 'brand_tier' in df.columns else df['brand'].map(BRAND_TIER).fillna(1).astype(int)
    df['age_x_brand_tier'] = age * tier
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_phone_age(df)
    df = add_price_retention(df)
    df = add_brand_tier(df)
    df = add_camera_total(df)
    df = add_specs_per_price(df)
    df = add_log_transforms(df)
    df = add_interaction_terms(df)
    return df
