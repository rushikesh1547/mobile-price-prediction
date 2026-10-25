"""
generate_data.py — Create a realistic synthetic used-phone dataset.

Labels the data clearly as SYNTHETIC.
Generates ~3500 rows with believable price relationships and ~7% missing values.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import set_seed, DATA_RAW

set_seed(42)

N = 3500

# ── Brand & OS definitions ──────────────────────────────────────────────────
BRANDS = {
    # brand: (market_share_weight, tier_mean, os)
    'Samsung':  (0.20, 0.55, 'Android'),
    'Apple':    (0.15, 0.80, 'iOS'),
    'Xiaomi':   (0.15, 0.35, 'Android'),
    'OnePlus':  (0.08, 0.60, 'Android'),
    'Huawei':   (0.07, 0.50, 'Android'),
    'Oppo':     (0.08, 0.40, 'Android'),
    'Vivo':     (0.08, 0.38, 'Android'),
    'Realme':   (0.07, 0.30, 'Android'),
    'Nokia':    (0.04, 0.28, 'Android'),
    'Motorola': (0.04, 0.32, 'Android'),
    'Google':   (0.02, 0.65, 'Android'),
    'Sony':     (0.01, 0.55, 'Android'),
    'LG':       (0.01, 0.35, 'Android'),
}

brand_names = list(BRANDS.keys())
weights = np.array([BRANDS[b][0] for b in brand_names])
weights /= weights.sum()
tier_means = {b: BRANDS[b][1] for b in brand_names}
os_map = {b: BRANDS[b][2] for b in brand_names}

# ── Sample brands ────────────────────────────────────────────────────────────
brands = np.random.choice(brand_names, size=N, p=weights)
os_col = np.array([os_map[b] for b in brands])

# ── Spec generation (correlated with brand tier) ─────────────────────────────
tier_vals = np.array([tier_means[b] for b in brands])

# Release year: 2015-2024, weighted toward recent
release_year = np.random.choice(range(2015, 2025), size=N,
                                 p=[0.03, 0.04, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.16])

# Days used: 30-2500, correlated with release year (older = more days)
max_days = (2024 - release_year) * 365 + 180
days_used = np.clip(
    (max_days * np.random.beta(2, 3, N)).astype(int),
    30, 2500
)

# Screen size: 4.5-7.2 inches
screen_size = np.round(np.clip(
    np.random.normal(6.2, 0.5, N) + (tier_vals - 0.5) * 0.3,
    4.5, 7.2
), 1)

# RAM (GB): correlated with tier
ram_options = np.array([2, 3, 4, 6, 8, 12, 16])
ram_probs_base = np.array([0.05, 0.08, 0.20, 0.25, 0.22, 0.15, 0.05])
ram = np.zeros(N, dtype=int)
for i in range(N):
    shift = int((tier_vals[i] - 0.4) * 3)
    probs = np.roll(ram_probs_base, shift)
    probs = np.clip(probs, 0.01, None)
    probs /= probs.sum()
    ram[i] = np.random.choice(ram_options, p=probs)

# Internal memory (GB)
mem_options = np.array([16, 32, 64, 128, 256, 512])
mem_probs_base = np.array([0.05, 0.10, 0.25, 0.30, 0.20, 0.10])
internal_memory = np.zeros(N, dtype=int)
for i in range(N):
    shift = int((tier_vals[i] - 0.4) * 2)
    probs = np.roll(mem_probs_base, shift)
    probs = np.clip(probs, 0.01, None)
    probs /= probs.sum()
    internal_memory[i] = np.random.choice(mem_options, p=probs)

# Battery (mAh): 2000-6000
battery = np.clip(
    np.random.normal(4200, 700, N) + (tier_vals - 0.5) * 500,
    2000, 6500
).astype(int)

# Main camera (MP): 8-200
main_camera = np.clip(
    np.random.choice([8, 12, 13, 16, 48, 50, 64, 108, 200], N,
                      p=[0.03, 0.08, 0.07, 0.10, 0.25, 0.22, 0.12, 0.10, 0.03])
    + (tier_vals * 10).astype(int),
    8, 200
).astype(int)

# Selfie camera (MP): 2-32
selfie_camera = np.clip(
    np.random.choice([2, 5, 8, 12, 16, 20, 25, 32], N,
                      p=[0.03, 0.07, 0.12, 0.18, 0.25, 0.18, 0.10, 0.07])
    + (tier_vals * 5).astype(int),
    2, 44
).astype(int)

# Weight (grams): 130-240
weight = np.clip(
    np.random.normal(180, 20, N) + (screen_size - 6.0) * 10,
    120, 250
).astype(int)

# 4G: almost all yes, except very old budget
has_4g = np.where((release_year >= 2016) | (np.random.rand(N) > 0.3), 'yes', 'no')

# 5G: only newer and higher-tier
has_5g = np.where(
    (release_year >= 2020) & (tier_vals > 0.4) & (np.random.rand(N) > 0.3),
    'yes', 'no'
)

# ── Price generation ─────────────────────────────────────────────────────────
# Normalized new price (0-1 range) based on brand tier + specs
new_price = (
    tier_vals * 0.35 +
    (ram / 16) * 0.15 +
    (internal_memory / 512) * 0.10 +
    (main_camera / 200) * 0.08 +
    (battery / 6500) * 0.05 +
    (screen_size / 7.2) * 0.05 +
    np.where(has_5g == 'yes', 0.08, 0.0) +
    np.random.normal(0, 0.04, N)  # noise
)
new_price = np.clip(new_price, 0.05, 1.0)

# Used price: depreciates with days_used, brand affects retention
# Apple and OnePlus retain value better
retention_brand = np.array([
    {'Samsung': 0.70, 'Apple': 0.82, 'Xiaomi': 0.60, 'OnePlus': 0.72,
     'Huawei': 0.55, 'Oppo': 0.58, 'Vivo': 0.57, 'Realme': 0.55,
     'Nokia': 0.50, 'Motorola': 0.52, 'Google': 0.65, 'Sony': 0.55, 'LG': 0.45
    }[b] for b in brands
])

# Depreciation curve: exponential decay
depreciation = retention_brand * np.exp(-days_used / 1800) + (1 - retention_brand) * np.exp(-days_used / 600)
used_price = new_price * depreciation + np.random.normal(0, 0.03, N)
used_price = np.clip(used_price, 0.01, new_price * 0.98)  # can't exceed new price

# ── Assemble DataFrame ──────────────────────────────────────────────────────
df = pd.DataFrame({
    'brand': brands,
    'os': os_col,
    'screen_size': screen_size,
    '4g': has_4g,
    '5g': has_5g,
    'main_camera_mp': main_camera,
    'selfie_camera_mp': selfie_camera,
    'internal_memory': internal_memory,
    'ram': ram,
    'battery': battery,
    'weight': weight,
    'release_year': release_year,
    'days_used': days_used,
    'normalized_new_price': np.round(new_price, 4),
    'normalized_used_price': np.round(used_price, 4),
})

# ── Inject ~7% missing values in selected columns ───────────────────────────
missing_cols = ['ram', 'battery', 'main_camera_mp', 'selfie_camera_mp',
                'internal_memory', 'weight', 'screen_size']
for col in missing_cols:
    mask = np.random.rand(N) < 0.07
    df.loc[mask, col] = np.nan

# ── Add metadata ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  SYNTHETIC USED PHONE DATASET GENERATOR")
print("=" * 70)
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n  NOTE: This dataset is SYNTHETIC (generated for educational purposes).")
print(f"  Price relationships are designed to be realistic but are simulated.\n")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_RAW, 'used_phones.csv')
df.to_csv(out_path, index=False)
print(f"  Saved to: {out_path}")

# Quick stats
print(f"\n  Price stats:")
print(f"    New price:  mean={df['normalized_new_price'].mean():.3f}, "
      f"std={df['normalized_new_price'].std():.3f}")
print(f"    Used price: mean={df['normalized_used_price'].mean():.3f}, "
      f"std={df['normalized_used_price'].std():.3f}")
print(f"\n  Brand distribution:")
print(df['brand'].value_counts().to_string())
