# ============================================================
# EXTRACT VISSIM PARAMETERS PER VEHICLE CLASS
#
# Input : S:\Second\behavior_metrics_clean.csv
# Output: S:\Second\vissim_parameters_by_class.csv
#
# For each vehicle class (Car, CNG, Rickshaw, Truck, Bus, Bike)
# this script computes:
#   - CC0  : standstill distance (m)          -> Wiedemann 99
#   - CC1  : gap time headway (s)             -> Wiedemann 99
#   - CC8  : increased acceleration from rest -> Wiedemann 99
#   - Lateral minimum distance at 0 km/h (m)  -> Lateral tab
#   - Lateral minimum distance at 50 km/h (m) -> Lateral tab
# ============================================================

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------
IN_PATH  = r"S:\Second\behavior_metrics_clean.csv"
OUT_PATH = r"S:\Second\vissim_parameters_by_class.csv"

if not os.path.exists(IN_PATH):
    raise SystemExit(f"Input not found: {IN_PATH}")

df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df):,} rows, classes: {df['class_name'].unique().tolist()}")

# ------------------------------------------------------------
# 2. THRESHOLDS
# ------------------------------------------------------------
# These are used to slice the data into physically meaningful bands.
# All speeds in km/h.

NEAR_STOP_KMH  = 3.0     # "vehicle is basically stopped"
LOW_SPEED_KMH  = 10.0    # "vehicle is accelerating away from stop"
VERY_LOW_KMH   = 3.0     # for lateral gap at standstill

# Percentile choices (justified in comments)
CC0_PERCENTILE          = 0.10   # 10th percentile of spacing at slow speed
CC1_PERCENTILE          = 0.50   # median of headway
CC8_PERCENTILE          = 0.90   # 90th percentile of positive accel at start
LAT0_PERCENTILE         = 0.05   # 5th percentile of lateral gap at slow speed
LAT50_PERCENTILE        = 0.10   # 10th percentile of lateral gap at high speed

# ------------------------------------------------------------
# 3. LOOP OVER CLASSES
# ------------------------------------------------------------
results = []

for cls, g in df.groupby('class_name'):
    n_total = len(g)

    # ------------------------------------------------------
    # CC0 — standstill distance
    # ------------------------------------------------------
    # Use spacing at low speed (vehicle is crawling or stopped).
    # In non-lane-based traffic this value is smaller than in
    # lane-based traffic because vehicles nose into any gap.
    slow = g[g['speed_kmph'] < NEAR_STOP_KMH]
    if len(slow) >= 20:
        cc0 = slow['spacing_m'].dropna().quantile(CC0_PERCENTILE)
    else:
        # Fallback if too few slow samples: use global low percentile
        cc0 = g['spacing_m'].dropna().quantile(0.05)

    # ------------------------------------------------------
    # CC1 — gap time headway
    # ------------------------------------------------------
    # Median of headway across all speeds for this class.
    # This is the most robust measure of "how many seconds back".
    cc1 = g['headway_s'].dropna().median()

    # ------------------------------------------------------
    # CC8 — increased acceleration from standstill
    # ------------------------------------------------------
    # Look at positive accelerations when the vehicle is pulling
    # away from a stop (speed just above 0 and below 10 km/h).
    starting = g[
        (g['speed_kmph'] > 0.5) &
        (g['speed_kmph'] < LOW_SPEED_KMH) &
        (g['acc_mps2'] > 0.0)
    ]
    if len(starting) >= 20:
        cc8 = starting['acc_mps2'].quantile(CC8_PERCENTILE)
    else:
        # Fallback: overall 90th percentile of positive accel
        pos = g[g['acc_mps2'] > 0]
        cc8 = pos['acc_mps2'].quantile(0.90) if len(pos) > 0 else 1.0

    # ------------------------------------------------------
    # Lateral minimum distance at 0 km/h (standing)
    # ------------------------------------------------------
    # 5th percentile of lateral gap at very low speed. This is
    # the tightest squeeze vehicles accept when stopped or crawling.
    slow_lat = g[g['speed_kmph'] < VERY_LOW_KMH]['lateral_gap_m'].dropna()
    if len(slow_lat) >= 20:
        lat0 = slow_lat.quantile(LAT0_PERCENTILE)
    else:
        lat0 = g['lateral_gap_m'].dropna().quantile(LAT0_PERCENTILE)

    # ------------------------------------------------------
    # Lateral minimum distance at 50 km/h (driving)
    # ------------------------------------------------------
    # Your data peaks at ~30 km/h for bikes and ~20 km/h for
    # others, so we cannot directly observe 50 km/h. Instead we
    # use the 10th percentile of lateral gap at the top 15% of
    # speeds for this class. VISSIM will linearly interpolate
    # between this value and the standing value.
    speeds = g['speed_kmph'].dropna()
    if len(speeds) > 0:
        high_thr = speeds.quantile(0.85)
        fast = g[g['speed_kmph'] >= high_thr]['lateral_gap_m'].dropna()
        if len(fast) >= 10:
            lat50 = fast.quantile(LAT50_PERCENTILE)
        else:
            lat50 = g['lateral_gap_m'].dropna().quantile(LAT50_PERCENTILE)
    else:
        lat50 = g['lateral_gap_m'].dropna().quantile(LAT50_PERCENTILE)

    # ------------------------------------------------------
    # Store
    # ------------------------------------------------------
    results.append({
        'class_name'         : cls,
        'CC0_standstill_m'   : round(cc0, 2)  if pd.notna(cc0)  else None,
        'CC1_headway_s'      : round(cc1, 2)  if pd.notna(cc1)  else None,
        'CC8_accel_mps2'     : round(cc8, 2)  if pd.notna(cc8)  else None,
        'LateralDist_0kmph'  : round(lat0, 2) if pd.notna(lat0) else None,
        'LateralDist_50kmph' : round(lat50, 2) if pd.notna(lat50) else None,
        'n_samples'          : n_total,
    })

# ------------------------------------------------------------
# 4. SAVE AND PRINT
# ------------------------------------------------------------
out = pd.DataFrame(results).sort_values('class_name').reset_index(drop=True)

print()
print("=" * 90)
print("VISSIM PARAMETERS BY VEHICLE CLASS")
print("=" * 90)
print(out.to_string(index=False))
print("=" * 90)

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")