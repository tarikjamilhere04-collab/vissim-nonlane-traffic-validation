# ============================================================
# VISSIM CLASS-OVERRIDE TABLE
#
# Produces the exact values you type into VISSIM's
#   Car following model tab  ->  bottom table:
#       columns: VehClass, W99cc0, W99cc1Distr, IncrsAccel
#   Lateral tab  ->  per Vehicle Type:
#       Distance standing (at 0 km/h)
#       Distance driving  (at 50 km/h)
#
# INPUT  : S:\Second\behavior_metrics_clean.csv
# OUTPUT : S:\Second\vissim_class_overrides.csv
# ============================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. PATHS AND THRESHOLDS
# ------------------------------------------------------------
IN_PATH  = r"S:\Second\behavior_metrics_clean.csv"
OUT_PATH = r"S:\Second\vissim_class_overrides.csv"

if not os.path.exists(IN_PATH):
    raise SystemExit(f"Input file not found: {IN_PATH}")

# Speed bands (km/h)
SLOW_SPEED_MAX = 5.0    # "near-stop" for CC0 and lateral standing
LOW_SPEED_MAX  = 15.0   # "accelerating" for CC8
HIGH_SPEED_Q   = 0.75   # "fast" band for lateral driving = top 25%

# Minimum sample count to trust a percentile
MIN_SAMPLES = 20

# ------------------------------------------------------------
# 2. LOAD
# ------------------------------------------------------------
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df):,} rows, classes: {sorted(df['class_name'].unique())}")

# ------------------------------------------------------------
# 3. COMPUTE PER-CLASS VALUES
# ------------------------------------------------------------
rows = []

for cls, g in df.groupby('class_name'):

    # ---- CC0 : standstill distance ----
    # Take the 10th percentile of spacing when the vehicle is
    # nearly stopped (speed < 5 km/h). This is the smallest gap
    # drivers accept when queued up.
    slow = g[g['speed_kmph'] < SLOW_SPEED_MAX]['spacing_m'].dropna()
    if len(slow) >= MIN_SAMPLES:
        cc0 = slow.quantile(0.10)
    else:
        cc0 = g['spacing_m'].dropna().quantile(0.10)

    # ---- CC1 : gap time headway (s) ----
    # Median headway. This is stable and robust.
    cc1 = g['headway_s'].dropna().median()

    # ---- CC8 : acceleration from standstill (m/s^2) ----
    # Look for positive accelerations when the vehicle is moving
    # slowly (pulling away from a stop). Take the MEDIAN (not the
    # 90th percentile) because we want the typical pull-away
    # acceleration, not the rare burst.
    pull_away = g[
        (g['speed_kmph'] > 0.5) &
        (g['speed_kmph'] < LOW_SPEED_MAX) &
        (g['acc_mps2'] > 0.0)
    ]['acc_mps2'].dropna()

    if len(pull_away) >= MIN_SAMPLES:
        cc8 = pull_away.median()
    else:
        # Fallback: median of ALL positive accelerations
        pos = g[g['acc_mps2'] > 0]['acc_mps2'].dropna()
        cc8 = pos.median() if len(pos) > 0 else 1.0

    # Clamp to physically sane range
    cc8 = float(np.clip(cc8, 0.8, 2.5))

    # ---- Lateral distance at 0 km/h (standing) ----
    # 5th percentile of lateral gap when nearly stopped.
    lat_stand = g[g['speed_kmph'] < SLOW_SPEED_MAX]['lateral_gap_m'].dropna()
    if len(lat_stand) >= MIN_SAMPLES:
        lat0 = lat_stand.quantile(0.05)
    else:
        lat0 = g['lateral_gap_m'].dropna().quantile(0.05)

    # ---- Lateral distance at 50 km/h (driving) ----
    # 10th percentile of lateral gap in the top 25% speed band of
    # this class. This is a proxy for "high-speed" behavior even
    # though few vehicles reach 50 km/h.
    speed_thr = g['speed_kmph'].quantile(HIGH_SPEED_Q)
    fast = g[g['speed_kmph'] >= speed_thr]['lateral_gap_m'].dropna()
    if len(fast) >= MIN_SAMPLES:
        lat50 = fast.quantile(0.10)
    else:
        lat50 = g['lateral_gap_m'].dropna().quantile(0.10)

    # ---- Sanity: driving lateral distance must exceed standing ----
    if pd.isna(lat50) or lat50 < lat0 * 1.3:
        lat50 = lat0 * 1.5

    # Guarantee absolute minimum (VISSIM dislikes < 0.1 m)
    lat0  = max(lat0,  0.10)
    lat50 = max(lat50, 0.20)

    rows.append({
        'VehClass'         : cls,
        'W99cc0'           : round(cc0, 2),
        'W99cc1Distr'      : round(cc1, 2),
        'IncrsAccel'       : round(cc8, 2),
        'LateralStand_m'   : round(lat0, 2),
        'LateralDrive_m'   : round(lat50, 2),
        'n_samples'        : len(g),
    })

# ------------------------------------------------------------
# 4. SAVE AND PRINT
# ------------------------------------------------------------
out = pd.DataFrame(rows).sort_values('VehClass').reset_index(drop=True)

print()
print("=" * 100)
print("VISSIM CLASS-OVERRIDE TABLE  (copy values into the Driving Behaviour dialog)")
print("=" * 100)
print(out.to_string(index=False))
print("=" * 100)

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")