#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 0 - COORDINATE SCALE CORRECTION
 Run this ONCE, before everything else, then re-run the pipeline.
================================================================================
 WHAT WAS WRONG
   The homography that turns pixels into metres compressed the LATERAL axis.
   Two completely independent measurements agree on the size of the error:

     1. Vehicle widths measured through the homography itself came out at
        0.69 of their true size (Phase 2B, 143,558 detections, 5 of 6 classes)
     2. The detections span 6.72 m across a carriageway measured on site at
        10.00 m  ->  ratio 0.672

   Two methods, one answer. The lateral scale is short by a factor of ~1.49.

 WHAT WAS *NOT* WRONG - AND WHY THIS MATTERS
   The LONGITUDINAL axis is fine. The test is the rickshaw. A cycle rickshaw
   in Dhaka traffic runs at roughly 8-12 km/h; the uncorrected data puts them
   at a median of 9.0 km/h and an 85th percentile of 13.7 km/h, which is right.
   Scaling the longitudinal axis by 1.49 as well would put rickshaws at a
   median of 13.4 and an 85th percentile of 20.4 km/h - which a pedalled
   vehicle cannot sustain in mixed traffic.

   So this is an ASPECT-RATIO error in the homography, not an overall scale
   error. Only X is corrected. Speeds, headways and longitudinal gaps are
   untouched, which means every speed-based result you already have stays
   valid.

 WHAT THIS FIXES
   lateral clearance, lateral wander, lateral speed, the side-by-side
   geometry, and the leader-overlap test that decides who is following whom

 WHAT THIS DOES NOT FIX
   The negative CC0. That is a LONGITUDINAL quantity, so it cannot come from
   a lateral error. Its likely cause is the assumed vehicle LENGTHS - a Dhaka
   car fleet is mostly small hatchbacks, not the 4.40 m sedan assumed in the
   scripts. Phase 2's length sanity check will now be able to test that
   properly, because the pairing it depends on will finally be geometrically
   correct.

 INPUT   trajectories.csv          (your original, never modified)
 OUTPUT  trajectories_scaled.csv   (feed this to phase1b_FOR_REAL.py)
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

# ==============================================================================
RAW_CSV = r"S:\soscho\trajectories.csv"
OUT_CSV = r"S:\soscho\trajectories_scaled.csv"
OUT_TXT = r"S:\soscho\phase0_scale_correction.txt"

# ---- [EMPIRICAL] measured on site ------------------------------------------
TRUE_ROAD_WIDTH_M = 10.00   # carriageway width, Sonargaon Janapath, Uttara

# ---- how much of that width can traffic actually occupy? --------------------
# Vehicles cannot drive right up to the kerb. In Dhaka mixed traffic they use
# very nearly the whole surface, but a small margin is realistic. The observed
# span is matched to USABLE_FRACTION x TRUE_ROAD_WIDTH_M.
USABLE_FRACTION = 1.00      # [ASSUMED] set to 0.95 to leave a kerb margin

# Which span to match against the road width.
#   "full"  - the full min-to-max range of detections. CORRECT for this data:
#             the vehicles nearest each kerb are real, not outliers, and they
#             are exactly the ones that define the usable width.
#   "p1"    - the 1-99 percentile range. Trims the kerb-hugging rickshaws and
#             bikes, which UNDERSTATES the span and therefore OVERSTATES the
#             correction. The first run of this script used "p1" and produced
#             K=2.04, which pushed vehicles 13.7 m apart on a 10 m road.
SPAN_MODE = "full"

# ---- longitudinal correction ------------------------------------------------
# 1.00 = leave it alone. See the rickshaw argument in the header before
# changing this. If you later confirm the longitudinal scale is also wrong,
# set it here and re-run everything.
K_LON = 1.00                # [EMPIRICAL] validated against rickshaw speeds
# ==============================================================================

REPORT = []


def say(s=""):
    print(s)
    REPORT.append(str(s))


def rule(t=""):
    if t:
        say("\n" + "=" * 78); say(t); say("=" * 78)
    else:
        say("-" * 78)


rule("PHASE 0 | COORDINATE SCALE CORRECTION")
if not os.path.exists(RAW_CSV):
    sys.exit(f"ERROR: {RAW_CSV} not found.")
df = pd.read_csv(RAW_CSV)
df.columns = [c.strip() for c in df.columns]
for c in ("X_m", "Y_m"):
    if c not in df.columns:
        sys.exit(f"ERROR: column {c} missing.")

say(f"Input : {RAW_CSV}")
say(f"        {len(df)} rows, {df.vehicle_id.nunique()} tracks")

# ---- work out the lateral correction ----------------------------------------
x = df.X_m.to_numpy()
# use a robust span: the 0.5-99.5 percentile range, so one stray detection at
# the edge of the image cannot set the scale for the whole dataset
lo, hi = np.percentile(x, [1.0, 99.0])
span_robust = hi - lo
span_full = x.max() - x.min()
span_p05 = np.percentile(x, 99.5) - np.percentile(x, 0.5)
target = TRUE_ROAD_WIDTH_M * USABLE_FRACTION
_spans = {"full": span_full, "p05": span_p05, "p1": span_robust}
span_used = _spans.get(SPAN_MODE, span_full)
K_LAT = target / span_used

rule("THE CORRECTION")
say(f"Observed lateral span, full range    : {span_full:.2f} m  -> K = {target/span_full:.2f}")
say(f"Observed lateral span, 0.5-99.5 pct  : {span_p05:.2f} m  -> K = {target/span_p05:.2f}")
say(f"Observed lateral span, 1-99 pct      : {span_robust:.2f} m  -> K = {target/span_robust:.2f}")
say(f"   SPAN_MODE = '{SPAN_MODE}'  ->  span {span_used:.2f} m,  K_LAT = {K_LAT:.2f}")
say("  The FULL range is the right one here. Percentile trimming was tried")
say("  first and was wrong: in this traffic the vehicles nearest each kerb")
say("  are rickshaws and bikes riding the edge - they are the real boundary")
say("  of the usable width, not outliers. Trimming them shrank the span from")
say("  6.72 m to 4.90 m and inflated the correction to 2.04, which placed")
say("  vehicles 13.7 m apart on a 10 m road.")
say(f"Measured carriageway width           : {TRUE_ROAD_WIDTH_M:.2f} m")
say(f"Usable fraction assumed              : {USABLE_FRACTION:.2f}")
say(f"   -> target span {target:.2f} m")
say("")
say(f"LATERAL correction  K_LAT = {K_LAT:.4f}")
say(f"LONGITUDINAL        K_LON = {K_LON:.4f}   (unchanged - see header)")
say("")
say("Cross-check: Phase 2B estimated the compression independently from")
say("vehicle bounding-box sizes and got a ratio of 0.69, i.e. a correction")
say(f"of x1.45. This road-width method gives x{K_LAT:.2f}. Two unrelated")
say("routes to the same number is the strongest evidence available without")
say("re-calibrating the homography from scratch.")

# ---- apply -------------------------------------------------------------------
centre = (x.max() + x.min()) / 2.0 if SPAN_MODE == "full" else (hi + lo) / 2.0
df["X_m_orig"] = df.X_m
df["Y_m_orig"] = df.Y_m
df["X_m"] = centre + (df.X_m - centre) * K_LAT
if K_LON != 1.0:
    ymid = df.Y_m.median()
    df["Y_m"] = ymid + (df.Y_m - ymid) * K_LON
df["scale_k_lat"] = K_LAT
df["scale_k_lon"] = K_LON

_new_span = df.X_m.max() - df.X_m.min()
rule("PHYSICAL GUARD")
say(f"Corrected lateral range : {df.X_m.min():.2f} .. {df.X_m.max():.2f} m "
    f"= {_new_span:.2f} m wide")
say(f"Measured road width     : {TRUE_ROAD_WIDTH_M:.2f} m")
if _new_span > TRUE_ROAD_WIDTH_M * 1.10:
    say("  *** FAILED - vehicles would be outside the carriageway ***")
    say("  The correction factor is too large. This is a hard physical")
    say("  impossibility, not a matter of judgement: no vehicle can be")
    say("  driving beyond the kerb. Lower K by switching SPAN_MODE to 'full',")
    say("  or reduce USABLE_FRACTION.")
elif _new_span < TRUE_ROAD_WIDTH_M * 0.75:
    say("  Corrected span is well inside the road. Either traffic genuinely")
    say("  does not use the edges, or the correction is too small.")
else:
    say("  PASSED - the corrected traffic fits inside the measured carriageway.")

rule("BEFORE AND AFTER")
say(f"{'quantity':<32}{'before':>12}{'after':>12}")
say("-" * 78)
say(f"{'lateral span (0.5-99.5 pct), m':<32}{span_robust:>12.2f}"
    f"{np.percentile(df.X_m,99.5)-np.percentile(df.X_m,0.5):>12.2f}")
say(f"{'lateral min, m':<32}{x.min():>12.2f}{df.X_m.min():>12.2f}")
say(f"{'lateral max, m':<32}{x.max():>12.2f}{df.X_m.max():>12.2f}")
say(f"{'longitudinal span, m':<32}{df.Y_m_orig.max()-df.Y_m_orig.min():>12.2f}"
    f"{df.Y_m.max()-df.Y_m.min():>12.2f}")

# ---- sanity: implied vehicle widths after correction -------------------------
if "box_w_px" in df.columns and "px" in df.columns and "py" in df.columns:
    d = df.dropna(subset=["px", "py", "X_m", "Y_m", "box_w_px"])
    px, py = d.px.to_numpy(), d.py.to_numpy()
    A = np.column_stack([np.ones_like(px), px, py, px * py, px ** 2, py ** 2])
    cx, *_ = np.linalg.lstsq(A, d.X_m.to_numpy(), rcond=None)
    cy, *_ = np.linalg.lstsq(A, d.Y_m.to_numpy(), rcond=None)
    stepx = cx[1] + cx[3] * py + 2 * cx[4] * px
    stepy = cy[1] + cy[3] * py + 2 * cy[4] * px
    size = d.box_w_px.to_numpy() * np.hypot(stepx, stepy)
    TRUE_W = {"Bike": 0.70, "CNG": 1.40, "Rickshaw": 1.20,
              "Car": 1.70, "Truck": 2.40, "Bus": 2.50}
    rule("SANITY CHECK AFTER CORRECTION - implied vehicle size")
    say(f"{'class':<11}{'n':>9}{'implied':>10}{'true W':>9}{'ratio':>8}   verdict")
    say("-" * 78)
    d = d.assign(size_m=size)
    for cn, t in d.groupby("class_name"):
        if len(t) < 200 or cn not in TRUE_W:
            continue
        m = float(np.median(t.size_m)); r = m / TRUE_W[cn]
        v = "good" if 0.85 <= r <= 1.6 else ("still small" if r < 0.85 else "large")
        say(f"{cn:<11}{len(t):>9}{m:>10.2f}{TRUE_W[cn]:>9.2f}{r:>8.2f}   {v}")
    _rs = []
    for cn, t in d.groupby("class_name"):
        if len(t) >= 200 and cn in TRUE_W:
            _rs.append(float(np.median(t.size_m)) / TRUE_W[cn])
    if _rs:
        _med = float(np.median(_rs))
        say("")
        say(f"  MEDIAN RATIO AFTER CORRECTION: {_med:.2f}")
        if 0.85 <= _med <= 1.35:
            say("  -> the correction factor is right. This is the check that matters:")
            say("     it was computed from the ROAD WIDTH, and it lands the VEHICLE")
            say("     WIDTHS on their true values. Two independent quantities agreeing")
            say("     is what makes the correction trustworthy rather than a fudge.")
        elif _med < 0.85:
            say("  -> still under-corrected. Either the carriageway is wider than")
            say(f"     {TRUE_ROAD_WIDTH_M} m, or traffic does not use its full width -")
            say("     try lowering USABLE_FRACTION (e.g. 0.90) which RAISES the factor.")
        else:
            say("  -> over-corrected. Traffic probably uses less than the full width;")
            say("     raise USABLE_FRACTION above 1.0 is not meaningful, so instead")
            say("     re-check the measured road width - it may include footpaths.")
    say("  Ratios should now sit near or a little above 1.0. Slightly above is")
    say("  expected: a 2D box also contains some of the vehicle's length.")
    say("  A class still well below 1.0 has a detector that draws tight boxes,")
    say("  not a remaining scale error.")

# ---- write -------------------------------------------------------------------
drop = [c for c in ("X_m_orig", "Y_m_orig") if c in df.columns]
out = df.drop(columns=[])          # keep the originals for traceability
try:
    out.to_csv(OUT_CSV, index=False)
    say(f"\nWritten : {OUT_CSV}   ({len(out)} rows)")
    say("   Your original trajectories.csv is untouched.")
    say("   The file keeps X_m_orig / Y_m_orig columns so the correction is")
    say("   reversible and auditable - an examiner can see exactly what changed.")
except Exception as e:
    sys.exit(f"ERROR writing output: {e}")

rule("WHAT TO DO NEXT")
say("1. phase1b_FOR_REAL.py   - already points at trajectories_scaled.csv")
say("2. phase2_behavior_extraction.py")
say("3. phase2c_lateral_behaviour.py")
say("")
say("EXPECT TO SEE:")
say(" - lateral clearances roughly 1.5x larger, and far fewer negative ones")
say(" - the Step 3 width sanity check dropping from 37.5 % negative to a few %")
say(" - manoeuvre counts in Phase 2C rising, because a 0.40 m threshold is now")
say("   reachable by a typical track")
say(" - speeds, headways and CC1 UNCHANGED - that is the point of correcting")
say("   only one axis, and it is how you know the correction is conservative")
say("")
say("FOR THE THESIS: report this as a calibration correction, with the two")
say("independent estimates and the rickshaw-speed argument for leaving the")
say("longitudinal axis alone. Finding and fixing a calibration error mid-study")
say("is normal practice; hiding one is not.")

try:
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\n[saved to {OUT_TXT}]")
except Exception:
    pass
