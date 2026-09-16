#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 2B - COORDINATE SCALE / HOMOGRAPHY DIAGNOSTIC
================================================================================
 WHY THIS EXISTS
   Phase 2 produced physically impossible car-following parameters: negative
   standstill distances, and gaps that do not grow with speed. Two independent
   checks both pointed at the same cause - the world coordinates appear to be
   SMALLER than reality, so every distance, gap and clearance is understated.

   This script settles the question using data you already have. It needs no
   fieldwork and no new video.

 THE THREE INDEPENDENT ESTIMATES
   A) From the DETECTION BOXES. Your raw file stores both the image position
      (px, py) and the world position (X_m, Y_m) of every detection. Fitting a
      smooth surface to that pair gives the local ground-metres-per-pixel. Apply
      it to box_w_px and you get each vehicle's size in metres, measured
      entirely through your own homography.
   B) From VEHICLES PASSING SIDE BY SIDE. Two vehicles cannot occupy the same
      space, so the closest centre-to-centre lateral distance ever observed
      between two vehicles of the same class is an upper bound on their width.
   C) From the ROAD ITSELF. The lateral spread of all detections is the width
      of the used carriageway. Compare it with the real road width.

   If A and B agree with each other but disagree with the known vehicle sizes,
   the homography scale is wrong, and the correction factor is the ratio.

 INPUT   trajectories.csv            (raw - it has px, py, box_w_px)
         trajectories_filtered.csv   (for the side-by-side check)
 OUTPUT  phase2b_scale_report.txt
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

# ==============================================================================
RAW_CSV      = r"S:\soscho\trajectories.csv"
FILTERED_CSV = r"S:\soscho\trajectories_filtered.csv"
OUT_TXT      = r"S:\soscho\phase2b_scale_report.txt"
# CONTROL GROUP. The synthetic filtered file has a KNOWN-CORRECT scale by
# construction. Running the same side-by-side check on it shows how much that
# check is biased by traffic density alone. Only the DIFFERENCE between real
# and control is evidence of a scale error. Set to None to skip.
CONTROL_CSV  = r"S:\soscho\synthetic_noisy_filtered.csv" 

# ---- [ASSUMED] true vehicle dimensions, Bangladeshi fleet, from literature ---
TRUE_W = {"Bike": 0.70, "CNG": 1.40, "Rickshaw": 1.20,
          "Car": 1.70, "Truck": 2.40, "Bus": 2.50}
TRUE_L = {"Bike": 1.90, "CNG": 2.60, "Rickshaw": 2.00,
          "Car": 4.40, "Truck": 7.50, "Bus": 11.0}

MIN_N = 200          # minimum detections for a class to be reported
SIDE_LON_FRAC = 0.6  # |lon difference| < this x mean length => abreast
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


rule("PHASE 2B | COORDINATE SCALE DIAGNOSTIC")
if not os.path.exists(RAW_CSV):
    sys.exit(f"ERROR: {RAW_CSV} not found.")
raw = pd.read_csv(RAW_CSV)
raw.columns = [c.strip() for c in raw.columns]
need = ["px", "py", "X_m", "Y_m", "box_w_px", "box_h_px", "class_name"]
miss = [c for c in need if c not in raw.columns]
if miss:
    sys.exit(f"ERROR: raw file is missing {miss}. This check needs the pixel columns.")
d = raw.dropna(subset=need).copy()
say(f"Raw detections usable: {len(d)}")

# ==============================================================================
# A) SCALE FROM THE IMAGE-TO-WORLD MAP
# ==============================================================================
rule("A) VEHICLE SIZE MEASURED THROUGH YOUR OWN HOMOGRAPHY")
px, py = d.px.to_numpy(), d.py.to_numpy()
A = np.column_stack([np.ones_like(px), px, py, px * py, px ** 2, py ** 2])
cx, *_ = np.linalg.lstsq(A, d.X_m.to_numpy(), rcond=None)
cy, *_ = np.linalg.lstsq(A, d.Y_m.to_numpy(), rcond=None)
rx = d.X_m.to_numpy() - A @ cx
ry = d.Y_m.to_numpy() - A @ cy
say(f"Surface fit residual : X {np.sqrt(np.mean(rx**2)):.4f} m | "
    f"Y {np.sqrt(np.mean(ry**2)):.4f} m")
if np.sqrt(np.mean(rx ** 2)) > 0.25:
    say("  WARNING: the fit is poor, so the pixel-to-metre map is not a smooth")
    say("  surface. Estimate A below is unreliable; rely on B and C.")
else:
    say("  A small residual means (px,py) -> (X_m,Y_m) is a clean smooth map,")
    say("  so its local gradient is a trustworthy metres-per-pixel scale.")

dXdpx = cx[1] + cx[3] * py + 2 * cx[4] * px
dYdpx = cy[1] + cy[3] * py + 2 * cy[4] * px
# an oblique camera moves the ground point in BOTH axes per horizontal pixel
step = np.hypot(dXdpx, dYdpx)
say(f"\nGround distance per horizontal pixel:")
say(f"   lateral component only : {np.median(np.abs(dXdpx))*100:.2f} cm/px")
say(f"   full ground magnitude  : {np.median(step)*100:.2f} cm/px")
say(f"   obliquity factor       : {np.median(step)/max(np.median(np.abs(dXdpx)),1e-9):.2f}x")

d["size_m"] = d.box_w_px.to_numpy() * step
say(f"\n{'class':<11}{'n':>8}{'measured size':>15}{'true width':>12}{'ratio':>8}")
say("-" * 78)
ratios_A = {}
for cn, t in d.groupby("class_name"):
    if len(t) < MIN_N:
        continue
    m = float(np.median(t.size_m))
    tw = TRUE_W.get(cn, np.nan)
    if np.isfinite(tw) and tw > 0:
        ratios_A[cn] = m / tw
        say(f"{cn:<11}{len(t):>8}{m:>15.2f}{tw:>12.2f}{m/tw:>8.2f}")
say("  'measured size' is the detection box converted to metres by your own")
say("  homography. If it is consistently BELOW the true width, world distances")
say("  are compressed and every gap in Phase 2 is understated.")
say("  Caveat: a 2D box also contains the vehicle's length foreshortened, which")
say("  makes this estimate an OVER-estimate if anything - so a ratio below 1.0")
say("  is strong evidence, while a ratio slightly above 1.0 is inconclusive.")

# ==============================================================================
# B) SCALE FROM VEHICLES PASSING SIDE BY SIDE
# ==============================================================================
rule("B) VEHICLE WIDTH FROM THE CLOSEST SIDE-BY-SIDE PASSES")
def side_by_side_ratios(path, label):
    """1st-percentile centre-to-centre distance for same-class abreast pairs."""
    if not path or not os.path.exists(path):
        return {}
    f = pd.read_csv(path)
    if "direction" in f.columns:
        f = f[f.direction == "with_flow"]
    f["L"] = f.class_name.map(lambda c: TRUE_L.get(c, 3.0))
    say(f"{label}: {len(f)} rows")
    closest = {}
    for fr, g in f.groupby("frame"):
        if len(g) < 2:
            continue
        x = g.lon_m.to_numpy(); y = g.lat_m.to_numpy()
        L = g.L.to_numpy(); cl = g.class_name.to_numpy()
        dx = np.abs(x[None, :] - x[:, None])
        dy = np.abs(y[None, :] - y[:, None])
        lensum = (L[:, None] + L[None, :]) / 2.0
        ab = (dx < lensum * SIDE_LON_FRAC)
        np.fill_diagonal(ab, False)
        ii, jj = np.where(np.triu(ab))
        for a_, b_ in zip(ii, jj):
            if cl[a_] != cl[b_]:
                continue
            closest.setdefault(cl[a_], []).append(dy[a_, b_])
    out = {}
    for cn, v in sorted(closest.items()):
        if len(v) < 50:
            continue
        tw = TRUE_W.get(cn, np.nan)
        if np.isfinite(tw) and tw > 0:
            out[cn] = (float(np.percentile(v, 1)) / tw, len(v))
    return out


real_B = side_by_side_ratios(FILTERED_CSV, "REAL")
ctrl_B = side_by_side_ratios(CONTROL_CSV, "CONTROL (synthetic, scale exact)")
ratios_B = {}
if real_B:
    say(f"\n{'class':<11}{'REAL ratio':>12}{'n':>8}{'CONTROL ratio':>15}{'n':>8}"
        f"{'real/control':>14}")
    say("-" * 78)
    for cn in sorted(real_B):
        rr, rn = real_B[cn]
        if cn in ctrl_B:
            cr, cn_ = ctrl_B[cn]
            rel = rr / cr if cr else np.nan
            ratios_B[cn] = rel
            say(f"{cn:<11}{rr:>12.2f}{rn:>8}{cr:>15.2f}{cn_:>8}{rel:>14.2f}")
        else:
            say(f"{cn:<11}{rr:>12.2f}{rn:>8}{'--':>15}{'--':>8}{'--':>14}")
    say("  This check is biased LOW by traffic density alone - vehicles in a")
    say("  dense stream pass closer than their nominal width suggests, and the")
    say("  detector's centre wanders. The CONTROL column measures that bias on")
    say("  synthetic data whose scale is exact by construction.")
    say("  Only the LAST column is evidence: if real/control is near 1.0 the")
    say("  scale is fine; if it is well below 1.0 the real world coordinates")
    say("  are compressed relative to a dataset of the same density.")
    if not ctrl_B:
        say("  NOTE: no control file found, so the last column is missing and")
        say("  this check cannot stand on its own. Generate the synthetic data")
        say("  and filter it first, or rely on check A.")
else:
    say(f"{FILTERED_CSV} not found - skipping check B.")

# ==============================================================================
# C) THE ROAD ITSELF
# ==============================================================================
rule("C) IMPLIED ROAD WIDTH")
span = d.X_m.max() - d.X_m.min()
say(f"Lateral spread of all detections : {span:.2f} m")
say(f"   (from {d.X_m.min():.2f} m to {d.X_m.max():.2f} m)")

# ==============================================================================
# VERDICT
# ==============================================================================
rule("VERDICT")
# Check A is absolute; check B is only meaningful relative to its control.
allr = dict(ratios_A)
if ratios_B:
    say("Check B (relative to control) per class:")
    for cn, r in sorted(ratios_B.items()):
        say(f"   {cn:<11} {r:.2f}")
    say("")
say("Check A (box size through your homography) is the primary evidence,")
say("because it is absolute rather than relative. The verdict below uses it.")
say("")
if allr:
    med = float(np.median(list(allr.values())))
    say(f"Size ratio (measured / true) per class:")
    for cn, r in sorted(allr.items(), key=lambda kv: kv[1]):
        say(f"   {cn:<11} {r:.2f}")
    say(f"\nMEDIAN RATIO: {med:.2f}")
    say("")
    if med < 0.85:
        k = 1.0 / med
        say(f"World distances look COMPRESSED by about {med:.2f}x.")
        say(f"   Implied scale correction factor: x{k:.2f}")
        say(f"   Implied TRUE road width: {span:.2f} x {k:.2f} = {span*k:.2f} m")
        say("")
        say("WHAT TO DO - in order:")
        say(f" 1. Measure the real carriageway width at the site, or on Google")
        say(f"    Earth. If it is close to {span*k:.2f} m rather than {span:.2f} m,")
        say("    the homography reference dimensions were wrong and this is")
        say("    confirmed.")
        say(" 2. If confirmed, redo the homography with the correct reference")
        say("    distances and re-run the pipeline from Phase 1. Everything")
        say("    downstream - gaps, CC0, lateral clearance, AND SPEEDS - is")
        say("    scaled by the same factor, so nothing is lost, but nothing")
        say("    can be quoted until it is fixed.")
        say(" 3. Do NOT patch it by editing DIMS. That would make the gaps look")
        say("    sensible while leaving the speeds wrong, which is worse than")
        say("    an obvious error.")
        say("")
        say("SANITY CHECK ON THE OTHER AXIS: if the scale is wrong in BOTH")
        say(f"   directions, your measured speeds are also low by x{k:.2f}.")
        say("   Per-class mean speeds would become:")
        for cn, v in [("Car", 10.0), ("Rickshaw", 9.3), ("Bike", 15.8)]:
            say(f"      {cn:<10} {v:.1f} -> {v*k:.1f} km/h")
        say("   Judge those against what you see in the video. Rickshaws in")
        say("   Dhaka traffic run about 8-12 km/h; if the corrected figure is")
        say("   far above that, the error is LATERAL ONLY, not both axes -")
        say("   which points at the homography's aspect ratio rather than its")
        say("   overall scale.")
    elif med > 1.15:
        say("World distances look STRETCHED. The same logic applies in reverse.")
    else:
        say("Scale looks consistent with the true vehicle sizes. The car-following")
        say("problems in Phase 2 are therefore NOT a scale artefact - they are")
        say("behavioural, and the flat gap-speed relation is a real finding.")
else:
    say("Not enough data for a verdict.")

try:
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    print(f"\n[saved to {OUT_TXT}]")
except Exception as e:
    print(f"[warn] {e}")
