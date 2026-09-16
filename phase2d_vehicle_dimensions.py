#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 2D - VEHICLE DIMENSIONS FROM THE DATA
================================================================================
 WHY THIS EXISTS
   After the Phase 0 scale correction, one problem remains: CC0. For Car the
   fit gives CC0 = -0.14 m with CC1 = 0.92 s. The headway is entirely credible;
   the standstill distance is 14 cm on the wrong side of zero.

   That is not a behavioural result, it is an arithmetic one. Every gap is

        gap  =  (centre-to-centre distance)  -  (half length A + half length B)

   so an assumed length that is 30 cm too long drops every gap by 30 cm and
   pushes CC0 straight through zero. The lengths in DIMS came from literature,
   not from this road.

 THE HONEST DIFFICULTY - READ THIS BEFORE USING THE OUTPUT
   The trajectories give the distance between vehicle CENTRES. At standstill,

        centre distance  =  length  +  CC0

   The data pins the SUM. It cannot, on its own, separate the two. A longer
   assumed vehicle means a smaller standstill gap and vice versa, and no
   amount of statistics breaks that tie.

   So this script does NOT hand you a single answer. It reports the trade-off
   explicitly: for a range of assumed minimum physical clearances, what length
   and what CC0 follow. You choose a point on that curve and DECLARE it. That
   is the defensible way to report a quantity the measurement cannot separate.

 WHAT IS GENUINELY MEASURED HERE
   - the closest centre-to-centre distance two same-class vehicles ever reach
     while queued (a hard geometric bound on length + clearance)
   - the same quantity side by side, which bounds WIDTH the same way
   Both are distributions, not single numbers, and both are reported as such.

 INPUT   trajectories_filtered.csv, phase2_pairs.csv
 OUTPUT  phase2d_dimensions.txt  +  a DIMS block to paste into the scripts
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

# ==============================================================================
FILTERED_CSV = r"S:\soscho\trajectories_filtered.csv"
PAIRS_CSV    = r"S:\soscho\phase2_pairs.csv"
OUT_TXT      = r"S:\soscho\phase2d_dimensions.txt"

# the dimensions currently assumed by the pipeline (length, width)
DIMS_NOW = {"Bike": (1.90, 0.70), "CNG": (2.60, 1.40), "Rickshaw": (2.00, 1.20),
            "Car": (4.40, 1.70), "Truck": (7.50, 2.40), "Bus": (11.0, 2.50)}

# [ASSUMED] the tightest physical clearance a driver in a Dhaka queue leaves.
# This is the ONE external number needed to split length from standstill gap.
# The script reports the whole trade-off curve so you can see how much it matters.
CLEARANCE_GRID = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
CLEARANCE_PICK = 0.30

CRAWL_MPS   = 1.00   # both vehicles below this count as "queued"
LOW_PCT     = 2.0    # percentile of the closest-approach distribution to use
MIN_SAMPLES = 40
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


rule("PHASE 2D | VEHICLE DIMENSIONS FROM THE DATA")
for f in (FILTERED_CSV, PAIRS_CSV):
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} not found. Run Phase 2 first.")
P = pd.read_csv(PAIRS_CSV)
df = pd.read_csv(FILTERED_CSV)
idcol = "track_id" if "track_id" in df.columns else "vehicle_id"
say(f"Following pairs : {len(P)}")
say(f"Trajectories    : {len(df)} rows")

# ==============================================================================
# LENGTH - from the closest approach of two queued vehicles of the same class
# ==============================================================================
rule("LENGTH: CLOSEST APPROACH WHEN QUEUED")
say("For two vehicles of the same class, one behind the other, both crawling:")
say("   centre-to-centre distance  =  length  +  gap")
say("The smallest distance ever observed therefore bounds length + the tightest")
say("gap anyone accepts. Reported as the 2nd percentile, so a single bad")
say("detection cannot set the answer.\n")
say(f"{'class':<11}{'n queued':>10}{'closest c-c':>13}{'assumed L':>11}   status")
say("-" * 78)

cc_len = {}
for cn in sorted(set(P.follower_class)):
    s = P[(P.follower_class == cn) & (P.leader_class == cn)
          & (P.v_follower < CRAWL_MPS) & (P.v_leader < CRAWL_MPS)]
    if len(s) < MIN_SAMPLES:
        say(f"{cn:<11}{len(s):>10}{'--':>13}{DIMS_NOW.get(cn,(0,0))[0]:>11.2f}"
            f"   too few queued pairs")
        continue
    # rebuild centre distance from the gap the pipeline computed
    L_assumed = DIMS_NOW.get(cn, (3.0, 1.5))[0]
    cdist = s.gap_m + L_assumed
    val = float(np.percentile(cdist, LOW_PCT))
    cc_len[cn] = (val, len(s))
    st = "consistent" if val >= L_assumed else "ASSUMED LENGTH TOO LONG"
    say(f"{cn:<11}{len(s):>10}{val:>13.2f}{L_assumed:>11.2f}   {st}")
say("  'closest c-c' is length + clearance. If it is BELOW the assumed length,")
say("  the assumption is impossible: it would require a negative gap.")

# ==============================================================================
# THE TRADE-OFF
# ==============================================================================
rule("THE TRADE-OFF BETWEEN LENGTH AND STANDSTILL GAP")
say("The measurement fixes (length + clearance). Splitting them needs one")
say("assumption. Here is what each choice implies:\n")
if cc_len:
    hdr = f"{'class':<11}{'measured sum':>14}"
    for c in CLEARANCE_GRID:
        hdr += f"{'L @ ' + format(c, '.2f'):>11}"
    say(hdr)
    say("-" * 78)
    for cn, (val, n) in sorted(cc_len.items()):
        row = f"{cn:<11}{val:>14.2f}"
        for c in CLEARANCE_GRID:
            row += f"{val - c:>11.2f}"
        say(row)
    say("  Each column is the vehicle length implied if the tightest clearance")
    say("  in a queue is that many metres. Compare against what you know a")
    say("  vehicle of that class actually measures - that is the check.")
else:
    say("No class had enough queued same-class pairs to estimate this.")

# ==============================================================================
# WIDTH - the same logic, side by side
# ==============================================================================
rule("WIDTH: CLOSEST SIDE-BY-SIDE APPROACH")
lat_path = PAIRS_CSV.replace("pairs", "lateral")
if os.path.exists(lat_path):
    S = pd.read_csv(lat_path)
    say(f"{'class':<11}{'n abreast':>11}{'closest c-c':>13}{'assumed W':>11}"
        f"{'implied W':>11}")
    say("-" * 78)
    for cn in sorted(set(S.class_a) | set(S.class_b)):
        s = S[(S.class_a == cn) & (S.class_b == cn)]
        if len(s) < MIN_SAMPLES:
            continue
        W_assumed = DIMS_NOW.get(cn, (3.0, 1.5))[1]
        cdist = s.lat_clear_m + W_assumed          # rebuild centre distance
        val = float(np.percentile(cdist, LOW_PCT))
        say(f"{cn:<11}{len(s):>11}{val:>13.2f}{W_assumed:>11.2f}"
            f"{val - CLEARANCE_PICK:>11.2f}")
    say(f"  'implied W' assumes the same {CLEARANCE_PICK} m tightest clearance")
    say("  sideways as longitudinally. In Dhaka traffic the sideways clearance")
    say("  is usually TIGHTER than the longitudinal one, so this is a lower")
    say("  bound on width - treat it as such.")
else:
    say(f"{lat_path} not found - skipping width.")

# ==============================================================================
# THE RECOMMENDATION
# ==============================================================================
rule(f"SUGGESTED DIMS  (at a {CLEARANCE_PICK} m tightest clearance)")
say("Paste this into phase2_behavior_extraction.py and phase2c, replacing the")
say("DIMS block. Keep the old values in a comment so the change is visible.\n")
say("DIMS = {")
for cn in ("Bike", "CNG", "Rickshaw", "Car", "Truck", "Bus"):
    L_old, W_old = DIMS_NOW.get(cn, (3.0, 1.5))
    if cn in cc_len:
        L_new = round(cc_len[cn][0] - CLEARANCE_PICK, 2)
        tag = f"# was {L_old}, measured"
    else:
        L_new = L_old
        tag = "# unchanged - not enough queued pairs"
    say(f'    "{cn}": ({L_new:.2f}, {W_old:.2f}),   {tag}')
say("}")

rule("WHAT THIS DOES AND DOES NOT SETTLE")
say("SETTLED: the lengths are no longer imported from a European vehicle")
say("   catalogue. They come from how close these vehicles actually get on")
say("   this road, which is the right source for a Dhaka thesis.")
say("")
say("NOT SETTLED: length and CC0 remain tied together. If you report CC0 after")
say("   this, you must also report the clearance assumption that produced it.")
say("   The honest sentence is: 'the data determines the sum of vehicle length")
say("   and standstill distance as X m; assuming a minimum physical clearance")
say(f"   of {CLEARANCE_PICK} m gives a length of L and a standstill distance of CC0.'")
say("")
say("SENSITIVITY: the trade-off table above IS the sensitivity analysis. Quote")
say("   it. A reader can then see exactly how much CC0 moves if they disagree")
say("   with the clearance assumption - which is far stronger than a single")
say("   number with no stated uncertainty.")
say("")
say("A WARNING ABOUT CIRCULARITY: do not tune the clearance until CC0 comes")
say("   out at a value you like. Pick it from physical reasoning about how")
say("   close vehicles can actually stop, fix it, and report whatever CC0")
say("   follows - including if it stays near zero. A standstill distance that")
say("   really is very small would itself be a finding about Dhaka traffic.")

try:
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\n[saved to {OUT_TXT}]")
except Exception as e:
    print(f"[warn] {e}")
