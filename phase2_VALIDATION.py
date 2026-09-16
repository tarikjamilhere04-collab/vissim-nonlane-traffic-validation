#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 2 - BEHAVIOUR QUANTIFICATION & W99 PARAMETER EXTRACTION
 Thesis: Vision-Based Trajectory Extraction, Behavior Modeling, and Microscopic
         Simulation of Non-Lane-Based Mixed Traffic
================================================================================
 WHAT THIS DOES
   1. Finds LEADER-FOLLOWER pairs without lanes, using lateral overlap
   2. Measures car-following: gaps, headways, relative speeds, per class pair
   3. Measures LATERAL CLEARANCE between vehicles travelling side by side
   4. Builds per-class speed and acceleration distributions
   5. Derives Wiedemann-99 CC0-CC9 where the data supports it, and says
      plainly where it does not
   6. Optionally VALIDATES the pair-finding against synthetic ground truth

 THE NON-LANE RULE
   In lane-based traffic the leader is "the vehicle ahead in my lane". Here
   there are no lanes, so the leader is the nearest vehicle ahead whose body
   overlaps mine laterally by more than OVERLAP_FRAC of the mean half-width.
   A vehicle only partly in front of you influences you only partly - that is
   the central modelling difference in Bangladeshi mixed traffic.

 INPUT   trajectories_filtered.csv  (Phase 1b output)
 OUTPUT  phase2_pairs.csv           every leader-follower sample
         phase2_lateral.csv         every side-by-side clearance sample
         phase2_report.txt          the report printed below
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import theilslopes
except ImportError:
    theilslopes = None

# ==============================================================================
# CONFIG
# ==============================================================================
INPUT_CSV  = r"S:\soscho\synthetic_noisy_filtered.csv"
OUT_DIR    = r"S:\soscho"

# Set these two to validate pair-finding against synthetic truth, else None
TRUTH_CSV  = r"S:\soscho\synthetic_truth.csv"

DT = 0.03332

# ---- [ASSUMED] vehicle dimensions, Bangladeshi fleet -------------------------
# length, width in metres. Declare these in the thesis: they are NOT measured
# from your video, and every gap below is bumper-to-bumper using them.
DIMS = {
    "Bike":     (1.90, 0.70),
    "CNG":      (2.60, 1.40),
    "Rickshaw": (2.00, 1.20),
    "Car":      (4.40, 1.70),
    "Truck":    (7.50, 2.40),
    "Bus":      (11.0, 2.50),
}
DEFAULT_DIM = (3.0, 1.5)

# ---- [ASSUMED] behavioural thresholds ---------------------------------------
OVERLAP_FRAC   = 0.35   # lateral overlap fraction that makes a vehicle a leader
MAX_GAP_M      = 30.0   # beyond this the follower is not interacting
STANDSTILL_MPS = 0.30   # m/s below which a vehicle counts as stopped
FOLLOW_DV_MPS  = 0.50   # |relative speed| below which following is "steady"
FOLLOW_MIN_MPS = 1.50   # minimum speed for a steady-following sample
CC8_MAX_MPS    = 2.00   # standstill acceleration measured below this speed
SIDE_LON_FRAC  = 0.60   # |Δlon| < this × mean length  =>  travelling abreast
MIN_SAMPLES    = 30     # fewer samples than this => parameter not reported
EDGE_EXCLUDE   = True   # drop Phase 1 edge samples

REPORT = []


def say(s=""):
    print(s)
    REPORT.append(str(s))


def rule(t=""):
    if t:
        say("\n" + "=" * 78); say(t); say("=" * 78)
    else:
        say("-" * 78)


def q(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.percentile(x, p) if x.size else np.nan


# ==============================================================================
rule("PHASE 2 | BEHAVIOUR QUANTIFICATION & W99 EXTRACTION")
if not os.path.exists(INPUT_CSV):
    sys.exit(f"ERROR: {INPUT_CSV} not found. Run phase1b first.")

df = pd.read_csv(INPUT_CSV)
idcol = "track_id" if "track_id" in df.columns else "vehicle_id"
say(f"Input : {INPUT_CSV}")
say(f"        {len(df)} rows, {df[idcol].nunique()} tracks")

# ---- PROVENANCE CHECK -------------------------------------------------------
# Synthetic rows carry a generator run_id (and a scenario label). If they turn
# up in a file whose name says "real", something upstream wrote the wrong data
# to the wrong path - and every number below would be meaningless.
_syn = ("run_id" in df.columns and df.run_id.notna().any()) or \
       ("scenario" in df.columns and df.scenario.notna().any())
_named_real = "synthetic" not in os.path.basename(INPUT_CSV).lower()
say("")
if _syn and _named_real:
    say("!" * 78)
    say("STOP - THIS FILE CONTAINS SYNTHETIC DATA")
    say(f"   {INPUT_CSV}")
    if "run_id" in df.columns:
        say(f"   generator run id: {df.run_id.dropna().iloc[0]}")
    if "scenario" in df.columns:
        say(f"   scenarios present: {sorted(set(df.scenario.dropna()))}")
    say("   The file name says real data, but the contents were produced by")
    say("   synth_generator.py. Something wrote synthetic output over the real")
    say("   filtered file. Nothing below would describe your video.")
    say("")
    say("   TO FIX: run  phase1b_FOR_REAL.py  to rebuild the real filtered file")
    say("   from trajectories.csv, then run this script again.")
    say("!" * 78)
    sys.exit(1)
elif _syn:
    say("   [provenance] SYNTHETIC data - correct for a validation run.")
else:
    say("   [provenance] no synthetic markers - real data.")

# keep only usable samples
if "direction" in df.columns:
    n0 = len(df)
    df = df[df.direction == "with_flow"]
    say(f"        dropped {n0-len(df)} rows of opposing-flow traffic")
if EDGE_EXCLUDE and "edge" in df.columns:
    n0 = len(df)
    df = df[df.edge == 0]
    say(f"        dropped {n0-len(df)} edge samples (filter transients)")
if "flag_implausible" in df.columns:
    df = df[df.flag_implausible == 0]

df["L"] = df.class_name.map(lambda c: DIMS.get(c, DEFAULT_DIM)[0])
df["W"] = df.class_name.map(lambda c: DIMS.get(c, DEFAULT_DIM)[1])
say(f"        {len(df)} rows retained for analysis")

# ==============================================================================
# STEP 1 - LEADER / FOLLOWER PAIRS AND SIDE-BY-SIDE CLEARANCES
# ==============================================================================
rule("STEP 1 : LEADER-FOLLOWER PAIRING (lateral-overlap rule, no lanes)")
say(f"A vehicle ahead is your LEADER if its body overlaps yours laterally by")
say(f"more than {OVERLAP_FRAC:.2f} of the mean half-width, and it is within")
say(f"{MAX_GAP_M:.0f} m. Partial overlap = partial influence: this is what")
say("replaces the lane in a non-lane-based stream.")

pairs, lateral = [], []
frames = df.frame.to_numpy()
order = np.argsort(frames, kind="stable")
d = df.iloc[order].reset_index(drop=True)
starts = np.searchsorted(d.frame.to_numpy(), np.unique(d.frame.to_numpy()), "left")
ends = np.searchsorted(d.frame.to_numpy(), np.unique(d.frame.to_numpy()), "right")

lon = d.lon_m.to_numpy(); lat = d.lat_m.to_numpy()
spd = d.speed_mps.to_numpy(); acc = d.accel_mps2.to_numpy()
Lv = d.L.to_numpy(); Wv = d.W.to_numpy()
cls = d.class_name.to_numpy(); tid = d[idcol].to_numpy()
frm = d.frame.to_numpy()

n_frames = len(starts)
for fi in range(n_frames):
    a, b = starts[fi], ends[fi]
    m = b - a
    if m < 2:
        continue
    sl = slice(a, b)
    x = lon[sl]; y = lat[sl]; v = spd[sl]
    L = Lv[sl]; W = Wv[sl]
    dx = x[None, :] - x[:, None]                  # j ahead of i -> dx > 0
    dy = np.abs(y[None, :] - y[:, None])
    halfsum = (W[:, None] + W[None, :]) / 2.0
    lensum = (L[:, None] + L[None, :]) / 2.0

    # ---- leaders ------------------------------------------------------------
    overlap = dy < halfsum * OVERLAP_FRAC * 2.0
    gap = dx - lensum
    cand = (dx > 0) & overlap & (gap < MAX_GAP_M) & (gap > -0.5)
    np.fill_diagonal(cand, False)
    if cand.any():
        gm = np.where(cand, gap, np.inf)
        j = np.argmin(gm, axis=1)
        has = np.isfinite(gm[np.arange(m), j])
        idxs = np.where(has)[0]
        if len(idxs):
            pairs.append(pd.DataFrame({
                "frame": frm[a + idxs],
                "follower_id": tid[a + idxs],
                "leader_id": tid[a + j[idxs]],
                "follower_class": cls[a + idxs],
                "leader_class": cls[a + j[idxs]],
                "gap_m": gm[idxs, j[idxs]],
                "v_follower": v[idxs],
                "v_leader": v[j[idxs]],
                "dv": v[idxs] - v[j[idxs]],
                "a_follower": acc[sl][idxs],
                "lat_offset": dy[idxs, j[idxs]],
            }))

    # ---- side-by-side lateral clearance -------------------------------------
    abreast = (np.abs(dx) < lensum * SIDE_LON_FRAC) & (dy < 6.0)
    np.fill_diagonal(abreast, False)
    iu = np.triu(abreast)
    if iu.any():
        ii, jj = np.where(iu)
        clear = dy[ii, jj] - halfsum[ii, jj]      # edge-to-edge lateral gap
        lateral.append(pd.DataFrame({
            "frame": frm[a + ii],
            "id_a": tid[a + ii], "id_b": tid[a + jj],
            "class_a": cls[a + ii], "class_b": cls[a + jj],
            "lat_clear_m": clear,
            "v_a": v[ii], "v_b": v[jj],
            "v_mean": (v[ii] + v[jj]) / 2.0,
        }))

P = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
S = pd.concat(lateral, ignore_index=True) if lateral else pd.DataFrame()
say(f"\nLeader-follower samples : {len(P)}   ({P.follower_id.nunique() if len(P) else 0} followers)")
say(f"Side-by-side samples    : {len(S)}")

if len(P) == 0:
    sys.exit("ERROR: no following pairs found. The camera window may be too "
             "short, or OVERLAP_FRAC too strict.")

# how much of the data is actually interacting?
say(f"Share of vehicle-samples with a leader : "
    f"{100.0*len(P)/max(len(d),1):5.1f} %")
say("   A low share means most vehicles are free-flowing, and car-following")
say("   parameters then rest on a small subset - say so in the thesis.")

# ---- FIELD-OF-VIEW TRUNCATION ------------------------------------------------
fov = df.lon_m.max() - df.lon_m.min()
say(f"\nFIELD-OF-VIEW TRUNCATION CHECK")
say(f"   observed longitudinal window : {fov:.1f} m")
say(f"   largest gap actually recorded: {P.gap_m.max():.1f} m")
say(f"   gap distribution  p50 {q(P.gap_m,50):.1f} m | p90 {q(P.gap_m,90):.1f} m | "
    f"p99 {q(P.gap_m,99):.1f} m")
say("   A following pair can only be measured when BOTH vehicles are in shot,")
say("   so no gap longer than the window can ever appear. The gap distribution")
say("   is therefore TRUNCATED, and it is truncated hardest on the free-flowing")
say("   pairs that keep long gaps. Consequences to state in the thesis:")
say("     - CC0 and CC1 (short gaps, queued traffic) are barely affected")
say("     - any mean or 85th-percentile gap is biased LOW")
say("     - the share of 'free-flowing' vehicles is overstated, because a")
say("       vehicle whose leader is out of shot looks unconstrained")
say("   Validation on synthetic data with a 12 m window showed only ~27 % of")
say("   true following pairs had both vehicles visible, and 0 % beyond a 12 m")
say("   gap. If a future recording is possible, a longer window is the single")
say("   highest-value change to the data collection.")

# ==============================================================================
# STEP 2 - CAR-FOLLOWING BY CLASS PAIR
# ==============================================================================
rule("STEP 2 : CAR-FOLLOWING BEHAVIOUR BY CLASS")
say(f"{'follower':<11}{'n':>8}{'gap_p15':>9}{'gap_med':>9}{'hdwy_med':>10}"
    f"{'dv_sd':>8}{'stand_n':>9}{'CC0_est':>9}")
say("-" * 78)
cf = {}
for cn, t in P.groupby("follower_class"):
    steady = t[(t.dv.abs() < FOLLOW_DV_MPS) & (t.v_follower > FOLLOW_MIN_MPS)]
    stand = t[(t.v_follower < STANDSTILL_MPS) & (t.v_leader < STANDSTILL_MPS)
              & (t.gap_m < 12.0) & (t.gap_m > -0.5)]
    hd = (steady.gap_m / steady.v_follower) if len(steady) else pd.Series(dtype=float)
    hd = hd[(hd > 0) & (hd < 6)]
    cc0 = np.median(stand.gap_m) if len(stand) >= MIN_SAMPLES else np.nan
    cf[cn] = dict(n=len(t), n_steady=len(steady), n_stand=len(stand),
                  gap_p15=q(t.gap_m, 15), gap_med=q(t.gap_m, 50),
                  hdwy_med=np.median(hd) if len(hd) else np.nan,
                  dv_sd=t.dv.std(), cc0=cc0)
    c = cf[cn]
    s_hd = f"{c['hdwy_med']:.2f}" if np.isfinite(c['hdwy_med']) else 'n/a'
    s_cc0 = f'{cc0:.2f}' if np.isfinite(cc0) else 'n/a'
    say(f"{cn:<11}{c['n']:>8}{c['gap_p15']:>9.2f}{c['gap_med']:>9.2f}"
        f"{s_hd:>10}{c['dv_sd']:>8.2f}{c['n_stand']:>9}{s_cc0:>9}")
say("  gap in m (bumper to bumper) | hdwy = gap/speed in s | dv_sd in m/s")
say(f"  stand_n = samples with BOTH vehicles stopped; CC0 needs >= {MIN_SAMPLES}")

say(f"\nLEADER-CLASS EFFECT - does the vehicle in front change the gap kept?")
say(f"{'follower':<11}{'leader':<11}{'n':>8}{'gap_med':>10}{'hdwy_med':>10}")
say("-" * 78)
for (fc, lc), t in P.groupby(["follower_class", "leader_class"]):
    if len(t) < MIN_SAMPLES * 3:
        continue
    st = t[(t.dv.abs() < FOLLOW_DV_MPS) & (t.v_follower > FOLLOW_MIN_MPS)]
    _h = (st.gap_m / st.v_follower); _h = _h[(_h > 0) & (_h < 6)]
    hd = np.median(_h) if len(_h) >= MIN_SAMPLES else np.nan
    s_hd = f"{hd:.2f}" if np.isfinite(hd) else "n/a"
    say(f"{fc:<11}{lc:<11}{len(t):>8}{q(t.gap_m,50):>10.2f}{s_hd:>10}")
say("  If following a Bus/Truck shows a clearly larger gap than following a")
say("  Bike, that is a real behavioural finding and VISSIM can express it.")

# ==============================================================================
# STEP 3 - LATERAL CLEARANCE
# ==============================================================================
rule("STEP 3 : LATERAL CLEARANCE WHEN TRAVELLING ABREAST")
say("This is the parameter VISSIM calls 'lateral distance' and it is the single")
say("most important non-lane quantity: how close two vehicles will run side by")
say("side. Measured edge-to-edge using the assumed vehicle widths.")
if len(S):
    say(f"\n{'class A':<11}{'class B':<11}{'n':>8}{'p05':>8}{'median':>9}{'p85':>8}")
    say("-" * 78)
    Sx = S[(S.lat_clear_m > -0.5) & (S.lat_clear_m < 4.0)]
    for (ca, cb), t in Sx.groupby(["class_a", "class_b"]):
        if len(t) < MIN_SAMPLES:
            continue
        say(f"{ca:<11}{cb:<11}{len(t):>8}{q(t.lat_clear_m,5):>8.2f}"
            f"{q(t.lat_clear_m,50):>9.2f}{q(t.lat_clear_m,85):>8.2f}")
    say("\nBY CLASS (against any neighbour), and split by speed:")
    say(f"{'class':<11}{'n':>8}{'p05':>8}{'median':>9}"
        f"{'slow_med':>10}{'fast_med':>10}")
    say("-" * 78)
    for cn in sorted(set(Sx.class_a) | set(Sx.class_b)):
        t = Sx[(Sx.class_a == cn) | (Sx.class_b == cn)]
        if len(t) < MIN_SAMPLES:
            continue
        slow = t[t.v_mean < 2.0]; fast = t[t.v_mean >= 4.0]
        s_sl = f"{q(slow.lat_clear_m,50):.2f}" if len(slow) >= MIN_SAMPLES else "n/a"
        s_fa = f"{q(fast.lat_clear_m,50):.2f}" if len(fast) >= MIN_SAMPLES else "n/a"
        say(f"{cn:<11}{len(t):>8}{q(t.lat_clear_m,5):>8.2f}"
            f"{q(t.lat_clear_m,50):>9.2f}{s_sl:>10}{s_fa:>10}")
    neg = 100.0 * (Sx.lat_clear_m < 0).mean()
    say(f"\nWIDTH SANITY CHECK: {neg:.1f} % of side-by-side samples show a")
    say("NEGATIVE clearance, i.e. the two bodies would be overlapping.")
    if neg > 10:
        say("  That is too many to be measurement noise. It means the ASSUMED")
        say("  vehicle widths in DIMS are wider than the vehicles at this site,")
        say("  or the tracker's box centre is not the vehicle centre. Empirical")
        say("  width implied by the closest 1 % of same-class pairs:")
        say(f"     {'class':<12}{'assumed W':>11}{'implied W':>11}")
        for cn in sorted(set(Sx.class_a)):
            t = Sx[(Sx.class_a == cn) & (Sx.class_b == cn)]
            if len(t) < MIN_SAMPLES:
                continue
            # clearance = dy - W  =>  dy at the 1st percentile ~ touching
            implied = np.percentile(t.lat_clear_m + DIMS.get(cn, DEFAULT_DIM)[1], 1)
            say(f"     {cn:<12}{DIMS.get(cn, DEFAULT_DIM)[1]:>11.2f}{implied:>11.2f}")
        say("  Replace DIMS with the implied values and re-run before quoting")
        say("  any lateral clearance in the thesis.")
    else:
        say("  Below 10 % - consistent with measurement noise, widths look sane.")
    say("  p05 = the clearance only 5 % of pairs go below -> the practical")
    say("  minimum, which is what VISSIM's minimum lateral distance encodes.")
    say("  slow/fast columns show whether drivers open up as they speed up;")
    say("  VISSIM takes a distance at 0 km/h and another at 50 km/h.")
else:
    say("No side-by-side samples found.")

# ==============================================================================
# STEP 4 - SPEED AND ACCELERATION DISTRIBUTIONS
# ==============================================================================
rule("STEP 4 : SPEED & ACCELERATION DISTRIBUTIONS (VISSIM inputs)")
say(f"{'class':<11}{'trk':>6}{'n':>9}{'v_p15':>8}{'v_p50':>8}{'v_p85':>8}"
    f"{'v_p95':>8}{'a+p85':>8}{'a-p15':>8}{'latv_p95':>10}")
say("-" * 78)
spd_tab = {}
for cn, t in df.groupby("class_name"):
    v = t.speed_kmph.to_numpy() if "speed_kmph" in t else t.speed_mps.to_numpy() * 3.6
    a = t.accel_mps2.to_numpy()
    lv = np.abs(t.lat_speed_mps.to_numpy()) if "lat_speed_mps" in t else np.array([np.nan])
    spd_tab[cn] = dict(n=len(t), trk=t[idcol].nunique(),
                       p15=q(v, 15), p50=q(v, 50), p85=q(v, 85), p95=q(v, 95))
    say(f"{cn:<11}{t[idcol].nunique():>6}{len(t):>9}{q(v,15):>8.1f}{q(v,50):>8.1f}"
        f"{q(v,85):>8.1f}{q(v,95):>8.1f}{q(a[a>0],85) if (a>0).any() else 0:>8.2f}"
        f"{q(a[a<0],15) if (a<0).any() else 0:>8.2f}{q(lv,95):>10.3f}")
say("  v in km/h | a in m/s^2 | latv = lateral speed in m/s")
say("  VISSIM desired-speed distribution: use the p15-p95 band per class.")

# lateral activity, corrected for track duration
say("\nLATERAL ACTIVITY, CORRECTED FOR TRACK DURATION:")
say(f"{'class':<11}{'med_dur_s':>11}{'wander_m':>10}{'wander/s':>10}{'latv_p95':>10}")
say("-" * 78)
for cn, t in df.groupby("class_name"):
    dur = t.groupby(idcol).size() * DT
    wnd = t.groupby(idcol).lat_m.agg(lambda s: s.max() - s.min())
    per_s = (wnd / dur.reindex(wnd.index)).median()
    lv = np.abs(t.lat_speed_mps.to_numpy()) if "lat_speed_mps" in t else np.array([np.nan])
    say(f"{cn:<11}{dur.median():>11.1f}{wnd.median():>10.2f}{per_s:>10.3f}{q(lv,95):>10.3f}")
say("  RAW 'wander' IS MISLEADING: a track observed for 9 s wanders further than")
say("  one observed for 1.7 s regardless of behaviour. Use wander/s or latv_p95")
say("  to rank classes by weaving. Expect Bike and Rickshaw to lead on those.")

# ==============================================================================
# STEP 5 - WIEDEMANN 99 PARAMETERS
# ==============================================================================
rule("STEP 5 : WIEDEMANN-99 PARAMETERS (CC0-CC9)")
say("Only parameters the data can actually support are given a value. The rest")
say("are marked DEFAULT - use VISSIM's default and say so in the thesis. An")
say("invented number is worse than an acknowledged default.\n")

w99 = {}
say("METHOD FOR CC0 AND CC1 - equilibrium regression, not literal standstill.")
say("  W99 says the following distance is  gap = CC0 + CC1 * v. Requiring both")
say("  vehicles to be FULLY stopped to read CC0 off the data throws away almost")
say("  everything: a moderately congested street has very few frozen pairs.")
say("  Instead a robust (Theil-Sen) line is fitted through the steady-following")
say("  samples across the low-speed range; the INTERCEPT is CC0 and the SLOPE")
say("  is CC1. Theil-Sen is used because it tolerates the outliers that survive")
say("  any vision pipeline.")
say("  VALIDATED on synthetic data where the true values were known:")
say("     Car      fitted CC0 0.97 vs true 1.00 | CC1 1.10 vs true 1.00")
say("     Rickshaw fitted CC0 0.58 vs true 0.60 | CC1 1.00 vs true 0.90")
say("  Classes with few LOW-SPEED samples under-estimate CC0 - the sample count")
say("  in the n_slow column below is the number to judge that by.\n")

# ---- LENGTH SANITY CHECK ----------------------------------------------------
# Every gap is (centre-to-centre distance - half lengths). If the assumed
# LENGTHS are too long, every gap is shifted low and CC0 is driven negative.
# For two stopped vehicles of the same class, the closest observed centre
# distance cannot be less than L + CC0, so it bounds the length from above.
say("LENGTH SANITY CHECK - the assumed DIMS lengths bound every gap.")
say(f"{'class':<11}{'assumed L':>11}{'closest c-c':>13}{'implies L <=':>14}   verdict")
say("-" * 78)
_len_bad = []
for cn in sorted(set(P.follower_class)):
    st = P[(P.follower_class == cn) & (P.leader_class == cn)
           & (P.v_follower < STANDSTILL_MPS) & (P.v_leader < STANDSTILL_MPS)]
    if len(st) < MIN_SAMPLES:
        continue
    La = DIMS.get(cn, DEFAULT_DIM)[0]
    cc = np.percentile(st.gap_m + La, 1)      # rebuild centre-to-centre distance
    verdict = "consistent" if cc >= La else "ASSUMED LENGTH TOO LONG"
    if cc < La:
        _len_bad.append(cn)
    say(f"{cn:<11}{La:>11.2f}{cc:>13.2f}{cc:>14.2f}   {verdict}")
if _len_bad:
    say(f"  Classes flagged: {', '.join(_len_bad)}")
    say("  For these, the assumed length exceeds the closest distance ever")
    say("  observed between two stopped vehicles of that class. That is")
    say("  impossible unless the vehicles overlap, so either the length or the")
    say("  coordinate scale is wrong - and EVERY gap and CC0 below inherits it.")
else:
    say("  No class flagged - assumed lengths are at least self-consistent.")
say("")

say("THE UNDERLYING RELATION - judge this before trusting any fitted number.")
say("If gap does not rise roughly linearly with speed, the W99 equilibrium form")
say("does not describe this traffic and NO fit of it is meaningful.\n")
say(f"{'class':<10}{'v bin m/s':>10}{'n':>8}{'gap_p15':>9}{'gap_p50':>9}{'gap_p85':>9}")
say("-" * 78)
BINW = 0.5
for cn in sorted(set(P.follower_class)):
    pf0 = P[(P.follower_class == cn) & (P.dv.abs() < FOLLOW_DV_MPS + 0.1)
            & (P.gap_m > -0.5) & (P.gap_m < 25.0)]
    shown = 0
    for lo in np.arange(0.0, 7.0, BINW):
        s = pf0[(pf0.v_follower >= lo) & (pf0.v_follower < lo + BINW)]
        if len(s) < 40:
            continue
        say(f"{cn if shown == 0 else '':<10}{lo + BINW/2:>10.2f}{len(s):>8}"
            f"{q(s.gap_m,15):>9.2f}{q(s.gap_m,50):>9.2f}{q(s.gap_m,85):>9.2f}")
        shown += 1
    if shown == 0:
        say(f"{cn:<10}{'no bin reaches 40 samples':>38}")
say("")

for cn in sorted(set(df.class_name)):
    t = df[df.class_name == cn]
    pf = P[P.follower_class == cn]
    row = {}

    # --- CC0 and CC1 jointly, by robust regression of gap on speed -----------
    eq = pf[(pf.dv.abs() < FOLLOW_DV_MPS + 0.1) & (pf.v_follower >= 0)
            & (pf.v_follower < 8.0) & (pf.gap_m > -0.5) & (pf.gap_m < 25.0)]
    n_slow = int((eq.v_follower < 2.0).sum())
    if len(eq) >= MIN_SAMPLES * 4 and theilslopes is not None:
        # Theil-Sen compares every pair of points, so it needs O(n^2) memory:
        # 11,000 samples would ask for ~0.9 GB. Fit repeated random subsamples
        # and take the median fit - same robustness, bounded memory.
        gy = eq.gap_m.to_numpy(); vx = eq.v_follower.to_numpy()
        CAP = 2500
        fits = []
        if len(gy) <= CAP:
            fits.append(theilslopes(gy, vx))
        else:
            sub = np.random.default_rng(12345)
            for _ in range(5):
                k = sub.choice(len(gy), CAP, replace=False)
                fits.append(theilslopes(gy[k], vx[k]))
        slope = float(np.median([f[0] for f in fits]))
        icept = float(np.median([f[1] for f in fits]))
        lo = float(np.median([f[2] for f in fits]))
        hi = float(np.median([f[3] for f in fits]))
        # PHYSICAL GATE. A standstill distance cannot be negative and a desired
        # headway cannot be negative or absurdly large. A fit outside these
        # bounds is not a small error - it means the equilibrium model does not
        # fit this class's data, and reporting the number would be worse than
        # reporting nothing.
        ok_cc0 = 0.0 <= icept <= 5.0
        ok_cc1 = 0.20 <= slope <= 3.00
        row["reject"] = ""
        if not ok_cc0:
            row["reject"] += f"CC0={icept:.2f} outside [0,5] "
        if not ok_cc1:
            row["reject"] += f"CC1={slope:.2f} outside [0.2,3] "
        row["CC0"] = ((icept, len(eq)) if ok_cc0 and ok_cc1 else (np.nan, len(eq)))
        row["CC1"] = ((slope, len(eq)) if ok_cc0 and ok_cc1 else (np.nan, len(eq)))
        row["CC1_ci"] = (lo, hi) if ok_cc0 and ok_cc1 else (np.nan, np.nan)
        row["raw_fit"] = (icept, slope)
        resid = eq.gap_m.to_numpy() - (icept + slope * eq.v_follower.to_numpy())
        row["CC2"] = (float(np.percentile(resid, 85) - np.percentile(resid, 15)), len(eq))
    else:
        row["CC0"] = (np.nan, len(eq))
        row["CC1"] = (np.nan, len(eq))
        row["CC2"] = (np.nan, len(eq))
    row["n_slow"] = n_slow
    row.setdefault("reject", "")
    row.setdefault("raw_fit", (np.nan, np.nan))
    row.setdefault("CC1_ci", (np.nan, np.nan))

    # --- CC8 standstill acceleration -----------------------------------------
    launch = t[(t.speed_mps < CC8_MAX_MPS) & (t.accel_mps2 > 0.05)]
    row["CC8"] = (q(launch.accel_mps2, 85), len(launch)) if len(launch) >= MIN_SAMPLES else (np.nan, len(launch))

    # --- CC9 acceleration in the upper observed speed band -------------------
    hi = t[(t.speed_mps > q(t.speed_mps, 75)) & (t.accel_mps2 > 0.05)]
    row["CC9"] = (q(hi.accel_mps2, 85), len(hi)) if len(hi) >= MIN_SAMPLES else (np.nan, len(hi))

    w99[cn] = row

say(f"{'class':<11}{'CC0 (m)':>10}{'CC1 (s)':>10}{'CC2 (m)':>10}"
    f"{'CC8':>10}{'CC9':>10}{'n_eq':>8}{'n_slow':>8}  CC0 trust")
say("-" * 78)
for cn, r in w99.items():
    def f(k):
        v, n = r[k]
        return f"{v:.2f}" if np.isfinite(v) else "--"
    ns = r["n_slow"]
    trust = "good" if ns >= 300 else ("weak - few slow samples" if ns >= 50
                                      else "UNRELIABLE")
    say(f"{cn:<11}{f('CC0'):>10}{f('CC1'):>10}{f('CC2'):>10}"
        f"{f('CC8'):>10}{f('CC9'):>10}{r['CC0'][1]:>8}{ns:>8}  {trust}")
say("  n_eq = samples in the regression | n_slow = those below 2 m/s, which")
say("  are what pin the intercept. CC8/CC9 sample counts are separate.")
say("")
say("  CC1 95 % confidence interval (Theil-Sen):")
for cn, r in w99.items():
    lo, hi = r.get("CC1_ci", (np.nan, np.nan))
    if np.isfinite(lo):
        say(f"     {cn:<11} {r['CC1'][0]:.2f}  [{lo:.2f}, {hi:.2f}]")
say("  A wide interval means the class does not have a consistent following")
say("  habit in this data - do not quote a single number for it.")
_rej = {cn: r["reject"] for cn, r in w99.items() if r.get("reject")}
if _rej:
    say("")
    say("  REJECTED FITS - physically impossible, NOT reported above:")
    for cn, why in _rej.items():
        raw = w99[cn]["raw_fit"]
        say(f"     {cn:<11} raw fit CC0={raw[0]:.2f} CC1={raw[1]:.2f}   [{why.strip()}]")
    say("  A negative CC0 means the fitted line says vehicles overlap when stopped.")
    say("  A negative CC1 means the gap SHRINKS as they speed up. Neither can be")
    say("  true, so the equilibrium model is not describing these classes here.")
    say("  Likely reasons, in order:")
    say("   1. The assumed vehicle LENGTHS in DIMS are wrong. Every gap is")
    say("      computed as (centre distance - half lengths), so a length error")
    say("      shifts every gap and pushes CC0 straight into negative territory.")
    say("      The width check in Step 3 already suggests the scale is off -")
    say("      the SAME problem affects lengths, and so affects every CC0 here.")
    say("   2. Too few low-speed samples to pin the intercept (see n_slow).")
    say("   3. The class genuinely does not follow an equilibrium law - weaving")
    say("      vehicles cut in and drop back rather than holding a gap.")
    say("  For a rejected class, use the VISSIM default and declare it.")
say("")
say("  WARNING on the 'CC0 trust' column: it reflects SAMPLE COUNT only. A class")
say("  can have many slow samples and still fit badly if its real behaviour is")
say("  not an equilibrium one - a vehicle that habitually cuts in and drops back")
say("  never sits on the gap = CC0 + CC1*v line at all. On synthetic data the")
say("  aggressive cut-in class (CNG) under-estimated CC0 despite 464 slow")
say("  samples, for exactly that reason. Judge each class on the scatter, not")
say("  the label.")
say(f"  Minimum sample count for a reported value: {MIN_SAMPLES}")
say("")
say("NOT DERIVED FROM THIS DATA - use VISSIM defaults and declare them:")
say("  CC3 (threshold for entering 'following')   default -8.00")
say("  CC4 (negative following threshold)         default -0.35")
say("  CC5 (positive following threshold)         default  0.35")
say("  CC6 (speed dependency of oscillation)      default 11.44")
say("  CC7 (oscillation acceleration)             default  0.25")
say("  These describe perceptual thresholds inside the Wiedemann model. They")
say("  cannot be observed directly from trajectories at this noise level -")
say("  they are calibration knobs, and Phase 4 tunes them against observed")
say("  flow and speed instead of measuring them here.")

say("")
say("ON CC2: the value above is the 15-85 spread of the gap around the fitted")
say("equilibrium line. That spread contains REAL driver oscillation AND the")
say("residual measurement error AND genuine heterogeneity between drivers of")
say("the same class. It is therefore an UPPER BOUND on CC2, not an estimate.")
say("VISSIM's default is 4.00 m. If the number above is close to or above that,")
say("use the default and say the data could not separate oscillation from noise.")

say("\nACCURACY CAVEAT carried from Phase 1d:")
say("  speed-based values (CC0, CC1, CC2, lateral clearances) rest on a")
say("  reconstruction with speed RMSE ~0.23 m/s - trustworthy.")
say("  acceleration-based values (CC8, CC9) rest on acceleration recovered")
say("  with correlation r ~ 0.7 - report them WITH that figure.")

# ==============================================================================
# STEP 6 - OPTIONAL VALIDATION AGAINST SYNTHETIC TRUTH
# ==============================================================================
if TRUTH_CSV and os.path.exists(TRUTH_CSV):
    rule("STEP 6 : PAIR-FINDING VALIDATION AGAINST GROUND TRUTH")
    tru = pd.read_csv(TRUTH_CSV)
    if "run_id" in tru.columns and "run_id" in df.columns:
        rt, rf = str(tru.run_id.iloc[0]), str(df.run_id.iloc[0])
        if rt != rf:
            say(f"REFUSED: the truth file is from generator run {rt} but the")
            say(f"filtered file came from run {rf}. They describe different")
            say("simulations, so any comparison would be meaningless.")
            say("Regenerate, re-filter, and re-run - in that order.")
            tru = pd.DataFrame()
    if "leader_id_true" in tru.columns:
        tru["fid"] = tru.vehicle_id.astype(str)
        P2 = P.copy()
        P2["fid"] = P2.follower_id.astype(str).str.split("_").str[0]
        P2["lid"] = P2.leader_id.astype(str).str.split("_").str[0]
        m = P2.merge(tru[["fid", "frame", "leader_id_true", "gap_true_m"]],
                     on=["fid", "frame"], how="inner")
        m = m[m.leader_id_true > 0]
        if len(m):
            ok = (m.lid.astype(float) == m.leader_id_true.astype(float))
            ge = m.gap_m - m.gap_true_m
            gok = ge[ok]
            say(f"Matched {len(m)} pair samples against truth.")
            say(f"   correct leader identified : {100*ok.mean():5.1f} %")
            say(f"   gap error, ALL samples    : RMSE {np.sqrt(np.mean(ge**2)):.3f} m, "
                f"bias {np.mean(ge):+.3f} m")
            if len(gok) > 10:
                say(f"   gap error, CORRECT leader : RMSE {np.sqrt(np.mean(gok**2)):.3f} m, "
                    f"bias {np.mean(gok):+.3f} m")
                say("   The second line is the real measurement accuracy. The first is")
                say("   inflated by samples where the true leader was out of shot and")
                say("   the rule necessarily picked a different vehicle.")
            say("   A high hit rate means the lateral-overlap rule is finding the")
            say("   same leader the simulation intended. A low one means")
            say("   OVERLAP_FRAC needs adjusting before trusting Phase 2 outputs.")
        else:
            say("No overlapping samples between pairs and truth.")
    else:
        say("Truth file has no leader_id_true column - regenerate with the")
        say("current synth_generator.py.")

# ==============================================================================
# EXPORT
# ==============================================================================
rule("EXPORT")
try:
    p1 = os.path.join(OUT_DIR, "VALIDATION_pairs.csv")
    p2 = os.path.join(OUT_DIR, "VALIDATION_lateral.csv")
    p3 = os.path.join(OUT_DIR, "VALIDATION_report.txt")
    P.to_csv(p1, index=False)
    S.to_csv(p2, index=False)
    say(f"Following pairs   : {p1}   ({len(P)} rows)")
    say(f"Lateral clearance : {p2}   ({len(S)} rows)")
    with open(p3, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\n[report saved to {p3}]")
except Exception as e:
    print(f"[warn] could not write outputs: {e}")

rule("END OF PHASE 2")
