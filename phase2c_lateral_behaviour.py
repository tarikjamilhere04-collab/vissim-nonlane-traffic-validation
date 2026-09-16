#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 2C - LATERAL MANOEUVRE & OVERTAKING BEHAVIOUR
 ("lane changing" in a stream that has no lanes)
================================================================================
 WHY THIS IS A SEPARATE SCRIPT
   Phase 2 measures how close vehicles run side by side. It does NOT measure the
   ACT of moving sideways. In Bangladeshi mixed traffic that act is the defining
   behaviour: a bike or CNG does not wait behind a slow rickshaw, it slides
   around it. VISSIM calls the settings for this "lateral behaviour" and
   "overtake on same lane", and they need their own measurements.

 WHAT IS MEASURED
   1. MANOEUVRE EVENTS - each sustained sideways movement, found by hysteresis
      thresholding on lateral speed, then required to accumulate a real shift
   2. For each event: which side, how far, how long, how fast sideways, and the
      longitudinal speed while it happened
   3. OVERTAKES - manoeuvres where the vehicle was behind another and ended up
      in front of it. Includes which side it passed on
   4. GAP ACCEPTANCE - the lead and lag gaps present in the target lateral
      position at the moment the driver committed. This is what VISSIM's
      lane-change safety settings encode
   5. Per-class manoeuvre RATES, and a direct VISSIM parameter mapping

 ROBUSTNESS TO THE OPEN SCALE QUESTION
   Phase 2B is still checking whether the world coordinates are correctly
   scaled. Results here are split accordingly:
     SCALE-FREE  (trustworthy either way): manoeuvre rate, duration, side
                 preference, share of vehicles that ever manoeuvre, overtake
                 rate, and the ranking of classes by aggression
     SCALE-DEPENDENT (must wait for Phase 2B): shift amplitude in metres,
                 lateral speed in m/s, and all gap distances
   Every table below is labelled with which kind it is.

 INPUT   trajectories_filtered.csv    (Phase 1b output)
 OUTPUT  phase2c_manoeuvres.csv       one row per detected event
         phase2c_report.txt
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

# ==============================================================================
INPUT_CSV = r"S:\soscho\trajectories_filtered.csv"
OUT_DIR   = r"S:\soscho"
DT = 0.03332

# ---- [ASSUMED] manoeuvre detection thresholds -------------------------------
V_ON        = 0.25   # m/s  lateral speed that STARTS an event
V_OFF       = 0.10   # m/s  lateral speed that ENDS it (hysteresis)
MIN_SHIFT   = 0.40   # m    an event must accumulate at least this much shift
MAX_DUR_S   = 6.0    # s    longer than this is drift, not a manoeuvre
MIN_DUR_S   = 0.30   # s    shorter than this is noise
MIN_TRACK_S = 1.0    # s    tracks shorter than this cannot show a manoeuvre
OT_WINDOW_S = 4.0    # s    how long after the sideways move to look for the pass

# ---- [ASSUMED] vehicle dimensions, for gap and overlap geometry -------------
DIMS = {"Bike": (1.90, 0.70), "CNG": (2.60, 1.40), "Rickshaw": (2.00, 1.20),
        "Car": (4.40, 1.70), "Truck": (7.50, 2.40), "Bus": (11.0, 2.50)}
DEFAULT_DIM = (3.0, 1.5)
OVERLAP_FRAC = 0.35
MIN_SAMPLES = 20
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


def q(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.percentile(x, p) if x.size else np.nan


# ==============================================================================
rule("PHASE 2C | LATERAL MANOEUVRE & OVERTAKING BEHAVIOUR")
if not os.path.exists(INPUT_CSV):
    sys.exit(f"ERROR: {INPUT_CSV} not found. Run phase1b_FOR_REAL.py first.")
df = pd.read_csv(INPUT_CSV)
idcol = "track_id" if "track_id" in df.columns else "vehicle_id"
say(f"Input : {INPUT_CSV}")
say(f"        {len(df)} rows, {df[idcol].nunique()} tracks")

_syn = ("run_id" in df.columns and df.run_id.notna().any())
if _syn and "synthetic" not in os.path.basename(INPUT_CSV).lower():
    say("\nSTOP - this file contains SYNTHETIC data but is named as real.")
    say("Run phase1b_FOR_REAL.py to rebuild it, then try again.")
    sys.exit(1)

if "direction" in df.columns:
    df = df[df.direction == "with_flow"]
if "flag_implausible" in df.columns:
    df = df[df.flag_implausible == 0]
if "lat_speed_mps" not in df.columns:
    sys.exit("ERROR: no lat_speed_mps column. Re-run phase1b.")
df["L"] = df.class_name.map(lambda c: DIMS.get(c, DEFAULT_DIM)[0])
df["W"] = df.class_name.map(lambda c: DIMS.get(c, DEFAULT_DIM)[1])
say(f"        {len(df)} rows retained")

# ==============================================================================
# STEP 1 - DETECT MANOEUVRE EVENTS
# ==============================================================================
rule("STEP 1 : MANOEUVRE DETECTION")
say(f"An event starts when lateral speed exceeds {V_ON} m/s and ends when it")
say(f"falls below {V_OFF} m/s. Two thresholds rather than one (hysteresis) stops")
say("a single wobble around the threshold from being counted many times.")
say(f"The event is kept only if it accumulates at least {MIN_SHIFT} m of shift")
say(f"and lasts between {MIN_DUR_S} and {MAX_DUR_S} s.")

events = []
n_tracks_ok = 0
for tid, t in df.groupby(idcol):
    t = t.sort_values("frame")
    if len(t) * DT < MIN_TRACK_S:
        continue
    n_tracks_ok += 1
    vy = t.lat_speed_mps.to_numpy()
    lat = t.lat_m.to_numpy()
    lon = t.lon_m.to_numpy()
    spd = t.speed_mps.to_numpy()
    frm = t.frame.to_numpy()
    n = len(t)

    active = False
    k0 = 0
    for k in range(n):
        a = abs(vy[k])
        if not active and a > V_ON:
            active, k0 = True, k
        elif active and a < V_OFF:
            dur = (k - k0) * DT
            shift = lat[k] - lat[k0]
            if MIN_DUR_S <= dur <= MAX_DUR_S and abs(shift) >= MIN_SHIFT:
                seg = slice(k0, k + 1)
                events.append(dict(
                    track_id=tid, cls=t.class_name.iloc[0],
                    f_start=frm[k0], f_end=frm[k],
                    lat_start=lat[k0], lat_end=lat[k],
                    lon_start=lon[k0], lon_end=lon[k],
                    shift_m=shift, dur_s=dur,
                    side=("right" if shift > 0 else "left"),
                    vy_peak=float(np.max(np.abs(vy[seg]))),
                    v_mean=float(np.mean(spd[seg])),
                ))
            active = False
    # a manoeuvre still in progress when the track ends is discarded: its
    # true extent is unknown, and counting it would bias the amplitude low

E = pd.DataFrame(events)
say(f"\nTracks long enough to assess : {n_tracks_ok}")
say(f"Manoeuvre events detected    : {len(E)}")
if len(E) == 0:
    sys.exit("No events found. Lower V_ON / MIN_SHIFT, or the data is too smooth.")

# ==============================================================================
# STEP 2 - OVERTAKES AND GAP ACCEPTANCE
# ==============================================================================
rule("STEP 2 : OVERTAKES AND THE GAPS ACCEPTED")
say("A manoeuvre counts as an OVERTAKE if, at the moment it began, another")
say("vehicle was ahead of the subject with lateral overlap, and within")
say(f"{OT_WINDOW_S:.0f} s afterwards the subject has got in front of that same vehicle.")
say("The LEAD and LAG gaps are measured at the instant the manoeuvre begins,")
say("in the lateral position the driver is moving INTO. Those two numbers are")
say("what a lane-change safety rule is built from.")

# index the data by frame for neighbour queries
d = df.sort_values("frame").reset_index(drop=True)
fr = d.frame.to_numpy()
uf = np.unique(fr)
st = np.searchsorted(fr, uf, "left"); en = np.searchsorted(fr, uf, "right")
fidx = {int(f): (int(a), int(b)) for f, a, b in zip(uf, st, en)}
D_lon = d.lon_m.to_numpy(); D_lat = d.lat_m.to_numpy()
D_v = d.speed_mps.to_numpy(); D_L = d.L.to_numpy(); D_W = d.W.to_numpy()
D_id = d[idcol].to_numpy()

lead_gap, lag_gap, lead_dv, lag_dv, is_ot, ot_side = [], [], [], [], [], []
for _, e in E.iterrows():
    f0 = int(e.f_start)
    rng = fidx.get(f0)
    if rng is None:
        lead_gap.append(np.nan); lag_gap.append(np.nan)
        lead_dv.append(np.nan); lag_dv.append(np.nan)
        is_ot.append(False); ot_side.append(""); continue
    a, b = rng
    mine = D_id[a:b] == e.track_id
    if not mine.any():
        lead_gap.append(np.nan); lag_gap.append(np.nan)
        lead_dv.append(np.nan); lag_dv.append(np.nan)
        is_ot.append(False); ot_side.append(""); continue
    i = a + int(np.where(mine)[0][0])
    myL, myW, myV, myLon = D_L[i], D_W[i], D_v[i], D_lon[i]
    tgt = e.lat_end                       # the position being moved into

    others = np.arange(a, b)
    others = others[others != i]
    if len(others) == 0:
        lead_gap.append(np.nan); lag_gap.append(np.nan)
        lead_dv.append(np.nan); lag_dv.append(np.nan)
        is_ot.append(False); ot_side.append(""); continue

    dy = np.abs(D_lat[others] - tgt)
    half = (myW + D_W[others]) / 2.0
    ov = dy < half * OVERLAP_FRAC * 2.0
    dx = D_lon[others] - myLon
    gap = np.abs(dx) - (myL + D_L[others]) / 2.0

    ahead = ov & (dx > 0)
    behind = ov & (dx < 0)
    lead_gap.append(float(np.min(gap[ahead])) if ahead.any() else np.nan)
    lag_gap.append(float(np.min(gap[behind])) if behind.any() else np.nan)
    lead_dv.append(float(myV - D_v[others][ahead][np.argmin(gap[ahead])]) if ahead.any() else np.nan)
    lag_dv.append(float(D_v[others][behind][np.argmin(gap[behind])] - myV) if behind.any() else np.nan)

    # overtake test: was someone ahead in my ORIGINAL position, and am I now
    # ahead of them?
    dy0 = np.abs(D_lat[others] - e.lat_start)
    ov0 = dy0 < half * OVERLAP_FRAC * 2.0
    cand = ov0 & (dx > 0) & (gap < 15.0)
    done = False
    if cand.any():
        j = others[cand][np.argmin(gap[cand])]
        vid_j = D_id[j]
        # the sideways move only STARTS the pass - completing it takes longer,
        # so look forward for a while rather than judging at the instant the
        # lateral motion stops
        for fq in range(int(e.f_end), int(e.f_end) + int(OT_WINDOW_S / DT), 5):
            r2 = fidx.get(fq)
            if not r2:
                continue
            a2, b2 = r2
            sel = np.where(D_id[a2:b2] == vid_j)[0]
            me2 = np.where(D_id[a2:b2] == e.track_id)[0]
            if len(sel) and len(me2):
                if D_lon[a2 + me2[0]] > D_lon[a2 + sel[0]]:
                    done = True
                    break
    is_ot.append(done)
    ot_side.append(e.side if done else "")

E["lead_gap_m"] = lead_gap
E["lag_gap_m"] = lag_gap
E["lead_dv"] = lead_dv
E["lag_dv"] = lag_dv
E["overtake"] = is_ot
E["ot_side"] = ot_side

say(f"\nOvertakes identified : {int(E.overtake.sum())} "
    f"({100*E.overtake.mean():.1f} % of manoeuvres)")

# ==============================================================================
# STEP 3 - SCALE-FREE RESULTS
# ==============================================================================
rule("STEP 3 : SCALE-FREE RESULTS (valid whatever Phase 2B concludes)")
expo = df.groupby([idcol, "class_name"]).size().reset_index(name="n")
expo["sec"] = expo.n * DT
cls_sec = expo.groupby("class_name").sec.sum()
cls_trk = expo.groupby("class_name").size()

say(f"{'class':<11}{'tracks':>8}{'veh-sec':>10}{'events':>8}{'per veh-min':>13}"
    f"{'% tracks':>10}{'dur_s':>8}")
say("-" * 78)
for cn in sorted(set(df.class_name)):
    ev = E[E.cls == cn]
    sec = cls_sec.get(cn, 0.0)
    trk = cls_trk.get(cn, 0)
    if sec <= 0:
        continue
    rate = len(ev) / (sec / 60.0)
    pct = 100.0 * ev[idcol if idcol in ev.columns else "track_id"].nunique() / max(trk, 1)
    say(f"{cn:<11}{trk:>8}{sec:>10.0f}{len(ev):>8}{rate:>13.2f}{pct:>10.1f}"
        f"{ev.dur_s.median() if len(ev) else np.nan:>8.2f}")
say("  per veh-min = manoeuvres per vehicle-minute of observation. This is the")
say("  cleanest measure of how restless a class is: it does not depend on the")
say("  coordinate scale, nor on how long the tracks happen to be.")
say("  % tracks = share of that class's tracks showing at least one manoeuvre.")

say(f"\nSIDE PREFERENCE (scale-free) - VISSIM 'overtake on same lane L/R':")
say(f"{'class':<11}{'events':>8}{'left %':>9}{'right %':>9}{'overtakes':>11}"
    f"{'ot left %':>11}{'ot right %':>12}")
say("-" * 78)
for cn in sorted(set(E.cls)):
    ev = E[E.cls == cn]
    if len(ev) < MIN_SAMPLES:
        continue
    ot = ev[ev.overtake]
    l = 100.0 * (ev.side == "left").mean()
    say(f"{cn:<11}{len(ev):>8}{l:>9.1f}{100-l:>9.1f}{len(ot):>11}"
        f"{(100.0*(ot.side=='left').mean() if len(ot) else np.nan):>11.1f}"
        f"{(100.0*(ot.side=='right').mean() if len(ot) else np.nan):>12.1f}")
say("  A strong asymmetry is a real behavioural finding and maps straight onto")
say("  VISSIM's two 'overtake on same lane' checkboxes. Near 50/50 means both")
say("  should be enabled.")

# ==============================================================================
# STEP 4 - SCALE-DEPENDENT RESULTS
# ==============================================================================
rule("STEP 4 : SCALE-DEPENDENT RESULTS (hold until Phase 2B is settled)")
say("Every number in this section is in metres or m/s, so if Phase 2B finds the")
say("world coordinates are compressed, every value here scales by the same")
say("factor. The RANKING between classes stays valid either way.\n")
say(f"{'class':<11}{'events':>8}{'shift_p50':>11}{'shift_p85':>11}"
    f"{'vy_peak_p50':>13}{'v_lon_p50':>11}")
say("-" * 78)
for cn in sorted(set(E.cls)):
    ev = E[E.cls == cn]
    if len(ev) < MIN_SAMPLES:
        continue
    say(f"{cn:<11}{len(ev):>8}{q(ev.shift_m.abs(),50):>11.2f}"
        f"{q(ev.shift_m.abs(),85):>11.2f}{q(ev.vy_peak,50):>13.2f}"
        f"{q(ev.v_mean,50)*3.6:>11.1f}")
say("  shift = how far sideways in one manoeuvre, m | vy_peak = peak lateral")
say("  speed, m/s | v_lon = longitudinal speed during it, km/h")

say(f"\nGAP ACCEPTANCE at the moment of committing (scale-dependent):")
say(f"{'class':<11}{'n lead':>8}{'lead p15':>10}{'lead p50':>10}"
    f"{'n lag':>8}{'lag p15':>10}{'lag p50':>10}")
say("-" * 78)
for cn in sorted(set(E.cls)):
    ev = E[E.cls == cn]
    ld = ev.lead_gap_m.dropna(); lg = ev.lag_gap_m.dropna()
    if len(ld) < MIN_SAMPLES and len(lg) < MIN_SAMPLES:
        continue
    say(f"{cn:<11}{len(ld):>8}{q(ld,15):>10.2f}{q(ld,50):>10.2f}"
        f"{len(lg):>8}{q(lg,15):>10.2f}{q(lg,50):>10.2f}")
say("  lead = space ahead in the target position; lag = space behind.")
say("  The 15th percentile is the tight end - the gap drivers of that class are")
say("  still willing to take. That is the number a safety rule should encode,")
say("  not the median.")

# ==============================================================================
# STEP 5 - VISSIM MAPPING
# ==============================================================================
rule("STEP 5 : WHAT TO ENTER IN VISSIM")
say("Driving Behaviour -> Lateral tab:")
say("  'Desired position at free flow'  : set to 'any' - your data shows")
say("     vehicles using the full width, not hugging a lane centre.")
say("  'Overtake on same lane: left'    : see the side-preference table")
say("  'Overtake on same lane: right'   : same")
say("  'Minimum lateral distance' at 0 km/h and at 50 km/h : take these from")
say("     Phase 2 Step 3 (the p05 column, by class) ONCE the scale is settled.")
say("  'Consider next turn'             : off for a mid-block section.")
say("")
say("Driving Behaviour -> Lane Change tab:")
say("  VISSIM's lane-change model assumes discrete lanes, so on a non-lane")
say("  link it is the LATERAL settings above that do the work. Use the lane")
say("  change tab only if you model the carriageway as several narrow lanes.")
say("  If you do, the gap-acceptance table above gives the safety distances.")
say("")
say("Per-class aggression ordering (from the per-veh-min column) is what you")
say("should reproduce in the calibrated model. If the simulated ordering comes")
say("out different in Phase 4, the lateral parameters are wrong even when the")
say("speeds match.")

# ==============================================================================
try:
    p1 = os.path.join(OUT_DIR, "phase2c_manoeuvres.csv")
    p2 = os.path.join(OUT_DIR, "phase2c_report.txt")
    E.to_csv(p1, index=False)
    with open(p2, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    say(f"\nEvents written : {p1}   ({len(E)} rows)")
    print(f"\n[report saved to {p2}]")
except Exception as e:
    print(f"[warn] {e}")

rule("END OF PHASE 2C")
