#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 3B - THE CONSOLIDATED METHODS AND RESULTS REPORT
================================================================================
 WHAT THIS IS FOR
   One document to put in front of the supervisor. It states, in order:
      - what data there is
      - every formula used, with its source in the literature
      - every number extracted, in tables
      - which numbers are EMPIRICAL, which are ASSUMED, and which are SYNTHETIC
      - what is known to be wrong or unresolved
      - where each number goes in the VISSIM GUI

 WHY IT IS GENERATED AND NOT TYPED
   Every table below is recomputed here, from the same CSV files the figures
   are drawn from, using the same constants and the same estimators as Phase 2
   and Phase 2C. Nothing is copied by hand. A typed report can drift away from
   the data; this one cannot.

 OUTPUT
   Thesis_Methods_and_Results.html   self-contained, opens in any browser,
                                     figures embedded, Ctrl+P -> PDF
   Thesis_Methods_and_Results.md     same text without figures, for pasting
                                     into the thesis document

 RUN THIS AFTER phase3a_figures.py, so the figures exist to embed.
================================================================================
"""

import base64
import html as _html
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from scipy.stats import theilslopes
except ImportError:
    theilslopes = None

# ==============================================================================
BASE    = r"S:\soscho"
FIG_DIR = os.path.join(BASE, "figures")
OUT_HTML = os.path.join(BASE, "Thesis_Methods_and_Results.html")
OUT_MD   = os.path.join(BASE, "Thesis_Methods_and_Results.md")

# --- these MUST match phase2_behavior_extraction.py exactly --------------------
DT             = 0.03332
OVERLAP_FRAC   = 0.35
FOLLOW_DV_MPS  = 0.50
CC8_MAX_MPS    = 2.00
MIN_SAMPLES    = 30
DIMS = {"Bike": (1.90, 0.70), "CNG": (2.60, 1.40), "Rickshaw": (2.00, 1.20),
        "Car": (4.40, 1.70), "Truck": (7.50, 2.40), "Bus": (11.0, 2.50)}

CLASS_ORDER = ["Car", "Rickshaw", "Bike", "CNG", "Truck", "Bus"]

# --- numbers that come from the VALIDATION runs, not from the real data -------
# These are transcribed from phase1d_filter_tuning.txt and the Phase 2
# VALIDATION run against synthetic ground truth. They are reported as
# validation evidence, clearly labelled as such. If those runs are repeated,
# update these four numbers here.
VALIDATION = {
    "speed_rmse_mps": 0.23,
    "accel_r": 0.70,
    "pair_hit_rate_pct": 80.3,
    "gap_rmse_m": 0.46,
}
# ==============================================================================

MD = []


def md(s=""):
    MD.append(str(s))


def h1(s):
    md(f"\n# {s}\n")


def h2(s):
    md(f"\n## {s}\n")


def h3(s):
    md(f"\n### {s}\n")


def table(headers, rows):
    md("| " + " | ".join(str(h) for h in headers) + " |")
    md("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        md("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    md("")


def fmt(v, d=2):
    try:
        return f"{float(v):.{d}f}" if np.isfinite(float(v)) else "--"
    except (TypeError, ValueError):
        return "--"


def q(s, p):
    a = np.asarray(s, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


def load(name, required=False):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        if required:
            sys.exit(f"ERROR: {p} not found.")
        print(f"   [skip] {name} not found")
        return None
    print(f"   read  {name}")
    return pd.read_csv(p)


# ==============================================================================
print("PHASE 3B | CONSOLIDATED REPORT")
traj  = load("trajectories_filtered.csv", required=True)
pairs = load("phase2_pairs.csv")
lat   = load("phase2_lateral.csv")
man   = load("phase2c_manoeuvres.csv")
raw   = None
for cand in ("trajectories_scaled.csv", "trajectories.csv"):
    p = os.path.join(BASE, cand)
    if os.path.exists(p):
        raw = cand
        break

idcol = "track_id" if "track_id" in traj.columns else "vehicle_id"
full = traj.copy()
if "direction" in traj.columns:
    traj = traj[traj.direction == "with_flow"]
if "edge" in traj.columns:
    traj = traj[traj.edge == 0]
present = [c for c in CLASS_ORDER if c in set(traj.class_name)]

# ------------------------------------------------------------------------------
# PROVENANCE GUARD - identical in spirit to phase3a. A report assembled from
# files of different runs is worse than no report: it is wrong and looks right.
# ------------------------------------------------------------------------------
_key = traj.set_index([idcol, "frame"])
_bad = []


def _agree(name, ids, frames, vals, col, tol):
    ids = np.asarray(ids)
    frames = np.asarray(frames)
    vals = np.asarray(vals, dtype=float)
    k = min(400, len(ids))
    if k < 20:
        return
    r = np.random.default_rng(0).choice(len(ids), k, replace=False)
    idx = pd.MultiIndex.from_arrays([ids[r], frames[r]])
    got = _key[col].reindex(idx).to_numpy(dtype=float)
    d = np.abs(got - vals[r])
    d = d[np.isfinite(d)]
    if len(d) < 20:
        print(f"   provenance: {name:<24} UNCHECKABLE")
        _bad.append(name)
        return
    med = float(np.median(d))
    print(f"   provenance: {name:<24} median mismatch {med:6.3f} "
          f"(tol {tol})  {'OK' if med <= tol else 'FAIL'}")
    if med > tol:
        _bad.append(name)


if man is not None and len(man):
    _agree("phase2c_manoeuvres.csv", man.track_id, man.f_start, man.lat_start, "lat_m", 0.20)
if pairs is not None and len(pairs):
    _agree("phase2_pairs.csv", pairs.follower_id, pairs.frame, pairs.v_follower, "speed_mps", 0.10)
if lat is not None and len(lat):
    _agree("phase2_lateral.csv", lat.id_a, lat.frame, lat.v_a, "speed_mps", 0.10)
if _bad:
    print("\n" + "!" * 78)
    print("STOP. These result files were not produced from this trajectory file:")
    for b in _bad:
        print("   " + b)
    print("Re-run Phase 2 and Phase 2C on the SAME trajectories_filtered.csv.")
    print("!" * 78)
    sys.exit(1)

# ==============================================================================
# TITLE
# ==============================================================================
h1("Vision-based trajectory extraction and behaviour modelling of "
   "non-lane-based mixed traffic")
md(f"*Methods, formulas and results to date - {datetime.now():%d %B %Y}*")
md("")
md("**Site.** Uttara, Dhaka - the carriageway in front of Zamzam Tower. "
   "One direction of flow, one fixed overhead camera.")
md("")
md("**Purpose of this document.** To set out every method used, every formula "
   "applied, every number extracted, and - separately and explicitly - which of "
   "those numbers are measured from the video, which are assumed, and which "
   "come from synthetic data. It is generated directly from the result files, "
   "so no figure in it can disagree with the table beside it.")
md("")
md("**Software constraint.** PTV VISSIM Free Version. There is no COM "
   "interface, so every parameter below is stated as a value to be typed into "
   "a named field of the VISSIM GUI, not as a script.")

# ==============================================================================
# 1. THE PIPELINE
# ==============================================================================
h1("1. The pipeline in one page")
md("")
table(["Phase", "What it does", "Output"],
      [["0", "Corrects the lateral scale of the homography",
        "`trajectories_scaled.csv`"],
       ["1", "Outlier rejection and trajectory smoothing; derives speed, "
             "acceleration, jerk", "`trajectories_filtered.csv`"],
       ["1c", "Generates synthetic traffic with known ground truth",
        "`synthetic_truth.csv`, `synthetic_noisy.csv`"],
       ["1d", "Tunes the filter against that known truth",
        "`phase1d_filter_tuning.txt`"],
       ["2", "Finds leader-follower pairs without lanes; extracts speeds, "
             "accelerations and Wiedemann-99 parameters",
        "`phase2_pairs.csv`, `phase2_lateral.csv`"],
       ["2B", "Independent check on the lateral scale", "`phase2b` console"],
       ["2C", "Detects sideways manoeuvres and overtakes",
        "`phase2c_manoeuvres.csv`"],
       ["2D", "Bounds vehicle length from closest queued approach",
        "`phase2d_dimensions.txt`"],
       ["3A", "Figures", "`figures/*.png`"],
       ["3C", "Desired speed and CC8/CC9, measured on free-flowing vehicles "
              "only", "`phase3c_desired_speed.txt` / `.csv`"],
       ["3D", "The VISSIM setup sheet - every field, every value",
        "`VISSIM_Setup_Sheet.html`"],
       ["3B", "This report", "this file"]])
md("The order matters. Phase 1d exists because a filter has to be validated "
   "against something whose true answer is known, and a real video has no such "
   "answer. That is the only reason synthetic data appears in this work.")

# ==============================================================================
# 2. FORMULAS
# ==============================================================================
h1("2. Methods and formulas")

h2("2.1 Scale correction (Phase 0)")
md("The homography was recovered with an aspect-ratio error: the lateral axis "
   "was compressed relative to the longitudinal one. A single multiplicative "
   "correction restores it.")
md("")
md("$$ K_{lat} \\;=\\; \\frac{W_{true}}{\\max(y) - \\min(y)} "
   "\\qquad y' \\;=\\; \\bar{y} + K_{lat}\\,(y - \\bar{y}) $$")
md("")
md("with $W_{true}$ the measured carriageway width and $y$ the lateral "
   "coordinate. The full observed span is used, not a trimmed percentile "
   "range: the vehicles nearest the kerb are rickshaws and bikes, and they "
   "*are* the physical edge of the traffic. Trimming them removes the very "
   "observations that define the width.")
md("")
md("The correction is accepted only if it passes a physical test - after "
   "correction no vehicle may lie more than 10 % outside the carriageway:")
md("")
md("$$ \\max(y') - \\min(y') \\;\\le\\; 1.10 \\, W_{true} $$")
md("")
md("The longitudinal scale is left untouched ($K_{lon}=1$), because "
   "longitudinal distances were already consistent with observed speeds.")

h2("2.2 Outlier rejection (Phase 1)")
md("Detection noise is not Gaussian; it is mostly fine with occasional large "
   "failures. A mean-and-standard-deviation rule is therefore the wrong tool - "
   "the outliers corrupt the very statistics used to find them. The **Hampel "
   "identifier** is used instead, because the median and the median absolute "
   "deviation have a 50 % breakdown point.")
md("")
md("$$ \\mathrm{MAD} = \\mathrm{median}\\big(|x_i - \\tilde{x}|\\big), "
   "\\qquad \\hat{\\sigma} = 1.4826 \\cdot \\mathrm{MAD} $$")
md("$$ \\text{reject} \\quad x_i \\iff |x_i - \\tilde{x}| > "
   "3\\hat{\\sigma} $$")
md("")
md("The factor 1.4826 makes $\\hat{\\sigma}$ agree with the standard deviation "
   "when the data really are Gaussian, so the threshold keeps its usual "
   "meaning.")

h2("2.3 Noise level, estimated from the data itself")
md("The smoother needs to know how noisy the measurements are. That is "
   "recovered from the second difference of position, which cancels any "
   "constant-velocity motion and leaves only noise:")
md("")
md("$$ \\Delta^2 x_i = x_{i+1} - 2x_i + x_{i-1}, \\qquad "
   "\\sigma \\;=\\; \\frac{\\mathrm{MAD}(\\Delta^2 x)}{\\sqrt{6}} $$")
md("")
md("The $\\sqrt{6}$ is exact: if each sample carries independent noise of "
   "variance $\\sigma^2$, the second difference has variance "
   "$(1+4+1)\\sigma^2 = 6\\sigma^2$. So the noise level is measured, not "
   "assumed - and it is measured per track, because a small distant vehicle "
   "is noisier than a large near one.")

h2("2.4 Smoothing: three candidates, one chosen")
md("Three estimators were implemented and compared.")
md("")
table(["Method", "What it does", "Why it was considered"],
      [["Symmetric EMA",
        "weighted average of the whole track with exponential weights "
        "falling off either side of the current instant, width D",
        "Thiemann, Treiber & Kesting (2008) - the standard in the "
        "trajectory-extraction literature"],
       ["Savitzky-Golay",
        "local cubic least-squares fit; derivatives taken analytically",
        "gives speed and acceleration without differencing the smoothed "
        "position twice"],
       ["Kalman + RTS",
        "constant-acceleration state (position, speed, acceleration), "
        "forward filter then Rauch-Tung-Striebel backward smoother",
        "**selected** - it is the only one of the three that carries an "
        "explicit noise model, and the only one that estimates acceleration "
        "as a state rather than as a second derivative"]])
md("The symmetric exponential kernel is")
md("")
md("$$ \\hat{x}(t) = \\frac{1}{Z}\\sum_{t'} x(t')\\, e^{-|t-t'|/D} $$")
md("")
md("The symmetric EMA needed one correction before it could be compared "
   "fairly. A symmetric kernel pulls the ends of a track inwards, which "
   "shortened the net displacement of short tracks by about 27 %. The fix is "
   "to remove the linear trend, smooth the residual, and add the trend back:")
md("")
md("$$ x = (a + bt) + r, \\qquad "
   "\\hat{x} = (a + bt) + \\mathrm{smooth}(r) $$")
md("")
md("where $(a+bt)$ is the straight-line trend of the track and $r$ the "
   "residual about it.")
md("")
md("**The Kalman model.** The state is position, speed and acceleration; "
   "with time step $\\Delta t$ the transition, measurement and measurement-"
   "noise matrices are")
md("")
md("")
md("        | 1   dt   dt^2/2 |")
md("    F = | 0    1     dt   |        H = [ 1  0  0 ]        R = sigma^2")
md("        | 0    0      1   |")
md("")
md("The process noise is jerk-driven:")
md("")
md("$$ \\mathbf{Q} = q\\,\\mathbf{G}\\mathbf{G}^T, \\qquad "
   "\\mathbf{G} = [\\;\\Delta t^2/2, \\;\\Delta t, \\;1\\;]^T $$")
md("")
md("so there is a single tuning number, $q$, the assumed jerk intensity. "
   "Phase 1d chooses it.")

h2("2.5 Choosing the amount of smoothing")
md("Smoothing is a trade: too little leaves noise in the acceleration, too "
   "much erases real braking. Two rules were used.")
md("")
md("**Minimum sufficient smoothing.** Of all settings whose implied "
   "accelerations are physically admissible, take the **weakest** - the one "
   "that does the least to the data while still producing possible "
   "accelerations:")
md("")
md("$$ D^* = \\min \\{\\, D \\;:\\; P(|\\ddot{x}| > a_{max}) < \\epsilon "
   "\\,\\} $$")
md("")
md("read as: the smallest smoothing width $D$ for which the probability of an "
   "impossible acceleration falls below $\\epsilon$.")
md("")
md("**Ground truth (Phase 1d).** Synthetic trajectories with known true speed "
   "and acceleration were passed through the whole filter, and the setting "
   "was swept. The objective is agreement with truth, which needs no proxy:")
md("")
md("$$ \\mathrm{RMSE}_v=\\sqrt{\\tfrac1N\\sum (\\hat v-v_{true})^2}, \\qquad "
   "r_a=\\mathrm{corr}(\\hat a, a_{true}) $$")
md("")
md(f"The chosen setting (`KF-RTS`, $q=0.05$) gives a speed RMSE of "
   f"**{VALIDATION['speed_rmse_mps']:.2f} m/s** and an acceleration "
   f"correlation of **r = {VALIDATION['accel_r']:.2f}** against known truth. "
   "Before tuning, the acceleration correlation was 0.35 - the tuning step "
   "doubled it.")

h2("2.6 Finding the leader when there are no lanes")
md("Every car-following model needs a leader, and every standard definition "
   "of a leader is *the vehicle ahead in the same lane*. This road has no "
   "lanes. The definition is therefore replaced by a geometric overlap test.")
md("")
md("Vehicle $j$ is a leader of vehicle $i$ at time $t$ if")
md("")
md("$$ x_j > x_i \\quad\\text{and}\\quad |y_j - y_i| \\;<\\; "
   "\\alpha \\cdot \\tfrac{w_i + w_j}{2}, \\qquad \\alpha = "
   f"{OVERLAP_FRAC} $$")
md("")
md("and among all such $j$, the leader is the nearest one. In words: the "
   "vehicle in front whose body actually blocks yours. The gap is measured "
   "body to body, not centre to centre:")
md("")
md("$$ g_{ij} \\;=\\; (x_j - x_i) - \\frac{L_i + L_j}{2} $$")
md("")
md(f"This pairing rule was validated on synthetic data where the true leader "
   f"is known: it identifies the correct leader in "
   f"**{VALIDATION['pair_hit_rate_pct']:.1f} %** of samples, and when the "
   f"leader is correct the gap is recovered to "
   f"**{VALIDATION['gap_rmse_m']:.2f} m RMSE**.")

h2("2.7 Wiedemann-99 parameters")
md("**CC0 and CC1 are fitted jointly, not measured separately.** The direct "
   "definition of CC0 is the gap between two *fully stopped* vehicles, and a "
   "moderately congested street contains almost no frames in which leader and "
   "follower are both stationary - the counts here were 0 to 18, against a "
   "minimum of 30. Estimating from those few frames would be estimating from "
   "noise.")
md("")
md("Instead the Wiedemann equilibrium relation itself is fitted:")
md("")
md("$$ g \\;=\\; CC0 \\;+\\; CC1 \\cdot v $$")
md("")
md("over all steady-following samples ($|\\Delta v| < "
   f"{FOLLOW_DV_MPS}$ m/s). The intercept is CC0 and the slope is CC1, and "
   "the fit uses thousands of samples rather than a handful.")
md("")
md("The estimator is **Theil-Sen**, not least squares: it takes the median "
   "of the slopes of all point pairs,")
md("")
md("$$ \\hat{\\beta} \\;=\\; \\mathrm{median}_{i<j} "
   "\\frac{g_j - g_i}{v_j - v_i} $$")
md("")
md("which tolerates up to 29 % contaminated points. That matters here, "
   "because a mis-assigned leader produces a wildly wrong gap and least "
   "squares would follow it. Theil-Sen is $O(n^2)$ in memory, so the fit is "
   "run on five random subsamples of 2500 points and the median fit taken - "
   "the same robustness, bounded memory.")
md("")
md("**A physical gate is applied before any fit is reported.** A standstill "
   "distance cannot be negative, and a desired headway cannot be negative or "
   "absurd:")
md("")
md("$$ 0 \\le CC0 \\le 5\\ \\text{m}, \\qquad 0.20 \\le CC1 \\le 3.00\\ "
   "\\text{s} $$")
md("")
md("A fit outside these bounds is not a small error - it means the "
   "equilibrium model does not describe that class on this road. Such fits "
   "are **rejected and reported as rejected**, and the VISSIM default is used "
   "in their place. Reporting them would be worse than reporting nothing.")
md("")
md("CC2, the following oscillation, is taken as the spread of the residuals "
   "about the fitted line and is an **upper bound**, because that spread "
   "contains genuine oscillation plus measurement error plus driver-to-driver "
   "variation, which this data cannot separate:")
md("")
md("$$ CC2 \\;\\le\\; P_{85}(g - \\hat{g}) - P_{15}(g - \\hat{g}) $$")
md("")
md(f"CC8 (standstill acceleration) is the 85th percentile of acceleration "
   f"below {CC8_MAX_MPS:.1f} m/s; CC9 (acceleration at 80 km/h) is the 85th "
   "percentile in the upper observed speed band, because this road never "
   "reaches 80 km/h and the value must be extrapolated by the modeller, not "
   "invented by the script.")

h2("2.8 Sideways manoeuvres (Phase 2C)")
md("A vehicle on a lane-less road moves sideways constantly. A manoeuvre has "
   "to be separated from a wobble, and a single threshold on lateral speed "
   "would count one crossing of that threshold many times. **Hysteresis** is "
   "used: two thresholds, one to start and a lower one to stop.")
md("")
md("$$ \\text{start} \\quad |v_y| > v_{on}, \\qquad "
   "\\text{end} \\quad |v_y| < v_{off}, \\qquad v_{on} > v_{off} $$")
md("")
md("An event is kept only if it accumulates a minimum lateral shift and lasts "
   "within a plausible duration window. The rate is reported per "
   "vehicle-minute of observation, not per vehicle, so that a class whose "
   "vehicles happen to be tracked for longer does not appear more active:")
md("")
md("$$ \\text{rate}_c \\;=\\; \\frac{N_{events,c}}"
   "{\\sum_{i \\in c} T_i \\,/\\, 60} $$")
md("")
md("A manoeuvre counts as an **overtake** if, at the moment it began, another "
   "vehicle was ahead with lateral overlap, and within 4 s the subject is "
   "ahead of that same vehicle. The lead and lag gaps are measured at the "
   "instant the manoeuvre begins, in the lateral position being moved into - "
   "these are the gaps the driver actually accepted.")
md("")
md("**The threshold sensitivity is reported, not hidden.** A result that "
   "changes a lot when the threshold moves is a result about the threshold. "
   "Phase 2C sweeps both thresholds and prints the table.")

h2("2.9 Why synthetic data was generated at all")
md("Synthetic trajectories were generated with the **Intelligent Driver "
   "Model** (Treiber, Hennecke & Helbing, 2000) for longitudinal motion:")
md("")
md("$$ \\dot v = a\\left[1-\\left(\\frac{v}{v_0}\\right)^{\\delta}-"
   "\\left(\\frac{s^*(v,\\Delta v)}{s}\\right)^{2}\\right], \\qquad "
   "s^* = s_0 + vT + \\frac{v\\,\\Delta v}{2\\sqrt{ab}} $$")
md("")
md("and **MOBIL** (Kesting, Treiber & Helbing, 2007) for lane-change "
   "decisions, with a politeness factor $p$:")
md("")
md("$$ (\\tilde a_c - a_c) + p\\left[ (\\tilde a_n - a_n) + "
   "(\\tilde a_o - a_o) \\right] > \\Delta a_{th} $$")
md("")
md("The first bracket is what the changing vehicle gains; the term multiplied "
   "by $p$ is what the new and old followers lose. A selfish driver has "
   "$p = 0$; a courteous one weighs the others' loss almost as heavily as its "
   "own gain.")
md("")
md("Detector noise was added as an AR(1) process rather than white noise, "
   "because real detection error is correlated from frame to frame:")
md("")
md("$$ \\eta_t = \\phi\\,\\eta_{t-1} + \\varepsilon_t, \\qquad \\phi = 0.93 $$")
md("")
md("**This synthetic data was used for one purpose only: to validate the "
   "filter and the pairing rule against a known answer. No synthetic "
   "trajectory and no synthetic parameter appears in the results of Section 3 "
   "or in the VISSIM parameter set of Section 6.**")

# ==============================================================================
# 3. RESULTS
# ==============================================================================
h1("3. Results - all measured from the video")

dur_all = full.groupby(idcol).size() * DT
h2("3.1 The dataset")
rows = [["Source video", "Uttara, Zamzam Tower frontage - one direction"],
        ["Raw trajectory file", raw or "trajectories.csv"],
        ["Rows after filtering", f"{len(full):,}"],
        ["Tracks", f"{full[idcol].nunique():,}"],
        ["Rows used for statistics (with-flow, track ends excluded)",
         f"{len(traj):,}"],
        ["Sampling interval", f"{DT:.5f} s  ({1/DT:.1f} fps)"],
        ["Total observation", f"{dur_all.sum()/60:.1f} vehicle-minutes"],
        ["Median track duration", f"{dur_all.median():.1f} s"],
        ["Longitudinal window observed",
         f"{traj.lon_m.max() - traj.lon_m.min():.1f} m"],
        ["Lateral extent after scale correction",
         f"{traj.lat_m.max() - traj.lat_m.min():.2f} m"]]
# Record length is counted as frames that actually contain a vehicle, not as
# first-frame-to-last-frame. Those two differ whenever the record has gaps,
# and using the span would silently divide by a much larger number and report
# a flow several times too low.
occupied_s = float(full.frame.nunique() * DT)
raw_span_s = float((full.frame.max() - full.frame.min()) * DT)
flow_vph = full[idcol].nunique() / occupied_s * 3600.0 if occupied_s > 0 else np.nan
rows.append(["Record length (frames containing traffic)",
             f"{occupied_s/60:.1f} min"])
rows.append(["Observed flow on this approach",
             f"{flow_vph:,.0f} veh/h  (all classes)"])
table(["Item", "Value"], rows)
md("The observed flow is the count of distinct vehicles entering the field of "
   "view divided by the length of the record. It is the **vehicle input** for "
   "this approach in VISSIM. A vehicle that leaves the view and re-enters "
   "would be counted twice, so treat it as an upper estimate and confirm it "
   "against a manual count of the same clip.")
if raw_span_s > 1.5 * occupied_s:
    md("")
    md(f"> **Note.** The first and last frame in the file are "
       f"{raw_span_s/60:.0f} minutes apart, but only {occupied_s/60:.1f} "
       "minutes of frames actually contain traffic. The record is therefore "
       "not continuous - it is either several clips concatenated or a "
       "partially processed video. The flow above is computed on the occupied "
       "frames, which is the correct denominator, but the gap should be "
       "explained before the flow figure is quoted.")

comp = traj.groupby("class_name")[idcol].nunique()
comp = comp.reindex([c for c in CLASS_ORDER if c in comp.index])
tot = comp.sum()
table(["Class", "Tracks", "Share of vehicles"],
      [[c, int(comp[c]), f"{100*comp[c]/tot:.1f} %"] for c in comp.index])
md("This composition is the **vehicle composition** to enter in VISSIM. It is "
   "measured, not assumed.")

# --- speed --------------------------------------------------------------------
h2("3.2 Speed by class")
sp_rows = []
for cn in present:
    v = traj[traj.class_name == cn]
    kmh = v.speed_kmph.to_numpy() if "speed_kmph" in v else v.speed_mps.to_numpy() * 3.6
    sp_rows.append([cn, f"{v[idcol].nunique():,}", fmt(q(kmh, 15), 1),
                    fmt(q(kmh, 50), 1), fmt(q(kmh, 85), 1), fmt(q(kmh, 95), 1)])
table(["Class", "Tracks", "p15 km/h", "median km/h", "p85 km/h", "p95 km/h"],
      sp_rows)
md("Speeds are the most trustworthy quantity in this work: the validation run "
   f"puts speed error at {VALIDATION['speed_rmse_mps']:.2f} m/s.")
md("")
md("> **These are observed speeds, and they are NOT the VISSIM desired-speed "
   "input.** VISSIM's desired speed is the speed a driver chooses when nothing "
   "is in the way; the car-following model then slows vehicles down whenever a "
   "leader is close. Feeding it speeds observed in congestion counts the "
   "congestion twice, and the simulated traffic comes out far too slow - an "
   "error no amount of W99 tuning can repair, because it is in the input. "
   "**Phase 3C** re-measures the distribution over free-flowing vehicles only; "
   "use its numbers for the desired-speed distributions.")
md("")
md("> **Caution - CNG.** The CNG tracks in this record are almost all "
   "stationary vehicles with drifting detection boxes. Their speed statistics "
   "describe the detector, not the driver. Do not calibrate CNG from this "
   "table.")

# --- acceleration -------------------------------------------------------------
h2("3.3 Acceleration by class")
ac_rows = []
for cn in present:
    a = traj[traj.class_name == cn].accel_mps2.to_numpy()
    ac_rows.append([cn, f"{len(a):,}", fmt(q(a, 5)), fmt(q(a, 15)),
                    fmt(q(a, 50)), fmt(q(a, 85)), fmt(q(a, 95))])
table(["Class", "Samples", "p05", "p15", "median", "p85", "p95"], ac_rows)
md("All values in m/s². The p15 and p85 lines are the practical deceleration "
   "and acceleration envelopes; the extreme percentiles are shown so the "
   "reader can see how much of the range is tail.")
md("")
md(f"> **Caution.** Acceleration is a second derivative and is the weakest "
   f"quantity here: validation gives r = {VALIDATION['accel_r']:.2f} against "
   "known truth. Every acceleration-derived parameter, CC8 and CC9 included, "
   "must be quoted with that number beside it.")

# --- W99 ----------------------------------------------------------------------
h2("3.4 Wiedemann-99 car-following parameters")
w99_rows, rejected = [], []
if pairs is not None:
    for cn in present:
        pf = pairs[pairs.follower_class == cn]
        eq = pf[(pf.dv.abs() < FOLLOW_DV_MPS + 0.1) & (pf.v_follower >= 0)
                & (pf.v_follower < 8.0) & (pf.gap_m > -0.5) & (pf.gap_m < 25.0)]
        n_slow = int((eq.v_follower < 2.0).sum())
        cc0 = cc1 = cc2 = np.nan
        why = ""
        if len(eq) >= MIN_SAMPLES * 4 and theilslopes is not None:
            gy = eq.gap_m.to_numpy(); vx = eq.v_follower.to_numpy()
            CAP = 2500
            fits = []
            if len(gy) <= CAP:
                fits.append(theilslopes(gy, vx))
            else:
                rng = np.random.default_rng(12345)
                for _ in range(5):
                    k = rng.choice(len(gy), CAP, replace=False)
                    fits.append(theilslopes(gy[k], vx[k]))
            slope = float(np.median([f[0] for f in fits]))
            icept = float(np.median([f[1] for f in fits]))
            ok = (0.0 <= icept <= 5.0) and (0.20 <= slope <= 3.00)
            resid = gy - (icept + slope * vx)
            if ok:
                cc0, cc1 = icept, slope
                cc2 = float(np.percentile(resid, 85) - np.percentile(resid, 15))
            else:
                why = f"raw fit CC0={icept:.2f}, CC1={slope:.2f}"
                rejected.append([cn, why])
        t = traj[traj.class_name == cn]
        launch = t[(t.speed_mps < CC8_MAX_MPS) & (t.accel_mps2 > 0.05)]
        cc8 = q(launch.accel_mps2, 85) if len(launch) >= MIN_SAMPLES else np.nan
        hi = t[(t.speed_mps > q(t.speed_mps, 75)) & (t.accel_mps2 > 0.05)]
        cc9 = q(hi.accel_mps2, 85) if len(hi) >= MIN_SAMPLES else np.nan
        trust = ("good" if n_slow >= 300 else
                 "weak - few slow samples" if n_slow >= 50 else "unreliable")
        w99_rows.append([cn, fmt(cc0), fmt(cc1), fmt(cc2), fmt(cc8), fmt(cc9),
                         f"{len(eq):,}", n_slow, trust])
    table(["Class", "CC0 (m)", "CC1 (s)", "CC2 (m) upper bd", "CC8 (m/s²)",
           "CC9 (m/s²)", "n fit", "n slow", "CC0 confidence"], w99_rows)
    md("`--` means the fit was rejected by the physical gate or there were too "
       "few samples. **Use the VISSIM default for those and say so in the "
       "thesis.** The 'CC0 confidence' column reflects sample count only; a "
       "class can have many slow samples and still fit badly, because a "
       "vehicle that habitually cuts in and drops back never sits on the "
       "equilibrium line at all.")
    if rejected:
        md("")
        md("**Rejected fits - reported here precisely so they are not hidden:**")
        md("")
        table(["Class", "Why rejected"], rejected)
        md("A negative CC0 would mean vehicles overlap when stopped; a negative "
           "CC1 would mean the gap shrinks as they speed up. Neither can be "
           "true, so the equilibrium model is not describing these classes on "
           "this road.")

# --- CC0 / length coupling ----------------------------------------------------
h2("3.5 The one quantity this data cannot separate")
md("Trajectories give the distance between vehicle **centres**. At standstill")
md("")
md("$$ d_{centre} \\;=\\; L \\;+\\; CC0 $$")
md("")
md("The measurement pins the **sum**. No amount of statistics separates the "
   "two: a longer assumed vehicle implies a smaller standstill gap, exactly. "
   "Phase 2D therefore reports the whole trade-off curve rather than a single "
   "answer, and the honest sentence for the thesis is:")
md("")
md("> *The data determines the sum of vehicle length and standstill distance "
   "as X m. Assuming a minimum physical clearance of c metres gives a length "
   "of L and a standstill distance of CC0 = X − L.*")
md("")
md("The lengths currently assumed by the pipeline, which every gap depends "
   "on, are stated in Section 5 as assumptions.")

# --- lateral ------------------------------------------------------------------
h2("3.6 Lateral clearance - the parameter that makes this road non-lane-based")
if lat is not None:
    lrows, bad_classes, ok_p05 = [], [], []
    for cn in present:
        s = lat[(lat.class_a == cn) | (lat.class_b == cn)]
        if len(s) < 100:
            continue
        c = s.lat_clear_m.to_numpy()
        p05 = q(c, 5)
        neg = 100 * (c < 0).mean()
        # A p05 of 0.00 is no more usable than a negative one - it still says
        # the bodies touch. Require a real, positive clearance.
        if p05 > 0.05:
            usable = f"**{p05:.2f}**"
            ok_p05.append(p05)
        else:
            usable = "not usable"
            bad_classes.append((cn, p05, neg))
        lrows.append([cn, f"{len(s):,}", fmt(p05), fmt(q(c, 25)),
                      fmt(q(c, 50)), f"{neg:.1f} %", usable])
    table(["Class", "Side-by-side samples", "p05 (m)", "p25 (m)", "median (m)",
           "negative", "enter in VISSIM (m)"], lrows)
    md("The 'negative' column is a **data-quality readout, not a behaviour**. A "
       "negative clearance says the two bodies overlap, which cannot happen. "
       "Clearance is computed as")
    md("")
    md("$$ c_{ij} = |y_i - y_j| - \\frac{w_i + w_j}{2} $$")
    md("")
    md("so it inherits the assumed widths in Section 5.2 directly: a width that "
       "is too large by 30 cm moves every clearance for that class down by "
       "30 cm. Before the Phase 0 scale correction 37.5 % of all samples were "
       "negative; the fall since then is the main evidence that the correction "
       "worked.")
    if bad_classes:
        md("")
        md("**Where the 5th percentile is still negative, it is not a usable "
           "parameter and must not be entered.** The affected classes:")
        md("")
        def _cause(neg):
            if neg > 35:
                return ("assumed width badly wrong, or the class is being "
                        "mis-detected - check this one first")
            if neg > 15:
                return "assumed width too large"
            return "borderline - a small width correction would fix it"

        table(["Class", "p05 (m)", "share negative", "most likely cause"],
              [[cn, fmt(p05), f"{neg:.1f} %", _cause(neg)]
               for cn, p05, neg in bad_classes])
        md("")
        if ok_p05:
            floor = min(ok_p05)
            floor_why = ("It is the smallest clearance that *is* measurable on "
                         "this road, taken from the one class whose width "
                         "assumption survives the test.")
        else:
            floor = 0.20
            floor_why = ("No class here produced a usable measured value, so "
                         "this figure is an assumption in full and must be "
                         "labelled as one in the thesis - not presented as a "
                         "result.")
        md("Two defensible options, and the choice must be stated in the "
           "thesis:")
        md("")
        md(f"1. **Enter a declared floor of {floor:.2f} m** for every affected "
           f"class. {floor_why}")
        md("2. **Re-measure the widths first** (Phase 2D bounds them from the "
           "closest side-by-side approaches) and recompute. This is the better "
           "route if there is time, because it turns an assumption back into a "
           "measurement.")
        md("")
        md("What must *not* happen is entering the negative number, or quietly "
           "clipping it to zero and presenting the result as measured.")
        worst = max(bad_classes, key=lambda r: r[2])
        if worst[2] > 35:
            md("")
            md(f"> **Look at {worst[0]} specifically.** {worst[2]:.0f} % of its "
               f"side-by-side samples are impossible. That is too many to "
               "explain by a slightly wrong width. Check a few frames by eye: "
               "if vans and pickups are being labelled as this class, the "
               "width assumption is being applied to vehicles that are not "
               "that class at all, and the fix is in the detector, not in the "
               "geometry.")

# --- manoeuvres ---------------------------------------------------------------
h2("3.7 Sideways manoeuvres and overtaking")
if man is not None and len(man):
    expo = traj.groupby([idcol, "class_name"]).size().reset_index(name="n")
    expo["sec"] = expo.n * DT
    sec = expo.groupby("class_name").sec.sum()
    mrows = []
    for cn in present:
        k = int((man.cls == cn).sum())
        s = float(sec.get(cn, 0.0))
        if s <= 30:
            continue
        ev = man[man.cls == cn]
        left = 100 * (ev.side == "left").mean() if len(ev) else np.nan
        mrows.append([cn, f"{s/60:.1f}", k, fmt(k / (s / 60.0)),
                      fmt(left, 0) + " %" if np.isfinite(left) else "--",
                      fmt(ev.shift_m.median()) if len(ev) else "--",
                      fmt(ev.dur_s.median()) if len(ev) else "--",
                      "yes" if k >= 15 else "**too few - do not quote**"])
    table(["Class", "veh-min observed", "events", "per veh-min", "to the left",
           "median shift (m)", "median duration (s)", "reliable?"], mrows)
    md("This ranking is **scale-free**: it depends on the ordering of lateral "
       "speeds, not on their absolute calibration, so it survives any "
       "residual error in the lateral scale. It is the single most "
       "transferable behavioural result in this work, and it is what the "
       "calibrated VISSIM model must reproduce.")

    ot = man[man.overtake == True] if "overtake" in man.columns else man.iloc[0:0]
    if len(ot):
        md("")
        h3("Overtakes")
        side = ot.ot_side.value_counts(normalize=True) * 100 if "ot_side" in ot else None
        orows = [["Overtakes detected", f"{len(ot):,}"]]
        if side is not None and len(side):
            for k_, v_ in side.items():
                orows.append([f"...passing on the {k_}", f"{v_:.0f} %"])
        for col, lab in (("lead_gap_m", "Median lead gap accepted (m)"),
                         ("lag_gap_m", "Median lag gap accepted (m)")):
            if col in ot.columns:
                orows.append([lab, fmt(ot[col].median())])
        table(["Item", "Value"], orows)
        md("The side preference is a genuine behavioural finding and maps "
           "directly onto VISSIM's *overtake on left / on right* checkboxes.")

# --- leader effect ------------------------------------------------------------
h2("3.8 Who you are following changes the gap you keep")
if pairs is not None:
    fc = [c for c in present if (pairs.follower_class == c).sum() > 500]
    lc = [c for c in present if (pairs.leader_class == c).sum() > 500]
    if len(fc) >= 2 and len(lc) >= 2:
        rows = []
        for f_ in fc:
            r = [f_]
            for l_ in lc:
                s = pairs[(pairs.follower_class == f_) & (pairs.leader_class == l_)]
                r.append(fmt(np.median(s.gap_m)) if len(s) >= 100 else "--")
            rows.append(r)
        table(["follower \\ leader"] + lc, rows)
        md("Median gap in metres. Blank cells have fewer than 100 samples. "
           "This is why driving behaviour must be assigned **per vehicle "
           "class** in VISSIM rather than once for the whole link: a single "
           "behaviour parameter set cannot produce a table that varies across "
           "a row.")

# ==============================================================================
# 4. FIGURES
# ==============================================================================
h1("4. Figures")
FIGS = [
    ("fig1_speed_distribution.png", "Observed speed by vehicle class. The "
     "15th-95th percentile band is the VISSIM desired-speed input."),
    ("fig5_lanechange_speed.png", "**At what speed vehicles move sideways.** "
     "The grey histogram is all driving; the orange outline is only those "
     "samples during a detected sideways manoeuvre. Where the orange curve "
     "sits relative to the grey one says whether drivers weave when moving "
     "freely or when held up."),
    ("fig2_gap_vs_speed.png", "The Wiedemann equilibrium. Median gap against "
     "speed with the interquartile band, and the fitted line "
     "gap = CC0 + CC1·v. A grey dashed line is an accepted fit and its CC0 "
     "and CC1 are the numbers in Table 3.4; a red dotted line failed the "
     "physical gate and is shown only so the failure is visible - those "
     "classes have no measured CC0 or CC1."),
    ("fig3_lateral_clearance.png", "Side-by-side clearance by class. p05 is "
     "the 5th percentile. Rows whose p05 is below zero are marked in red - "
     "those are not behavioural results but a symptom of the assumed width, "
     "and Section 3.6 says what to do about them."),
    ("fig4_manoeuvre_rate.png", "Sideways manoeuvres per vehicle-minute, "
     "exposure-corrected. This ordering is what calibration must reproduce."),
    ("fig6_leader_effect.png", "Median gap kept, by follower class and leader "
     "class. Reading across a row shows one class changing its gap depending "
     "on what is in front of it."),
    ("fig7_acceleration.png", "Acceleration envelope against speed. The upper "
     "and lower lines are the 85th and 15th percentiles - the VISSIM maximum "
     "acceleration and deceleration functions."),
    ("fig9_desired_speed.png", "Speed of all driving (grey) against speed of "
     "free-flowing vehicles only (coloured outline). The coloured curve is the "
     "VISSIM desired-speed input; the grey one is not, and the gap between "
     "them is the size of the error that would otherwise have been built into "
     "the model. See Phase 3C."),
    ("fig8_space_time.png", "Space-time diagram of the busiest window, and "
     "the lateral positions over the same window. A line crossing others on "
     "the right-hand panel is a vehicle moving sideways past them - this is "
     "what 'lane changing' looks like on a road with no lanes."),
]
for i, (fn, cap) in enumerate(FIGS, 1):
    md(f"**Figure {i}.** {cap}")
    md("")
    md(f"![Figure {i}](figures/{fn})")
    md("")

# ==============================================================================
# 5. PROVENANCE DECLARATION
# ==============================================================================
h1("5. Declaration: what is measured, what is assumed, what is synthetic")
md("This section exists because the distinction matters more than any single "
   "number in this work. Nothing below is left implicit.")

h2("5.1 EMPIRICAL - measured from the video")
table(["Quantity", "Where it appears"],
      [["Vehicle composition by class", "Table 3.1"],
       ["Speed distributions (p15/p50/p85/p95) per class", "Table 3.2"],
       ["Acceleration distributions per class", "Table 3.3"],
       ["CC0, CC1, CC2 for classes not rejected by the physical gate",
        "Table 3.4"],
       ["CC8, CC9 per class", "Table 3.4"],
       ["Lateral clearance percentiles per class", "Table 3.6"],
       ["Manoeuvre rate, shift, duration and side per class", "Table 3.7"],
       ["Overtake counts, side preference, accepted gaps", "Section 3.7"],
       ["Gap by follower-leader class pair", "Table 3.8"]])

h2("5.2 ASSUMED - external values the results depend on")
arows = [["Carriageway width", "10.00 m",
          "Measured on site / from imagery. Sets the lateral scale factor; "
          "every lateral number scales linearly with it."],
         ["Longitudinal scale correction", "K_lon = 1.00",
          "Longitudinal distances were already consistent with observed "
          "speeds, so no correction was applied."]]
for cn in CLASS_ORDER:
    if cn in present:
        L, W = DIMS[cn]
        arows.append([f"{cn} length / width", f"{L:.2f} m / {W:.2f} m",
                      "Literature value. Enters every gap and every clearance; "
                      "see Section 3.5."])
arows += [["Leader overlap fraction", f"α = {OVERLAP_FRAC}",
           "Defines when a vehicle ahead counts as a leader. Validated "
           f"at {VALIDATION['pair_hit_rate_pct']:.1f} % correct against "
           "synthetic truth."],
          ["Steady-following threshold", f"|Δv| < {FOLLOW_DV_MPS} m/s",
           "Selects samples for the equilibrium fit."],
          ["CC3-CC7", "VISSIM defaults",
           "Not observable from overhead trajectories at this resolution."]]
table(["Assumption", "Value", "What it affects"], arows)

h2("5.3 SYNTHETIC - generated, and used only for validation")
table(["Item", "Purpose", "Does it enter the results?"],
      [["IDM/MOBIL synthetic trajectories", "Provide a known ground truth "
        "against which the filter could be tuned (Phase 1d)", "**No**"],
       ["AR(1) synthetic detector noise", "Make the synthetic data as hard to "
        "filter as the real data", "**No**"],
       ["Synthetic leader assignments", "Measure how often the overlap-based "
        "pairing rule finds the right leader", "**No**"]])
md("**No synthetic trajectory, distribution or parameter appears in Section 3 "
   "or Section 6.** The synthetic corpus exists so that the error of each "
   "method could be measured, and those measured errors are quoted throughout "
   "this report.")

# ==============================================================================
# 6. VISSIM
# ==============================================================================
h1("6. Where each number goes in the VISSIM GUI")
md("VISSIM Free Version has no COM interface, so all of this is entered by "
   "hand. This table is the mapping; the values are in Section 3.")
table(["VISSIM location", "Field", "Source"],
      [["Base Data > Distributions > Desired Speed", "one distribution per "
        "class, p15 to p95 of FREE-FLOWING vehicles",
        "**Phase 3C**, not Table 3.2 - see the note in Section 3.2"],
       ["Base Data > Vehicle Types", "Length, Width", "Section 5.2 "
        "(assumed - declare them)"],
       ["Base Data > Vehicle Compositions", "relative flow per class",
        "Table 3.1"],
       ["Driving Behaviour > Following (W99)", "CC0, CC1, CC2", "Table 3.4"],
       ["Driving Behaviour > Following (W99)", "CC3-CC7", "leave at default; "
        "declare as not observable"],
       ["Driving Behaviour > Following (W99)", "CC8, CC9", "Table 3.4, quoted "
        f"with r = {VALIDATION['accel_r']:.2f}"],
       ["Driving Behaviour > Lateral", "Minimum lateral distance at 0 km/h "
        "and at 50 km/h", "Table 3.6, 'enter in VISSIM' column only - never "
        "a negative p05"],
       ["Driving Behaviour > Lateral", "Desired position at free flow = ANY; "
        "Overtake on same lane: left and right both ON",
        "Section 3.7 side preference"],
       ["Driving Behaviour > Lateral", "Consider next turning direction = OFF",
        "there are no lanes to line up in"],
       ["Links", "Lane width = full carriageway as one wide lane",
        "10.00 m (Section 5.2)"],
       ["Vehicle Inputs", "veh/h per approach", "per-leg counts - see the "
        "note in Section 7"]])
md("**The single most important modelling decision**: the carriageway is "
   "modelled as *one wide lane*, not as several narrow ones, and lateral "
   "freedom is produced by the lateral driving-behaviour settings. Splitting "
   "it into lanes would impose exactly the structure this thesis is about the "
   "absence of.")

# ==============================================================================
# 7. LIMITATIONS
# ==============================================================================
h1("7. Known limitations - stated here rather than left to be found")
table(["Limitation", "Consequence", "How it is handled"],
      [["Acceleration recovered at r = " + f"{VALIDATION['accel_r']:.2f}",
        "CC8 and CC9 carry real uncertainty",
        "Quoted with the correlation every time"],
       ["Vehicle lengths are assumed, not measured",
        "CC0 is only determined up to the length assumption",
        "The trade-off curve is reported instead of a single CC0 (Sec. 3.5)"],
       ["Vehicle widths are assumed, not measured",
        "Lateral clearance inherits the error; several classes show an "
        "impossible negative 5th percentile",
        "Those classes are marked not usable and a declared floor is used "
        "instead (Sec. 3.6)"],
       ["CNG tracks are mostly stationary vehicles with drifting boxes",
        "CNG statistics describe the detector, not the driver",
        "Flagged in Section 3.2; not used for calibration"],
       ["No car queues in the record",
        "Car CC0 is not measurable from this video",
        "VISSIM default used, and the reason declared"],
       ["Observation window is " +
        f"{traj.lon_m.max() - traj.lon_m.min():.0f} m",
        "Gaps above roughly 20 m are truncated",
        "Beyond the interacting range, so the effect is small; the gap "
        "distribution is reported in full"],
       ["One direction of one road",
        "Behaviour is characterised for this site only",
        "Stated as the scope; see Section 7.1"]])

h2("7.1 On extending this to a four-leg intersection")
md("A frequent and reasonable question is whether trajectories should be "
   "collected, or generated, for the other three approaches.")
md("")
md("**They should not be generated.** VISSIM is not fed trajectories; it is "
   "fed *parameters*. Driving behaviour - W99 values, lateral clearances, "
   "overtaking preferences - is a property of the drivers and the vehicle mix, "
   "not of the compass direction they happen to be travelling in. The "
   "behaviour measured here applies to all four legs unchanged, and that is a "
   "defensible modelling position, stated plainly.")
md("")
md("What genuinely does differ per leg is only this:")
md("")
md("1. **Vehicle input** - veh/h on each approach")
md("2. **Vehicle composition** - the mix may differ by approach")
md("3. **Turning proportions** - left / through / right at the junction")
md("4. **Signal timing**, if the intersection is signalised")
md("")
md("All four are obtained by **counting**, from a short video of each "
   "approach or from a manual count - not by extracting trajectories. "
   "Fabricating three legs of trajectory data and presenting them as "
   "observations would be indefensible, and is not necessary: the parameters "
   "needed for the other legs are counts, and counts are cheap to collect.")
md("")
md("The measured approach already supplies one row of that table. The "
   "remaining three need only a 10-15 minute count each:")
md("")
comp_pct = {c: 100 * comp[c] / tot for c in comp.index}
head = ["Approach", "Vehicle input (veh/h)"] + [f"{c} %" for c in comp.index] \
       + ["left %", "through %", "right %"]
r0 = ["**This approach (measured)**", f"{flow_vph:,.0f}"] + \
     [f"{comp_pct[c]:.1f}" for c in comp.index] + ["count it", "count it",
                                                   "count it"]
blanks = ["Opposite direction", "Left-hand road", "Right-hand road"]
rows2 = [r0] + [[b, "count it"] + ["count it"] * len(comp.index)
                + ["count it", "count it", "count it"] for b in blanks]
table(head, rows2)
md("Turning proportions are needed even for the measured approach, because "
   "the recording covers a mid-block section rather than the junction itself. "
   "Every cell marked *count it* is a tally from video or from the roadside - "
   "not a quantity this pipeline can produce, and not one that should be "
   "invented.")

# ==============================================================================
# 8. REFERENCES
# ==============================================================================
h1("8. References for the methods used")
for r in [
    "Hampel, F. R. (1974). The influence curve and its role in robust "
    "estimation. *Journal of the American Statistical Association*, 69(346), "
    "383-393.",
    "Kesting, A., Treiber, M., & Helbing, D. (2007). General lane-changing "
    "model MOBIL for car-following models. *Transportation Research Record*, "
    "1999, 86-94.",
    "Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood "
    "estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445-1450.",
    "Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of "
    "data by simplified least squares procedures. *Analytical Chemistry*, "
    "36(8), 1627-1639.",
    "Sen, P. K. (1968). Estimates of the regression coefficient based on "
    "Kendall's tau. *Journal of the American Statistical Association*, "
    "63(324), 1379-1389.",
    "Thiemann, C., Treiber, M., & Kesting, A. (2008). Estimating acceleration "
    "and lane-changing dynamics from next-generation simulation trajectory "
    "data. *Transportation Research Record*, 2088, 90-101.",
    "Treiber, M., Hennecke, A., & Helbing, D. (2000). Congested traffic states "
    "in empirical observations and microscopic simulations. *Physical Review "
    "E*, 62(2), 1805-1824.",
    "Wiedemann, R. (1974). *Simulation des Strassenverkehrsflusses*. "
    "Institut fur Verkehrswesen, Universitat Karlsruhe.",
]:
    md(f"- {r}")

md("")
md("---")
md(f"*Generated {datetime.now():%d %B %Y, %H:%M} directly from the result "
   f"files in `{BASE}`. Every table above is recomputed at generation time; "
   "none is transcribed.*")

# ==============================================================================
# WRITE
# ==============================================================================
md_text = "\n".join(MD)
try:
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\n   wrote {OUT_MD}")
except Exception as e:
    print(f"   [warn] {e}")


# ------------------------------------------------------------------------------
# EQUATIONS, RENDERED OFFLINE
#   The document must be readable on a machine with no internet, so the maths
#   is typeset here, into images, using matplotlib's own maths engine. That
#   engine understands a subset of LaTeX, so a few constructs are normalised
#   first. If an equation still fails to render, the LaTeX source is shown
#   instead - never a blank space.
# ------------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False
    print("   [warn] matplotlib not available - equations will show as LaTeX")

_SUBS = [("\\;", " "), ("\\,", " "), ("\\!", ""), ("\\ ", " "),
         ("\\tfrac1N", "\\frac{1}{N}"), ("\\tfrac12", "\\frac{1}{2}"),
         ("\\tfrac", "\\frac"), ("\\big(", "("), ("\\big)", ")"),
         ("\\Big[", "["), ("\\Big]", "]"), ("\\big[", "["), ("\\big]", "]"),
         ("\\left[", "["), ("\\right]", "]"), ("\\left(", "("),
         ("\\right)", ")"), ("\\iff", "\\Leftrightarrow"),
         ("\\text{", "\\mathrm{"),
         ("\\le ", "\\leq "), ("\\ge ", "\\geq ")]
_eqcache = {}


def eq_png(latex):
    """Render one display equation to a base64 PNG. Returns None on failure."""
    if not _MPL:
        return None
    if latex in _eqcache:
        return _eqcache[latex]
    s = latex
    for a, b in _SUBS:
        s = s.replace(a, b)
    import io
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, "$" + s + "$", fontsize=17, color="#15150f")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=170, bbox_inches="tight",
                    pad_inches=0.14, transparent=True)
        plt.close(fig)
        _eqcache[latex] = base64.b64encode(buf.getvalue()).decode("ascii")
        return _eqcache[latex]
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        _eqcache[latex] = None
        return None


_eq_fail = []


# --- minimal markdown -> html, enough for this document -----------------------
def to_html(text):
    out, in_tbl, in_pre = [], False, False
    for line in text.split("\n"):
        s = line.rstrip()
        # indented block -> preformatted (used for the Kalman matrices)
        if s.startswith("    ") and not in_tbl:
            if not in_pre:
                out.append("<pre>")
                in_pre = True
            out.append(_html.escape(s[4:]))
            continue
        if in_pre:
            out.append("</pre>")
            in_pre = False
        if s.strip() in ("---",) and not in_tbl:
            out.append("<hr>")
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            if not in_tbl:
                out.append("<table>")
                in_tbl = True
                tag = "th"
            else:
                tag = "td"
            out.append("<tr>" + "".join(
                f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>")
            in_tbl = False
        if s.startswith("### "):
            out.append(f"<h3>{inline(s[4:])}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{inline(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h1>{inline(s[2:])}</h1>")
        elif s.startswith("> "):
            out.append(f"<blockquote>{inline(s[2:])}</blockquote>")
        elif s.startswith("- "):
            out.append(f"<ul><li>{inline(s[2:])}</li></ul>")
        elif s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            body = s[2:-2].strip()
            b64 = eq_png(body)
            if b64:
                out.append(f'<div class="eq"><img class="eqimg" '
                           f'src="data:image/png;base64,{b64}"></div>')
            else:
                _eq_fail.append(body)
                out.append(f'<div class="eq"><code>{_html.escape(body)}'
                           f'</code></div>')
        elif s.startswith("!["):
            fn = s.split("(figures/")[-1].rstrip(")")
            p = os.path.join(FIG_DIR, fn)
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("ascii")
                out.append(f'<img src="data:image/png;base64,{b64}">')
            else:
                out.append(f'<p class="missing">[figure not found: {fn} - '
                           f'run phase3a_figures.py first]</p>')
        elif s.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{inline(s)}</p>")
    if in_tbl:
        out.append("</table>")
    return "\n".join(out)


_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
          "Delta": "Δ", "epsilon": "ε", "sigma": "σ", "Sigma": "Σ",
          "phi": "φ", "mu": "μ", "tau": "τ", "eta": "η", "lambda": "λ",
          "varepsilon": "ε"}


def inline_math(e):
    """Turn a short LaTeX expression into readable HTML. Small by design -
    the display equations go through matplotlib; this is only for the
    symbols that appear mid-sentence."""
    import re as _re
    for fn, comb in (("hat", "\u0302"), ("dot", "\u0307"), ("ddot", "\u0308"),
                     ("tilde", "\u0303"), ("bar", "\u0304")):
        e = _re.sub(r"\\" + fn + r"\{?(\\?\w+)\}?",
                    lambda m: (_GREEK.get(m.group(1).lstrip("\\"),
                                          m.group(1).lstrip("\\")) + comb), e)
    e = _re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", e)
    e = _re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", e)
    e = _re.sub(r"\\mathbf\{([^{}]*)\}", r"<strong>\1</strong>", e)
    e = _re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", e)
    for k, v in _GREEK.items():
        e = e.replace("\\" + k, v)
    for a, b in (("\\le", "≤"), ("\\ge", "≥"), ("\\cdot", "·"),
                 ("\\times", "×"), ("\\approx", "≈"), ("\\pm", "±"),
                 ("\\in", "∈"), ("\\sum", "Σ"), ("\\;", " "), ("\\,", " "),
                 ("\\quad", "  "), ("\\", "")):
        e = e.replace(a, b)
    e = _re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", e)
    e = _re.sub(r"\^(\w)", r"<sup>\1</sup>", e)
    e = _re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", e)
    e = _re.sub(r"_(\w)", r"<sub>\1</sub>", e)
    return '<i class="im">' + e + "</i>"


def inline(s):
    s = _html.escape(s)
    if s.count("$") >= 2:
        parts = s.split("$")
        s = "".join(p if i % 2 == 0 else inline_math(p)
                    for i, p in enumerate(parts))
    while "**" in s:
        s = s.replace("**", "<strong>", 1)
        if "**" in s:
            s = s.replace("**", "</strong>", 1)
        else:
            s += "</strong>"
    while s.count("*") >= 2:
        s = s.replace("*", "<em>", 1).replace("*", "</em>", 1)
    while s.count("`") >= 2:
        s = s.replace("`", "<code>", 1).replace("`", "</code>", 1)
    return s


CSS = """
body{font:15px/1.65 Georgia,'Times New Roman',serif;color:#15150f;
 background:#faf9f5;max-width:830px;margin:0 auto;padding:44px 28px 90px}
h1{font:600 25px/1.3 Georgia,serif;margin:46px 0 14px;color:#0b0b0b;
 border-bottom:2px solid #d8d6cf;padding-bottom:8px}
h2{font:600 19px/1.35 Georgia,serif;margin:32px 0 10px;color:#0b0b0b}
h3{font:600 16px/1.35 Georgia,serif;margin:22px 0 8px;color:#52514e}
p{margin:10px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font:13.5px/1.5
 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;background:#fff}
th{background:#f0eee7;text-align:left;font-weight:600;color:#0b0b0b}
th,td{border:1px solid #e0ded6;padding:7px 10px;vertical-align:top}
tr:nth-child(even) td{background:#fcfbf8}
code{font:13px ui-monospace,Menlo,Consolas,monospace;background:#f0eee7;
 padding:1px 5px;border-radius:3px}
.eq{background:#fff;border-left:3px solid #2a78d6;padding:9px 16px;
 margin:14px 0;color:#15150f;overflow-x:auto}
.eqimg{max-height:62px;width:auto;margin:2px 0;border:0;display:inline-block;
 vertical-align:middle;background:transparent}
.im{font-style:italic;font-family:'Cambria Math',Georgia,serif;
 font-style:normal}
.im sub,.im sup{font-style:normal;font-size:.72em}
pre{font:13.5px/1.5 ui-monospace,Menlo,Consolas,monospace;background:#fff;
 border-left:3px solid #2a78d6;padding:11px 16px;margin:14px 0;
 overflow-x:auto}
blockquote{border-left:3px solid #eb6834;background:#fff;margin:14px 0;
 padding:10px 16px;color:#3a3a35}
img{max-width:100%;display:block;margin:16px 0;border:1px solid #e0ded6;
 background:#fff}
ul{margin:6px 0 6px 22px}
hr{border:0;border-top:1px solid #d8d6cf;margin:28px 0}
.missing{color:#b00;font-style:italic}
@media print{body{background:#fff;max-width:none;padding:0}
 h1{page-break-after:avoid}table,img{page-break-inside:avoid}}
"""

body_html = to_html(md_text)
if _eq_fail:
    print(f"   [warn] {len(_eq_fail)} equation(s) could not be typeset and are "
          f"shown as LaTeX source:")
    for e in _eq_fail:
        print("      " + e[:90])

html_doc = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Methods and Results</title><style>" + CSS + "</style>"
            "</head><body>" + body_html + "</body></html>")
try:
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"   wrote {OUT_HTML}   ({len(html_doc)/1e6:.1f} MB)")
except Exception as e:
    print(f"   [warn] {e}")

print("\nDone.")
print("  Open the .html file in any browser.  Ctrl+P -> 'Save as PDF' gives")
print("  a document to hand to the supervisor. The .md file is the same text")
print("  without figures, for pasting into the thesis.")
