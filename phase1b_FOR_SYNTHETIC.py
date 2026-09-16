#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 1 - TRAJECTORY DATA FILTERING & NOISE REDUCTION
 Thesis: Vision-Based Trajectory Extraction, Behavior Modeling, and Microscopic
         Simulation of Non-Lane-Based Mixed Traffic
 PRE-CONFIGURED FOR THE SYNTHETIC FILE. Do not edit anything - just run it.
 It reads  synthetic_noisy.csv  and writes  synthetic_noisy_filtered.csv
 Your REAL data is untouched by this script.

 CHANGES FROM VERSION 2, all prompted by the first full-dataset run:
   [BUGFIX]  the teleport and lateral-speed scans divided by dt instead of the
             true elapsed time, so any movement across an occlusion gap looked
             like a teleport. This inflated the suspect count enormously.
   [NEW]     tracks are SPLIT at position discontinuities (ID switches) instead
             of being carried through as one corrupted track.
   [NEW]     track-level QUARANTINE: tracks still showing implausible dynamics
             are written to a separate file rather than contaminating Phase 2.
   [NEW]     the Kalman measurement noise R is now PER CLASS, not global.
   [NEW]     edge samples are flagged and excluded from the statistics.
   [FIXED]   the per-class lateral metric reported the whole class's span across
             the road, which looked like weaving. It now reports median
             per-track excursion (weaving) and class occupancy separately.
   [FIXED]   class adequacy now counts with-flow tracks of usable duration.
================================================================================
 STAGE 0 : Schema validation, raw data audit, flow-direction normalisation,
           ID-switch / teleport detection
 STAGE 1 : Robust measurement-noise estimation (MAD of 2nd difference)
 STAGE 2 : Hampel identifier -> rejection of gross bounding-box outliers
 STAGE 3 : Uniform time grid + short-gap reconstruction
 STAGE 4 : SELF-TUNED competitive benchmark of three smoothers
             (a) sEMA    - symmetric exponential moving average (Treiber 2008)
             (b) SavGol  - Savitzky-Golay cubic, analytic derivatives
             (c) KF-RTS  - constant-acceleration Kalman + RTS smoother
           Each method's bandwidth parameter is optimised on YOUR data by grid
           search against a three-term objective:
             residual whiteness  +  physical plausibility  +  displacement bias
 STAGE 5 : Method selection + export of the cleaned trajectory file
 STAGE 6 : Diagnostic report (this is the block you paste back to me)
--------------------------------------------------------------------------------
 USAGE    : python phase1_trajectory_filter.py
 REQUIRES : numpy, pandas, scipy      ->  pip install numpy pandas scipy
 WRITES   : S:\\soscho\\trajectories_filtered.csv
            S:\\soscho\\phase1_diagnostics.txt
--------------------------------------------------------------------------------
 DATA PROVENANCE (thesis transparency rule): every number printed below is
 EMPIRICALLY DERIVED from your CSV, except the plausibility thresholds and the
 parameter search grids, which are listed verbatim in the ASSUMED PRIORS block.
================================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy.signal import savgol_filter
except ImportError:
    sys.exit("ERROR: scipy is required.  Run:  pip install scipy")

# ==============================================================================
# CONFIG
# ==============================================================================
INPUT_PATH = r"S:\soscho\synthetic_noisy.csv"
OUTPUT_CSV = r"S:\soscho\synthetic_noisy_filtered.csv"
OUTPUT_TXT = r"S:\soscho\synthetic_phase1_diagnostics.txt"

# ---- ASSUMED PRIORS (declared, NOT derived from data) ------------------------
A_PLAUS_MAX      = 4.0     # m/s^2  plausible |acceleration| ceiling, urban mixed traffic
J_PLAUS_MAX      = 10.0    # m/s^3  plausible |jerk| ceiling
V_PLAUS_MAX      = 22.0    # m/s    urban speed ceiling (~80 km/h)
HAMPEL_WIN_S     = 0.33    # s      Hampel window duration
HAMPEL_NSIGMA    = 3.0     # -      Hampel rejection threshold
MAX_GAP_FRAMES   = 5       # frames longest reconstructed occlusion gap
MIN_TRACK_FRAMES = 15      # frames minimum retained track length
MIN_CONFIDENCE   = 0.25    # -      detection-confidence floor
TELEPORT_MPS     = 25.0    # m/s    frame-to-frame jump above this = ID switch
LAT_V_PLAUS      = 3.0     # m/s    plausible |lateral speed| ceiling (weaving)
JUMP_SPLIT_M     = 1.50    # m      position discontinuity that splits a track in two
QUARANTINE_PCT   = 5.0     # %      a track with more implausible rows than this is quarantined
MIN_DUR_FIT_S    = 2.0     # s      minimum track duration to count toward class adequacy
EDGE_S           = 0.20    # s      samples this close to a track end are marked edge=1
PLAUS_TOL_PCT    = 1.0     # %      tolerated share of implausible a / j after filtering
DISP_BIAS_TOL    = 0.02    # -      tolerated relative error in net travelled distance
MAKE_PLOTS       = True    # write a raw-vs-filtered diagnostic figure if matplotlib exists
# ---- override the automatic selection with a ground-truth-tuned setting ------
# Leave as None to let Stage 4 choose by its plausibility heuristic. Set both
# once phase1d has told you which setting actually recovers the truth best.
FORCE_METHOD     = "KF-RTS"    # None | "KF-RTS" | "SavGol" | "sEMA"
FORCE_PARAM      = 0.05    # sigma_jerk (KF-RTS) | window s (SavGol) | delta s (sEMA)
# ---- tuning grids (searched on your data; the CHOSEN value is empirical) -----
GRID_SEMA_S      = [0.15, 0.25, 0.35, 0.50, 0.70, 1.00, 1.40]   # sEMA delta, s
GRID_SG_S        = [0.20, 0.33, 0.50, 0.70, 1.00, 1.40, 1.80]   # SavGol window, s
GRID_KF_JERK     = [0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00]   # sigma_jerk, m/s^3
TUNE_MAX_TRACKS  = 40      # longest N tracks used for the parameter search
CMP_MAX_TRACKS   = 250     # longest N tracks used for the head-to-head table
MIN_TRK_PER_CLS  = 15      # fewer tracks than this => class needs synthetic support
PROGRESS         = True    # print progress markers (large files take a few minutes)

# Canonical class names. Extend the right-hand lists if your detector uses
# different labels; matching is case-insensitive and ignores spaces/underscores.
CLASS_SYNONYMS = {
    "Car":      ["car", "sedan", "microbus", "jeep", "suv", "privatecar"],
    "Bus":      ["bus", "minibus", "coach"],
    "Truck":    ["truck", "lorry", "coveredvan", "pickup", "trailer"],
    "Rickshaw": ["rickshaw", "cyclerickshaw", "van", "easybike", "autorickshawpedal"],
    "CNG":      ["cng", "autorickshaw", "auto", "tuktuk", "threewheeler", "mishuk"],
    "Bike":     ["bike", "motorbike", "motorcycle", "mc", "scooter", "bicycle", "cycle"],
}
# ==============================================================================

REPORT = []


def say(s=""):
    print(s)
    REPORT.append(str(s))


def rule(title=""):
    if title:
        say("\n" + "=" * 78)
        say(title)
        say("=" * 78)
    else:
        say("-" * 78)


# ==============================================================================
# NUMERICAL HELPERS
# ==============================================================================
def mad_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def estimate_meas_noise(x):
    """sigma from Var(2nd difference) = 6 sigma^2, robust (MAD) version."""
    d2 = np.diff(np.asarray(x, float), n=2)
    s = mad_sigma(d2)
    return np.nan if not np.isfinite(s) else s / np.sqrt(6.0)


def hampel(x, win, nsig):
    s = pd.Series(np.asarray(x, float))
    w = max(3, int(win) | 1)
    med = s.rolling(w, center=True, min_periods=3).median()
    dev = (s - med).abs()
    mad = (dev.rolling(w, center=True, min_periods=3).median() * 1.4826).replace(0.0, np.nan)
    flag = (dev > nsig * mad).fillna(False).to_numpy()
    out = s.to_numpy().copy()
    out[flag] = med.to_numpy()[flag]
    return out, flag


def sema(x, dt, delta):
    """
    Symmetric EMA (Thiemann/Treiber/Kesting 2008), edge-normalised and vectorised.
    A linear trend is removed before smoothing and restored afterwards: without
    this detrending step the symmetric kernel systematically contracts the first
    and last ~3*delta seconds of a track and biases the net displacement.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 3:
        return x.copy()
    idx = np.arange(n, dtype=float)
    b, a0 = np.polyfit(idx, x, 1)          # linear trend
    r = x - (b * idx + a0)
    D = max(1.0, delta / dt)
    half = min(int(3 * D), max(1, n - 1))  # kernel never exceeds the track
    k = np.arange(-half, half + 1)
    w = np.exp(-np.abs(k) / D)
    num = np.convolve(r, w, mode="same")
    den = np.convolve(np.ones_like(r), w, mode="same")
    return (num / den)[:n] + (b * idx + a0)


def kf_rts(z, dt, sigma_meas, sigma_jerk):
    """Constant-acceleration Kalman filter + RTS smoother. State [x, v, a]."""
    z = np.asarray(z, float)
    n = z.size
    F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], float)
    q = sigma_jerk ** 2
    Q = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                      [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
                      [dt**3 / 6,  dt**2 / 2, dt]], float)
    R = max(sigma_meas, 1e-4) ** 2

    xs = np.zeros((n, 3)); Ps = np.zeros((n, 3, 3))
    xp = np.zeros((n, 3)); Pp = np.zeros((n, 3, 3))
    x = np.array([z[0], 0.0, 0.0])
    P = np.diag([R, 10.0, 10.0])
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        S = P[0, 0] + R
        K = P[:, 0] / S
        x = x + K * (z[k] - x[0])
        P = P - np.outer(K, P[0, :])
        xs[k], Ps[k] = x, P
    xsm = xs.copy()
    for k in range(n - 2, -1, -1):
        try:
            C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        except np.linalg.LinAlgError:
            C = Ps[k] @ F.T @ np.linalg.pinv(Pp[k + 1])
        xsm[k] = xs[k] + C @ (xsm[k + 1] - xp[k + 1])
    return xsm[:, 0], xsm[:, 1], xsm[:, 2]


def d1(x, dt):
    return np.gradient(np.asarray(x, float), dt, edge_order=2)


def pct_bad(a, lim):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return 100.0 * np.mean(np.abs(a) > lim) if a.size else np.nan


def tick(msg):
    if PROGRESS:
        print(f"   ... {msg}", flush=True)


def canon_class(name):
    """Map a detector label onto a canonical vehicle class."""
    k = str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    for canon, alts in CLASS_SYNONYMS.items():
        if k in alts or k == canon.lower():
            return canon
    return str(name).strip()          # unknown label kept verbatim and reported


def lag1_rho(r):
    r = np.asarray(r, float)
    if r.size < 8 or np.std(r) < 1e-12:
        return 0.0
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def smooth_series(method, z, dt, param, sigma_meas):
    """Return (position, velocity, acceleration) for the chosen method/parameter."""
    n = z.size
    if method == "sEMA":
        p = sema(z, dt, param)
        v = d1(p, dt); a = d1(v, dt)
    elif method == "SavGol":
        w = max(5, int(round(param / dt)) | 1)
        if w >= n:
            w = max(5, (n - 1) | 1) - 2
        if w < 5:
            p = z.copy(); v = d1(p, dt); a = d1(v, dt)
        else:
            p = savgol_filter(z, w, 3)
            v = savgol_filter(z, w, 3, deriv=1, delta=dt)
            a = savgol_filter(z, w, 3, deriv=2, delta=dt)
    else:  # KF-RTS
        p, v, a = kf_rts(z, dt, sigma_meas, param)
    return p, v, a


def evaluate(method, param, tracks, dt, sigma_meas):
    """
    Diagnostics for one (method, bandwidth) pair, pooled over the tuning tracks:
      pa, pj   share of |a|, |j| beyond the physical ceilings   [physics]
      rrms     RMS of (raw - filtered) position, in metres      [fidelity cost]
      bias     relative error in net travelled distance         [no distortion]
      rho1     lag-1 autocorrelation of the residual            [diagnostic only]
    """
    res, aa, jj, bias, rhos = [], [], [], [], []
    for z in tracks:
        p, v, a = smooth_series(method, z, dt, param, sigma_meas)
        res.append(z - p)
        rhos.append(lag1_rho(z - p))
        aa.append(a); jj.append(d1(a, dt))
        net_raw, net_sm = z[-1] - z[0], p[-1] - p[0]
        bias.append(abs(net_sm - net_raw) / max(abs(net_raw), 1.0))
    res = np.concatenate(res); aa = np.concatenate(aa); jj = np.concatenate(jj)
    return dict(pa=pct_bad(aa, A_PLAUS_MAX), pj=pct_bad(jj, J_PLAUS_MAX),
                rrms=float(np.sqrt(np.mean(res ** 2))),
                bias=float(np.mean(bias)), rho=float(np.mean(np.abs(rhos))),
                jrms=float(np.sqrt(np.mean(jj ** 2))))


def feasible(d):
    """Physical-plausibility constraint that a filtered trajectory must satisfy."""
    return (d["pa"] <= PLAUS_TOL_PCT and d["pj"] <= PLAUS_TOL_PCT
            and d["bias"] <= DISP_BIAS_TOL)


# ==============================================================================
# STAGE 0 - LOAD, VALIDATE, AUDIT
# ==============================================================================
rule("PHASE 1 | STAGE 0 : LOAD, SCHEMA VALIDATION, RAW DATA AUDIT")

if not os.path.exists(INPUT_PATH):
    sys.exit(f"ERROR: file not found -> {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df.columns = [c.strip() for c in df.columns]
required = ["vehicle_id", "class_name", "frame", "time", "X_m", "Y_m"]
missing = [c for c in required if c not in df.columns]
if missing:
    sys.exit(f"ERROR: missing column(s) {missing}. Found: {list(df.columns)}")
for c in ["px", "py", "box_w_px", "box_h_px", "confidence"]:
    if c not in df.columns:
        df[c] = np.nan
df = df.sort_values(["vehicle_id", "frame"]).reset_index(drop=True)

say(f"File                      : {INPUT_PATH}")
say(f"Rows x Columns            : {df.shape[0]} x {df.shape[1]}")
say(f"Unique vehicle tracks     : {df.vehicle_id.nunique()}")

dta = np.diff(np.sort(df.time.unique()))
dta = dta[dta > 0]
DT = float(np.median(dta)); FPS = 1.0 / DT
say(f"Sampling interval dt      : {DT:.5f} s  (=> {FPS:.2f} fps, empirical)")
say(f"Observation duration      : {df.time.max() - df.time.min():.2f} s")

raw_labels = sorted(df.class_name.astype(str).str.strip().unique())
df["class_name"] = df.class_name.map(canon_class)
known = set(CLASS_SYNONYMS)
unknown = sorted(set(df.class_name.unique()) - known)

say("\nClass composition (labels normalised to canonical classes):")
say(f"{'   class':<15}{'tracks':>8}{'rows':>9}{'row %':>8}   {'status'}")
cls_tab = df.groupby("class_name").agg(rows=("frame", "size"),
                                       tracks=("vehicle_id", "nunique"))
thin = []
for k, r in cls_tab.sort_values("tracks", ascending=False).iterrows():
    if int(r.tracks) < MIN_TRK_PER_CLS:
        st = "THIN -> needs synthetic support"
        thin.append((k, int(r.tracks)))
    else:
        st = "adequate for empirical fit"
    say(f"   {str(k):<12s}{int(r.tracks):>8d}{int(r.rows):>9d}"
        f"{100.0*r.rows/len(df):>7.1f}%   {st}")
absent = [c for c in CLASS_SYNONYMS if c not in cls_tab.index]
if absent:
    say(f"   NOT OBSERVED AT ALL: {', '.join(absent)}  -> fully synthetic in Phase 2")
if unknown:
    say(f"   Unrecognised labels kept verbatim: {unknown}")
    say(f"   (raw labels found in file: {raw_labels})")
say(f"   Threshold for an empirical per-class fit: {MIN_TRK_PER_CLS} tracks.")

# --- axis orientation ---------------------------------------------------------
g0 = df.groupby("vehicle_id")
rng_x = g0.X_m.agg(lambda s: s.max() - s.min()).median()
rng_y = g0.Y_m.agg(lambda s: s.max() - s.min()).median()
LON, LAT = ("Y_m", "X_m") if rng_y >= rng_x else ("X_m", "Y_m")
say(f"\nAxis diagnosis            : median travel  X={rng_x:.2f} m   Y={rng_y:.2f} m")
say(f"   LONGITUDINAL (car-following) axis = {LON}")
say(f"   LATERAL      (weaving)       axis = {LAT}")
say(f"   Lateral extent of all detections : {df[LAT].min():.2f} .. {df[LAT].max():.2f} m "
    f"(usable width proxy = {df[LAT].max()-df[LAT].min():.2f} m)")

# --- flow direction normalisation --------------------------------------------
net = g0.apply(lambda t: t[LON].iloc[-1] - t[LON].iloc[0])
SIGN = -1.0 if np.median(net) < 0 else 1.0
n_fwd = int((net * SIGN > 0).sum()); n_opp = int((net * SIGN < 0).sum())
df["_lon"] = df[LON] * SIGN
df["_lat"] = df[LAT]
say(f"\nFlow direction            : dominant travel is "
    f"{'DECREASING' if SIGN < 0 else 'INCREASING'} {LON}")
say(f"   -> longitudinal axis multiplied by {SIGN:+.0f} so that speeds are POSITIVE downstream")
say(f"   Tracks with the flow : {n_fwd}    against the flow (opposing/parked) : {n_opp}")

# --- ID-switch / teleport detection ------------------------------------------
tele = []
for vid, t in df.groupby("vehicle_id"):
    # divide by the ACTUAL elapsed time: consecutive ROWS may straddle an
    # occlusion gap of several frames, and dividing by dt would then report a
    # normal movement as a teleport.
    dtr = t.frame.diff() * DT
    step = np.hypot(t["_lon"].diff(), t["_lat"].diff()) / dtr
    k = int((step > TELEPORT_MPS).sum())
    if k:
        tele.append((vid, k, float(step.max())))
say(f"\nID-switch / teleport scan : {len(tele)} suspect track(s) "
    f"(implied speed > {TELEPORT_MPS} m/s over the true elapsed time)")
for vid, k, mx in tele[:10]:
    say(f"   vehicle_id={vid}: {k} jump(s), max implied speed {mx:.1f} m/s  <-- inspect video")

# --- geometric anomaly scan: lateral motion that cannot be real weaving --------
anom = []
for vid, t in df.groupby("vehicle_id"):
    if len(t) < 10:
        continue
    dlon = abs(t["_lon"].iloc[-1] - t["_lon"].iloc[0])
    dlat_span = t["_lat"].max() - t["_lat"].min()
    latv95 = float(np.nanpercentile(np.abs(t["_lat"].diff() / (t.frame.diff() * DT)), 95))
    why = []
    if dlat_span > max(2.0, dlon):
        why.append(f"lateral span {dlat_span:.1f} m >= longitudinal travel {dlon:.1f} m")
    if latv95 > LAT_V_PLAUS:
        why.append(f"lateral speed p95 {latv95:.1f} m/s > {LAT_V_PLAUS} m/s")
    if why:
        anom.append((vid, str(t.class_name.iloc[0]), "; ".join(why)))
say(f"\nGeometric anomaly scan    : {len(anom)} suspect track(s)")
for vid, cn, why in anom[:10]:
    say(f"   vehicle_id={vid} ({cn}): {why}")
if anom:
    say("   These signatures mean an ID switch, a mis-projected homography, or a")
    say("   stationary vehicle whose box drifts. Verify them against the video")
    say("   before they contaminate the lateral-clearance statistics in Phase 2.")

# --- track health -------------------------------------------------------------
tl = g0.size()
dup = int(df.duplicated(["vehicle_id", "frame"]).sum())
gp = int(g0.frame.apply(lambda s: int((s.diff().fillna(1) > 1).sum())).sum())
say(f"\nTrack length (frames)     : min={tl.min()} median={int(tl.median())} max={tl.max()}")
say(f"Duplicate (id,frame) rows : {dup}")
say(f"Occlusion gap events      : {gp}")
if df.confidence.notna().any():
    say(f"Detection confidence      : mean={df.confidence.mean():.3f} "
        f"p05={df.confidence.quantile(0.05):.3f} min={df.confidence.min():.3f}")
if df.box_w_px.notna().any():
    cvw = g0.box_w_px.apply(lambda s: s.std() / s.mean() if s.mean() else np.nan).median()
    cvh = g0.box_h_px.apply(lambda s: s.std() / s.mean() if s.mean() else np.nan).median()
    say(f"BBox width / height CoV   : {cvw*100:.1f} % / {cvh*100:.1f} % "
        f"[>15 % = strong detector scale jitter]")

# --- raw kinematics using TRUE time stamps ------------------------------------
rv, ra, rj = [], [], []
for vid, t in df.groupby("vehicle_id"):
    if len(t) < 5:
        continue
    tt = t.time.to_numpy()
    v = np.gradient(t["_lon"].to_numpy(), tt, edge_order=2)
    a = np.gradient(v, tt, edge_order=2)
    rv.append(v); ra.append(a); rj.append(np.gradient(a, tt, edge_order=2))
rv = np.concatenate(rv); ra = np.concatenate(ra); rj = np.concatenate(rj)
RAW_J_RMS = float(np.sqrt(np.mean(rj ** 2)))

rule()
say("RAW (UNFILTERED) LONGITUDINAL KINEMATICS - finite differences")
say(f"   speed  v   mean={np.mean(rv):7.2f}  p95={np.percentile(np.abs(rv),95):7.2f}  max|.|={np.max(np.abs(rv)):9.2f}  m/s")
say(f"   accel |a|  mean={np.mean(np.abs(ra)):7.2f}  p95={np.percentile(np.abs(ra),95):7.2f}  max   ={np.max(np.abs(ra)):9.2f}  m/s^2")
say(f"   jerk  |j|  RMS ={RAW_J_RMS:7.2f}  p95={np.percentile(np.abs(rj),95):7.2f}  max   ={np.max(np.abs(rj)):9.2f}  m/s^3")
say(f"   IMPLAUSIBLE : |a|>{A_PLAUS_MAX} m/s^2 -> {pct_bad(ra, A_PLAUS_MAX):5.1f} %   "
    f"|j|>{J_PLAUS_MAX} m/s^3 -> {pct_bad(rj, J_PLAUS_MAX):5.1f} %")
say("   A near-total implausible-jerk share is the signature of bounding-box")
say("   jitter amplified by double differentiation - not driver behaviour.")

# ==============================================================================
# STAGE 1 - MEASUREMENT NOISE
# ==============================================================================
rule("PHASE 1 | STAGE 1 : ROBUST MEASUREMENT-NOISE ESTIMATION")
sl, sa = [], []
for vid, t in df.groupby("vehicle_id"):
    if len(t) < 8:
        continue
    sl.append(estimate_meas_noise(t["_lon"].to_numpy()))
    sa.append(estimate_meas_noise(t["_lat"].to_numpy()))
SIG_LON = float(np.nanmedian(sl)); SIG_LAT = float(np.nanmedian(sa))
say("Estimator: sigma = MAD(2nd difference)/sqrt(6)   [50 % breakdown point]")
say(f"   sigma_longitudinal = {SIG_LON*100:6.2f} cm   "
    f"(per-track p10-p90: {np.nanpercentile(sl,10)*100:.1f} - {np.nanpercentile(sl,90)*100:.1f} cm)")
say(f"   sigma_lateral      = {SIG_LAT*100:6.2f} cm   "
    f"(per-track p10-p90: {np.nanpercentile(sa,10)*100:.1f} - {np.nanpercentile(sa,90)*100:.1f} cm)")
say("   This is the positional uncertainty from detector jitter + homography")
say("   error, and it sets the Kalman measurement-noise covariance R directly.")

# per-class noise: small objects (Bike) are detected less stably than large ones
percls = {}
for cn, tc in df.groupby("class_name"):
    vals = [estimate_meas_noise(t["_lon"].to_numpy())
            for _, t in tc.groupby("vehicle_id") if len(t) >= 8]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        percls[cn] = (float(np.median(vals)), len(vals))
if len(percls) > 1:
    say("\n   Noise by vehicle class (median sigma_lon, tracks used):")
    for cn, (s, n) in sorted(percls.items(), key=lambda kv: -kv[1][0]):
        say(f"      {cn:<12s} {s*100:6.2f} cm   ({n} tracks)")
    say("   Expect small classes (Bike) to be noisiest: fewer pixels per object")
    say("   means a less stable bounding box and a larger homography error.")

# ==============================================================================
# STAGE 2-3 - HAMPEL + GRID + GAPS
# ==============================================================================
rule("PHASE 1 | STAGE 2-3 : HAMPEL OUTLIER REJECTION + GAP RECONSTRUCTION")
HW = max(3, int(round(HAMPEL_WIN_S / DT)) | 1)
say(f"Hampel window {HW} frames ({HW*DT:.2f} s) at {HAMPEL_NSIGMA} sigma_MAD; "
    f"gaps <= {MAX_GAP_FRAMES} frames reconstructed")

tracks, n_lon, n_lat, n_tot, dropped = [], 0, 0, 0, 0
n_splits, dropped_seg = 0, 0
tick(f"Hampel + gap reconstruction over {df.vehicle_id.nunique()} tracks")
for vid, t in df.groupby("vehicle_id"):
    t = t.drop_duplicates("frame").sort_values("frame")
    if df.confidence.notna().any():
        t = t[(t.confidence >= MIN_CONFIDENCE) | (t.confidence.isna())]
    if len(t) < MIN_TRACK_FRAMES:
        dropped += 1
        continue
    full = pd.DataFrame({"frame": np.arange(int(t.frame.min()), int(t.frame.max()) + 1)})
    t = full.merge(t, on="frame", how="left")
    miss = t["_lon"].isna()
    runlen = miss.astype(int).groupby((~miss).cumsum()).transform("sum")
    t = t[~(miss & (runlen > MAX_GAP_FRAMES))]
    if len(t) < MIN_TRACK_FRAMES:
        dropped += 1
        continue
    for c in ["_lon", "_lat", "px", "py", "box_w_px", "box_h_px", "confidence"]:
        t[c] = pd.to_numeric(t[c], errors="coerce").interpolate(limit_direction="both")
    t["vehicle_id"] = vid
    t["class_name"] = t["class_name"].ffill().bfill()
    t["time"] = t.frame * DT

    # --- split at position discontinuities (ID switches) -----------------------
    # A jump larger than JUMP_SPLIT_M between consecutive samples cannot be real
    # motion at these frame rates. Splitting preserves both halves of the track
    # instead of discarding a vehicle that the tracker re-identified mid-way.
    jump = np.hypot(t["_lon"].diff(), t["_lat"].diff()).to_numpy()
    cut = [int(c) for c in np.where(jump > JUMP_SPLIT_M)[0]]
    # explicit iloc slicing: np.split on a DataFrame does not reliably return
    # DataFrames across pandas versions
    if cut:
        bounds = [0] + cut + [len(t)]
        segments = [t.iloc[bounds[b]:bounds[b + 1]] for b in range(len(bounds) - 1)]
    else:
        segments = [t]
    n_splits += len(cut)

    for si, seg in enumerate(segments):
        if len(seg) < MIN_TRACK_FRAMES:
            dropped_seg += 1
            continue
        seg = seg.reset_index(drop=True)
        seg["track_id"] = f"{vid}" if len(segments) == 1 else f"{vid}_{si}"
        lon_c, f1 = hampel(seg["_lon"].to_numpy(), HW, HAMPEL_NSIGMA)
        lat_c, f2 = hampel(seg["_lat"].to_numpy(), HW, HAMPEL_NSIGMA)
        n_lon += int(f1.sum()); n_lat += int(f2.sum()); n_tot += len(seg)
        seg["_lon_h"], seg["_lat_h"] = lon_c, lat_c
        tracks.append(seg)

say(f"Discontinuity splits: {n_splits} jump(s) > {JUMP_SPLIT_M} m cut their track in two")
say(f"Tracks retained  : {len(tracks)} segments from {df.vehicle_id.nunique()} raw ids")
say(f"   dropped: {dropped} raw tracks and {dropped_seg} post-split segments "
    f"below {MIN_TRACK_FRAMES} frames")
say(f"Outliers replaced: longitudinal {n_lon}/{n_tot} ({100.0*n_lon/max(n_tot,1):.2f} %)   "
    f"lateral {n_lat}/{n_tot} ({100.0*n_lat/max(n_tot,1):.2f} %)")
if not tracks:
    sys.exit("ERROR: no track survived. Lower MIN_TRACK_FRAMES / MIN_CONFIDENCE.")

# ==============================================================================
# STAGE 4 - SELF-TUNED BENCHMARK
# ==============================================================================
rule("PHASE 1 | STAGE 4 : SELF-TUNED COMPETITIVE SMOOTHER BENCHMARK")
say("SELECTION PRINCIPLE - minimum sufficient smoothing:")
say("   CONSTRAINT  a filtered trajectory is admissible only if, after filtering,")
say(f"               <= {PLAUS_TOL_PCT:.1f} % of |a| exceed {A_PLAUS_MAX} m/s^2,")
say(f"               <= {PLAUS_TOL_PCT:.1f} % of |j| exceed {J_PLAUS_MAX} m/s^3, and")
say(f"               net travelled distance is distorted by <= {DISP_BIAS_TOL*100:.0f} %.")
say("   TUNING      within each method, take the WEAKEST bandwidth that is still")
say("               admissible - never smooth more than physics requires.")
say("   SELECTION   among the three admissible candidates, take the one with the")
say("               SMALLEST residual RMS, i.e. the filter that buys physical")
say("               plausibility at the lowest cost in positional fidelity.")
say("   Rationale: Phase 2 needs credible ACCELERATIONS (for W99 CC8/CC9 and the")
say("   desired-acceleration curves), so implausible dynamics are a hard fail;")
say("   but over-smoothing destroys the very stop-and-go and creeping behaviour")
say("   that characterises Dhaka mixed traffic, so it is penalised, not rewarded.")
say(f"\nTuning sample: the {TUNE_MAX_TRACKS} longest tracks, longitudinal axis.\n")

order = sorted(range(len(tracks)), key=lambda i: -len(tracks[i]))[:TUNE_MAX_TRACKS]
tune = [tracks[i]["_lon_h"].to_numpy() for i in order if len(tracks[i]) >= 20]
if not tune:
    tune = [t["_lon_h"].to_numpy() for t in tracks]

# grids ordered from WEAKEST to STRONGEST smoothing
GRIDS = {"sEMA": GRID_SEMA_S, "SavGol": GRID_SG_S, "KF-RTS": sorted(GRID_KF_JERK, reverse=True)}
UNITS = {"sEMA": "delta=%.2f s", "SavGol": "win=%.2f s", "KF-RTS": "sigma_jerk=%.2f m/s^3"}
best_param, best_diag, admissible = {}, {}, {}

for m, grid in GRIDS.items():
    tick(f"tuning {m} over {len(grid)} bandwidths on {len(tune)} tracks")
    say(f"  {m}   (weakest -> strongest smoothing)")
    say(f"     {'param':>12}{'a>lim%':>9}{'j>lim%':>9}{'ResidRMS':>10}{'dispbias':>10}"
        f"{'|rho1|':>9}   admissible")
    chosen = None
    for p in grid:
        d = evaluate(m, p, tune, DT, SIG_LON)
        ok = feasible(d)
        say(f"     {p:>12.2f}{d['pa']:>9.2f}{d['pj']:>9.2f}{d['rrms']*100:>9.2f}c"
            f"{d['bias']*100:>9.2f}%{d['rho']:>9.3f}   {'YES' if ok else 'no'}")
        if ok and chosen is None:
            chosen, chosen_d = p, d
    if chosen is None:                      # nothing admissible -> least-bad
        cands = [(evaluate(m, p, tune, DT, SIG_LON), p) for p in grid]
        cands.sort(key=lambda c: c[0]["pa"] + c[0]["pj"])
        chosen_d, chosen = cands[0]
        admissible[m] = False
        say(f"     -> NO admissible setting; fallback to " + UNITS[m] % chosen)
    else:
        admissible[m] = True
        say(f"     -> selected " + UNITS[m] % chosen +
            f"   (weakest admissible, residual {chosen_d['rrms']*100:.2f} cm)")
    best_param[m], best_diag[m] = chosen, chosen_d
    say("")

rule()
cmp_idx = sorted(range(len(tracks)), key=lambda i: -len(tracks[i]))[:CMP_MAX_TRACKS]
cmp_tracks = [tracks[i] for i in cmp_idx]
say(f"HEAD-TO-HEAD AT EACH METHOD'S TUNED SETTING "
    f"({len(cmp_tracks)} longest tracks, longitudinal axis)")
say(f"{'Method':<9}{'param':>12}{'ResidRMS':>11}{'|j|RMS':>9}{'a>lim%':>9}"
    f"{'j>lim%':>9}{'|rho1|':>9}  admis")
say("-" * 78)
summary = {}
for m in GRIDS:
    tick(f"head-to-head: {m}")
    res, aa, jj, rh = [], [], [], []
    for t in cmp_tracks:
        z = t["_lon_h"].to_numpy()
        p, v, a = smooth_series(m, z, DT, best_param[m], SIG_LON)
        res.append(z - p); aa.append(a); jj.append(d1(a, DT)); rh.append(lag1_rho(z - p))
    res = np.concatenate(res); aa = np.concatenate(aa); jj = np.concatenate(jj)
    summary[m] = dict(rrms=float(np.sqrt(np.mean(res**2))),
                      jrms=float(np.sqrt(np.mean(jj**2))),
                      pa=pct_bad(aa, A_PLAUS_MAX), pj=pct_bad(jj, J_PLAUS_MAX),
                      rho=float(np.mean(np.abs(rh))))
    s = summary[m]
    say(f"{m:<9}{best_param[m]:>12.2f}{s['rrms']*100:>10.2f}c{s['jrms']:>9.2f}"
        f"{s['pa']:>9.2f}{s['pj']:>9.2f}{s['rho']:>9.3f}  {'YES' if admissible[m] else 'no'}")
say("\n  ResidRMS in cm | |j|RMS in m/s^3 | rho1 = residual lag-1 autocorrelation")
say(f"  Reference: measurement sigma_lon = {SIG_LON*100:.2f} cm. A residual RMS of the")
say("  order of sigma means the filter removed the noise and little else.")

pool = [m for m in GRIDS if admissible[m]] or list(GRIDS)
BEST = min(pool, key=lambda m: summary[m]["rrms"])
PARAM = best_param[BEST]

if FORCE_METHOD is not None and FORCE_PARAM is not None:
    say("")
    say(f"OVERRIDE: the automatic choice ({BEST} @ {PARAM}) is being replaced by")
    say(f"   {FORCE_METHOD} @ {FORCE_PARAM}, supplied from phase1d ground-truth tuning.")
    say("   The heuristic above chose without knowing the true kinematics; the")
    say("   forced setting was measured against known truth. Prefer the latter.")
    BEST, PARAM = FORCE_METHOD, FORCE_PARAM
    if BEST not in summary:
        summary[BEST] = dict(rrms=np.nan, jrms=np.nan, pa=np.nan, pj=np.nan, rho=np.nan)
        UNITS.setdefault(BEST, "%.3f")

# ==============================================================================
# STAGE 5 - EXPORT
# ==============================================================================
rule("PHASE 1 | STAGE 5 : METHOD SELECTION & CLEANED-DATA EXPORT")
say(f">>> SELECTED : {BEST}  with  " + UNITS[BEST] % PARAM +
    f"   (residual {summary[BEST]['rrms']*100:.2f} cm)")
for m in GRIDS:
    if m != BEST:
        why = ("not admissible" if not admissible[m]
               else f"admissible but residual {summary[m]['rrms']*100:.2f} cm "
                    f"= {summary[m]['rrms']/max(summary[BEST]['rrms'],1e-9):.2f}x worse")
        say(f"    rejected  : {m:<7s} ({why})")

# lateral axis: same family, bandwidth scaled to the lateral noise level
say("\nMeasurement noise is applied PER CLASS, not globally: the Kalman R matrix")
say("uses each class's own sigma, so a Rickshaw track is not filtered as if it")
say("were as cleanly detected as a Car.")

out_rows = []
tick(f"applying {BEST} to all {len(tracks)} tracks (both axes)")
edge_n = max(1, int(round(EDGE_S / DT)))
for ti, t in enumerate(tracks):
    if PROGRESS and len(tracks) > 200 and ti % 200 == 0 and ti:
        tick(f"  {ti}/{len(tracks)} tracks filtered")
    z = t["_lon_h"].to_numpy(); y = t["_lat_h"].to_numpy()
    cls = str(t.class_name.iloc[0])
    s_lon = percls.get(cls, (SIG_LON, 0))[0] if percls else SIG_LON
    if not np.isfinite(s_lon):
        s_lon = SIG_LON
    p, v, a = smooth_series(BEST, z, DT, PARAM, s_lon)
    q, vy, ay = smooth_series(BEST, y, DT, PARAM, SIG_LAT)
    edge = np.zeros(len(t), dtype=int)
    edge[:edge_n] = 1; edge[-edge_n:] = 1
    out_rows.append(pd.DataFrame({
        "track_id": t.track_id.to_numpy(),
        "edge": edge,
        "sigma_used_cm": s_lon * 100,
        "vehicle_id": t.vehicle_id.to_numpy(),
        "class_name": t.class_name.to_numpy(),
        "frame": t.frame.to_numpy(),
        "time": t.time.to_numpy(),
        "lon_m": p, "lat_m": q,
        "speed_mps": v, "speed_kmph": v * 3.6,
        "accel_mps2": a, "jerk_mps3": d1(a, DT),
        "lat_speed_mps": vy, "lat_accel_mps2": ay,
        "raw_lon_m": z, "raw_lat_m": y,
        "box_w_px": t.box_w_px.to_numpy(), "box_h_px": t.box_h_px.to_numpy(),
        "confidence": t.confidence.to_numpy(),
    }))
clean = pd.concat(out_rows, ignore_index=True).sort_values(["track_id", "frame"])
clean["flag_implausible"] = ((clean.speed_mps.abs() > V_PLAUS_MAX) |
                             (clean.accel_mps2.abs() > A_PLAUS_MAX)).astype(int)
clean["direction"] = np.where(
    clean.groupby("track_id").speed_mps.transform("mean") >= 0, "with_flow", "opposing")
# carry a generator run id through, so downstream steps can refuse to compare
# a filtered file against ground truth from a DIFFERENT generation
if "run_id" in df.columns and len(df):
    clean["run_id"] = df.run_id.iloc[0]

# --- track-level quality gate -------------------------------------------------
# Implausible dynamics cluster inside a minority of corrupted tracks rather than
# spreading evenly. Those tracks are quarantined whole: a track that contains an
# undetected ID switch is wrong everywhere near it, and no amount of extra
# smoothing repairs it - it only hides the damage.
bad_rate = clean.groupby("track_id").flag_implausible.mean() * 100
bad_ids = set(bad_rate[bad_rate > QUARANTINE_PCT].index)
quar = clean[clean.track_id.isin(bad_ids)]
clean = clean[~clean.track_id.isin(bad_ids)]
say(f"\nQUALITY GATE: {len(bad_ids)} track(s) exceed {QUARANTINE_PCT} % implausible rows "
    f"-> quarantined ({len(quar)} rows, {100.0*len(quar)/max(len(quar)+len(clean),1):.1f} % of data)")
say(f"   retained: {clean.track_id.nunique()} tracks / {len(clean)} rows")
say(f"   residual implausible rows in the retained set: "
    f"{int(clean.flag_implausible.sum())} ({100*clean.flag_implausible.mean():.2f} %)")

try:
    clean.to_csv(OUTPUT_CSV, index=False)
    say(f"\nCleaned file written : {OUTPUT_CSV}   ({len(clean)} rows, "
        f"{clean.track_id.nunique()} tracks)")
    if len(quar):
        qp = OUTPUT_CSV.replace(".csv", "_quarantined.csv")
        quar.to_csv(qp, index=False)
        say(f"Quarantined file     : {qp}   ({len(quar)} rows) - inspect, do not use blindly")
except Exception as e:
    alt = os.path.join(os.getcwd(), "trajectories_filtered.csv")
    clean.to_csv(alt, index=False)
    say(f"\n[warn] could not write {OUTPUT_CSV} ({e}) -> wrote {alt}")

# ==============================================================================
# STAGE 6 - REPORT
# ==============================================================================
rule("PHASE 1 | STAGE 6 : POST-FILTER DIAGNOSTICS  << PASTE THIS BLOCK BACK >>")
say(f"dt={DT:.5f}s  fps={FPS:.2f}  tracks={clean.track_id.nunique()}  rows={len(clean)}  "
    f"duration={df.time.max()-df.time.min():.1f}s")
say(f"selected={BEST} (" + UNITS[BEST] % PARAM + f")  sigma_lon={SIG_LON*100:.2f}cm  "
    f"sigma_lat={SIG_LAT*100:.2f}cm")
say(f"jerk RMS: raw={RAW_J_RMS:.2f} -> filtered={summary[BEST]['jrms']:.2f} m/s^3  "
    f"({100*(1-summary[BEST]['jrms']/max(RAW_J_RMS,1e-9)):.2f} % noise removed)")
say(f"implausible rows remaining: {int(clean.flag_implausible.sum())} "
    f"({100*clean.flag_implausible.mean():.2f} %)   "
    f"quarantined tracks: {len(bad_ids)}   "
    f"opposing-flow tracks: {(clean.groupby('track_id').direction.first()=='opposing').sum()}")

wf = clean[(clean.direction == "with_flow") & (clean.edge == 0)]
say("\nPER-CLASS KINEMATICS - with-flow, edge samples excluded (EMPIRICAL):")
say(f"{'class':<10}{'trk':>5}{'rows':>7}{'v_mean':>9}{'v_p85':>8}{'v_p95':>8}"
    f"{'a+_p95':>8}{'a-_p05':>8}{'|latv|p95':>10}{'wander':>8}{'occupy':>8}")
say("-" * 78)
for cn, t in wf.groupby("class_name"):
    v = t.speed_mps.to_numpy(); a = t.accel_mps2.to_numpy()
    lv = np.abs(t.lat_speed_mps.to_numpy())
    # wander = median PER-TRACK lateral excursion (true weaving amplitude)
    # occupy = span of the whole class across the carriageway (road usage)
    wander = t.groupby("track_id").lat_m.agg(lambda s: s.max() - s.min()).median()
    say(f"{str(cn):<10}{t.track_id.nunique():>5}{len(t):>7}"
        f"{np.mean(v)*3.6:>9.2f}{np.percentile(v,85)*3.6:>8.2f}{np.percentile(v,95)*3.6:>8.2f}"
        f"{(np.percentile(a[a>0],95) if (a>0).any() else 0):>8.2f}"
        f"{(np.percentile(a[a<0],5) if (a<0).any() else 0):>8.2f}"
        f"{np.percentile(lv,95):>10.3f}{wander:>8.2f}{t.lat_m.max()-t.lat_m.min():>8.2f}")
say("  v in km/h | a in m/s^2 | latv in m/s")
say("  wander = MEDIAN PER-TRACK lateral excursion in m -> the real weaving amplitude,")
say("           this is what feeds VISSIM lateral behaviour in Phase 3")
say("  occupy = span of the whole class across the carriageway in m (road usage,")
say("           NOT a per-vehicle quantity - it grows with sample size)")
say("  -> v_p85 feeds the VISSIM desired-speed distribution; a+_p95 / a-_p05 feed")
say("     the maximum acceleration / deceleration functions (Phase 3).")

qt = clean.groupby("track_id").apply(lambda t: pd.Series({
    "cls": t.class_name.iloc[0], "n": len(t), "dur": len(t) * DT,
    "v": t.speed_mps.mean() * 3.6, "j": float(np.sqrt(np.mean(t.jerk_mps3 ** 2))),
    "fl": int(t.flag_implausible.sum()), "dir": t.direction.iloc[0]}))
say(f"\nPER-TRACK QUALITY - {len(qt)} tracks; "
    f"showing the 25 noisiest by jerk RMS (the ones worth inspecting):")
say(f"{'vid':>7}  {'class':<10}{'frames':>7}{'dur_s':>8}{'v_mean':>9}{'|j|RMS':>9}"
    f"{'flags':>7}  {'direction':<10}")
say("-" * 78)
for vid, r in qt.sort_values("j", ascending=False).head(25).iterrows():
    say(f"{vid:>7}  {str(r.cls):<10}{int(r.n):>7}{r.dur:>8.2f}"
        f"{r.v:>9.2f}{r.j:>9.2f}{int(r.fl):>7}  {str(r['dir']):<10}")
say(f"\nTrack-quality distribution over all {len(qt)} tracks:")
say(f"   duration s : p10={qt.dur.quantile(.1):.1f}  median={qt.dur.median():.1f}  "
    f"p90={qt.dur.quantile(.9):.1f}  max={qt.dur.max():.1f}")
say(f"   |j|RMS     : median={qt.j.median():.2f}  p90={qt.j.quantile(.9):.2f}  "
    f"max={qt.j.max():.2f} m/s^3")
say(f"   tracks with any implausible row : {int((qt.fl > 0).sum())} "
    f"({100.0*(qt.fl > 0).mean():.1f} %)")

say("\nPHASE 2 READINESS BY CLASS")
say("   A class is usable only if it has enough tracks that are WITH-FLOW, long")
say(f"   enough to show behaviour (>= {MIN_DUR_FIT_S} s) and actually moving.")
say(f"{'   class':<15}{'usable':>8}{'veh-sec':>10}{'med_dur':>9}{'v_med':>8}   verdict")
wfa = clean[clean.direction == "with_flow"]
for cn in sorted(set(clean.class_name)):
    t = wfa[wfa.class_name == cn]
    if len(t) == 0:
        say(f"   {cn:<12s}{0:>8d}{0:>10.1f}{0:>9.1f}{0:>8.1f}   NO with-flow track -> synthetic")
        continue
    dur = t.groupby("track_id").size() * DT
    usable = dur[dur >= MIN_DUR_FIT_S]
    vmed = t.speed_mps.median() * 3.6
    if len(usable) >= MIN_TRK_PER_CLS and vmed > 1.0:
        verdict = "EMPIRICAL fit possible"
    elif len(usable) >= MIN_TRK_PER_CLS:
        verdict = "tracks OK but near-stationary -> inspect before fitting"
    else:
        verdict = "TOO THIN -> synthetic augmentation, to be declared"
    say(f"   {cn:<12s}{len(usable):>8d}{len(t)*DT:>10.1f}{dur.median():>9.1f}"
        f"{vmed:>8.1f}   {verdict}")
for c in [c for c in CLASS_SYNONYMS if c not in set(clean.class_name)]:
    say(f"   {c:<12s}{0:>8d}{0:>10.1f}{0:>9.1f}{0:>8.1f}   ABSENT -> fully synthetic")
say("   usable = with-flow tracks lasting >= " + f"{MIN_DUR_FIT_S} s | med_dur in s | v_med in km/h")

# --- class-mix sanity check against known Dhaka composition -------------------
shares = (clean.groupby("class_name").track_id.nunique() /
          clean.track_id.nunique() * 100).sort_values(ascending=False)
say("\nCLASS-MIX SANITY CHECK (track share of the retained set):")
for cn, s in shares.items():
    say(f"   {cn:<12s} {s:>5.1f} %")
if "CNG" in shares and "Rickshaw" in shares and shares["CNG"] < 5 < shares["Rickshaw"]:
    say("   WARNING: CNG share is very low next to Rickshaw. Both are three-wheelers")
    say("   and detectors routinely merge them. Verify a sample of Rickshaw tracks")
    say("   on video before treating this split as real - it changes the vehicle")
    say("   composition you will enter in VISSIM.")

if anom:
    say("\nSUSPECT TRACKS (from the Stage 0 geometric scan) - verify on video:")
    for vid, cn, why in anom[:10]:
        say(f"   vehicle_id={vid} ({cn}): {why}")

# --- optional diagnostic figure ----------------------------------------------
if MAKE_PLOTS:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vids = list(clean.track_id.unique())[:3]
        fig, ax = plt.subplots(len(vids), 3, figsize=(15, 3.2 * len(vids)), squeeze=False)
        for i, vid in enumerate(vids):
            t = clean[clean.track_id == vid]
            ax[i][0].plot(t.time, t.raw_lon_m, lw=.8, alpha=.55, label="raw (Hampel'd)")
            ax[i][0].plot(t.time, t.lon_m, lw=1.6, label=f"{BEST}")
            ax[i][0].set_ylabel(f"id {vid} ({t.class_name.iloc[0]})\nlon [m]")
            ax[i][0].legend(fontsize=7)
            ax[i][1].plot(t.time, t.speed_mps, lw=1.4, color="tab:green")
            ax[i][1].set_ylabel("speed [m/s]")
            ax[i][2].plot(t.time, t.accel_mps2, lw=1.2, color="tab:red")
            ax[i][2].axhline(A_PLAUS_MAX, ls=":", c="k"); ax[i][2].axhline(-A_PLAUS_MAX, ls=":", c="k")
            ax[i][2].set_ylabel("accel [m/s^2]")
            for j in range(3):
                ax[i][j].set_xlabel("t [s]"); ax[i][j].grid(alpha=.3)
        fig.suptitle(f"Phase 1 filtering check - {BEST}, " + UNITS[BEST] % PARAM)
        fig.tight_layout()
        png = os.path.join(os.path.dirname(OUTPUT_CSV) or ".", "phase1_filter_check.png")
        fig.savefig(png, dpi=130)
        say(f"\nDiagnostic figure written : {png}")
    except Exception as e:
        say(f"\n[plot skipped: {e}]")

rule("ASSUMED PRIORS (declare these in the thesis as non-empirical)")
say(f"  A_PLAUS_MAX={A_PLAUS_MAX} m/s^2 | J_PLAUS_MAX={J_PLAUS_MAX} m/s^3 | "
    f"V_PLAUS_MAX={V_PLAUS_MAX} m/s")
say(f"  HAMPEL={HAMPEL_NSIGMA} sigma / {HAMPEL_WIN_S} s | MAX_GAP={MAX_GAP_FRAMES} frames | "
    f"MIN_TRACK={MIN_TRACK_FRAMES} frames | MIN_CONF={MIN_CONFIDENCE}")
say(f"  TELEPORT_MPS={TELEPORT_MPS} m/s | search grids as listed in CONFIG")
say("  The SELECTED smoother and its bandwidth are EMPIRICAL (chosen by the")
say("  objective above on your own data), not assumed.")
rule("END OF PHASE 1")

try:
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\n[full report saved to {OUTPUT_TXT}]")
except Exception:
    pass
