#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 1D - FILTER TUNING AGAINST KNOWN GROUND TRUTH
================================================================================
 In Phase 1 the smoother bandwidth was chosen by an INDIRECT rule ("minimum
 sufficient smoothing"): the weakest setting whose output is physically
 plausible. That rule was necessary because, with real data, the true speed and
 acceleration are unknown - there is nothing to compare against.

 The synthetic dataset removes that limitation. We know the true kinematics, so
 we can sweep every candidate filter and measure DIRECTLY which one recovers
 the truth best. The winning setting is then carried back to the real data.

 This is the payoff of generating synthetic data with ground truth, and it is a
 defensible thesis contribution in its own right: a vision-trajectory filter
 calibrated against known truth rather than against a plausibility heuristic.

 INPUTS   synthetic_noisy.csv  +  synthetic_truth.csv   (from synth_generator)
 OUTPUT   a table of accuracy vs filter setting, and a recommended setting
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except ImportError:
    sys.exit("ERROR: scipy required.  pip install scipy")

# ==============================================================================
NOISY_CSV = r"S:\soscho\synthetic_noisy.csv"
TRUTH_CSV = r"S:\soscho\synthetic_truth.csv"
OUT_TXT   = r"S:\soscho\phase1d_filter_tuning.txt"

DT = 0.03332
HAMPEL_WIN_S = 0.33
HAMPEL_NSIGMA = 3.0
MAX_GAP_FRAMES = 5
MIN_TRACK_FRAMES = 30
JUMP_SPLIT_M = 1.50
EDGE_S = 0.20

# candidate settings to sweep
GRID_KF   = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]  # sigma_jerk
GRID_SG   = [1.00, 1.40, 1.80, 2.40, 3.20]                # window, s
GRID_SEMA = [0.35, 0.50, 0.70, 1.00]                      # delta, s
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


# ---------------- numerical helpers (identical to Phase 1) --------------------
def mad_sigma(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.nan if x.size < 3 else 1.4826 * np.median(np.abs(x - np.median(x)))


def estimate_meas_noise(x):
    s = mad_sigma(np.diff(np.asarray(x, float), n=2))
    return np.nan if not np.isfinite(s) else s / np.sqrt(6.0)


def hampel(x, win, nsig):
    s = pd.Series(np.asarray(x, float)); w = max(3, int(win) | 1)
    med = s.rolling(w, center=True, min_periods=3).median()
    dev = (s - med).abs()
    mad = (dev.rolling(w, center=True, min_periods=3).median() * 1.4826).replace(0.0, np.nan)
    flag = (dev > nsig * mad).fillna(False).to_numpy()
    out = s.to_numpy().copy(); out[flag] = med.to_numpy()[flag]
    return out


def sema(x, dt, delta):
    x = np.asarray(x, float); n = x.size
    if n < 3:
        return x.copy()
    idx = np.arange(n, dtype=float)
    b, a0 = np.polyfit(idx, x, 1)
    r = x - (b * idx + a0)
    D = max(1.0, delta / dt); half = min(int(3 * D), max(1, n - 1))
    w = np.exp(-np.abs(np.arange(-half, half + 1)) / D)
    return (np.convolve(r, w, "same") / np.convolve(np.ones_like(r), w, "same"))[:n] \
        + (b * idx + a0)


def kf_rts(z, dt, sm, sj):
    z = np.asarray(z, float); n = z.size
    F = np.array([[1, dt, .5 * dt * dt], [0, 1, dt], [0, 0, 1]], float)
    Q = sj ** 2 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                            [dt**4/8, dt**3/3, dt**2/2],
                            [dt**3/6, dt**2/2, dt]], float)
    R = max(sm, 1e-4) ** 2
    xs = np.zeros((n, 3)); Ps = np.zeros((n, 3, 3))
    xp = np.zeros((n, 3)); Pp = np.zeros((n, 3, 3))
    x = np.array([z[0], 0., 0.]); P = np.diag([R, 10., 10.])
    for k in range(n):
        if k > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        S = P[0, 0] + R; K = P[:, 0] / S
        x = x + K * (z[k] - x[0]); P = P - np.outer(K, P[0, :])
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


def smooth(method, z, dt, param, sm):
    n = z.size
    if method == "KF-RTS":
        return kf_rts(z, dt, sm, param)
    if method == "sEMA":
        p = sema(z, dt, param); v = d1(p, dt); return p, v, d1(v, dt)
    w = max(5, int(round(param / dt)) | 1)
    if w >= n:
        w = max(5, (n - 1) | 1) - 2
    if w < 5:
        p = z.copy(); v = d1(p, dt); return p, v, d1(v, dt)
    return (savgol_filter(z, w, 3),
            savgol_filter(z, w, 3, deriv=1, delta=dt),
            savgol_filter(z, w, 3, deriv=2, delta=dt))


# ==============================================================================
rule("PHASE 1D | FILTER TUNING AGAINST GROUND TRUTH")
for f in (NOISY_CSV, TRUTH_CSV):
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} not found. Run synth_generator.py MODE='corpus' first.")

noisy = pd.read_csv(NOISY_CSV)
truth = pd.read_csv(TRUTH_CSV)
say(f"Noisy : {len(noisy)} rows, {noisy.vehicle_id.nunique()} tracks")
say(f"Truth : {len(truth)} rows")

# flow direction normalisation, same convention as Phase 1
net = noisy.groupby("vehicle_id").apply(lambda t: t.Y_m.iloc[-1] - t.Y_m.iloc[0])
SIGN = -1.0 if np.median(net) < 0 else 1.0
noisy["_lon"] = noisy.Y_m * SIGN
noisy["_lat"] = noisy.X_m

HW = max(3, int(round(HAMPEL_WIN_S / DT)) | 1)
edge_n = max(1, int(round(EDGE_S / DT)))

# ---- preprocess once: Hampel + uniform grid + split at jumps ------------------
tracks = []
for vid, t in noisy.groupby("vehicle_id"):
    t = t.drop_duplicates("frame").sort_values("frame")
    full = pd.DataFrame({"frame": np.arange(int(t.frame.min()), int(t.frame.max()) + 1)})
    t = full.merge(t, on="frame", how="left")
    miss = t["_lon"].isna()
    runlen = miss.astype(int).groupby((~miss).cumsum()).transform("sum")
    t = t[~(miss & (runlen > MAX_GAP_FRAMES))]
    if len(t) < MIN_TRACK_FRAMES:
        continue
    for c in ["_lon", "_lat"]:
        t[c] = pd.to_numeric(t[c], errors="coerce").interpolate(limit_direction="both")
    t["class_name"] = t["class_name"].ffill().bfill()
    t["vehicle_id"] = vid
    jump = np.hypot(t["_lon"].diff(), t["_lat"].diff()).to_numpy()
    cut = [int(c) for c in np.where(jump > JUMP_SPLIT_M)[0]]
    bounds = [0] + cut + [len(t)]
    for b in range(len(bounds) - 1):
        seg = t.iloc[bounds[b]:bounds[b + 1]]
        if len(seg) < MIN_TRACK_FRAMES:
            continue
        seg = seg.reset_index(drop=True)
        seg["_lon_h"] = hampel(seg["_lon"].to_numpy(), HW, HAMPEL_NSIGMA)
        tracks.append(seg)
say(f"Usable segments after preprocessing: {len(tracks)}")

# per-class measurement noise, as Phase 1 estimates it
sig = {}
for cn in noisy.class_name.unique():
    vals = [estimate_meas_noise(t["_lon_h"].to_numpy())
            for t in tracks if t.class_name.iloc[0] == cn and len(t) >= 20]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        sig[cn] = float(np.median(vals))
SIG_GLOBAL = float(np.median(list(sig.values()))) if sig else 0.03

truth_idx = truth.set_index(["vehicle_id", "frame"])[["v_true", "a_true"]]


def score(method, param):
    """Filter every track at this setting and compare against known truth."""
    V, A, VT, AT = [], [], [], []
    for t in tracks:
        z = t["_lon_h"].to_numpy()
        sm = sig.get(t.class_name.iloc[0], SIG_GLOBAL)
        _, v, a = smooth(method, z, DT, param, sm)
        keep = np.ones(len(t), bool); keep[:edge_n] = False; keep[-edge_n:] = False
        key = pd.MultiIndex.from_arrays([t.vehicle_id.to_numpy()[keep],
                                         t.frame.to_numpy()[keep]])
        tr = truth_idx.reindex(key)
        ok = tr.v_true.notna().to_numpy()
        if ok.sum() < 5:
            continue
        V.append(v[keep][ok]); A.append(a[keep][ok])
        VT.append(tr.v_true.to_numpy()[ok]); AT.append(tr.a_true.to_numpy()[ok])
    if not V:
        return None
    v = np.concatenate(V); a = np.concatenate(A)
    vt = np.concatenate(VT) * (1.0)          # truth speed is positive downstream
    at = np.concatenate(AT)
    return dict(n=len(v),
                v_rmse=float(np.sqrt(np.mean((v - vt) ** 2))),
                a_rmse=float(np.sqrt(np.mean((a - at) ** 2))),
                a_r=float(np.corrcoef(a, at)[0, 1]),
                v_r=float(np.corrcoef(v, vt)[0, 1]))


results = []
for method, grid in (("KF-RTS", GRID_KF), ("SavGol", GRID_SG), ("sEMA", GRID_SEMA)):
    rule()
    say(f"{method}")
    say(f"{'param':>10}{'n':>10}{'speed RMSE':>13}{'speed r':>10}"
        f"{'accel RMSE':>13}{'accel r':>10}")
    for p in grid:
        r = score(method, p)
        if r is None:
            continue
        results.append((method, p, r))
        say(f"{p:>10.3f}{r['n']:>10}{r['v_rmse']:>13.3f}{r['v_r']:>10.3f}"
            f"{r['a_rmse']:>13.3f}{r['a_r']:>10.3f}")

rule("RECOMMENDATION")
best_a = max(results, key=lambda x: x[2]["a_r"])
best_v = min(results, key=lambda x: x[2]["v_rmse"])
say(f"Best ACCELERATION recovery : {best_a[0]} @ {best_a[1]}  "
    f"(accel r = {best_a[2]['a_r']:.3f}, accel RMSE = {best_a[2]['a_rmse']:.3f} m/s^2)")
say(f"Best SPEED recovery        : {best_v[0]} @ {best_v[1]}  "
    f"(speed RMSE = {best_v[2]['v_rmse']:.3f} m/s)")

say("")
if best_a[0] == best_v[0] and best_a[1] == best_v[1]:
    say("The same setting wins on both. Use it for the real data.")
else:
    say("Speed and acceleration prefer DIFFERENT settings - this is expected.")
    say("Acceleration needs more smoothing than speed does, because it is a")
    say("second derivative and inherits twice the noise amplification.")
    say("")
    say("SUPERVISOR DECISION: run the real data TWICE and keep both outputs.")
    say(f"   - speed/position parameters (desired speed, CC0, CC1, lateral")
    say(f"     clearances) from the {best_v[0]} @ {best_v[1]} output")
    say(f"   - acceleration parameters (CC8, CC9, desired-acceleration curves)")
    say(f"     from the {best_a[0]} @ {best_a[1]} output")
    say("   Using one filter for both means one of the two parameter families")
    say("   is extracted from a sub-optimal reconstruction.")

HEURISTIC_PICK = ("KF-RTS", 0.25)      # what phase1b chose on the REAL data
hp = [r for r in results if r[0] == HEURISTIC_PICK[0] and abs(r[1] - HEURISTIC_PICK[1]) < 1e-9]
say("")
say("WHAT DID TUNING BUY?")
if hp:
    h = hp[0][2]; b = best_a[2]
    say(f"   plausibility heuristic ({HEURISTIC_PICK[0]} @ {HEURISTIC_PICK[1]}): "
        f"accel r = {h['a_r']:.3f}, accel RMSE = {h['a_rmse']:.3f}")
    say(f"   ground-truth tuned     ({best_a[0]} @ {best_a[1]}): "
        f"accel r = {b['a_r']:.3f}, accel RMSE = {b['a_rmse']:.3f}")
    say(f"   improvement            : accel r {b['a_r']-h['a_r']:+.3f}, "
        f"RMSE {100*(h['a_rmse']-b['a_rmse'])/h['a_rmse']:+.1f} %")
say("")
say("CAVEAT TO STATE IN THE THESIS: the optimal bandwidth depends on how much")
say("jerk the TRUE motion contains. It was tuned on synthetic truth whose jerk")
say("RMS is ~1.6 m/s^3, which matches the urban stop-and-go range reported in")
say("the literature. If real Dhaka driving is jerkier than that, the true")
say("optimum sits slightly higher than the value recommended here.")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT))
print(f"\n[saved to {OUT_TXT}]")
