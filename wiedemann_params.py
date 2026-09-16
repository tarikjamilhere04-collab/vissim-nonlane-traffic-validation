"""
wiedemann_params.py  (v5)
-------------------------
Estimate Wiedemann 99 car-following parameters from trajectory CSV.

v5 changes:
  - Tighter following regime for regression (spacing 2-20 m, |dX|<1.0)
  - Plausibility band for every W99 parameter
  - Data value used only if inside band; else W99 default
  - 'source' column tells the reader which parameters are real
  - Full fallback list written to w99_notes.txt

Reads : S:\soscho\trajectories.csv
Writes: S:\Second\w99_overall.csv
        S:\Second\w99_by_class.csv
        S:\Second\w99_notes.txt
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ----------------- SETTINGS -----------------
IN_CSV   = r"S:\soscho\trajectories.csv"
OUT_ALL  = r"S:\Second\w99_overall.csv"
OUT_CLS  = r"S:\Second\w99_by_class.csv"
OUT_NOTE = r"S:\Second\w99_notes.txt"

FPS      = 30.0
SG_WIN   = 15
SG_ORDER = 3

LANE_TOL = 1.50
MAX_GAP  = 60.0
MIN_GAP  = 0.30

A_MAX    = 5.0
V_MIN    = 0.3
V_MAX    = 40.0

MIN_FOLLOW_S = 1.0
MIN_SAMPLES  = 100

# Tighter regime used ONLY for the CC0/CC1 regression
REG_SPACING_LO = 2.0
REG_SPACING_HI = 20.0
REG_DX_MAX     = 1.0
REG_DV_MAX     = 1.5
REG_V_MIN      = 1.5

COL_ID  = "vehicle_id"
COL_CLS = "class_name"
COL_T   = "time"
COL_X   = "X_m"
COL_Y   = "Y_m"
# --------------------------------------------

# --- W99 plausibility bands: (lower, upper, default-if-outside) ---
W99_BANDS = {
    "CC0": (1.0,   3.0,   1.50),
    "CC1": (0.6,   1.8,   0.90),
    "CC2": (2.0,   6.0,   4.00),
    "CC3": (-12.0, -3.0, -8.00),
    "CC4": (-1.5, -0.10, -0.35),
    "CC5": ( 0.10, 1.5,   0.35),
    "CC6": ( 5.0, 15.0,  11.44),
    "CC7": ( 0.10, 0.50,  0.25),
    "CC8": ( 1.5,  4.0,   3.50),
    "CC9": ( 0.5,  3.0,   1.50),
}


def clip_or_default(value, bounds):
    """If value inside [lo,hi] return (value,'data'). Else (default,'default')."""
    lo, hi, default = bounds
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default, "default"
    if lo <= value <= hi:
        return float(value), "data"
    return float(default), "default"


# -------------- safe file writer --------------
def safe_to_csv(df, path):
    try:
        df.to_csv(path, index=False)
        print("Saved:", path)
        return path
    except PermissionError:
        alt = path.replace(".csv", "_new.csv")
        df.to_csv(alt, index=False)
        print(f"WARNING: {path} locked. Saved instead to: {alt}")
        return alt


# ---------------- smoothing -------------------
def smooth_and_diff(df):
    out = []
    for vid, sub in df.groupby(COL_ID):
        sub = sub.sort_values(COL_T).reset_index(drop=True)
        n = len(sub)
        if n < SG_WIN:
            continue
        t = sub[COL_T].to_numpy(dtype=float)
        x = sub[COL_X].to_numpy(dtype=float)
        y = sub[COL_Y].to_numpy(dtype=float)

        if not np.all(np.diff(t) > 0):
            keep = np.concatenate(([True], np.diff(t) > 0))
            t, x, y = t[keep], x[keep], y[keep]
            sub = sub.loc[keep].reset_index(drop=True)
            n = len(sub)
            if n < SG_WIN:
                continue

        dt_mean = float(np.mean(np.diff(t)))
        if dt_mean <= 0:
            continue

        xs = savgol_filter(x, SG_WIN, SG_ORDER, deriv=0)
        ys = savgol_filter(y, SG_WIN, SG_ORDER, deriv=0)
        vx = savgol_filter(x, SG_WIN, SG_ORDER, deriv=1) / dt_mean
        vy = savgol_filter(y, SG_WIN, SG_ORDER, deriv=1) / dt_mean
        ax = savgol_filter(x, SG_WIN, SG_ORDER, deriv=2) / (dt_mean**2)
        ay = savgol_filter(y, SG_WIN, SG_ORDER, deriv=2) / (dt_mean**2)

        v = np.sqrt(vx**2 + vy**2)
        bad = (v < V_MIN) | (v > V_MAX) | (np.abs(ax) > A_MAX) | (np.abs(ay) > A_MAX)
        keep = ~bad
        if keep.sum() < SG_WIN:
            continue

        tmp = sub.loc[keep].copy()
        tmp["Xs"] = xs[keep]
        tmp["Ys"] = ys[keep]
        tmp["vx"] = vx[keep]
        tmp["vy"] = vy[keep]
        tmp["ax"] = ax[keep]
        tmp["ay"] = ay[keep]
        tmp["v"]  = v[keep]
        out.append(tmp)

    if not out:
        return pd.DataFrame(columns=list(df.columns) +
                            ["Xs", "Ys", "vx", "vy", "ax", "ay", "v"])
    return pd.concat(out, ignore_index=True)


# -------------- sustained pairs ---------------
def build_sustained_pairs(df):
    pairs = []
    for t, fr in df.groupby(COL_T):
        if len(fr) < 2:
            continue
        ids = fr[COL_ID].to_numpy()
        X   = fr["Xs"].to_numpy()
        Y   = fr["Ys"].to_numpy()
        v   = fr["v"].to_numpy()
        vy  = fr["vy"].to_numpy()
        ay  = fr["ay"].to_numpy()
        cls = fr[COL_CLS].to_numpy()

        for i in range(len(fr)):
            dY = Y - Y[i]
            dX = X - X[i]
            m = (dY > MIN_GAP) & (dY <= MAX_GAP) & (np.abs(dX) < LANE_TOL)
            m[i] = False
            if not m.any():
                continue
            cand = np.where(m)[0]
            k = cand[np.argmin(dY[cand])]
            pairs.append({
                "t": t, "follower": ids[i], "leader": ids[k],
                "class_f": cls[i], "class_l": cls[k],
                "spacing": dY[k], "dX": dX[k],
                "v_f": v[i], "v_l": v[k],
                "vy_f": vy[i], "vy_l": vy[k],
                "ay_f": ay[i],
                "dV": vy[i] - vy[k],
            })

    pairs = pd.DataFrame(pairs)
    if len(pairs) == 0:
        return pairs
    key = pairs.groupby(["follower", "leader"]).size()
    ok = key[key >= int(MIN_FOLLOW_S * FPS)].index
    if len(ok) == 0:
        return pd.DataFrame(columns=pairs.columns)
    return pairs.set_index(["follower", "leader"]).loc[ok].reset_index()


def following_regime(pairs):
    p = pairs.copy()
    p = p[(p["v_f"] > 0.55) & (p["spacing"] > 1.0) & (p["spacing"] < 40.0)]
    p = p[np.abs(p["dV"]) < 3.0]
    return p


def regression_regime(pairs):
    """Much stricter set of pairs used only for CC0/CC1 regression."""
    p = pairs.copy()
    p = p[(p["v_f"] > REG_V_MIN) &
          (p["spacing"] >= REG_SPACING_LO) &
          (p["spacing"] <= REG_SPACING_HI) &
          (np.abs(p["dX"]) < REG_DX_MAX) &
          (np.abs(p["dV"]) < REG_DV_MAX)]
    return p


# ---------------- estimator ------------------
def estimate_w99(df, pairs, label="", notes=None):
    """
    Return a dict with CC0..CC9 plus CC0_src..CC9_src
    ('data' = extracted within band, 'default' = W99 fallback).
    """
    out = {"subset": label, "n_pairs": int(len(pairs))}

    if label != "ALL":
        df_cls = df[df[COL_CLS] == label]
    else:
        df_cls = df

    follow = following_regime(pairs)
    reg    = regression_regime(pairs)

    # ---------------- CC0 + CC1 by regression ----------------
    cc0_raw, cc1_raw = np.nan, np.nan
    if len(reg) >= 200:
        x = reg["v_f"].to_numpy(dtype=float)
        y = reg["spacing"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        cc0_raw = float(intercept)
        cc1_raw = float(slope)
        if notes is not None:
            notes.append(
                f"{label}: CC0/CC1 regression (n={len(reg)}) "
                f"raw intercept={cc0_raw:.3f}, slope={cc1_raw:.3f}"
            )
    else:
        if notes is not None:
            notes.append(
                f"{label}: CC0/CC1 regression skipped (n={len(reg)} < 200)"
            )

    # standstill fallback for CC0 if regression produced nothing
    if np.isnan(cc0_raw):
        stand = pairs[(pairs["v_f"] < 0.6) & (pairs["spacing"] < 12.0)]
        if len(stand) >= 30:
            cc0_raw = float(np.percentile(stand["spacing"], 25))
            if notes is not None:
                notes.append(f"{label}: CC0 from standstill p25 (n={len(stand)})")

    # headway fallback for CC1 if regression produced nothing
    if np.isnan(cc1_raw) and len(follow) >= 100:
        hw = (follow["spacing"] - 1.5) / follow["v_f"].clip(lower=1.0)
        hw = hw[(hw > 0.2) & (hw < 4.0)]
        if len(hw) >= 100:
            cc1_raw = float(np.median(hw))
            if notes is not None:
                notes.append(f"{label}: CC1 from headway median (n={len(hw)})")

    cc0, cc0_src = clip_or_default(cc0_raw, W99_BANDS["CC0"])
    cc1, cc1_src = clip_or_default(cc1_raw, W99_BANDS["CC1"])
    if cc0_src == "default" and notes is not None and not np.isnan(cc0_raw):
        notes.append(f"{label}: CC0 raw={cc0_raw:.3f} out of band -> default 1.50")
    if cc1_src == "default" and notes is not None and not np.isnan(cc1_raw):
        notes.append(f"{label}: CC1 raw={cc1_raw:.3f} out of band -> default 0.90")

    out["CC0"] = cc0
    out["CC1"] = cc1

    # ---------------- CC2 ----------------
    if len(follow) >= MIN_SAMPLES:
        resid = follow["spacing"] - (cc0 + cc1 * follow["v_f"])
        cc2_raw = float(resid.std(ddof=1))
    else:
        cc2_raw = np.nan
    cc2, cc2_src = clip_or_default(cc2_raw, W99_BANDS["CC2"])
    out["CC2"] = cc2

    # ---------------- CC3 ----------------
    closing = pairs[(pairs["dV"] < -0.5) & (pairs["ay_f"] < 0)
                    & (pairs["spacing"] < 30)]
    if len(closing) >= MIN_SAMPLES:
        # use the 10th percentile (more negative) to capture braking intent
        cc3_raw = float(np.percentile(closing["ay_f"], 10))
    else:
        cc3_raw = np.nan
    cc3, cc3_src = clip_or_default(cc3_raw, W99_BANDS["CC3"])
    out["CC3"] = cc3

    # ---------------- CC4 / CC5 ----------------
    if len(follow) >= MIN_SAMPLES:
        neg = follow["dV"][follow["dV"] < 0]
        pos = follow["dV"][follow["dV"] > 0]
        cc4_raw = float(np.median(neg)) if len(neg) >= MIN_SAMPLES else np.nan
        cc5_raw = float(np.median(pos)) if len(pos) >= MIN_SAMPLES else np.nan
    else:
        cc4_raw = cc5_raw = np.nan
    cc4, cc4_src = clip_or_default(cc4_raw, W99_BANDS["CC4"])
    cc5, cc5_src = clip_or_default(cc5_raw, W99_BANDS["CC5"])
    out["CC4"] = cc4
    out["CC5"] = cc5

    # ---------------- CC6 ----------------
    cc6_raw = np.nan
    if len(follow) >= 3 * MIN_SAMPLES:
        bins = pd.cut(follow["v_f"], bins=6)
        grp  = follow.groupby(bins, observed=True)["spacing"].std(ddof=1)
        vs   = follow.groupby(bins, observed=True)["v_f"].mean()
        ok = grp.notna() & vs.notna()
        if ok.sum() >= 3:
            slope, _ = np.polyfit(vs[ok], grp[ok], 1)
            cc6_raw = float(slope)
    cc6, cc6_src = clip_or_default(cc6_raw, W99_BANDS["CC6"])
    out["CC6"] = cc6

    # ---------------- CC7 ----------------
    if len(follow) >= MIN_SAMPLES:
        closing2 = follow[np.abs(follow["ay_f"]) > 0.05]
        if len(closing2) >= MIN_SAMPLES:
            a = np.abs(closing2["ay_f"])
            q1, q3 = np.percentile(a, [25, 75])
            mid = a[(a >= q1) & (a <= q3)]
            cc7_raw = float(np.mean(mid)) if len(mid) else np.nan
        else:
            cc7_raw = np.nan
    else:
        cc7_raw = np.nan
    cc7, cc7_src = clip_or_default(cc7_raw, W99_BANDS["CC7"])
    out["CC7"] = cc7

    # ---------------- CC8 ----------------
    start = df_cls[(df_cls["v"] < 1.4) & (df_cls["ay"] > 0.1)]
    if len(start) >= MIN_SAMPLES:
        cc8_raw = float(np.median(start["ay"]))
    else:
        cc8_raw = np.nan
    cc8, cc8_src = clip_or_default(cc8_raw, W99_BANDS["CC8"])
    out["CC8"] = cc8

    # ---------------- CC9 ----------------
    fast = df_cls[(df_cls["v"] > 22.0) & (df_cls["ay"] > 0.1)]
    if len(fast) >= MIN_SAMPLES:
        cc9_raw = float(np.median(fast["ay"]))
    else:
        cc9_raw = np.nan
    cc9, cc9_src = clip_or_default(cc9_raw, W99_BANDS["CC9"])
    out["CC9"] = cc9

    # record sources
    out["CC0_src"] = cc0_src
    out["CC1_src"] = cc1_src
    out["CC2_src"] = cc2_src
    out["CC3_src"] = cc3_src
    out["CC4_src"] = cc4_src
    out["CC5_src"] = cc5_src
    out["CC6_src"] = cc6_src
    out["CC7_src"] = cc7_src
    out["CC8_src"] = cc8_src
    out["CC9_src"] = cc9_src

    return out


# ------------------- main --------------------
def main():
    notes = []
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV)
    df[COL_T] = df[COL_T].astype(float)

    print("Smoothing + differentiating (SG win=%d, order=%d)..."
          % (SG_WIN, SG_ORDER))
    df = smooth_and_diff(df)
    print("Rows after smoothing:", len(df),
          "| vehicles:", df[COL_ID].nunique())

    print("Building sustained leader/follower pairs ...")
    pairs = build_sustained_pairs(df)
    print("Sustained pair samples:", len(pairs))
    if len(pairs) == 0:
        print("No sustained pairs found.")
        return

    overall = estimate_w99(df, pairs, label="ALL", notes=notes)
    safe_to_csv(pd.DataFrame([overall]), OUT_ALL)
    print("\nOverall W99 (all classes):")
    for k, v in overall.items():
        print(f"  {k:<10} = {v}")

    rows = []
    for cls, sub in pairs.groupby("class_f"):
        if len(sub) < MIN_SAMPLES:
            continue
        rows.append(estimate_w99(df, sub, label=cls, notes=notes))
    per_cls = pd.DataFrame(rows)
    safe_to_csv(per_cls, OUT_CLS)

    # order columns: subset, n_pairs, CC0..CC9, sources
    cc_cols = [f"CC{i}" for i in range(10)]
    src_cols = [f"CC{i}_src" for i in range(10)]
    cols = ["subset", "n_pairs"] + cc_cols + src_cols
    cols = [c for c in cols if c in per_cls.columns]
    per_cls = per_cls[cols]

    print("\nW99 by follower class:")
    print(per_cls.to_string(index=False))

    try:
        with open(OUT_NOTE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(notes))
        print("\nNotes:", OUT_NOTE)
    except PermissionError:
        alt = OUT_NOTE.replace(".txt", "_new.txt")
        with open(alt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(notes))
        print(f"\nWARNING: {OUT_NOTE} locked. Saved instead to: {alt}")

    print("\nDone.")


if __name__ == "__main__":
    main()