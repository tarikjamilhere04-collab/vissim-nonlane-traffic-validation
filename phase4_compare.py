#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 4 - SIMULATED versus OBSERVED
================================================================================
 WHAT IT DOES
   Runs the SAME statistics on the VISSIM output that Phase 2, 2C and 3E ran on
   the real video, and puts the two side by side. That is the only way to answer
   the question the thesis has to answer: did the measured parameters, once
   entered into VISSIM, actually reproduce the behaviour they were measured from?

 WHAT IT COMPARES, AND WHY ONLY THESE
   speed of FREE vehicles        the desired-speed input, directly
   gap against speed             the car-following behaviour (CC0, CC1)
   weaving intensity             the lateral behaviour, split free / following
   two-regime ratio sd / MAD     whether the simulated weaving is a mixture too
   manoeuvre rate per veh-min    exposure-corrected, scale-free

   Raw average speed is NOT compared. The real road was congested by downstream
   friction; the model is congested by a traffic signal. Those are different
   mechanisms, so the raw averages have no reason to agree and agreement would
   mean nothing. Conditioning on free / following removes that difference and
   leaves the behaviour, which is what was calibrated.

 MISSING COLUMNS ARE HANDLED
   If the Vehicle Record was written without VehType, Speed or Acceleration,
   this script recovers them:
     - VehType from the .fhz file, joined on vehicle number
     - Speed and acceleration by differentiating position, which is exact here
       because simulated positions carry no measurement noise
   It says which ones it had to recover.

 INPUT   the VISSIM .fzp (and .fhz beside it), plus the real result files
 OUTPUT  phase4_comparison.txt  and  figures/fig20..fig22
================================================================================
"""

import io
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("ERROR: matplotlib required.  pip install matplotlib")
try:
    from scipy.stats import theilslopes
except ImportError:
    theilslopes = None

# ==============================================================================
# THE ONLY TWO PATHS YOU SHOULD NEED TO CHANGE
SIM_FZP  = r"S:\Vissim Tutorial\New Try\New One_048.fzp"
REAL_BASE = r"S:\soscho"
# ==============================================================================

FIG_DIR  = os.path.join(REAL_BASE, "figures")
OUT_TXT  = os.path.join(REAL_BASE, "phase4_comparison.txt")

# Which links in the model correspond to the road that was filmed.
# 1 and 7 are the two north-south approaches - same composition, same width.
SIM_LINKS   = [1, 7]
LANE_WIDTH  = 5.0      # m, as built in the model
WARMUP_S    = 60.0     # discard the start of the run (use 600 for a 3600 s run)
PAIR_HZ     = 1.0      # rebuild leader pairs at this rate - 1 Hz is plenty

# The filmed road had no signal, so "no leader ahead" meant free-flowing. The
# model HAS a signal, so a vehicle waiting at a red with an empty road in front
# of it would also be called free. Samples within this distance of the
# downstream end of an approach link are therefore excluded from the
# free-flow comparison.
SIGNAL_ZONE_M = 80.0

# A manoeuvre starts when lateral speed passes this, so the share of samples
# above it is the part of the weaving distribution VISSIM can actually produce.
TAIL_THRESH = 0.15

# --- these MUST match the real-data scripts ----------------------------------
FREE_GAP_M     = 15.0
MOVING_MPS     = 0.5
OVERLAP_FRAC   = 0.35
FOLLOW_DV_MPS  = 0.50
VY_ON          = 0.15   # as used by Phase 2C after its sensitivity sweep
VY_OFF         = 0.10
MIN_SHIFT_M    = 0.15
MIN_DUR_S      = 0.3
MAX_DUR_S      = 6.0
VY_CLIP        = 1.5
DIMS = {"Bike": (1.90, 0.70), "CNG": (2.60, 1.40), "Rickshaw": (2.00, 1.20),
        "Car": (4.40, 1.70), "Truck": (7.50, 2.40), "Bus": (11.0, 2.50)}

# VISSIM vehicle type number -> the class name used throughout this study
VTYPE = {100: "Car", 300: "Bus", 650: "CNG", 660: "Rickshaw",
         670: "Bike", 680: "Truck"}

CLASS_ORDER = ["Car", "Rickshaw", "Bike", "CNG", "Truck", "Bus"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK_1, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE, BAD = "#e6e5e1", "#ffffff", "#b3261e"
SIM_COLOUR = "#7a4fbf"          # one colour for "simulated", reserved
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


def q(a, p):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


def f(v, d=2):
    try:
        return f"{float(v):.{d}f}" if np.isfinite(float(v)) else "--"
    except (TypeError, ValueError):
        return "--"


def sd_mad(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return np.nan, np.nan
    return float(np.std(v)), 1.4826 * float(np.median(np.abs(v - np.median(v))))


def colour(cn):
    return PALETTE[CLASS_ORDER.index(cn) % len(PALETTE)] if cn in CLASS_ORDER else INK_3


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK_1, fontsize=11, loc="left", pad=8)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    say(f"   wrote {p}")


# ==============================================================================
# READING THE VISSIM OUTPUT
# ==============================================================================
def read_fzp(path):
    """Read a .fzp whatever columns it happens to contain."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")
    header, skip = None, 0
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith("$VEHICLE:"):
                header = line.strip().lstrip("$").split(":", 1)[1].split(";")
                skip = i + 1
                break
            if i > 200:
                break
    if header is None:
        sys.exit("ERROR: no '$VEHICLE:' header line found in the .fzp.")
    header = [h.strip().upper() for h in header if h.strip()]
    df = pd.read_csv(path, sep=";", skiprows=skip, header=None,
                     names=header, usecols=range(len(header)),
                     encoding="utf-8-sig", low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=[header[0]]), header


def read_fhz(fzp_path):
    """The 'vehicles entered' table - carries VehType per vehicle number."""
    cand = re.sub(r"\.fzp$", ".fhz", fzp_path, flags=re.I)
    if not os.path.exists(cand):
        return None
    raw = open(cand, encoding="utf-8-sig", errors="replace").read().splitlines()
    hdr = [i for i, l in enumerate(raw) if l.strip().startswith("Time;")]
    if not hdr:
        return None
    d = pd.read_csv(io.StringIO("\n".join(raw[hdr[0]:])), sep=";",
                    skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    keep = [c for c in ("VehNo", "VehType") if c in d.columns]
    if len(keep) < 2:
        return None
    d = d[keep].apply(pd.to_numeric, errors="coerce").dropna()
    return d.drop_duplicates(subset="VehNo")


rule("PHASE 4 | SIMULATED versus OBSERVED")
say(f"Simulation : {SIM_FZP}")
sim, cols = read_fzp(SIM_FZP)
say(f"             {len(sim):,} rows, columns present: {', '.join(cols)}")

C = {c: c for c in cols}
def col(*names):
    for n in names:
        for c in cols:
            if c.replace("\\", "").upper() == n.replace("\\", "").upper():
                return c
    return None

c_t   = col("SIMSEC")
c_no  = col("NO")
c_lnk = col("LANE\\LINK\\NO", "LINK")
c_lan = col("LANE\\INDEX", "LANE")
c_pos = col("POS")
c_lat = col("POSLAT")
for nm, c in (("SIMSEC", c_t), ("NO", c_no), ("LINK", c_lnk),
              ("LANE", c_lan), ("POS", c_pos), ("POSLAT", c_lat)):
    if c is None:
        sys.exit(f"ERROR: the .fzp has no {nm} column. Re-run with it enabled.")

sim = sim.rename(columns={c_t: "t", c_no: "no", c_lnk: "link",
                          c_lan: "lane", c_pos: "pos", c_lat: "poslat"})

recovered = []
c_typ = col("VEHTYPE")
if c_typ:
    sim = sim.rename(columns={c_typ: "vtype"})
else:
    fhz = read_fhz(SIM_FZP)
    if fhz is None:
        sys.exit("ERROR: the .fzp has no VEHTYPE and no .fhz was found beside "
                 "it.\nEnable VehType in the Vehicle Record, or keep the .fhz.")
    sim = sim.merge(fhz.rename(columns={"VehNo": "no", "VehType": "vtype"}),
                    on="no", how="left")
    recovered.append("VehType (from the .fhz file)")

sim["class_name"] = sim.vtype.map(VTYPE)
unknown = sim.vtype[sim.class_name.isna()].unique()
if len(unknown):
    say(f"   [warn] vehicle types not in the mapping, ignored: "
        f"{sorted(int(u) for u in unknown if np.isfinite(u))}")
sim = sim.dropna(subset=["class_name"])

# --- geometry ----------------------------------------------------------------
sim = sim.sort_values(["no", "t"]).reset_index(drop=True)
sim["lon_m"] = sim.pos
sim["lat_m"] = (sim.lane - 1) * LANE_WIDTH + sim.poslat * LANE_WIDTH

# --- speed / acceleration, computed only where the vehicle stayed on the link -
same = (sim.no == sim.no.shift()) & (sim.link == sim.link.shift())
dt = sim.t.diff()
ok = same & (dt > 0)

c_spd = col("SPEED", "V")
if c_spd:
    sim["speed_mps"] = pd.to_numeric(sim[c_spd], errors="coerce") / 3.6 \
        if sim[c_spd].max() > 45 else pd.to_numeric(sim[c_spd], errors="coerce")
else:
    sim["speed_mps"] = np.where(ok, sim.lon_m.diff() / dt, np.nan)
    recovered.append("Speed (differentiated from Pos)")

sim["lat_speed_mps"] = np.where(ok, sim.lat_m.diff() / dt, np.nan)

c_acc = col("ACCELERATION", "ACCEL", "A")
if c_acc:
    sim["accel_mps2"] = pd.to_numeric(sim[c_acc], errors="coerce")
else:
    sim["accel_mps2"] = np.where(ok, sim.speed_mps.diff() / dt, np.nan)
    recovered.append("Acceleration (differentiated from speed)")

if recovered:
    say("   recovered, because the Vehicle Record did not contain them:")
    for r in recovered:
        say(f"     - {r}")

# a lane change is continuous in this coordinate system, but guard anyway
sim.loc[sim.lat_speed_mps.abs() > VY_CLIP, "lat_speed_mps"] = np.nan
sim.loc[sim.speed_mps < -0.5, "speed_mps"] = np.nan

sim = sim[(sim.t >= WARMUP_S) & sim.link.isin(SIM_LINKS)]
say(f"   after warm-up and link filter: {len(sim):,} rows, "
    f"{sim.no.nunique():,} vehicles on links {SIM_LINKS}")
DT_SIM = float(np.median(np.diff(np.sort(sim.t.unique())))) or 0.1
say(f"   time step {DT_SIM:.2f} s")


# ==============================================================================
# LEADER PAIRING - the same geometric rule as Phase 2
# ==============================================================================
def build_pairs(d, idcol="no", tcol="t", hz=PAIR_HZ):
    """Nearest vehicle ahead with lateral body overlap, on the same link."""
    step = max(1, int(round((1.0 / hz) / DT_SIM)))
    times = np.sort(d[tcol].unique())[::step]
    sub = d[d[tcol].isin(times)]
    W = {k: v[1] for k, v in DIMS.items()}
    L = {k: v[0] for k, v in DIMS.items()}
    out = []
    for (_, _), g in sub.groupby([tcol, "link"], sort=False):
        if len(g) < 2:
            continue
        x = g.lon_m.to_numpy(); y = g.lat_m.to_numpy()
        cn = g.class_name.to_numpy()
        v = g.speed_mps.to_numpy(); ids = g[idcol].to_numpy()
        w = np.array([W.get(c, 1.5) for c in cn])
        ln = np.array([L.get(c, 3.0) for c in cn])
        dx = x[None, :] - x[:, None]                    # j ahead of i
        dy = np.abs(y[None, :] - y[:, None])
        thr = OVERLAP_FRAC * (w[:, None] + w[None, :]) / 2.0
        cand = (dx > 0) & (dy < thr)
        dxm = np.where(cand, dx, np.inf)
        j = np.argmin(dxm, axis=1)
        good = np.isfinite(dxm[np.arange(len(x)), j])
        if not good.any():
            continue
        i = np.where(good)[0]; jj = j[good]
        gap = dx[i, jj] - (ln[i] + ln[jj]) / 2.0
        out.append(pd.DataFrame({
            "follower_id": ids[i], "leader_id": ids[jj],
            "follower_class": cn[i], "leader_class": cn[jj],
            "gap_m": gap, "v_follower": v[i], "v_leader": v[jj],
            "dv": v[i] - v[jj], "t": g[tcol].iloc[0]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


say("\n   rebuilding leader pairs on the simulated traffic "
    "(same overlap rule as Phase 2)...")
simp = build_pairs(sim)
say(f"   {len(simp):,} leader-follower samples")


# ==============================================================================
# THE REAL DATA
# ==============================================================================
def load_real():
    tp = os.path.join(REAL_BASE, "trajectories_filtered.csv")
    pp = os.path.join(REAL_BASE, "phase2_pairs.csv")
    if not os.path.exists(tp):
        sys.exit(f"ERROR: {tp} not found.")
    t = pd.read_csv(tp)
    idc = "track_id" if "track_id" in t.columns else "vehicle_id"
    if "direction" in t.columns:
        t = t[t.direction == "with_flow"]
    if "edge" in t.columns:
        t = t[t.edge == 0]
    t = t.rename(columns={idc: "no"})
    p = pd.read_csv(pp) if os.path.exists(pp) else pd.DataFrame()
    return t, p


say("")
say(f"Real data  : {REAL_BASE}")
real, realp = load_real()
say(f"             {len(real):,} rows, {real.no.nunique():,} tracks")
DT_REAL = 0.03332


# ==============================================================================
# FREE / FOLLOWING, identically on both
# ==============================================================================
def mark_free(traj, pairs, idcol="no", tcol="t"):
    """Label every sample FREE or FOLLOWING.

    The real pairs exist at every frame, so they join exactly. The simulated
    pairs are rebuilt at PAIR_HZ to keep the cost down, so they are joined on
    the whole second instead - otherwise every unsampled row would find no
    pair and be called 'free', which would put the free share near 100 %."""
    traj = traj.copy()
    if len(pairs) == 0:
        traj["following"] = False
        return traj
    if tcol == "frame":
        near = (pairs.groupby(["follower_id", "frame"], as_index=False).gap_m.min()
                .rename(columns={"follower_id": idcol, "gap_m": "lead_gap"}))
        out = traj.merge(near, on=[idcol, "frame"], how="left")
    else:
        p = pairs.copy()
        p["tsec"] = np.floor(p.t).astype(int)
        near = (p.groupby(["follower_id", "tsec"], as_index=False).gap_m.min()
                .rename(columns={"follower_id": idcol, "gap_m": "lead_gap"}))
        traj["tsec"] = np.floor(traj[tcol]).astype(int)
        out = traj.merge(near, on=[idcol, "tsec"], how="left")
    out["following"] = out.lead_gap.notna() & (out.lead_gap <= FREE_GAP_M)
    return out


sim = mark_free(sim, simp, "no", "t")
if len(realp):
    realp = realp.rename(columns={"frame": "frame"})
    real = mark_free(real, realp, "no", "frame")
else:
    real["following"] = False

for nm, d in (("simulated", sim), ("observed", real)):
    fr = 100 * (~d.following).mean()
    say(f"   {nm:<10} free {fr:.1f} %  /  following {100-fr:.1f} %")


# ==============================================================================
# COMPARISON 1 - free-flow speed
# ==============================================================================
rule("1. DESIRED SPEED  (free-flowing vehicles only)")
say("This is the most direct check there is: the desired-speed distributions")
say("were typed into VISSIM from the free-flow measurement, so if the model is")
say("wired up correctly these two columns must agree closely.")
say("")

present = [c for c in CLASS_ORDER
           if (sim.class_name == c).sum() > 200 and (real.class_name == c).sum() > 200]

# --- keep only the part of each approach that the signal does not reach ------
linklen = sim.groupby("link").lon_m.max()
sim["to_stopline"] = sim.link.map(linklen) - sim.lon_m
simfree = sim[sim.to_stopline > SIGNAL_ZONE_M]
kept = 100 * len(simfree) / max(len(sim), 1)
say(f"The filmed road had no signal, so on the simulated side the last "
    f"{SIGNAL_ZONE_M:.0f} m")
say("before each stop line is excluded - a vehicle queued at a red has no")
say(f"leader and would otherwise count as free. {kept:.0f} % of simulated "
    f"samples survive.")
if kept < 12:
    say("")
    say("  *** WARNING: almost nothing survives. The approach is queued from")
    say("      the stop line all the way back, so this run contains NO")
    say("      free-flowing traffic and the table below cannot mean anything.")
    say("      Reduce the vehicle inputs and run again. ***")
say("")
say(f"{'class':<11}{'OBSERVED p50':>14}{'SIM p50':>10}{'diff':>8}"
    f"{'OBS p85':>10}{'SIM p85':>9}{'  n sim':>9}   verdict")
say("-" * 78)

spd = {}
for cn in present:
    rv = real[(real.class_name == cn) & (~real.following)
              & (real.speed_mps > MOVING_MPS)].speed_mps.to_numpy() * 3.6
    sv = simfree[(simfree.class_name == cn) & (~simfree.following)
                 & (simfree.speed_mps > MOVING_MPS)].speed_mps.to_numpy() * 3.6
    if len(rv) < 100 or len(sv) < 100:
        continue
    r50, s50 = q(rv, 50), q(sv, 50)
    d = (s50 - r50) / r50 * 100 if r50 > 0.5 else np.nan
    spd[cn] = (rv, sv)
    v = ("good" if abs(d) < 10 else "check" if abs(d) < 25 else "WRONG")
    say(f"{cn:<11}{r50:>14.1f}{s50:>10.1f}{d:>7.0f} %{q(rv,85):>10.1f}"
        f"{q(sv,85):>9.1f}{len(sv):>9,}   {v}")
say("")
say("  Within 10 % is good. More than 25 % means something is not connected:")
say("  check that the vehicle composition points at the right desired-speed")
say("  distribution for that class.")


# ==============================================================================
# COMPARISON 2 - car following
# ==============================================================================
rule("2. CAR FOLLOWING  (gap against speed)")
say("Fitted the same way on both: Theil-Sen regression of gap on speed over")
say("steady following, then the same physical gate. '--' means the fit was")
say("rejected as impossible, exactly as in the Phase 2 report.")
say("")
say(f"{'class':<11}{'OBS CC0':>9}{'SIM CC0':>9}{'OBS CC1':>9}{'SIM CC1':>9}"
    f"{'n obs':>9}{'n sim':>9}")
say("-" * 78)


def fit_cc(pairs, cls):
    if len(pairs) == 0 or theilslopes is None:
        return np.nan, np.nan, 0
    e = pairs[(pairs.follower_class == cls) & (pairs.dv.abs() < FOLLOW_DV_MPS + 0.1)
              & (pairs.v_follower >= 0) & (pairs.v_follower < 8.0)
              & (pairs.gap_m > -0.5) & (pairs.gap_m < 25.0)]
    if len(e) < 120:
        return np.nan, np.nan, len(e)
    gy = e.gap_m.to_numpy(); vx = e.v_follower.to_numpy()
    fits = []
    if len(gy) <= 2500:
        fits.append(theilslopes(gy, vx))
    else:
        rng = np.random.default_rng(12345)
        for _ in range(5):
            k = rng.choice(len(gy), 2500, replace=False)
            fits.append(theilslopes(gy[k], vx[k]))
    sl = float(np.median([x[0] for x in fits]))
    ic = float(np.median([x[1] for x in fits]))
    if not (0.0 <= ic <= 5.0 and 0.20 <= sl <= 3.00):
        return np.nan, np.nan, len(e)
    return ic, sl, len(e)


for cn in present:
    r0, r1, rn = fit_cc(realp, cn)
    s0, s1, sn = fit_cc(simp, cn)
    say(f"{cn:<11}{f(r0):>9}{f(s0):>9}{f(r1):>9}{f(s1):>9}{rn:>9,}{sn:>9,}")
say("")
say("  A class whose observed fit was rejected carries the VISSIM default in")
say("  the model, so its simulated CC0 will NOT match the observed raw fit -")
say("  it should sit near the default of 1.50 m. That is correct behaviour.")


# ==============================================================================
# COMPARISON 3 - weaving
# ==============================================================================
rule("3. WEAVING  (the lateral-speed distribution)")
say("Two widths again: the ordinary standard deviation, and the robust one")
say("that ignores rare large values. Their RATIO is the two-regime index - it")
say("is 1.00 for a true normal distribution.")
say("")
say(f"{'class':<11}{'OBS sd':>8}{'SIM sd':>8}{'OBS MAD':>9}{'SIM MAD':>9}"
    f"{'OBS ratio':>11}{'SIM ratio':>11}")
say("-" * 78)

def ratio(sd, mad):
    """sd / MAD, but a MAD of zero is itself the answer, not an error."""
    if not np.isfinite(sd) or not np.isfinite(mad):
        return np.nan
    return sd / mad if mad > 1e-6 else np.inf


wv, flat = {}, []
for cn in present:
    rv = real[real.class_name == cn].lat_speed_mps.to_numpy()
    sv = sim[sim.class_name == cn].lat_speed_mps.to_numpy()
    rsd, rmad = sd_mad(rv)
    ssd, smad = sd_mad(sv)
    if not np.isfinite(rmad) or not np.isfinite(smad):
        continue
    wv[cn] = (rv, sv, rmad, smad)
    if smad <= 1e-6:
        flat.append(cn)
    sr = ratio(ssd, smad)
    say(f"{cn:<11}{f(rsd,3):>8}{f(ssd,3):>8}{f(rmad,3):>9}{f(smad,3):>9}"
        f"{f(ratio(rsd,rmad)):>11}"
        f"{('infinite' if np.isinf(sr) else f(sr)):>11}")
# --- the share of samples that are moving sideways at all --------------------
say("")
say("Share of samples with NO sideways movement at all, and share above the")
say(f"manoeuvre threshold of {TAIL_THRESH} m/s:")
say("")
say(f"{'class':<11}{'OBS zero':>10}{'SIM zero':>10}{'OBS tail':>10}"
    f"{'SIM tail':>10}{'tail ratio':>12}")
say("-" * 78)
tails = {}
for cn in present:
    rv = real[real.class_name == cn].lat_speed_mps.to_numpy()
    sv = sim[sim.class_name == cn].lat_speed_mps.to_numpy()
    rv = rv[np.isfinite(rv)]; sv = sv[np.isfinite(sv)]
    if len(rv) < 200 or len(sv) < 200:
        continue
    rz = 100 * (np.abs(rv) < 1e-9).mean()
    sz = 100 * (np.abs(sv) < 1e-9).mean()
    rt = 100 * (np.abs(rv) > TAIL_THRESH).mean()
    st = 100 * (np.abs(sv) > TAIL_THRESH).mean()
    tails[cn] = (rt, st)
    say(f"{cn:<11}{rz:>9.1f}%{sz:>9.1f}%{rt:>9.1f}%{st:>9.1f}%"
        f"{(st/rt if rt > 0 else np.nan):>12.2f}")

if flat or any(True for _ in flat):
    pass
say("")
say("=" * 78)
say("READ THIS BEFORE TRYING TO TUNE THE WEAVING")
say("=" * 78)
say("If the simulated 'zero' share is very high - above about 80 % - and the")
say("simulated MAD width is 0.000, that is almost certainly NOT a settings")
say("mistake. It is how VISSIM's lateral model works.")
say("")
say("  A real vehicle on a lane-free road drifts sideways CONTINUOUSLY. Its")
say("  lateral speed is small but almost never exactly zero.")
say("")
say("  A VISSIM vehicle picks a lateral offset and HOLDS it. It moves sideways")
say("  only when it decides to - to overtake within the lane, or to change")
say("  lane. Between those decisions its lateral speed is exactly zero.")
say("")
say("So the model can reproduce the TAIL of the observed distribution - the")
say("deliberate manoeuvres - but not the CORE, the constant small wobble. That")
say("is a limitation of the tool, not of your calibration, and it belongs in")
say("the thesis as a finding rather than being hidden.")
say("")
say("CONSEQUENCE FOR CALIBRATION: stop trying to match the MAD width. Match")
say("these two instead, both of which the model genuinely can produce:")
say("   - the tail share in the table above")
say("   - the manoeuvre rate in section 4")
say("")
say("ONE CHECK BEFORE YOU ACCEPT THIS. Pick one vehicle number out of the")
say(".fzp and plot its lateral position against time. If it is a staircase -")
say("flat, step, flat - the behaviour is structural and nothing you set will")
say("change it. If it is a wobbly line, then it IS a settings problem after")
say("all and the three lateral settings are worth another look.")

say("")
say(f"{'class':<11}{'OBS free':>10}{'OBS follow':>12}{'SIM free':>10}"
    f"{'SIM follow':>12}   same direction?")
say("-" * 78)
for cn in present:
    ro = real[real.class_name == cn]
    si = sim[sim.class_name == cn]
    _, rf = sd_mad(ro[~ro.following].lat_speed_mps)
    _, rl = sd_mad(ro[ro.following].lat_speed_mps)
    _, sf = sd_mad(si[~si.following].lat_speed_mps)
    _, sl = sd_mad(si[si.following].lat_speed_mps)
    if not all(np.isfinite([rf, rl, sf, sl])):
        continue
    same_dir = "yes" if np.sign(rl - rf) == np.sign(sl - sf) else "NO"
    say(f"{cn:<11}{f(rf,3):>10}{f(rl,3):>12}{f(sf,3):>10}{f(sl,3):>12}"
        f"   {same_dir}")
say("")
say("  'same direction' asks whether being blocked changes weaving the same")
say("  way in both. That is a behavioural claim and it should transfer even")
say("  though the absolute widths do not.")


# ==============================================================================
# COMPARISON 4 - manoeuvre rate
# ==============================================================================
rule("4. MANOEUVRE RATE  (sideways manoeuvres per vehicle-minute)")


def manoeuvres(d, idcol, dt):
    """Hysteresis detection, identical settings to Phase 2C."""
    ev = []
    for (vid, cn), g in d.groupby([idcol, "class_name"], sort=False):
        v = g.lat_speed_mps.to_numpy()
        y = g.lat_m.to_numpy()
        if len(v) < 5:
            continue
        on = np.abs(v) > VY_ON
        i = 0
        while i < len(v):
            if not on[i]:
                i += 1; continue
            j = i
            while j < len(v) and np.abs(v[j]) > VY_OFF:
                j += 1
            dur = (j - i) * dt
            shift = abs(y[min(j, len(y) - 1)] - y[i])
            if MIN_DUR_S <= dur <= MAX_DUR_S and shift >= MIN_SHIFT_M:
                ev.append((vid, cn, dur, shift))
            i = max(j, i + 1)
    return pd.DataFrame(ev, columns=["id", "class_name", "dur_s", "shift_m"])


say("Detected with the same two thresholds and the same minimum shift as")
say("Phase 2C, then divided by observation time so the two are comparable.")
say("")
sm = manoeuvres(sim, "no", DT_SIM)
say(f"{'class':<11}{'OBS rate':>10}{'SIM rate':>10}{'ratio':>8}"
    f"{'n obs':>8}{'n sim':>8}   verdict")
say("-" * 78)

obs_man_path = os.path.join(REAL_BASE, "phase2c_manoeuvres.csv")
rm = pd.read_csv(obs_man_path) if os.path.exists(obs_man_path) else None
r_exp = real.groupby("class_name").size() * DT_REAL / 60.0
s_exp = sim.groupby("class_name").size() * DT_SIM / 60.0
rates = {}
for cn in present:
    sn = int((sm.class_name == cn).sum())
    srate = sn / s_exp.get(cn, np.nan) if s_exp.get(cn, 0) > 0.5 else np.nan
    if rm is not None and "cls" in rm.columns:
        rn = int((rm.cls == cn).sum())
        rrate = rn / r_exp.get(cn, np.nan) if r_exp.get(cn, 0) > 0.5 else np.nan
    else:
        rn, rrate = 0, np.nan
    if not np.isfinite(rrate) or rn < 15:
        say(f"{cn:<11}{'--':>10}{f(srate):>10}{'--':>8}{rn:>8}{sn:>8}"
            f"   too few observed to compare")
        continue
    ratio = srate / rrate if rrate > 0 else np.nan
    v = "good" if 0.5 <= ratio <= 2.0 else "check"
    rates[cn] = (rrate, srate)
    say(f"{cn:<11}{rrate:>10.2f}{srate:>10.2f}{ratio:>8.2f}{rn:>8}{sn:>8}   {v}")
say("")
say("  This is the single best calibration target, because it is scale-free.")
say("  If the simulated rate is far too low, raise the lateral freedom:")
say("  reduce the minimum lateral distance, or lower the minimum longitudinal")
say("  speed for lateral movement. If it is far too high, do the opposite.")


# ==============================================================================
# FIGURES
# ==============================================================================
rule("FIGURES")

# --- fig20 speed -------------------------------------------------------------
if spd:
    cls = list(spd.keys())
    ncol = min(3, len(cls)); nrow = int(np.ceil(len(cls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 2.9 * nrow),
                             squeeze=False)
    bins = np.arange(0, 40, 2.0)
    for k, cn in enumerate(cls):
        ax = axes[k // ncol][k % ncol]
        rv, sv = spd[cn]
        ax.hist(rv, bins=bins, density=True, color=colour(cn), alpha=0.32,
                edgecolor=SURFACE, linewidth=1.0, label="observed")
        ax.hist(sv, bins=bins, density=True, histtype="step",
                color=SIM_COLOUR, linewidth=2.2, label="simulated")
        style(ax, "speed (km/h)" if k + ncol >= len(cls) else "",
              "share" if k % ncol == 0 else "", cn)
        if k == 0:
            leg = ax.legend(frameon=False, fontsize=8.5)
            for t_ in leg.get_texts():
                t_.set_color(INK_2)
    for k in range(len(cls), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Free-flow speed: observed against simulated", color=INK_1,
                 fontsize=12, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.text(0.02, 0.012, "Free-flowing vehicles only, both sides. These two "
             "should sit on top of each other -\nthe simulated curve IS the "
             "distribution that was typed into VISSIM from the observed one.",
             color=INK_3, fontsize=8.5)
    save(fig, "fig20_speed_compare.png")

# --- fig21 weaving -----------------------------------------------------------
if wv:
    cls = list(wv.keys())
    ncol = min(3, len(cls)); nrow = int(np.ceil(len(cls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 2.9 * nrow),
                             squeeze=False)
    for k, cn in enumerate(cls):
        ax = axes[k // ncol][k % ncol]
        rv, sv, rmad, smad = wv[cn]
        R = float(np.clip(9 * max(rmad, smad), 0.3, 1.2))
        b = np.linspace(-R, R, 61)
        ax.hist(rv[np.isfinite(rv)], bins=b, density=True, color=colour(cn),
                alpha=0.32, edgecolor=SURFACE, linewidth=0.5, label="observed")
        ax.hist(sv[np.isfinite(sv)], bins=b, density=True, histtype="step",
                color=SIM_COLOUR, linewidth=2.0, label="simulated")
        ax.set_yscale("log"); ax.set_ylim(bottom=1e-3)
        style(ax, "lateral speed (m/s)" if k + ncol >= len(cls) else "",
              "density (log)" if k % ncol == 0 else "", cn)
        if k == 0:
            leg = ax.legend(frameon=False, fontsize=8.5)
            for t_ in leg.get_texts():
                t_.set_color(INK_2)
    for k in range(len(cls), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Weaving: observed against simulated", color=INK_1,
                 fontsize=12, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    fig.text(0.02, 0.012, "Log scale, so the tails are visible. The simulated "
             "core will be narrower - simulated positions carry\nno measurement "
             "noise. What matters is that the simulated curve also has heavy "
             "tails rather than\nfalling away like a normal distribution: that "
             "is the model reproducing decisive manoeuvres.",
             color=INK_3, fontsize=8.5)
    save(fig, "fig21_weaving_compare.png")

# --- fig22 manoeuvre rate ----------------------------------------------------
if rates:
    cls = list(rates.keys())
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(cls) + 2.6))
    y = np.arange(len(cls))
    for i, cn in enumerate(cls):
        r, s = rates[cn]
        ax.barh(y[i] + 0.19, r, height=0.34, color=colour(cn),
                edgecolor=SURFACE, linewidth=1.5)
        ax.barh(y[i] - 0.19, s, height=0.34, color=SIM_COLOUR,
                edgecolor=SURFACE, linewidth=1.5)
    ax.text(rates[cls[0]][0], y[0] + 0.19, "  observed", va="center",
            fontsize=9, color=INK_2)
    ax.text(rates[cls[0]][1], y[0] - 0.19, "  simulated", va="center",
            fontsize=9, color=SIM_COLOUR)
    ax.set_yticks(y); ax.set_yticklabels(cls, color=INK_1, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(max(v) for v in rates.values()) * 1.35)
    style(ax, "sideways manoeuvres per vehicle-minute", "",
          "Manoeuvre rate: observed against simulated")
    ax.text(0.0, -0.16, "The scale-free calibration target. Bars of similar "
            "length mean the model weaves as often as\nthe real traffic does. "
            "Only classes with at least 15 observed manoeuvres are shown.",
            transform=ax.transAxes, color=INK_3, fontsize=8.5, va="top")
    save(fig, "fig22_manoeuvre_compare.png")


# ==============================================================================
rule("HOW TO USE THIS")
say("Work down the four comparisons in order. Each one depends on the one")
say("above it being right, so there is no point tuning the weaving while the")
say("desired speed is still wrong.")
say("")
say("1. SPEED off by more than 25 %")
say("   Not a calibration problem - a wiring problem. The vehicle composition")
say("   is pointing at the wrong desired-speed distribution for that class.")
say("")
say("2. CC0 or CC1 far off for a class whose observed fit was ACCEPTED")
say("   Check that class has its own driving behaviour, and that the value was")
say("   actually typed in. A class using the shared default will show the")
say("   default, not the measurement.")
say("")
say("3. WEAVING ratio near 1.00 in the simulation but well above 1 in the")
say("   observation")
say("   The model is producing wobble but no decisive manoeuvres. Lower the")
say("   minimum longitudinal speed for lateral movement, and check that")
say("   overtaking on the same lane is enabled on both sides.")
say("")
say("4. MANOEUVRE RATE too low")
say("   Reduce the minimum lateral distance, or raise the number of")
say("   interaction objects so vehicles can see the gaps they could use.")
say("")
say("A NOTE ON WHAT THIS CAN AND CANNOT SHOW")
say("  The behaviour parameters were measured from this same road, so this is")
say("  a check that they were entered and are working - not independent")
say("  validation. Say that plainly in the thesis. Independent validation")
say("  would need a second site, or a second recording held back from the")
say("  calibration.")

try:
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    print(f"\n[saved to {OUT_TXT}]")
except Exception as e:
    print(f"[warn] {e}")
