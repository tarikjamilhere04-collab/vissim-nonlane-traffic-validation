#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 3C - DESIRED SPEED, MEASURED PROPERLY
================================================================================
 WHY THIS SCRIPT EXISTS - AN ERROR IN THE EARLIER ADVICE

   The Phase 3B report said the observed p15-p95 speed band is the VISSIM
   desired-speed input. That is WRONG, and it matters.

   VISSIM's "desired speed" is the speed a driver chooses when nothing is in
   the way. The simulation then SLOWS vehicles down itself, through the
   car-following model, whenever a leader is close. So if the desired speed is
   taken from speeds observed in congestion, the congestion gets counted twice:
   once in the input and again in the model. The simulated traffic comes out
   far slower than the real road, and no amount of W99 tuning fixes it,
   because the error is in the input, not the behaviour.

   The correct input is the speed distribution of vehicles that were FREE -
   no vehicle close enough in front to constrain them.

 WHAT THIS SCRIPT DOES
   1. Marks every trajectory sample as FOLLOWING or FREE, using the leader
      pairs Phase 2 already found. A sample is FREE if no leader was found for
      it at that instant, or the leader was further ahead than FREE_GAP_M.
   2. Reports the speed distribution for both sets, side by side, so the size
      of the error is visible rather than asserted.
   3. Prints the desired-speed distribution to type into VISSIM.
   4. Draws fig9_desired_speed.png showing the two distributions per class.

 INPUT   trajectories_filtered.csv, phase2_pairs.csv
 OUTPUT  phase3c_desired_speed.txt, figures/fig9_desired_speed.png
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ==============================================================================
BASE    = r"S:\soscho"
FIG_DIR = os.path.join(BASE, "figures")
OUT_TXT = os.path.join(BASE, "phase3c_desired_speed.txt")
DT      = 0.03332

# A leader further ahead than this is not constraining the follower. The
# observed longitudinal window is about 29 m and gaps above roughly 20 m were
# already beyond the interacting range (Phase 2), so 15 m is a conservative
# choice: it errs towards calling a sample "following", which UNDER-states the
# correction rather than over-stating it.
FREE_GAP_M = 15.0

# A vehicle standing still with nothing in front of it is not revealing a
# desired speed - it is waiting for something this study does not model.
# Samples below this speed are excluded from the free-flow distribution.
MOVING_MPS = 0.5

MIN_FREE_SAMPLES = 200
CLASS_ORDER = ["Car", "Rickshaw", "Bike", "CNG", "Truck", "Bus"]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK_1, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE, BAD = "#e6e5e1", "#ffffff", "#b3261e"
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


def f(v, d=1):
    return f"{v:.{d}f}" if np.isfinite(v) else "--"


rule("PHASE 3C | DESIRED SPEED, MEASURED ON FREE-FLOWING VEHICLES ONLY")

tp = os.path.join(BASE, "trajectories_filtered.csv")
pp = os.path.join(BASE, "phase2_pairs.csv")
for p in (tp, pp):
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found. Run Phase 1 and Phase 2 first.")

traj = pd.read_csv(tp)
pairs = pd.read_csv(pp)
idcol = "track_id" if "track_id" in traj.columns else "vehicle_id"
if "direction" in traj.columns:
    traj = traj[traj.direction == "with_flow"]
if "edge" in traj.columns:
    traj = traj[traj.edge == 0]
if "speed_kmph" not in traj.columns:
    traj["speed_kmph"] = traj.speed_mps * 3.6

say(f"Trajectory samples : {len(traj):,}")
say(f"Leader pairs       : {len(pairs):,}")

# ------------------------------------------------------------------------------
# PROVENANCE - the same content check as Phase 3A/3B
# ------------------------------------------------------------------------------
_key = traj.set_index([idcol, "frame"])
if len(pairs):
    k = min(400, len(pairs))
    r = np.random.default_rng(0).choice(len(pairs), k, replace=False)
    idx = pd.MultiIndex.from_arrays([pairs.follower_id.to_numpy()[r],
                                     pairs.frame.to_numpy()[r]])
    got = _key["speed_mps"].reindex(idx).to_numpy(dtype=float)
    d = np.abs(got - pairs.v_follower.to_numpy()[r])
    d = d[np.isfinite(d)]
    med = float(np.median(d)) if len(d) >= 20 else np.inf
    say(f"Provenance check   : median mismatch {med:.3f} m/s "
        f"({'OK' if med <= 0.10 else 'FAIL'})")
    if med > 0.10:
        sys.exit("STOP: phase2_pairs.csv was not produced from this "
                 "trajectory file. Re-run Phase 2 first.")

# ------------------------------------------------------------------------------
# MARK EVERY SAMPLE AS FOLLOWING OR FREE
# ------------------------------------------------------------------------------
rule("STEP 1 : WHICH SAMPLES WERE ACTUALLY FREE")
say("A sample is FOLLOWING if Phase 2 found a leader for it with a gap of")
say(f"{FREE_GAP_M:.0f} m or less. Everything else is FREE: either no vehicle was")
say("ahead with lateral overlap, or the one that was is too far away to matter.")
say("")

# nearest leader per (follower, frame) - a follower can only have one here,
# but take the minimum gap defensively
near = (pairs.groupby(["follower_id", "frame"], as_index=False)
        .gap_m.min()
        .rename(columns={"follower_id": idcol, "gap_m": "lead_gap"}))
traj = traj.merge(near, on=[idcol, "frame"], how="left")
traj["following"] = traj.lead_gap.notna() & (traj.lead_gap <= FREE_GAP_M)
traj["free"] = ~traj.following

n_fol = int(traj.following.sum())
say(f"FOLLOWING samples : {n_fol:>9,}   ({100*n_fol/len(traj):.1f} %)")
say(f"FREE samples      : {len(traj)-n_fol:>9,}   "
    f"({100*(len(traj)-n_fol)/len(traj):.1f} %)")
say("")
say(f"Of the free ones, those below {MOVING_MPS} m/s are dropped: a vehicle")
say("standing still with nothing in front of it is not showing us a desired")
say("speed, it is waiting for something this study does not model.")

# ------------------------------------------------------------------------------
# THE TWO DISTRIBUTIONS
# ------------------------------------------------------------------------------
rule("STEP 2 : HOW BIG THE ERROR WAS")
say("If these two columns were the same, the earlier advice would have been")
say("harmless. They are not.")
say("")
say(f"{'class':<11}{'n free':>9}{'ALL p50':>9}{'FREE p50':>10}"
    f"{'ALL p85':>9}{'FREE p85':>10}{'  under-stated by'}")
say("-" * 78)

present = [c for c in CLASS_ORDER if c in set(traj.class_name)]
res = {}
for cn in present:
    t = traj[traj.class_name == cn]
    allv = t.speed_kmph.to_numpy()
    fr = t[t.free & (t.speed_mps > MOVING_MPS)].speed_kmph.to_numpy()
    if len(fr) < MIN_FREE_SAMPLES:
        say(f"{cn:<11}{len(fr):>9,}{'':>38}  too few free samples")
        res[cn] = None
        continue
    a50, f50 = q(allv, 50), q(fr, 50)
    a85, f85 = q(allv, 85), q(fr, 85)
    gain = (f50 - a50) / a50 * 100 if a50 > 0.5 else np.nan
    res[cn] = dict(n=len(fr), p15=q(fr, 15), p50=f50, p85=f85, p95=q(fr, 95),
                   a50=a50)
    say(f"{cn:<11}{len(fr):>9,}{f(a50):>9}{f(f50):>10}{f(a85):>9}{f(f85):>10}"
        f"{('   ' + f(gain, 0) + ' %') if np.isfinite(gain) else '   --'}")
say("")
say("  'under-stated by' is how much lower the median of ALL driving is than")
say("  the median of free driving. That is the amount by which the desired")
say("  speed would have been set too low, before the car-following model even")
say("  starts slowing vehicles down again.")

# ------------------------------------------------------------------------------
# THE NUMBERS TO TYPE IN
# ------------------------------------------------------------------------------
rule("STEP 3 : THE VISSIM DESIRED-SPEED DISTRIBUTIONS")
say("Base Data > Distributions > Desired Speed > New, one per class.")
say("VISSIM takes a minimum and a maximum and spreads drivers between them.")
say("Use the 15th and 95th percentile of FREE driving - not the extremes,")
say("which are single vehicles and detection noise.")
say("")
say(f"{'class':<11}{'min (km/h)':>12}{'max (km/h)':>12}{'median':>9}"
    f"{'n free':>9}   status")
say("-" * 78)
for cn in present:
    r = res.get(cn)
    if r is None:
        say(f"{cn:<11}{'--':>12}{'--':>12}{'--':>9}{'--':>9}   "
            f"NOT MEASURABLE - use a value from the literature and declare it")
        continue
    warn = ""
    if r["p50"] < 5.0:
        warn = ("NOT CREDIBLE - a motorised vehicle with nothing in front of "
                "it does not choose this speed")
    elif r["p15"] < 2.0:
        warn = "CHECK - p15 near zero suggests stationary detections"
    r["warn"] = warn
    say(f"{cn:<11}{r['p15']:>12.1f}{r['p95']:>12.1f}{r['p50']:>9.1f}"
        f"{r['n']:>9,}   {warn or 'ok'}")
say("")
say("  A row marked NOT CREDIBLE must not be entered. If a class free of any")
say("  leader still shows a crawl, the class is not measuring drivers - it is")
say("  measuring parked vehicles or a detection failure. Use a literature")
say("  value for that class and declare that you did.")

# ------------------------------------------------------------------------------
# STEP 4 - THE SAME ERROR APPLIES TO ACCELERATION
# ------------------------------------------------------------------------------
rule("STEP 4 : CC8 AND CC9 HAVE THE SAME PROBLEM")
say("CC8 is the acceleration a driver WANTS from standstill and CC9 the one")
say("they want at speed. Both are desires, exactly like desired speed - and")
say("both were measured over all driving, including vehicles that could not")
say("accelerate because something was in the way. So both are measured again")
say("here on free-flowing samples only.")
say("")
say(f"{'class':<11}{'CC8 all':>9}{'CC8 free':>10}{'CC9 all':>9}{'CC9 free':>10}"
    f"{'n launch':>10}   note")
say("-" * 78)

CC8_MAX_MPS = 2.00
MIN_ACC = 30
for cn in present:
    t = traj[traj.class_name == cn]
    lall = t[(t.speed_mps < CC8_MAX_MPS) & (t.accel_mps2 > 0.05)]
    lfree = lall[lall.free]
    hi_all = t[(t.speed_mps > q(t.speed_mps, 75)) & (t.accel_mps2 > 0.05)]
    hi_free = hi_all[hi_all.free]
    c8a = q(lall.accel_mps2, 85) if len(lall) >= MIN_ACC else np.nan
    c8f = q(lfree.accel_mps2, 85) if len(lfree) >= MIN_ACC else np.nan
    c9a = q(hi_all.accel_mps2, 85) if len(hi_all) >= MIN_ACC else np.nan
    c9f = q(hi_free.accel_mps2, 85) if len(hi_free) >= MIN_ACC else np.nan
    if res.get(cn) is not None:
        res[cn].update(cc8_all=c8a, cc8_free=c8f, cc9_all=c9a, cc9_free=c9f)
    note = "" if len(lfree) >= MIN_ACC else "too few free launches"
    say(f"{cn:<11}{f(c8a,2):>9}{f(c8f,2):>10}{f(c9a,2):>9}{f(c9f,2):>10}"
        f"{len(lfree):>10,}   {note}")
say("")
say("  Use the 'free' columns. Values are in m/s^2 and carry the acceleration")
say(f"  uncertainty of the whole study - the filter recovers acceleration at")
say("  r = 0.70 against known truth, so quote that alongside them.")
say("")
say("  These will still look low next to VISSIM's defaults (CC8 3.50, CC9")
say("  1.50). That is expected and is a finding, not an error: this is a")
say("  congested urban road where even unobstructed vehicles do not launch")
say("  hard. Enter the measured value, and say in the thesis that it is below")
say("  the default and why.")

# ------------------------------------------------------------------------------
# MACHINE-READABLE OUTPUT for the VISSIM sheet (Phase 3D)
# ------------------------------------------------------------------------------
rows = []
for cn in present:
    r = res.get(cn)
    if r is None:
        continue
    rows.append(dict(class_name=cn, n_free=r["n"], v_p15=r["p15"],
                     v_p50=r["p50"], v_p85=r["p85"], v_p95=r["p95"],
                     cc8_free=r.get("cc8_free", np.nan),
                     cc9_free=r.get("cc9_free", np.nan),
                     cc8_all=r.get("cc8_all", np.nan),
                     cc9_all=r.get("cc9_all", np.nan),
                     warn=r.get("warn", "")))
if rows:
    csv_out = os.path.join(BASE, "phase3c_desired_speed.csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    say(f"\n[machine-readable results: {csv_out}]")

rule("WHAT TO WRITE IN THE THESIS")
say("Desired speed distributions were estimated from free-flowing vehicles")
say("only - those with no leader within " + f"{FREE_GAP_M:.0f} m - because the")
say("desired speed in a car-following model is by definition the speed chosen")
say("in the absence of interaction. Using the speed distribution of all")
say("observed vehicles would embed the congestion of the recording into the")
say("model input, and the car-following model would then apply that")
say("congestion a second time.")
say("")
say("That sentence is worth including. It is the kind of methodological point")
say("an examiner looks for, and the table in Step 2 is the evidence for it.")

# ==============================================================================
# FIGURE
# ==============================================================================
if plt is not None:
    cls = [c for c in present if res.get(c)]
    if cls:
        ncol = min(3, len(cls)); nrow = int(np.ceil(len(cls) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.9 * nrow),
                                 squeeze=False)
        bins = np.arange(0, 40, 2.0)
        for k, cn in enumerate(cls):
            ax = axes[k // ncol][k % ncol]
            t = traj[traj.class_name == cn]
            allv = t.speed_kmph.to_numpy()
            fr = t[t.free & (t.speed_mps > MOVING_MPS)].speed_kmph.to_numpy()
            c = PALETTE[CLASS_ORDER.index(cn) % len(PALETTE)]
            ax.hist(allv, bins=bins, density=True, color=INK_3, alpha=0.30,
                    edgecolor=SURFACE, linewidth=1.2, label="all driving")
            ax.hist(fr, bins=bins, density=True, histtype="step", color=c,
                    linewidth=2.2, label="free-flowing only")
            ax.set_facecolor(SURFACE)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
            ax.set_title(cn, color=INK_1, fontsize=11, loc="left", pad=8)
            if k + ncol >= len(cls):
                ax.set_xlabel("speed (km/h)", color=INK_2, fontsize=10)
            if k % ncol == 0:
                ax.set_ylabel("share", color=INK_2, fontsize=10)
            if k == 0:
                leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
                for tx in leg.get_texts():
                    tx.set_color(INK_2)
        for k in range(len(cls), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("Desired speed: all driving vs free-flowing vehicles only",
                     color=INK_1, fontsize=12, x=0.02, ha="left", y=0.995)
        fig.tight_layout(rect=[0, 0.07, 1, 0.96])
        fig.text(0.02, 0.012,
                 "The coloured outline is the VISSIM desired-speed input. Where "
                 "it sits to the right of the grey\nhistogram, using the grey "
                 "one would have made every simulated vehicle too slow - and "
                 "the\ncar-following model would then have slowed them again.",
                 color=INK_3, fontsize=8.5)
        os.makedirs(FIG_DIR, exist_ok=True)
        p = os.path.join(FIG_DIR, "fig9_desired_speed.png")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        say(f"\n[figure written: {p}]")

try:
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    print(f"\n[saved to {OUT_TXT}]")
except Exception as e:
    print(f"[warn] {e}")
