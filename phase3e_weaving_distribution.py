#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 3E - THE DISTRIBUTION OF WEAVING, BY VEHICLE CLASS
================================================================================
 WHAT THE SUPERVISOR ASKED FOR
   "A normal distribution of the weaving behaviour of vehicles by class, with
   respect to something that better explains it."

 WHAT "WEAVING" IS MEASURED AS
   The LATERAL SPEED, v_y - how fast a vehicle is sliding sideways across the
   road at each instant, in m/s. Positive is one way, negative the other. It is
   the right quantity because:
     - it exists at every instant, not only during a detected manoeuvre, so
       nothing depends on a detection threshold;
     - it is symmetric about zero (drivers move left as often as right), which
       is what makes a normal distribution a sensible thing to fit at all;
     - its spread is one number per class - a clean "weaving intensity".

 THE THREE FIGURES
   fig10  the normal fit itself - histogram of v_y per class with N(0, sigma)
          drawn over it. This is the figure that was asked for.
   fig11  the same data on a logarithmic vertical axis. This is where the
          honest finding is: the real distribution has HEAVIER TAILS than a
          normal. Weaving is not one behaviour, it is two - constant small
          wobble, plus rare decisive manoeuvres.
   fig12  what explains it - weaving intensity against speed, and weaving
          intensity when FREE versus when FOLLOWING another vehicle.

 TWO MEASURES OF SPREAD, AND WHY BOTH ARE REPORTED
   sigma_SD   the ordinary standard deviation - counts the rare big manoeuvres
   sigma_MAD  1.4826 x median absolute deviation - ignores them, so it measures
              only the everyday wobble
   For a true normal distribution the two are equal. The RATIO sigma_SD /
   sigma_MAD is therefore a direct, assumption-free measure of how much of a
   class's weaving comes from decisive manoeuvres rather than from drift.

 INPUT   trajectories_filtered.csv, phase2_pairs.csv
 OUTPUT  figures/fig10..fig12, phase3e_weaving.txt
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
    sys.exit("ERROR: matplotlib required.  pip install matplotlib")
try:
    from scipy import stats as sstats
except ImportError:
    sstats = None

# ==============================================================================
BASE    = r"S:\soscho"
FIG_DIR = os.path.join(BASE, "figures")
OUT_TXT = os.path.join(BASE, "phase3e_weaving.txt")
DT      = 0.03332
DPI     = 300

FREE_GAP_M = 15.0      # same definition as Phase 3C
VY_CLIP    = 1.5       # m/s - beyond this is not a vehicle, it is a bad frame
MIN_N      = 500

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
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    say(f"   wrote {p}")


def sd_mad(v):
    """Two spreads: the ordinary one, and one that ignores the rare big values."""
    sd = float(np.std(v))
    mad = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    return sd, mad


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


# ==============================================================================
rule("PHASE 3E | THE DISTRIBUTION OF WEAVING, BY VEHICLE CLASS")

tp = os.path.join(BASE, "trajectories_filtered.csv")
pp = os.path.join(BASE, "phase2_pairs.csv")
if not os.path.exists(tp):
    sys.exit(f"ERROR: {tp} not found.")
traj = pd.read_csv(tp)
idcol = "track_id" if "track_id" in traj.columns else "vehicle_id"
if "direction" in traj.columns:
    traj = traj[traj.direction == "with_flow"]
if "edge" in traj.columns:
    traj = traj[traj.edge == 0]
if "lat_speed_mps" not in traj.columns:
    sys.exit("ERROR: lat_speed_mps not in the filtered file. Re-run Phase 1.")
if "speed_kmph" not in traj.columns:
    traj["speed_kmph"] = traj.speed_mps * 3.6

traj = traj[traj.lat_speed_mps.abs() < VY_CLIP]
present = [c for c in CLASS_ORDER if (traj.class_name == c).sum() >= MIN_N]
say(f"Samples used : {len(traj):,}")
say(f"Classes      : {', '.join(present)}")
say(f"(lateral speeds beyond +/-{VY_CLIP} m/s dropped - a vehicle does not")
say(" cross the road that fast; those are bad frames, not behaviour.)")

# --- free vs following, reusing the Phase 3C definition -----------------------
have_free = False
if os.path.exists(pp):
    pairs = pd.read_csv(pp)
    near = (pairs.groupby(["follower_id", "frame"], as_index=False).gap_m.min()
            .rename(columns={"follower_id": idcol, "gap_m": "lead_gap"}))
    traj = traj.merge(near, on=[idcol, "frame"], how="left")
    traj["following"] = traj.lead_gap.notna() & (traj.lead_gap <= FREE_GAP_M)
    have_free = True
else:
    say("\n[note] phase2_pairs.csv not found - the free/following panel is skipped")

# ==============================================================================
rule("STEP 1 : IS WEAVING NORMALLY DISTRIBUTED?")
say("A normal distribution is described by two numbers: where it is centred,")
say("and how wide it is. If weaving really were normal, the two ways of")
say("measuring the width below would agree.")
say("")
say(f"{'class':<11}{'n':>9}{'mean':>8}{'sd':>8}{'MAD-sd':>9}{'ratio':>8}"
    f"{'skew':>8}{'ex.kurt':>9}   verdict")
say("-" * 78)

stat = {}
for cn in present:
    v = traj[traj.class_name == cn].lat_speed_mps.to_numpy()
    v = v[np.isfinite(v)]
    sd, mad = sd_mad(v)
    sk = float(sstats.skew(v)) if sstats else np.nan
    ku = float(sstats.kurtosis(v)) if sstats else np.nan
    ratio = sd / mad if mad > 1e-9 else np.nan
    verdict = ("close to normal" if ratio < 1.3 else
               "heavy tails" if ratio < 2.0 else "STRONGLY two-regime")
    stat[cn] = dict(n=len(v), mean=float(np.mean(v)), sd=sd, mad=mad,
                    ratio=ratio, skew=sk, kurt=ku, verdict=verdict)
    say(f"{cn:<11}{len(v):>9,}{np.mean(v):>8.3f}{sd:>8.3f}{mad:>9.3f}"
        f"{ratio:>8.2f}{sk:>8.2f}{ku:>9.1f}   {verdict}")

say("")
say("HOW TO READ THIS TABLE")
say("  mean    ~ 0 for every class. Drivers move left as often as right -")
say("            there is no systematic drift to one side. Good.")
say("  sd      the ordinary standard deviation. Counts everything.")
say("  MAD-sd  the same width measured in a way that ignores rare large")
say("            values. It sees only the everyday wobble.")
say("  ratio   sd / MAD-sd. For a TRUE normal distribution this is 1.00.")
say("  ex.kurt excess kurtosis. 0.0 for a normal. Positive means the")
say("            distribution has a sharper peak and fatter tails than normal.")
say("")
say("THE FINDING, IN ONE SENTENCE:")
say("  Weaving is NOT one behaviour with a normal distribution. It is TWO")
say("  behaviours added together - a constant small side-to-side wobble that")
say("  every vehicle does all the time, plus occasional decisive sideways")
say("  manoeuvres. The normal curve describes the wobble; the extra weight in")
say("  the tails is the manoeuvres.")
say("")
say("  The ratio column measures how decisive each class is. A class near 1.0")
say("  only wobbles. A class near 3.0 spends most of its time barely moving")
say("  sideways and then commits to a real manoeuvre.")

# ==============================================================================
# FIG 10 - the normal fit (what was asked for)
# ==============================================================================
rule("STEP 2 : THE FIGURES")
ncol = min(3, len(present)); nrow = int(np.ceil(len(present) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.0 * nrow),
                         squeeze=False)
for k, cn in enumerate(present):
    ax = axes[k // ncol][k % ncol]
    v = traj[traj.class_name == cn].lat_speed_mps.to_numpy()
    c = colour(cn)
    s = stat[cn]
    # Each class gets its own x-range. A shared one leaves the narrow classes
    # with three bars and nothing to judge the shape by.
    R = float(np.clip(5.0 * s["mad"], 0.25, 1.0))
    bins = np.linspace(-R, R, 61)
    xs = np.linspace(-R, R, 400)
    ax.hist(v, bins=bins, density=True, color=c, alpha=0.30,
            edgecolor=SURFACE, linewidth=0.4, label="observed")
    ax.plot(xs, normal_pdf(xs, 0.0, s["mad"]), color=INK_1, linewidth=2.0,
            label="normal fit (wobble)")
    ax.plot(xs, normal_pdf(xs, 0.0, s["sd"]), color=BAD, linewidth=1.4,
            linestyle="--", label="normal using full sd")
    ax.text(0.03, 0.96, f"σ = {s['mad']:.3f} m/s", transform=ax.transAxes,
            fontsize=9, color=INK_2, va="top")
    style(ax, "lateral speed  (m/s)" if k + ncol >= len(present) else "",
          "probability density" if k % ncol == 0 else "", cn)
    if k == 0:
        leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
        for t in leg.get_texts():
            t.set_color(INK_2)
for k in range(len(present), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
fig.suptitle("Weaving by vehicle class: the distribution of sideways speed",
             color=INK_1, fontsize=12, x=0.02, ha="left", y=0.995)
fig.tight_layout(rect=[0, 0.09, 1, 0.96])
fig.text(0.02, 0.012,
         "Black curve: a normal distribution whose width is the everyday "
         "wobble (robust estimate). Red dashed: a normal\nusing the full "
         "standard deviation - too wide in the middle, because the standard "
         "deviation is inflated by the rare\nlarge manoeuvres. The observed "
         "bars are taller than either curve at the centre: that is the "
         "two-regime effect.",
         color=INK_3, fontsize=8.5)
save(fig, "fig10_weaving_normal.png")

# ==============================================================================
# FIG 11 - the tails, on a log axis
# ==============================================================================
fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.0 * nrow),
                         squeeze=False)
for k, cn in enumerate(present):
    ax = axes[k // ncol][k % ncol]
    v = traj[traj.class_name == cn].lat_speed_mps.to_numpy()
    c = colour(cn)
    s = stat[cn]
    R = float(np.clip(9.0 * s["mad"], 0.4, 1.2))   # wider, to show the tails
    bins = np.linspace(-R, R, 61)
    xs = np.linspace(-R, R, 400)
    ax.hist(v, bins=bins, density=True, color=c, alpha=0.30,
            edgecolor=SURFACE, linewidth=0.4)
    ax.plot(xs, normal_pdf(xs, 0.0, s["mad"]), color=INK_1, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    ax.text(0.03, 0.96, f"ratio {s['ratio']:.2f}\n{s['verdict']}",
            transform=ax.transAxes, fontsize=8.5,
            color=(BAD if s["ratio"] >= 2.0 else INK_2), va="top")
    style(ax, "lateral speed  (m/s)" if k + ncol >= len(present) else "",
          "density (log scale)" if k % ncol == 0 else "", cn)
for k in range(len(present), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
fig.suptitle("The same data on a log scale: where the normal model breaks",
             color=INK_1, fontsize=12, x=0.02, ha="left", y=0.995)
fig.tight_layout(rect=[0, 0.09, 1, 0.96])
fig.text(0.02, 0.012,
         "On a log scale a normal distribution is an upside-down parabola. "
         "Wherever the coloured bars sit ABOVE the black\ncurve out at the "
         "sides, real vehicles are moving sideways far more often than a "
         "normal distribution allows.\nThose bars are the overtakes and "
         "gap-filling moves - the behaviour this thesis is about.",
         color=INK_3, fontsize=8.5)
save(fig, "fig11_weaving_tails.png")

# ==============================================================================
# FIG 12 - what explains it
# ==============================================================================
rule("STEP 3 : WHAT EXPLAINS THE WEAVING")
ncol2 = 2 if have_free else 1
fig, axes = plt.subplots(1, ncol2, figsize=(6.2 * ncol2, 4.3), squeeze=False)

# (a) weaving intensity against how fast the vehicle is going
ax = axes[0][0]
say("(a) Weaving intensity against longitudinal speed")
say("")
say(f"{'class':<11}" + "".join(f"{lo:>7.0f}" for lo in np.arange(0, 30, 5)))
say(f"{'speed band':<11}" + "".join(f"{'-' + str(lo + 5):>7}" for lo in np.arange(0, 30, 5)))
say("-" * 78)
for cn in present:
    t = traj[traj.class_name == cn]
    xs2, ys2 = [], []
    line = f"{cn:<11}"
    for lo in np.arange(0, 30, 5.0):
        s = t[(t.speed_kmph >= lo) & (t.speed_kmph < lo + 5)]
        if len(s) < 200:
            line += f"{'--':>7}"
            continue
        _, m = sd_mad(s.lat_speed_mps.to_numpy())
        xs2.append(lo + 2.5); ys2.append(m)
        line += f"{m:>7.3f}"
    say(line)
    if len(xs2) >= 2:
        ax.plot(xs2, ys2, color=colour(cn), linewidth=2.0, marker="o",
                markersize=5, markerfacecolor=SURFACE,
                markeredgecolor=colour(cn), markeredgewidth=1.6, label=cn)
leg = ax.legend(frameon=False, fontsize=9)
for t_ in leg.get_texts():
    t_.set_color(INK_2)
style(ax, "longitudinal speed  (km/h)", "weaving intensity σ  (m/s)",
      "Does weaving depend on how fast you are going?")
say("")
say("  Rising to the right: vehicles weave more when moving faster.")
say("  Falling to the right: vehicles weave most when held up in queues.")
say("  Flat: weaving is a habit, independent of speed.")

# (b) free vs following
if have_free:
    ax = axes[0][1]
    say("")
    say("(b) Weaving intensity when FREE versus when FOLLOWING")
    say("")
    say(f"{'class':<11}{'FREE σ':>10}{'FOLLOWING σ':>14}{'change':>10}")
    say("-" * 78)
    labels, fr_v, fo_v = [], [], []
    for cn in present:
        t = traj[traj.class_name == cn]
        a = t[~t.following].lat_speed_mps.to_numpy()
        b = t[t.following].lat_speed_mps.to_numpy()
        if len(a) < 200 or len(b) < 200:
            continue
        _, ma = sd_mad(a); _, mb = sd_mad(b)
        labels.append(cn); fr_v.append(ma); fo_v.append(mb)
        ch = (mb - ma) / ma * 100 if ma > 1e-9 else np.nan
        say(f"{cn:<11}{ma:>10.3f}{mb:>14.3f}{ch:>9.0f} %")
    y = np.arange(len(labels))
    ax.barh(y + 0.19, fr_v, height=0.34, color=INK_3, alpha=0.55,
            edgecolor=SURFACE, linewidth=1.5)
    for i, cn in enumerate(labels):
        ax.barh(y[i] - 0.19, fo_v[i], height=0.34, color=colour(cn),
                edgecolor=SURFACE, linewidth=1.5)
    # Direct labels on the first pair rather than a legend - a legend swatch
    # here would carry a class colour and read as if it named a class.
    if labels:
        ax.text(fr_v[0] + 0.006, y[0] + 0.19, "free", va="center",
                fontsize=9, color=INK_2)
        ax.text(fo_v[0] + 0.006, y[0] - 0.19, "following", va="center",
                fontsize=9, color=colour(labels[0]))
    ax.set_xlim(right=max(max(fr_v), max(fo_v)) * 1.22)
    ax.set_yticks(y); ax.set_yticklabels(labels, color=INK_1, fontsize=10)
    ax.grid(axis="y", visible=False)
    style(ax, "weaving intensity σ  (m/s)", "",
          "Does being blocked make vehicles weave more?")
    say("")
    say("  If the coloured bar is longer than the grey one, that class weaves")
    say("  MORE when something is in front of it - weaving is a response to")
    say("  being blocked, which is what a lane-change model assumes.")
    say("  If it is shorter, the class weaves when it has room, not when")
    say("  it is stuck - a different mechanism, and worth saying so.")

fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.text(0.02, 0.012,
         "Weaving intensity is the robust width (1.4826 x median absolute "
         "deviation) of the lateral-speed distribution.\nThe robust width is "
         "used rather than the standard deviation so that a handful of large "
         "manoeuvres cannot\nmove the answer on its own.",
         color=INK_3, fontsize=8.5)
save(fig, "fig12_weaving_explained.png")

# ==============================================================================
rule("WHAT TO SAY TO THE SUPERVISOR")
say("1. The weaving of each class was measured as the distribution of lateral")
say("   speed - the sideways component of velocity at every instant.")
say("")
say("2. It is centred on zero for every class, so there is no systematic")
say("   drift towards either kerb. A normal distribution is therefore a")
say("   reasonable first model.")
say("")
say("3. But it is NOT normal. The spread measured in the ordinary way is")
say("   about two to three times the spread measured robustly, and the excess")
say("   kurtosis is large and positive. That means the distribution is a")
say("   MIXTURE: a narrow normal core - continuous small wobble - plus a")
say("   heavy tail of deliberate manoeuvres.")
say("")
say("4. This matters for the simulation. VISSIM reproduces the two parts with")
say("   two different mechanisms: the lateral driving-behaviour settings")
say("   produce the wobble, and the overtaking rules produce the tail. They")
say("   have to be calibrated against the two parts SEPARATELY - matching")
say("   only the overall standard deviation would get both wrong.")
say("")
say("5. The honest limitation: measurement noise also widens the lateral")
say("   speed distribution. It inflates the core, not the tail, and it acts")
say("   on every class equally - so the RANKING between classes and the")
say("   comparison between free and following are sound, while the absolute")
say("   width of the core is an upper bound.")

try:
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\n[saved to {OUT_TXT}]")
except Exception as e:
    print(f"[warn] {e}")
