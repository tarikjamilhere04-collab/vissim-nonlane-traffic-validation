#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 3A - FIGURES FOR THE THESIS AND THE SUPERVISOR MEETING
================================================================================
 Produces eight publication-quality figures as PNG (300 dpi) from the files the
 pipeline has already written. Nothing is recomputed from scratch - every figure
 draws the same numbers that appear in the Phase 2 / 2C reports, so a figure and
 the text can never disagree.

 FIGURES
   fig1_speed_distribution.png    per-class speed bands - the VISSIM desired
                                  speed input, drawn as p15/p50/p85/p95
   fig2_gap_vs_speed.png          the Wiedemann equilibrium, one panel per class,
                                  with the fitted CC0 + CC1*v line
   fig3_lateral_clearance.png     how close vehicles run side by side
   fig4_manoeuvre_rate.png        sideways manoeuvres per vehicle-minute
   fig5_lanechange_speed.png      AT WHAT SPEED do vehicles change position
   fig6_leader_effect.png         gap kept, by follower x leader class
   fig7_acceleration.png          acceleration against speed, per class
   fig8_space_time.png            a space-time diagram - the classic traffic
                                  figure, and the one that shows weaving

 INPUT   trajectories_filtered.csv, phase2_pairs.csv, phase2_lateral.csv,
         phase2c_manoeuvres.csv
 OUTPUT  S:\\soscho\\figures\\*.png
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
    from matplotlib.ticker import MaxNLocator
except ImportError:
    sys.exit("ERROR: matplotlib required.  pip install matplotlib")

try:
    from scipy.stats import theilslopes
except ImportError:
    theilslopes = None

# ==============================================================================
BASE      = r"S:\soscho"
FIG_DIR   = os.path.join(BASE, "figures")
DT        = 0.03332
DPI       = 300
# ==============================================================================

# Validated categorical palette, assigned in fixed order and never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK_1, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e6e5e1"
SURFACE = "#ffffff"
# Status colour, reserved for "this result is not usable" - never a series hue.
BAD = "#b3261e"

# The SAME physical gate Phase 2 applies. A fit outside these bounds is not
# reported as a parameter, so the figure must not print it as one either.
GATE_CC0 = (0.0, 5.0)
GATE_CC1 = (0.20, 3.00)

CLASS_ORDER = ["Car", "Rickshaw", "Bike", "CNG", "Truck", "Bus"]


def colour(cn):
    return PALETTE[CLASS_ORDER.index(cn) % len(PALETTE)] if cn in CLASS_ORDER else INK_3


def style(ax, xlabel="", ylabel="", title=""):
    """Recessive grid and axes; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK_1, fontsize=11.5, loc="left", pad=10,
                     fontweight="medium")


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"   wrote {p}")


def need(path):
    if not os.path.exists(path):
        print(f"   [skip] {os.path.basename(path)} not found")
        return None
    return pd.read_csv(path)


print("PHASE 3A | FIGURES")
traj = need(os.path.join(BASE, "trajectories_filtered.csv"))
pairs = need(os.path.join(BASE, "phase2_pairs.csv"))
lat = need(os.path.join(BASE, "phase2_lateral.csv"))
man = need(os.path.join(BASE, "phase2c_manoeuvres.csv"))
if traj is None:
    sys.exit("ERROR: trajectories_filtered.csv is required.")
idcol = "track_id" if "track_id" in traj.columns else "vehicle_id"
if "direction" in traj.columns:
    traj = traj[traj.direction == "with_flow"]
if "edge" in traj.columns:
    traj = traj[traj.edge == 0]
present = [c for c in CLASS_ORDER if c in set(traj.class_name)]
print(f"   classes present: {', '.join(present)}")

# ------------------------------------------------------------------------------
# PROVENANCE GUARD
#   Every figure mixes the trajectory file with the Phase 2 / 2C result files.
#   If those came from different runs - real trajectories with synthetic
#   manoeuvres, say - the figures would be silently, plausibly wrong. The track
#   ids must overlap. This checks that they do and refuses to draw if they do not.
# ------------------------------------------------------------------------------
#   Comparing id LISTS is not enough - track ids are small integers and two
#   unrelated runs share them by coincidence. So this looks up each result row
#   in the trajectory file and checks the VALUE agrees: a manoeuvre that claims
#   the vehicle was at lateral position y at frame f must find that vehicle at
#   that position at that frame. Only the same run can pass that.
_key = traj.set_index([idcol, "frame"])
_bad = []


def _agree(name, ids, frames, vals, col, tol):
    """Median absolute disagreement between a result file and the trajectories."""
    k = min(400, len(ids))
    r = np.random.default_rng(0).choice(len(ids), k, replace=False)
    try:
        got = _key.loc[list(zip(np.asarray(ids)[r], np.asarray(frames)[r])), col]
    except KeyError:
        idx = pd.MultiIndex.from_arrays([np.asarray(ids)[r], np.asarray(frames)[r]])
        got = _key[col].reindex(idx)
    d = np.abs(got.to_numpy(dtype=float) - np.asarray(vals, dtype=float)[r])
    d = d[np.isfinite(d)]
    if len(d) < 20:
        print(f"   provenance: {name:<22} could not be checked (too few matches)")
        _bad.append(name)
        return
    med = float(np.median(d))
    ok = med <= tol
    print(f"   provenance: {name:<22} median mismatch {med:6.3f}  "
          f"(tolerance {tol})  {'OK' if ok else 'FAIL'}")
    if not ok:
        _bad.append(name)


if man is not None and len(man):
    _agree("phase2c_manoeuvres.csv", man.track_id, man.f_start, man.lat_start,
           "lat_m", 0.20)
if pairs is not None and len(pairs):
    _agree("phase2_pairs.csv", pairs.follower_id, pairs.frame, pairs.v_follower,
           "speed_mps", 0.10)
if lat is not None and len(lat):
    _agree("phase2_lateral.csv", lat.id_a, lat.frame, lat.v_a, "speed_mps", 0.10)
if _bad:
    print("\n" + "!" * 78)
    print("STOP. These files do not belong to the same run as the trajectories:")
    for b in _bad:
        print("   " + b)
    print("The positions and speeds they record do not match what the trajectory")
    print("file says the vehicle was doing. Drawing figures from a mixed set would")
    print("produce plots that look right and are not. Re-run Phase 2 and Phase 2C")
    print("against the SAME trajectories_filtered.csv, then run this again.")
    print("!" * 78)
    sys.exit(1)
print()


# ==============================================================================
# FIG 1 - speed bands, the VISSIM desired-speed input
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.2))
ypos = np.arange(len(present))[::-1]
for y, cn in zip(ypos, present):
    v = traj[traj.class_name == cn].speed_kmph.to_numpy() \
        if "speed_kmph" in traj else traj[traj.class_name == cn].speed_mps.to_numpy() * 3.6
    p15, p50, p85, p95 = np.percentile(v, [15, 50, 85, 95])
    c = colour(cn)
    # the p15-p95 band is what VISSIM's desired-speed distribution encodes
    ax.plot([p15, p95], [y, y], color=c, linewidth=2.0, solid_capstyle="round",
            alpha=0.35)
    ax.plot([p15, p85], [y, y], color=c, linewidth=6.0, solid_capstyle="round")
    ax.plot([p50], [y], "o", color=SURFACE, markersize=9, zorder=3)
    ax.plot([p50], [y], "o", color=c, markersize=6, zorder=4)
    ax.text(p95 + 0.7, y, f"{p50:.1f}", color=INK_2, fontsize=9,
            va="center", ha="left")
ax.set_yticks(ypos); ax.set_yticklabels(present, color=INK_1, fontsize=10)
ax.grid(axis="y", visible=False)
style(ax, "speed  (km/h)", "",
      "Observed speed by vehicle class")
ax.text(0.0, -0.20, "thick bar: 15th-85th percentile   |   thin bar: to 95th   |   "
        "dot and number: median\nThe 15th-95th band is the VISSIM desired-speed "
        "distribution input.",
        transform=ax.transAxes, color=INK_3, fontsize=8.5, va="top")
save(fig, "fig1_speed_distribution.png")


# ==============================================================================
# FIG 2 - the Wiedemann equilibrium, gap against speed, one panel per class
# ==============================================================================
if pairs is not None:
    cls = [c for c in present if (pairs.follower_class == c).sum() > 300]
    n_rejected = []
    n = len(cls)
    if n:
        ncol = min(3, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow),
                                 squeeze=False)
        for k, cn in enumerate(cls):
            ax = axes[k // ncol][k % ncol]
            t = pairs[(pairs.follower_class == cn) & (pairs.dv.abs() < 0.6)
                      & (pairs.gap_m > -0.5) & (pairs.gap_m < 20)
                      & (pairs.v_follower < 8)]
            c = colour(cn)
            xs, med, lo, hi = [], [], [], []
            for b in np.arange(0, 6.0, 0.5):
                s = t[(t.v_follower >= b) & (t.v_follower < b + 0.5)]
                if len(s) < 40:
                    continue
                xs.append(b + 0.25)
                med.append(np.median(s.gap_m))
                lo.append(np.percentile(s.gap_m, 25))
                hi.append(np.percentile(s.gap_m, 75))
            if xs:
                ax.fill_between(xs, lo, hi, color=c, alpha=0.16, linewidth=0)
                ax.plot(xs, med, color=c, linewidth=2.0, marker="o",
                        markersize=5, markerfacecolor=SURFACE,
                        markeredgecolor=c, markeredgewidth=1.6)
                if theilslopes is not None and len(t) > 200:
                    gy = t.gap_m.to_numpy(); vx = t.v_follower.to_numpy()
                    k2 = np.random.default_rng(1).choice(
                        len(gy), min(2000, len(gy)), replace=False)
                    sl, ic, _, _ = theilslopes(gy[k2], vx[k2])
                    xx = np.array([0, max(xs) + 0.3])
                    ok = (GATE_CC0[0] <= ic <= GATE_CC0[1]
                          and GATE_CC1[0] <= sl <= GATE_CC1[1])
                    if ok:
                        ax.plot(xx, ic + sl * xx, color=INK_2, linewidth=1.4,
                                linestyle="--")
                        ax.text(0.04, 0.94, f"CC0 {ic:.2f} m\nCC1 {sl:.2f} s",
                                transform=ax.transAxes, fontsize=8.5,
                                color=INK_2, va="top")
                    else:
                        # Rejected by the physical gate. Draw it faintly so the
                        # reader can see WHY it was rejected, and say plainly
                        # that it is not a parameter.
                        ax.plot(xx, ic + sl * xx, color=BAD, linewidth=1.2,
                                linestyle=":", alpha=0.75)
                        ax.text(0.04, 0.94,
                                f"raw fit {ic:.2f} m, {sl:.2f} s\n"
                                f"REJECTED - not a parameter",
                                transform=ax.transAxes, fontsize=8.5,
                                color=BAD, va="top")
                        n_rejected.append(cn)
            # label x on any panel with no panel beneath it, not just the last row
            style(ax, "speed  (m/s)" if k + ncol >= n else "",
                  "gap  (m)" if k % ncol == 0 else "", cn)
            ax.set_ylim(bottom=0)
        for k in range(n, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("Following gap against speed  -  the Wiedemann equilibrium",
                     color=INK_1, fontsize=12, x=0.02, ha="left", y=0.995)
        note = ("band: interquartile range   |   line: median   |   "
                "dashed grey: fitted  gap = CC0 + CC1 x v   (Theil-Sen)")
        if n_rejected:
            note += ("\nRed dotted: a fit that fails the physical gate "
                     "(CC0 < 0 or CC1 outside 0.2-3.0 s). Those classes have "
                     "no measured CC0/CC1;\nthe VISSIM default is used and "
                     "the reason is declared. The raw numbers are shown only "
                     "so the failure is visible.")
            fig.tight_layout(rect=[0, 0.11, 1, 0.96])
        else:
            fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        fig.text(0.02, 0.012, note, color=INK_3, fontsize=8.5)
        save(fig, "fig2_gap_vs_speed.png")
        if n_rejected:
            print(f"   note: fits rejected by the physical gate -> "
                  f"{', '.join(n_rejected)}")


# ==============================================================================
# FIG 3 - lateral clearance
# ==============================================================================
if lat is not None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    rows = []
    for cn in present:
        s = lat[((lat.class_a == cn) | (lat.class_b == cn))
                & (lat.lat_clear_m > -1) & (lat.lat_clear_m < 5)]
        if len(s) >= 100:
            rows.append((cn, s.lat_clear_m.to_numpy()))
    ypos = np.arange(len(rows))[::-1]
    xmax = max(np.percentile(v, 75) for _, v in rows)
    any_neg = False
    for y, (cn, v) in zip(ypos, rows):
        c = colour(cn)
        p05, p25, p50, p75 = np.percentile(v, [5, 25, 50, 75])
        negshare = 100.0 * (v < 0).mean()
        ax.plot([p05, p75], [y, y], color=c, linewidth=2.0, alpha=0.35,
                solid_capstyle="round")
        ax.plot([p25, p75], [y, y], color=c, linewidth=6.0,
                solid_capstyle="round")
        ax.plot([p50], [y], "o", color=SURFACE, markersize=9, zorder=3)
        ax.plot([p50], [y], "o", color=c, markersize=6, zorder=4)
        # Labels go to the RIGHT of every row, at a common x, so they cannot
        # collide with the class names however far left a row reaches.
        if p05 < 0:
            any_neg = True
            ax.text(xmax * 1.06, y, f"p05 {p05:.2f}   ({negshare:.0f} % overlap)",
                    color=BAD, fontsize=9, va="center", ha="left")
        else:
            ax.text(xmax * 1.06, y, f"p05 {p05:.2f}", color=INK_2, fontsize=9,
                    va="center", ha="left")
    ax.axvline(0, color=INK_3, linewidth=1.0, linestyle=":")
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows],
                                            color=INK_1, fontsize=10)
    ax.grid(axis="y", visible=False)
    style(ax, "edge-to-edge lateral clearance  (m)", "",
          "How close vehicles run side by side")
    cap = ("Dotted line at zero: bodies touching. p05 is the 5th percentile - "
           "the clearance drivers accept.")
    if any_neg:
        cap += ("\nA p05 below zero is NOT a behaviour: it says the two boxes "
                "overlap, which cannot happen. It means the\nassumed WIDTH for "
                "that class is too large (or the class is mis-detected). Those "
                "classes have no usable\nminimum lateral distance from this "
                "data - see the report.")
    ax.text(0.0, -0.20, cap, transform=ax.transAxes, color=INK_3, fontsize=8.5,
            va="top")
    save(fig, "fig3_lateral_clearance.png")


# ==============================================================================
# FIG 4 - manoeuvre rate
# ==============================================================================
if man is not None and len(man):
    expo = traj.groupby([idcol, "class_name"]).size().reset_index(name="n")
    expo["sec"] = expo.n * DT
    sec = expo.groupby("class_name").sec.sum()
    cls, rate, cnt = [], [], []
    for cn in present:
        k = int((man.cls == cn).sum())
        s = sec.get(cn, 0.0)
        if s > 30 and k >= 5:
            cls.append(cn); rate.append(k / (s / 60.0)); cnt.append(k)
    if cls:
        order = np.argsort(rate)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ypos = np.arange(len(cls))
        for i, o in enumerate(order):
            c = colour(cls[o])
            thin = cnt[o] < 15          # too few events for the rate to mean much
            ax.barh(i, rate[o], height=0.58,
                    color=(SURFACE if thin else c),
                    edgecolor=c, linewidth=(1.6 if thin else 2.0),
                    hatch=("///" if thin else None),
                    alpha=(0.9 if thin else 1.0))
            lbl = f"{rate[o]:.2f}   (n={cnt[o]})"
            if thin:
                lbl += "  too few - do not quote"
            ax.text(rate[o] + 0.05, i, lbl, va="center",
                    color=(BAD if thin else INK_2), fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([cls[o] for o in order], color=INK_1, fontsize=10)
        ax.grid(axis="y", visible=False)
        style(ax, "sideways manoeuvres per vehicle-minute", "",
              "Which classes keep changing position")
        ax.text(0.0, -0.24, "Exposure-corrected, so it does not depend on how "
                "long each track happened to be observed.\nHatched, open bars "
                "have fewer than 15 detected manoeuvres - the rate is noise and "
                "must not be quoted.",
                transform=ax.transAxes, color=INK_3, fontsize=8.5, va="top")
        save(fig, "fig4_manoeuvre_rate.png")


# ==============================================================================
# FIG 5 - AT WHAT SPEED do vehicles change position
# ==============================================================================
if man is not None and len(man) >= 30:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    allv = (traj.speed_kmph.to_numpy() if "speed_kmph" in traj
            else traj.speed_mps.to_numpy() * 3.6)
    allv = allv[(allv > -2) & (allv < 40)]
    mv = man.v_mean.to_numpy() * 3.6
    mv = mv[(mv > -2) & (mv < 40)]
    bins = np.arange(0, 36, 2.0)
    ax.hist(allv, bins=bins, density=True, color=INK_3, alpha=0.30,
            edgecolor=SURFACE, linewidth=1.5, label="all driving")
    ax.hist(mv, bins=bins, density=True, histtype="step", color=PALETTE[1],
            linewidth=2.2, label="while changing position")
    medv = float(np.median(mv))
    ax.axvline(medv, color=PALETTE[1], linewidth=1.2, linestyle="--")
    ax.text(medv + 0.5, ax.get_ylim()[1] * 0.74,
            f"median {medv:.1f} km/h", color=PALETTE[1], fontsize=9)
    style(ax, "speed  (km/h)", "share of samples",
          "At what speed do vehicles move sideways?")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    ax.text(0.0, -0.20, "If the orange curve sits to the RIGHT of the grey one, "
            "drivers weave when moving\nfreely. To the LEFT, they weave when "
            "held up. That distinction drives the VISSIM\nlateral settings.",
            transform=ax.transAxes, color=INK_3, fontsize=8.5, va="top")
    save(fig, "fig5_lanechange_speed.png")


# ==============================================================================
# FIG 6 - leader-class effect
# ==============================================================================
if pairs is not None:
    fc = [c for c in present if (pairs.follower_class == c).sum() > 500]
    lc = [c for c in present if (pairs.leader_class == c).sum() > 500]
    if len(fc) >= 2 and len(lc) >= 2:
        M = np.full((len(fc), len(lc)), np.nan)
        for i, f in enumerate(fc):
            for j, l in enumerate(lc):
                s = pairs[(pairs.follower_class == f) & (pairs.leader_class == l)]
                if len(s) >= 100:
                    M[i, j] = np.median(s.gap_m)
        fig, ax = plt.subplots(figsize=(1.15 * len(lc) + 3.2, 0.85 * len(fc) + 2.6))
        # sequential: one hue, light to dark - magnitude, not identity
        im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=np.nanmin(M),
                       vmax=np.nanmax(M))
        for i in range(len(fc)):
            for j in range(len(lc)):
                if np.isfinite(M[i, j]):
                    rel = (M[i, j] - np.nanmin(M)) / max(np.nanmax(M) - np.nanmin(M), 1e-9)
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                            fontsize=9.5,
                            color=("#ffffff" if rel > 0.6 else INK_1))
        ax.set_xticks(range(len(lc))); ax.set_xticklabels(lc, fontsize=9.5, color=INK_1)
        ax.set_yticks(range(len(fc))); ax.set_yticklabels(fc, fontsize=9.5, color=INK_1)
        ax.set_xlabel("vehicle IN FRONT", color=INK_2, fontsize=10)
        ax.set_ylabel("vehicle BEHIND", color=INK_2, fontsize=10)
        ax.set_title("Median gap kept, by who is in front  (m)", color=INK_1,
                     fontsize=11.5, loc="left", pad=10, fontweight="medium")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK_2, labelsize=9, length=0)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        fig.text(0.02, 0.012, "Darker = more room left. Blank cells: fewer than 100 "
                 "samples. Read across a row to see\nhow one class changes its gap "
                 "depending on what is in front - a real effect VISSIM\ncan only "
                 "reproduce if the driving behaviour is assigned per vehicle class.",
                 color=INK_3, fontsize=8.5)
        save(fig, "fig6_leader_effect.png")


# ==============================================================================
# FIG 7 - acceleration against speed
# ==============================================================================
cls = [c for c in present if (traj.class_name == c).sum() > 2000]
if cls:
    ncol = min(3, len(cls)); nrow = int(np.ceil(len(cls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for k, cn in enumerate(cls):
        ax = axes[k // ncol][k % ncol]
        t = traj[traj.class_name == cn]
        v = (t.speed_kmph.to_numpy() if "speed_kmph" in t
             else t.speed_mps.to_numpy() * 3.6)
        a = t.accel_mps2.to_numpy()
        c = colour(cn)
        xs, p85, p50, p15 = [], [], [], []
        for b in np.arange(0, 30, 2.0):
            s = (v >= b) & (v < b + 2.0)
            if s.sum() < 100:
                continue
            xs.append(b + 1.0)
            p85.append(np.percentile(a[s], 85))
            p50.append(np.percentile(a[s], 50))
            p15.append(np.percentile(a[s], 15))
        if xs:
            ax.fill_between(xs, p15, p85, color=c, alpha=0.18, linewidth=0)
            ax.plot(xs, p85, color=c, linewidth=1.8)
            ax.plot(xs, p15, color=c, linewidth=1.8)
            ax.plot(xs, p50, color=c, linewidth=1.2, linestyle=":")
        ax.axhline(0, color=INK_3, linewidth=0.9)
        style(ax, "speed (km/h)" if k + ncol >= len(cls) else "",
              "accel (m/s$^2$)" if k % ncol == 0 else "", cn)
        ax.tick_params(labelbottom=(k + ncol >= len(cls)))
    for k in range(len(cls), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Acceleration envelope by speed", color=INK_1, fontsize=12,
                 x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.text(0.02, 0.012, "Upper and lower lines: 85th and 15th percentile - the "
             "VISSIM maximum acceleration and\ndeceleration functions. Dotted: "
             "median. Recovered acceleration accuracy r = 0.70 (Phase 1d).",
             color=INK_3, fontsize=8.5)
    save(fig, "fig7_acceleration.png")


# ==============================================================================
# FIG 8 - space-time diagram
# ==============================================================================
# Choose the BUSIEST window rather than the first one. A quiet opening minute
# makes an empty diagram and says nothing about congestion.
WIN_S = 45.0
dur = traj.groupby(idcol).size() * DT
keep_ids = dur[dur > 1.0].index
sub = traj[traj[idcol].isin(keep_ids)]
if len(sub) > 200:
    tsec = sub.frame.to_numpy() * DT
    lo, hi = float(tsec.min()), float(tsec.max())
    if hi - lo <= WIN_S:
        t_start = lo
    else:
        starts = np.arange(lo, hi - WIN_S, max((hi - lo) / 120.0, 1.0))
        counts = [((tsec >= s) & (tsec < s + WIN_S)).sum() for s in starts]
        t_start = float(starts[int(np.argmax(counts))])
    win = sub[(sub.frame * DT >= t_start) & (sub.frame * DT < t_start + WIN_S)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    drawn = set()
    for tid, t in win.groupby(idcol):
        if len(t) < 20:
            continue
        cn = t.class_name.iloc[0]
        c = colour(cn)
        drawn.add(cn)
        tt = t.frame.to_numpy() * DT - t_start
        ax1.plot(tt, t.lon_m.to_numpy(), color=c, linewidth=1.1, alpha=0.85)
        ax2.plot(tt, t.lat_m.to_numpy(), color=c, linewidth=1.1, alpha=0.85)
    style(ax1, "time  (s)", "position along the road  (m)",
          "Space-time: who is following whom")
    style(ax2, "time  (s)", "position across the road  (m)",
          "Lateral position: the weaving")
    handles = [plt.Line2D([], [], color=colour(c), linewidth=2.4, label=c)
               for c in present if c in drawn]
    leg = fig.legend(handles=handles, frameon=False, fontsize=9,
                     loc="upper right", bbox_to_anchor=(0.995, 0.97), ncol=1)
    for tx in leg.get_texts():
        tx.set_color(INK_2)
    fig.tight_layout(rect=[0, 0.14, 0.93, 1])
    fig.text(0.02, 0.015,
             f"Busiest {WIN_S:.0f} s of the record, starting at "
             f"{int(t_start) // 60:d} min {int(t_start) % 60:02d} s. "
             f"{len(set(win[idcol])):d} vehicles shown.\n"
             "Left: parallel lines mean vehicles holding station; converging lines "
             "mean one catching another.\nRight: a line crossing others is a vehicle "
             "moving sideways past them - this is what 'lane\nchanging' looks like "
             "on a road that has no lanes.",
             color=INK_3, fontsize=8.5)
    save(fig, "fig8_space_time.png")

print("\nDone. All figures are in:")
print(f"   {FIG_DIR}")
print("\nThey are 300 dpi PNG, sized for a report page, and drawn from the same")
print("numbers as the Phase 2 / 2C text reports - a figure cannot disagree with")
print("the table it came from.")
