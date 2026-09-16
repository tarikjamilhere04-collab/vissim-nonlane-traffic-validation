#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 3D - THE VISSIM SETUP SHEET
================================================================================
 WHAT THIS IS
   VISSIM Free Version has no COM interface, so every parameter is typed in by
   hand. This produces the sheet to type from: the exact menu path, the exact
   field, and the exact number, for every value this study measured - plus an
   explicit list of what to leave alone and the sentence to write about it.

   Every number is recomputed here from the result files, with the same
   constants and the same physical gates as Phase 2 and Phase 3C. Nothing is
   transcribed, so the sheet cannot drift away from the report.

 RUN ORDER
   phase3a_figures.py  ->  phase3c_desired_speed.py  ->  THIS  ->  phase3b_report.py

 INPUT   trajectories_filtered.csv, phase2_pairs.csv, phase2_lateral.csv,
         phase2c_manoeuvres.csv, phase3c_desired_speed.csv
 OUTPUT  VISSIM_Setup_Sheet.html   (open in a browser; Ctrl+P for a printout
                                    to keep beside the keyboard)
================================================================================
"""

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
BASE     = r"S:\soscho"
OUT_HTML = os.path.join(BASE, "VISSIM_Setup_Sheet.html")

DT             = 0.03332
FOLLOW_DV_MPS  = 0.50
MIN_SAMPLES    = 30
CLASS_ORDER    = ["Car", "Rickshaw", "Bike", "CNG", "Truck", "Bus"]
DIMS = {"Bike": (1.90, 0.70), "CNG": (2.60, 1.40), "Rickshaw": (2.00, 1.20),
        "Car": (4.40, 1.70), "Truck": (7.50, 2.40), "Bus": (11.0, 2.50)}
ROAD_WIDTH_M = 10.00

# VISSIM Wiedemann-99 defaults, for the "what changed" column
VISSIM_DEF = {"CC0": 1.50, "CC1": 0.90, "CC2": 4.00, "CC3": -8.00, "CC4": -0.35,
              "CC5": 0.35, "CC6": 11.44, "CC7": 0.25, "CC8": 3.50, "CC9": 1.50}
# The clearance to use where the measurement is not usable. Declared, not measured.
LATERAL_FLOOR_M = 0.20
# ==============================================================================

DOC = []


def md(s=""):
    DOC.append(str(s))


def h1(s):
    md(f"\n# {s}\n")


def h2(s):
    md(f"\n## {s}\n")


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
            sys.exit(f"ERROR: {p} not found.\n"
                     f"Run the earlier phases first - see the run order at the "
                     f"top of this script.")
        print(f"   [skip] {name} not found")
        return None
    print(f"   read  {name}")
    return pd.read_csv(p)


print("PHASE 3D | VISSIM SETUP SHEET")
traj = load("trajectories_filtered.csv", required=True)
ds = load("phase3c_desired_speed.csv", required=True)
pairs = load("phase2_pairs.csv")
lat = load("phase2_lateral.csv")
man = load("phase2c_manoeuvres.csv")

idcol = "track_id" if "track_id" in traj.columns else "vehicle_id"
full = traj.copy()
if "direction" in traj.columns:
    traj = traj[traj.direction == "with_flow"]
if "edge" in traj.columns:
    traj = traj[traj.edge == 0]
present = [c for c in CLASS_ORDER if c in set(traj.class_name)]
DS = {r.class_name: r for r in ds.itertuples()}

# ------------------------------------------------------------------------------
# PROVENANCE
# ------------------------------------------------------------------------------
_key = traj.set_index([idcol, "frame"])
if pairs is not None and len(pairs):
    k = min(400, len(pairs))
    r = np.random.default_rng(0).choice(len(pairs), k, replace=False)
    idx = pd.MultiIndex.from_arrays([pairs.follower_id.to_numpy()[r],
                                     pairs.frame.to_numpy()[r]])
    got = _key["speed_mps"].reindex(idx).to_numpy(dtype=float)
    d = np.abs(got - pairs.v_follower.to_numpy()[r])
    d = d[np.isfinite(d)]
    med = float(np.median(d)) if len(d) >= 20 else np.inf
    print(f"   provenance: phase2_pairs median mismatch {med:.3f}  "
          f"{'OK' if med <= 0.10 else 'FAIL'}")
    if med > 0.10:
        sys.exit("STOP: the result files are from different runs. Re-run "
                 "Phase 2 against this trajectory file.")

# ==============================================================================
h1("VISSIM setup sheet")
md(f"*Non-lane-based mixed traffic, Uttara / Zamzam Tower - "
   f"generated {datetime.now():%d %B %Y}*")
md("")
md("Every number below was measured in this study unless the row says "
   "otherwise. Work down the sheet in order; VISSIM needs the distributions "
   "and vehicle types to exist before the composition and the driving "
   "behaviour can refer to them.")
md("")
md("**Before anything else, one modelling decision.** The carriageway is "
   "modelled as **one wide lane** of "
   f"**{ROAD_WIDTH_M:.2f} m**, not as several narrow ones. Lateral freedom "
   "then comes from the lateral driving-behaviour settings in Step 6. "
   "Splitting the road into lanes would impose exactly the structure this "
   "thesis is about the absence of, and the weaving in the observed data "
   "could not appear.")

# ==============================================================================
h2("Step 1 - Desired speed distributions")
md("**Base Data > Distributions > Desired Speed > New**, one per class. "
   "Name each one after the class. VISSIM asks for a minimum and a maximum "
   "and spreads drivers between them.")
md("")
rows, not_credible = [], []
for cn in present:
    r = DS.get(cn)
    if r is None:
        rows.append([cn, "--", "--", "--", "no free-flow samples - use a "
                     "literature value and declare it"])
        continue
    warn = str(getattr(r, "warn", "") or "")
    if warn.startswith("NOT CREDIBLE"):
        not_credible.append(cn)
        rows.append([cn, f"~~{r.v_p15:.1f}~~", f"~~{r.v_p95:.1f}~~",
                     f"{r.v_p50:.1f}", "**DO NOT ENTER** - see the note below"])
    else:
        rows.append([cn, f"**{r.v_p15:.1f}**", f"**{r.v_p95:.1f}**",
                     f"{r.v_p50:.1f}", f"measured, n = {int(r.n_free):,}"])
table(["Class", "Minimum (km/h)", "Maximum (km/h)", "median, for reference",
       "status"], rows)
md("Measured on **free-flowing vehicles only** - those with no leader within "
   "15 m. That is what a desired speed means: the speed chosen when nothing "
   "is in the way. VISSIM slows vehicles down again by itself through the "
   "car-following model.")
if not_credible:
    md("")
    md(f"> **{', '.join(not_credible)}** must not be entered from this data. "
       "A motorised vehicle with nothing in front of it does not choose to "
       "crawl - that number is measuring parked or mis-detected vehicles, not "
       "drivers. Use a published value for Dhaka traffic, or drop the class "
       "from the model, and say in the thesis which you did and why.")

# ==============================================================================
h2("Step 2 - Vehicle types")
md("**Base Data > Vehicle Types**. For each class set the model dimensions "
   "and attach the desired-speed distribution from Step 1.")
md("")
rows = []
for cn in present:
    L, W = DIMS.get(cn, (3.0, 1.5))
    r = DS.get(cn)
    c8 = fmt(getattr(r, "cc8_free", np.nan)) if r is not None else "--"
    rows.append([cn, f"{L:.2f}", f"{W:.2f}", cn + " (Step 1)", c8])
table(["Vehicle type", "Length (m)", "Width (m)", "Desired speed distribution",
       "Max accel at 0 km/h (m/s²)"], rows)
md("> **Length and width are assumptions, not measurements** - they come from "
   "a vehicle catalogue, not from this road. They are also the single largest "
   "source of error in the study: every following gap is "
   "`centre distance − half lengths`, and every lateral clearance is "
   "`centre distance − half widths`, so an error here moves CC0 and the "
   "minimum lateral distance directly. Declare them in the thesis as assumed "
   "inputs.")
md("")
md("**Acceleration functions.** In *Functions > Maximum Acceleration*, the "
   "left-hand end of the curve (0 km/h) is the value in the last column - the "
   "85th percentile of acceleration observed on **free-flowing** vehicles "
   "starting off. Leave the shape of the curve at the default and move only "
   "that end point; the data does not constrain the high-speed end, because "
   "this road never reaches those speeds.")

# ==============================================================================
h2("Step 3 - Vehicle classes and composition")
md("**Base Data > Vehicle Classes**: one class per vehicle type, so that "
   "driving behaviour can be assigned separately in Step 5. That separation "
   "is not optional here - the measured gaps differ by class, and a single "
   "behaviour set cannot reproduce that.")
md("")
md("**Base Data > Vehicle Compositions > New**. Add one row per vehicle type "
   "and enter the relative flow:")
md("")
comp = traj.groupby("class_name")[idcol].nunique()
comp = comp.reindex([c for c in CLASS_ORDER if c in comp.index])
tot = int(comp.sum())
table(["Vehicle type", "Relative flow (%)", "Tracks observed"],
      [[c, f"**{100*comp[c]/tot:.1f}**", int(comp[c])] for c in comp.index])
md("Relative flows need not sum to 100 in VISSIM - it normalises them - but "
   "entering them as percentages keeps the sheet readable.")

# ==============================================================================
h2("Step 4 - Links and vehicle inputs")
occupied_s = float(full.frame.nunique() * DT)
flow = full[idcol].nunique() / occupied_s * 3600.0 if occupied_s > 0 else np.nan
table(["Where", "Field", "Value"],
      [["Link (right-click > Edit)", "Number of lanes", "**1**"],
       ["Link", "Lane width", f"**{ROAD_WIDTH_M:.2f} m**"],
       ["Link", "Behaviour type", "the one built in Step 5"],
       ["Vehicle Inputs", "Volume (veh/h)", f"**{flow:,.0f}**"],
       ["Vehicle Inputs", "Vehicle composition", "the one from Step 3"]])
md(f"The volume is the count of distinct vehicles entering the field of view "
   f"divided by {occupied_s/60:.1f} minutes of recording. A vehicle that left "
   "the view and came back is counted twice, so treat it as an **upper "
   "estimate** and confirm it against a manual count of the same clip before "
   "it goes in the thesis.")

# ==============================================================================
h2("Step 5 - Driving behaviour: the Following (W99) tab")
md("**Base Data > Driving Behaviors > New**, one per class, then *Following* "
   "and set the car-following model to **Wiedemann 99**.")
md("")
w99 = {}
if pairs is not None:
    for cn in present:
        pf = pairs[pairs.follower_class == cn]
        eq = pf[(pf.dv.abs() < FOLLOW_DV_MPS + 0.1) & (pf.v_follower >= 0)
                & (pf.v_follower < 8.0) & (pf.gap_m > -0.5) & (pf.gap_m < 25.0)]
        cc0 = cc1 = np.nan
        if len(eq) >= MIN_SAMPLES * 4 and theilslopes is not None:
            gy = eq.gap_m.to_numpy(); vx = eq.v_follower.to_numpy()
            fits = []
            if len(gy) <= 2500:
                fits.append(theilslopes(gy, vx))
            else:
                rng = np.random.default_rng(12345)
                for _ in range(5):
                    k = rng.choice(len(gy), 2500, replace=False)
                    fits.append(theilslopes(gy[k], vx[k]))
            sl = float(np.median([f[0] for f in fits]))
            ic = float(np.median([f[1] for f in fits]))
            if 0.0 <= ic <= 5.0 and 0.20 <= sl <= 3.00:
                cc0, cc1 = ic, sl
        w99[cn] = (cc0, cc1)

# ---- CC8 / CC9, with two rules applied -------------------------------------
# 1. A class whose desired speed was judged NOT CREDIBLE in Phase 3C has no
#    credible acceleration either: both come from the same tracks. If those
#    vehicles are parked with a drifting detection box, the "acceleration" is
#    box jitter, not a driver.
# 2. Where a class has no usable value, the VISSIM default is NOT a neutral
#    choice here - CC8 3.50 m/s2 is roughly four times anything measured on
#    this road, so a class carrying the default would launch unlike every
#    other vehicle in the model. The median of the classes that WERE measured
#    is used instead, and labelled a declared stand-in.
acc = {}
for cn in present:
    r = DS.get(cn)
    bad = str(getattr(r, "warn", "") or "").startswith("NOT CREDIBLE") if r is not None else True
    c8 = getattr(r, "cc8_free", np.nan) if r is not None else np.nan
    c9 = getattr(r, "cc9_free", np.nan) if r is not None else np.nan
    if bad:
        c8 = c9 = np.nan
    acc[cn] = [c8, c9]
m8 = np.nanmedian([v[0] for v in acc.values()]) if any(
    np.isfinite(v[0]) for v in acc.values()) else np.nan
m9 = np.nanmedian([v[1] for v in acc.values()]) if any(
    np.isfinite(v[1]) for v in acc.values()) else np.nan

FIXED = ["CC2", "CC3", "CC4", "CC5", "CC6", "CC7"]
rows = []
for cn in present:
    cc0, cc1 = w99.get(cn, (np.nan, np.nan))
    c8, c9 = acc[cn]
    row = [cn,
           f"**{cc0:.2f}**" if np.isfinite(cc0) else f"*{VISSIM_DEF['CC0']:.2f}*",
           f"**{cc1:.2f}**" if np.isfinite(cc1) else f"*{VISSIM_DEF['CC1']:.2f}*"]
    for k in FIXED:
        row.append(f"*{VISSIM_DEF[k]:.2f}*")
    row.append(f"**{c8:.2f}**" if np.isfinite(c8)
               else (f"{m8:.2f} †" if np.isfinite(m8) else f"*{VISSIM_DEF['CC8']:.2f}*"))
    row.append(f"**{c9:.2f}**" if np.isfinite(c9)
               else (f"{m9:.2f} †" if np.isfinite(m9) else f"*{VISSIM_DEF['CC9']:.2f}*"))
    rows.append(row)
table(["Behaviour", "CC0 (m)", "CC1 (s)", "CC2 (m)", "CC3 (s)", "CC4", "CC5",
       "CC6", "CC7 (m/s²)", "CC8 (m/s²)", "CC9 (m/s²)"], rows)
md("**Bold = measured in this study.** *Italic = the VISSIM default*, used "
   "because the measurement failed a physical test, or there were too few "
   "samples, or the parameter is not observable from overhead trajectories at "
   "all. **† = a declared stand-in**, explained below. Which is which must be "
   "stated in the thesis; a table of parameters that does not say where each "
   "one came from is not a result.")
md("")
md("> **Check the italic values against your own VISSIM dialog before typing "
   "them.** The defaults above are the standard Wiedemann-99 set, but they "
   "can differ between versions, and VISSIM already pre-fills them - so for "
   "every italic cell the safest action is simply to **leave the field "
   "untouched** rather than retype it.")
if np.isfinite(m8):
    st = [c for c in present if not np.isfinite(acc[c][0])]
    if st:
        md("")
        md(f"**On the † stand-in.** {', '.join(st)} produced no usable "
           f"free-flow acceleration. Leaving the VISSIM default there would "
           f"be worse than it looks: CC8 = {VISSIM_DEF['CC8']:.2f} m/s² is "
           f"about four times anything measured on this road, so that class "
           f"would launch quite unlike every other vehicle in the model and "
           f"the mix would be wrong for a reason that has nothing to do with "
           f"behaviour. The median of the classes that *were* measured "
           f"({m8:.2f} and {m9:.2f} m/s²) is used instead. It is an "
           f"assumption, not a measurement, and must be labelled as one.")
md("")
md("On CC0 and CC1: they were fitted jointly by robust regression of gap on "
   "speed, because a congested street contains almost no frames in which "
   "leader and follower are both fully stopped. Any fit implying a negative "
   "standstill distance, or a gap that shrinks with speed, was **rejected** - "
   "those classes carry the default.")
md("")
md(f"On CC2: left at the VISSIM default of {VISSIM_DEF['CC2']:.2f} m. The "
   "data gives only an upper bound on it, because the spread around the "
   "fitted line contains real oscillation plus measurement error plus "
   "driver-to-driver variation, and this data cannot separate them.")
md("")
md("On CC8 and CC9: measured on free-flowing vehicles only, for the same "
   "reason as desired speed. They come out **below** the VISSIM defaults "
   "(CC8 3.50, CC9 1.50). That is a finding about this road, not a mistake: "
   "on a congested Dhaka carriageway even an unobstructed vehicle does not "
   "launch hard. Quote them with the acceleration accuracy of the study "
   "(r = 0.70 against known truth).")
md("")
md("**CC3, CC4, CC5, CC6, CC7: leave every one untouched.** They govern the "
   "perception thresholds inside the Wiedemann model - when a driver notices "
   "they are closing, and how much speed difference they tolerate before "
   "reacting. None of that is observable from overhead trajectories at this "
   "resolution. Suggested wording: *\"CC3–CC7 were retained at their default "
   "values, as the perception thresholds they represent cannot be identified "
   "from vision-based trajectory data.\"*")
md("")
md("What each fixed parameter means, for the thesis text:")
table(["Parameter", "Meaning", "Why it stays at the default"],
      [["CC2 (m)", "following oscillation - how much the gap is allowed to "
        "wander before the driver corrects",
        "the data gives only an upper bound; the spread around the fitted "
        "line mixes real oscillation with measurement error"],
       ["CC3 (s)", "how long before reaching the safety distance the driver "
        "starts to decelerate", "a perception threshold, not a distance"],
       ["CC4 / CC5", "the negative and positive speed-difference thresholds "
        "during close following",
        "needs speed differences resolved far finer than r = 0.70 "
        "acceleration allows"],
       ["CC6 (1/(m·s))", "how the speed oscillation grows with distance",
        "not identifiable over a 29 m observation window"],
       ["CC7 (m/s²)", "the small acceleration used during oscillation",
        "below the noise floor of the recovered acceleration"]])

# ==============================================================================
h2("Step 6 - Driving behaviour: the Lateral tab")
md("This tab is what makes the model non-lane-based. It matters more here "
   "than any W99 value.")
md("")
lat_rows, floor_classes = [], []
if lat is not None:
    for cn in present:
        s = lat[(lat.class_a == cn) | (lat.class_b == cn)]
        if len(s) < 100:
            lat_rows.append([cn, "--", f"{LATERAL_FLOOR_M:.2f} *(declared)*",
                             "too few side-by-side samples"])
            floor_classes.append(cn)
            continue
        c = s.lat_clear_m.to_numpy()
        p05 = q(c, 5)
        neg = 100 * (c < 0).mean()
        if p05 > 0.05:
            lat_rows.append([cn, fmt(p05), f"**{p05:.2f}**",
                             f"measured, {neg:.0f} % impossible"])
        else:
            floor_classes.append(cn)
            lat_rows.append([cn, fmt(p05), f"{LATERAL_FLOOR_M:.2f} *(declared)*",
                             f"**{neg:.0f} % of samples impossible** - not usable"])
table(["Class", "Measured p05 (m)", "Enter as min. lateral distance",
       "status"], lat_rows)
if floor_classes:
    md(f"A measured 5th percentile at or below zero says the two vehicle "
       f"bodies overlap, which cannot happen. It is a symptom of the assumed "
       f"**width**, not a behaviour. For those classes enter the declared "
       f"floor of **{LATERAL_FLOOR_M:.2f} m** and label it in the thesis as an "
       f"assumption. Do not enter a negative number, and do not silently clip "
       f"it to zero and call it measured.")
md("")
md("Set the same value at 0 km/h and at 50 km/h unless you have a reason not "
   "to - this road never reaches 50 km/h, so the second field is an "
   "extrapolation either way.")
md("")
otl = None
if man is not None and "ot_side" in getattr(man, "columns", []):
    ot = man[man.overtake == True] if "overtake" in man.columns else man.iloc[0:0]
    if len(ot):
        sh = ot.ot_side.value_counts(normalize=True) * 100
        otl = sh.to_dict()
table(["Field", "Setting", "Why"],
      [["Desired position at free flow", "**Any**",
        "vehicles do not seek a lane centre on this road"],
       ["Overtake on same lane: on the left", "**ticked**",
        (f"{otl.get('left', 0):.0f} % of observed overtakes passed on the left"
         if otl else "observed in the data")],
       ["Overtake on same lane: on the right", "**ticked**",
        (f"{otl.get('right', 0):.0f} % passed on the right - less common, but "
         "real, so it must be allowed" if otl else "observed in the data")],
       ["Consider next turning direction", "**unticked**",
        "there are no lanes to line up in"],
       ["Diamond shaped queuing", "**unticked**",
        "vehicles queue side by side here, not in a diamond"]])
if otl:
    md("Both directions are ticked even though the left dominates. A model "
       "that allows overtaking on one side only cannot reproduce the observed "
       "mix; the *preference* emerges from the geometry, and that is what "
       "calibration should be checked against.")

# ==============================================================================
h2("Step 7 - What to record, so the model can be calibrated")
md("Set these up before the first run, or the run has to be repeated.")
md("")
mrows = []
if man is not None and len(man):
    expo = traj.groupby([idcol, "class_name"]).size().reset_index(name="n")
    expo["sec"] = expo.n * DT
    sec = expo.groupby("class_name").sec.sum()
    for cn in present:
        k = int((man.cls == cn).sum())
        s = float(sec.get(cn, 0.0))
        if s > 30 and k >= 15:
            mrows.append([cn, f"{k / (s / 60.0):.2f}", f"n = {k}"])
table(["Evaluation", "What it gives", "Target from the observed data"],
      [["Data Collection Points", "flow and speed at a cross-section",
        f"{flow:,.0f} veh/h; class medians from Step 1"],
       ["Vehicle Record", "per-vehicle trajectories out of VISSIM",
        "lets you rebuild the same Phase 2/2C statistics on the simulated "
        "output and compare like with like"],
       ["Vehicle Travel Times", "section travel time",
        "derived from the observed speed distribution"]])
if mrows:
    md("")
    md("**The single best calibration target** is the sideways-manoeuvre rate "
       "per vehicle-minute, because it is scale-free: it depends on the "
       "ordering of lateral speeds, not on their absolute calibration, so it "
       "survives any residual error in the lateral scale.")
    md("")
    table(["Class", "Target (manoeuvres per vehicle-minute)", "evidence"], mrows)
    md("Classes with fewer than 15 observed manoeuvres are left out - their "
       "rates are noise and calibrating to them would be fitting noise.")

# ==============================================================================
h2("Step 8 - The honest checklist before you show a result")
for s in [
    "Every parameter in the model is either measured here, or a declared "
    "default - and the thesis says which, for each one.",
    "Lengths and widths are labelled as assumed inputs, not results.",
    "CC0 appears only for classes whose fit passed the physical gate.",
    "Desired speed came from free-flowing vehicles, and the thesis says so.",
    "No negative lateral distance was entered anywhere.",
    "The vehicle input was checked against a manual count, not just the "
    "trajectory count.",
    "Any class the data could not support was dropped or given a declared "
    "literature value, and the choice is stated.",
]:
    md(f"- [ ] {s}")

md("")
md("---")
md(f"*Generated {datetime.now():%d %B %Y, %H:%M} from the result files in "
   f"`{BASE}`. Every value is recomputed at generation time.*")

# ==============================================================================
# RENDER
# ==============================================================================
def inline(s):
    s = _html.escape(s)
    while s.count("**") >= 2:
        s = s.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    while s.count("~~") >= 2:
        s = s.replace("~~", "<del>", 1).replace("~~", "</del>", 1)
    while s.count("*") >= 2:
        s = s.replace("*", "<em>", 1).replace("*", "</em>", 1)
    while s.count("`") >= 2:
        s = s.replace("`", "<code>", 1).replace("`", "</code>", 1)
    return s


def to_html(text):
    out, in_tbl = [], False
    for line in text.split("\n"):
        s = line.rstrip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            if not in_tbl:
                out.append("<table>"); in_tbl = True; tag = "th"
            else:
                tag = "td"
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>"
                                        for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>"); in_tbl = False
        if s.strip() == "---":
            out.append("<hr>")
        elif s.startswith("## "):
            out.append(f"<h2>{inline(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h1>{inline(s[2:])}</h1>")
        elif s.startswith("> "):
            out.append(f"<blockquote>{inline(s[2:])}</blockquote>")
        elif s.startswith("- [ ] "):
            out.append(f'<div class="chk"><span class="box"></span>'
                       f'{inline(s[6:])}</div>')
        elif s.startswith("- "):
            out.append(f"<ul><li>{inline(s[2:])}</li></ul>")
        elif s.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{inline(s)}</p>")
    if in_tbl:
        out.append("</table>")
    return "\n".join(out)


CSS = """
body{font:15px/1.65 Georgia,'Times New Roman',serif;color:#15150f;
 background:#faf9f5;max-width:860px;margin:0 auto;padding:44px 28px 90px}
h1{font:600 26px/1.3 Georgia,serif;margin:10px 0 14px;color:#0b0b0b;
 border-bottom:2px solid #d8d6cf;padding-bottom:10px}
h2{font:600 19px/1.35 Georgia,serif;margin:38px 0 10px;color:#0b0b0b;
 background:#f0eee7;padding:9px 14px;border-left:4px solid #2a78d6}
p{margin:10px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font:13.5px/1.5
 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;background:#fff}
th{background:#f0eee7;text-align:left;font-weight:600;color:#0b0b0b}
th,td{border:1px solid #e0ded6;padding:7px 10px;vertical-align:top}
tr:nth-child(even) td{background:#fcfbf8}
td strong{color:#08306b;font-size:14.5px}
code{font:13px ui-monospace,Menlo,Consolas,monospace;background:#f0eee7;
 padding:1px 5px;border-radius:3px}
blockquote{border-left:3px solid #b3261e;background:#fff;margin:14px 0;
 padding:10px 16px;color:#3a3a35}
ul{margin:6px 0 6px 22px}
hr{border:0;border-top:1px solid #d8d6cf;margin:28px 0}
.chk{margin:7px 0;padding-left:30px;position:relative;font-size:14.5px}
.chk .box{position:absolute;left:0;top:3px;width:15px;height:15px;
 border:1.6px solid #52514e;border-radius:2px;background:#fff}
del{color:#8a8880}
@media print{body{background:#fff;max-width:none;padding:0;font-size:12px}
 h2{page-break-after:avoid}table{page-break-inside:avoid}}
"""

doc = ("<!doctype html><html><head><meta charset='utf-8'>"
       "<meta name='viewport' content='width=device-width,initial-scale=1'>"
       "<title>VISSIM Setup Sheet</title><style>" + CSS + "</style></head>"
       "<body>" + to_html("\n".join(DOC)) + "</body></html>")
try:
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\n   wrote {OUT_HTML}")
except Exception as e:
    sys.exit(f"[error] {e}")

print("\nDone. Open it in a browser and keep it beside the keyboard while you")
print("set VISSIM up. Ctrl+P gives a printable copy with tick boxes.")
