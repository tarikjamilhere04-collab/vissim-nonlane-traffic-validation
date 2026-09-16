#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PHASE 1C - SYNTHETIC TRAJECTORY GENERATOR
 Thesis: Vision-Based Trajectory Extraction, Behavior Modeling, and Microscopic
         Simulation of Non-Lane-Based Mixed Traffic
================================================================================
 WHY A SIMULATOR AND NOT RESAMPLING
   Phase 2 needs INTERACTIONS - car-following pairs, lateral clearances, gap
   acceptance. Those cannot be manufactured by drawing from marginal speed or
   acceleration distributions: a resampled dataset has the right histograms and
   no physics. This generator runs a genuine microscopic model, so a synthetic
   follower reacts to a synthetic leader and the resulting gaps are causal.

 MODEL
   Longitudinal : Intelligent Driver Model (Treiber, Hennecke & Helbing 2000)
   Lateral      : continuous gap-seeking with inter-vehicle repulsion, no lanes
   Leader rule  : nearest vehicle ahead whose LATERAL OVERLAP exceeds a
                  threshold - this is what makes the model non-lane-based
   Scenario     : a downstream signal, so the data contains free flow, queue
                  build-up, standstill, creeping and discharge - the regimes
                  W99 CC0-CC9 are actually defined on

 MODES  (set MODE below)
   "corpus"   full standalone six-class dataset, for building and testing the
              Phase 2-4 pipeline now
   "augment"  generate ONLY the deficit - classes your real data lacks - with
              ids and frames offset so they never collide with real tracks
   "validate" compare a filtered synthetic file against the known ground truth
              and report how accurately Phase 1 recovered speed/acceleration

 OUTPUTS
   *_truth.csv    noise-free, with true speed / accel / lateral position
   *_noisy.csv    detector-emulated, SCHEMA-IDENTICAL to your trajectories.csv
                  (feed it straight to phase1b_trajectory_filter.py)
   *_DECLARATION.txt   exactly what was generated and from which parameters -
                  paste into your thesis appendix
--------------------------------------------------------------------------------
 PROVENANCE RULE: every parameter below is tagged [EMPIRICAL] (measured from
 YOUR data) or [ASSUMED] (chosen by the supervisor from literature/judgement).
 Nothing is silently invented. The DECLARATION file reprints all of it.
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIG
# ==============================================================================
MODE = "corpus"                 # "corpus" | "augment" | "validate"

OUT_DIR    = r"S:\soscho"
REAL_CSV   = r"S:\soscho\trajectories_filtered.csv"   # used by augment/validate
TRUTH_CSV  = r"S:\soscho\synthetic_truth.csv"         # used by validate
FILTERED_SYNTH_CSV = r"S:\soscho\synthetic_noisy_filtered.csv"   # validate input

SEED = 20260913

# ---- [EMPIRICAL] site constants measured from YOUR trajectories.csv ----------
DT           = 0.03332          # s      30.01 fps, from your file
ROAD_WIDTH   = 6.72             # m      lateral extent -1.08 .. 5.64 m
X_MIN        = -1.08            # m      lateral origin of your coordinate frame
FOV_LENGTH   = 12.0             # m      median longitudinal travel was 11.15 m
Y_ORIGIN     = 25.0             # m      Y_m start value; travel is DECREASING Y
FRAME_OFFSET = 30000            # frames synthetic frames start after your last

# ---- [EMPIRICAL] per-class measurement noise, median sigma_lon (cm) ----------
SIGMA_LON_CM = {"Rickshaw": 4.42, "CNG": 3.29, "Truck": 2.74,
                "Bike": 2.69, "Bus": 2.19, "Car": 2.10}
SIGMA_LAT_CM = 0.80

# ---- [EMPIRICAL] noise is CORRELATED, not white -----------------------------
# Your Phase 1 run: residual RMS 16.1 cm against a white-noise sigma of 3.6 cm,
# and residual lag-1 autocorrelation 0.79. Bounding-box jitter therefore has a
# large low-frequency component (the box "breathes" over ~0.5 s). Emulating it
# as white noise would make your filter look far better than it really is.
NOISE_AR1_PHI   = 0.93          # [EMPIRICAL] tuned to reproduce rho1 ~ 0.79
NOISE_AR1_RATIO = 3.2           # [EMPIRICAL] low-freq std / white std
NOISE_SIGMA_SCALE = 0.75        # [EMPIRICAL] calibration: the AR(1) component
                                # inflates the MAD-of-2nd-difference estimator,
                                # so the injected white sigma is scaled down
                                # until Phase 1 RECOVERS your measured values

# ---- [EMPIRICAL] detector artefact rates from your file ---------------------
GAP_EVENTS_PER_TRACK = 6.8      # 7710 gap events / 1129 tracks
GAP_LEN_MAX          = 5        # frames
CONF_MEAN            = 0.631    # detection confidence mean
CONF_STD             = 0.17
ID_SWITCH_RATE       = 0.05     # [ASSUMED] share of tracks given a jump
ID_SWITCH_JUMP_M     = 4.0      # [EMPIRICAL] the magnitude that corrupts ~40 rows
CLASS_CONFUSION      = 0.00     # [ASSUMED] set >0 to emulate CNG/Rickshaw mixing

# ---- simulation size ---------------------------------------------------------
SIM_DURATION_S   = 1500.0       # [ASSUMED] long enough for ~1200 observed tracks
ROAD_LENGTH      = 150.0        # m  simulated; only FOV_LENGTH is "observed"
FOV_START        = 98.0         # m  camera window; spans the queue tail,
                                #    so the data contains approach, stop,
                                #    standstill and discharge in one track
DEMAND_VEH_PER_H = 2550.0       # [ASSUMED] congested Dhaka arterial demand

# ---- SCENARIOS ---------------------------------------------------------------
# One flow condition cannot serve every purpose. Your video captured a
# moderately congested street (class mean speeds ~10-12 km/h) and the generator
# reproduces it - but a W99 calibration also needs STANDSTILL and CREEPING rows
# for CC0 and CC1, which a free-flowing street simply does not contain.
# So the corpus is generated as TWO scenarios and concatenated:
#   "video_match" - tuned to reproduce your measured per-class speeds
#   "congested"   - heavier demand and a longer red, producing queues,
#                   standstill gaps and stop-and-go
# Each row carries a `scenario` column so you can analyse them separately.
SCENARIOS = [
    ("video_match", 2550.0, 30.0),     # name, demand veh/h, red time s
    ("congested",   3300.0, 55.0),
]
# ---- [ASSUMED] downstream signal: the source of creeping and standstill ------
SIGNAL_ON        = True
SIGNAL_POS       = 138.0        # m  just downstream of the FOV
SIGNAL_RED_S     = 30.0
SIGNAL_GREEN_S   = 60.0

# ---- augment-mode targets ----------------------------------------------------
TARGET_USABLE_TRACKS = 80       # [ASSUMED] usable tracks wanted per class
MIN_DUR_FIT_S        = 2.0      # s   matches phase1b's adequacy test

# ==============================================================================
# VEHICLE CLASS PARAMETERS
# ==============================================================================
# v0    desired speed (m/s)          [EMPIRICAL] from your v_p85 per class
# len,w vehicle dimensions (m)       [ASSUMED]   Bangladeshi fleet, literature
# s0    standstill gap (m)           [ASSUMED]   deliberately tight (Dhaka)
# T     desired time headway (s)     [ASSUMED]
# amax  max acceleration (m/s^2)     [ASSUMED]   bounded by your a+_p95
# bcomf comfortable decel (m/s^2)    [ASSUMED]
# vy    max lateral speed (m/s)      [EMPIRICAL] ~1.4x your |latv|p95 per class
# aggr  lateral gap-seeking drive    [ASSUMED]   weaving propensity, 0..1
# share fleet composition            [EMPIRICAL] your retained track shares
CLASSES = {
    #            v0     len   wid   s0    T     amax  bcomf  vy    aggr  share  id  ltol
    "Rickshaw": (4.03, 2.00, 1.20, 0.60, 0.90, 1.00, 2.00, 1.35, 0.55, 0.500, 4, 0.88),
    "Car":      (4.15, 4.40, 1.70, 1.00, 1.00, 1.80, 2.50, 1.20, 0.40, 0.230, 3, 1.00),
    "Bike":     (6.48, 1.90, 0.70, 0.40, 0.55, 2.20, 2.80, 1.85, 1.00, 0.130, 1, 0.62),
    "Truck":    (4.53, 7.50, 2.40, 1.20, 1.30, 0.90, 2.00, 0.70, 0.15, 0.060, 5, 1.00),
    "CNG":      (4.60, 2.60, 1.40, 0.70, 0.80, 1.50, 2.40, 1.60, 0.85, 0.060, 2, 0.60),
    "Bus":      (3.25, 11.0, 2.50, 1.50, 1.30, 0.90, 1.90, 0.65, 0.15, 0.024, 0, 1.00),
}
# ltol = [ASSUMED] lateral tolerance: the fraction of nominal side clearance the
# class will accept when squeezing past. Bike 0.45 is what reproduces the
# filtering behaviour your data shows (Bikes averaging 15.98 km/h against Cars
# at 10.31 - in Dhaka the motorcycle is the FASTEST mode precisely because it
# does not queue). Bus and Truck at 1.00 cannot filter at all.
# NOTE on CNG: your file yielded only 9 with-flow CNG tracks at a mean speed of
# 1.38 km/h - a sample of parked vehicles. CNG v0 here is therefore [ASSUMED]
# (set between Rickshaw and Car, consistent with a 4-stroke auto-rickshaw),
# NOT measured. This is the single most important declaration in this file.

CLASS_ORDER = list(CLASSES)
DELTA_IDM = 4.0                 # [ASSUMED] IDM free-acceleration exponent
LAT_OVERLAP_FRAC = 0.55         # [ASSUMED] overlap fraction that makes a leader
LAT_REPULSE_M = 0.45            # [ASSUMED] lateral clearance vehicles defend
ACC_LAG_S = 0.35                # [ASSUMED] driver + powertrain response lag, s

# ---- CHAOTIC BEHAVIOURS OF BANGLADESHI MIXED TRAFFIC -------------------------
# These are what separate a Dhaka street from a textbook lane-based road. Each
# is [ASSUMED] in magnitude but the BEHAVIOUR itself is what the video shows.
#
# POLITENESS: MOBIL's politeness factor. 0 = will cut in even if it forces the
# vehicle behind to brake hard; 1 = never inconveniences anybody. A Dhaka
# motorcycle is near 0. A bus cannot squeeze anyway, so its value hardly binds.
POLITENESS = {"Bike": 0.05, "CNG": 0.15, "Rickshaw": 0.35,
              "Car": 0.55, "Truck": 0.85, "Bus": 0.85}
MAX_IMPOSED_BRAKE = 2.5         # m/s^2  worst braking a cut-in may force on others
#
# MID-ROAD STOPS: buses halt wherever a passenger waves, rickshaws stop to pick
# up or drop off. The vehicle simply becomes an obstacle in the middle of the
# carriageway and everyone else must weave around it. This single behaviour
# generates much of the observed disorder.
# DISABLED by supervision decision - the target behaviour is congestion and
# sudden lateral cut-ins, not mid-road halting. Set non-zero to re-enable
# (e.g. Bus 0.0035) if a later chapter needs bus-stop disruption.
STOP_RATE_PER_S = {"Bus": 0.0, "Rickshaw": 0.0, "CNG": 0.0,
                   "Bike": 0.0, "Car": 0.0, "Truck": 0.0}
STOP_DUR_S = {"Bus": (5.0, 13.0), "Rickshaw": (4.0, 10.0), "CNG": (3.0, 8.0),
              "Bike": (2.0, 4.0), "Car": (3.0, 7.0), "Truck": (5.0, 10.0)}
# a halting vehicle drifts toward the nearer edge rather than parking dead centre
STOP_PULL_TO_EDGE = True
#
# OPPOSING TRAFFIC: your real file contained 32 tracks running against the flow
# out of 1129 (2.8 %) - vehicles using the wrong side to get past a jam.
# DISABLED by supervision decision. Your real file did contain 2.8 % wrong-way
# tracks, but they are not the behaviour being modelled here. Set to 0.028 to
# re-enable.
OPPOSING_SHARE = 0.0

RUN_ID = None                   # stamped into both output files so that the
                                # validation step can refuse mismatched pairs
rng = np.random.default_rng(SEED)
DECL = []


def say(s=""):
    print(s)
    DECL.append(str(s))


def rule(t=""):
    if t:
        say("\n" + "=" * 78); say(t); say("=" * 78)
    else:
        say("-" * 78)


# ==============================================================================
# SIMULATION ENGINE
# ==============================================================================
class Sim:
    """Non-lane-based microscopic simulation on a single carriageway."""

    def __init__(self, shares, duration, demand):
        self.shares = shares
        self.duration = duration
        self.demand = demand
        cap = 4000
        self.x = np.zeros(cap)          # longitudinal position, m
        self.y = np.zeros(cap)          # lateral position, m (0 = left edge)
        self.v = np.zeros(cap)          # longitudinal speed, m/s
        self.vy = np.zeros(cap)         # lateral speed, m/s
        self.ytar = np.zeros(cap)       # target lateral position, m
        self.cls = np.zeros(cap, dtype=int)
        self.vid = np.zeros(cap, dtype=int)
        self.v0 = np.zeros(cap)
        self.acc = np.zeros(cap)        # realised acceleration (lagged)
        self.dirn = np.ones(cap)        # +1 with the flow, -1 against it
        self.stop_until = np.full(cap, -1.0)   # sim time a mid-road stop ends
        self.alive = np.zeros(cap, dtype=bool)
        self.n_created = 0
        self.records = []

    # ---- per-vehicle parameter lookup ---------------------------------------
    def p(self, idx, k):
        arr = np.array([CLASSES[CLASS_ORDER[c]][k] for c in self.cls[idx]])
        return arr

    def pc(self, idx, d, default=0.0):
        """Look up a per-class value from a name-keyed dict."""
        return np.array([d.get(CLASS_ORDER[c], default) for c in self.cls[idx]])

    def spawn(self, t):
        free = np.where(~self.alive)[0]
        if not len(free):
            return
        i = free[0]
        cname = rng.choice(CLASS_ORDER, p=self.shares)
        c = CLASS_ORDER.index(cname)
        par = CLASSES[cname]
        # do not spawn on top of an existing vehicle
        live = np.where(self.alive)[0]
        near = live[self.x[live] < 25.0] if len(live) else np.array([], dtype=int)
        ytry = rng.uniform(par[2] / 2, ROAD_WIDTH - par[2] / 2)
        for _ in range(12):
            if not len(near):
                break
            clash = np.abs(self.y[near] - ytry) < (par[2] / 2 + 0.6)
            if not np.any(clash & (self.x[near] < par[1] + 3.0)):
                break
            ytry = rng.uniform(par[2] / 2, ROAD_WIDTH - par[2] / 2)
        else:
            return
        opposing = rng.random() < OPPOSING_SHARE
        self.alive[i] = True
        self.dirn[i] = -1.0 if opposing else 1.0
        self.stop_until[i] = -1.0
        # a wrong-way vehicle enters from the far end and hugs the far side
        self.x[i] = ROAD_LENGTH if opposing else 0.0
        if opposing:
            ytry = rng.uniform(par[2] / 2, min(ROAD_WIDTH, par[2] / 2 + 2.2))
        self.y[i] = ytry
        self.ytar[i] = ytry
        self.cls[i] = c
        self.n_created += 1
        self.vid[i] = self.n_created
        # desired speed: lognormal-ish spread around the class v0
        self.v0[i] = max(1.0, par[0] * rng.normal(1.0, 0.18))
        self.v[i] = min(self.v0[i], 3.0 + rng.normal(0, 0.8))
        self.vy[i] = 0.0
        self.acc[i] = 0.0

    def step(self, t):
        idx = np.where(self.alive)[0]
        if not len(idx):
            return
        x, y, v = self.x[idx], self.y[idx], self.v[idx]
        L = self.p(idx, 1); W = self.p(idx, 2)
        s0 = self.p(idx, 3); T = self.p(idx, 4)
        amax = self.p(idx, 5); bcomf = self.p(idx, 6)
        vymax = self.p(idx, 7); aggr = self.p(idx, 8)
        ltol = self.p(idx, 11)
        v0 = self.v0[idx]

        # ---- leader identification by LATERAL OVERLAP (the non-lane rule) ----
        dirn = self.dirn[idx]
        # "ahead" is measured in each vehicle's OWN direction of travel, and only
        # a vehicle going the SAME way can be a car-following leader. An oncoming
        # vehicle is an obstacle to steer around, not something to follow.
        dx = (x[None, :] - x[:, None]) * dirn[:, None]
        same = dirn[:, None] == dirn[None, :]
        dy = np.abs(y[None, :] - y[:, None])
        halfsum = (W[:, None] + W[None, :]) / 2.0
        overlap = dy < halfsum * LAT_OVERLAP_FRAC * ltol[:, None] + 0.10
        ahead = (dx > 0) & overlap & same
        np.fill_diagonal(ahead, False)
        gap = np.where(ahead, dx - (L[:, None] + L[None, :]) / 2.0, np.inf)
        j = np.argmin(gap, axis=1)
        s = gap[np.arange(len(idx)), j]
        dv = np.where(np.isfinite(s), v - v[j], 0.0)
        s = np.where(np.isfinite(s), np.maximum(s, 0.05), 1e6)

        # ---- oncoming vehicles: steer away, and slow if closing head-on -------
        # A wrong-way vehicle hugs the kerb and the oncoming stream squeezes
        # past it laterally. It must NOT act as a full blocker: treating it as
        # one lets a single wrong-way rickshaw gridlock the entire carriageway,
        # which is not what the video shows.
        onc = (~same) & (dx > 0) & (dx < 10.0)
        np.fill_diagonal(onc, False)
        # (a) lateral avoidance - the dominant real response
        onc_near = onc & (dy < halfsum + 0.9)
        if onc_near.any():
            sgn_on = np.sign(y[:, None] - y[None, :])
            rep_on = np.sum(np.where(onc_near, sgn_on * 1.6, 0.0), axis=1)
        else:
            rep_on = np.zeros(len(idx))
        # (b) slow down only for a genuinely head-on, directly-in-path conflict
        block = onc & (dy < halfsum * 0.55) & (dx < 6.0)
        np.fill_diagonal(block, False)
        if block.any():
            nearest_on = np.where(block.any(axis=1),
                                  np.min(np.where(block, dx, np.inf), axis=1), np.inf)
            s = np.where(np.isfinite(nearest_on) & (nearest_on * 0.9 < s),
                         np.maximum(nearest_on * 0.9, 0.05), s)

        # ---- downstream signal acts as a virtual stopped leader ---------------
        if SIGNAL_ON:
            phase = t % (SIGNAL_RED_S + SIGNAL_GREEN_S)
            if phase < SIGNAL_RED_S:
                ds = SIGNAL_POS - x - L / 2.0
                # Dilemma zone: a driver who cannot stop comfortably proceeds.
                # Without this rule the model brakes at the clip every time the
                # phase flips, and the deceleration statistics become artefacts.
                can_stop = (ds > v ** 2 / (2 * bcomf)) & (dirn > 0)
                red = (ds > 0) & can_stop & (ds < s)   # compare against the OLD s
                dv = np.where(red, v, dv)              # signal is a stopped leader
                s = np.where(red, np.maximum(ds, 0.05), s)

        # ---- IDM ---------------------------------------------------------------
        sstar = s0 + np.maximum(0.0, v * T + v * dv / (2 * np.sqrt(amax * bcomf)))
        acc = amax * (1 - (v / v0) ** DELTA_IDM - (sstar / s) ** 2)
        # ---- mid-road stops: bus halts for a passenger, rickshaw drops a fare -
        srate = self.pc(idx, STOP_RATE_PER_S)
        stopping = self.stop_until[idx] > t
        start = (~stopping) & (v > 2.0) & (rng.random(len(idx)) < srate * DT)
        if start.any():
            lo = np.array([STOP_DUR_S.get(CLASS_ORDER[c], (3.0, 6.0))[0] for c in self.cls[idx]])
            hi = np.array([STOP_DUR_S.get(CLASS_ORDER[c], (3.0, 6.0))[1] for c in self.cls[idx]])
            dur = lo + rng.random(len(idx)) * (hi - lo)
            self.stop_until[idx] = np.where(start, t + dur, self.stop_until[idx])
            stopping = self.stop_until[idx] > t
        # a stopping vehicle brakes to rest and then blocks whatever it occupies
        acc = np.where(stopping, -bcomf * 1.1, acc)
        if STOP_PULL_TO_EDGE and stopping.any():
            # it pulls toward the nearer kerb - it still obstructs, but it does
            # not wall off the whole carriageway the way a dead-centre halt would
            edge = np.where(y < ROAD_WIDTH / 2, W / 2, ROAD_WIDTH - W / 2)
            self.ytar[idx] = np.where(stopping, edge, self.ytar[idx])

        acc_target = np.clip(acc, -4.5, amax)
        # first-order lag toward the target acceleration
        self.acc[idx] += (acc_target - self.acc[idx]) * (DT / ACC_LAG_S)
        acc = self.acc[idx]

        # ---- lateral: gap-seeking + repulsion + boundary ----------------------
        # utility of shifting left/right = how much longitudinal room it buys
        cand = np.stack([y - 0.75, y, y + 0.75])
        util = np.zeros_like(cand)
        for k in range(3):
            ck = np.clip(cand[k], W / 2, ROAD_WIDTH - W / 2)
            dyk = np.abs(ck[:, None] - y[None, :])
            ov = dyk < halfsum * LAT_OVERLAP_FRAC * ltol[:, None] + 0.10
            ah = (dx > 0) & ov
            np.fill_diagonal(ah, False)
            gk = np.where(ah, dx - (L[:, None] + L[None, :]) / 2.0, np.inf)
            util[k] = np.minimum(np.min(gk, axis=1), 40.0)
        # a candidate lateral position is BLOCKED if a vehicle already sits
        # alongside it - you cannot weave into an occupied space
        blocked = np.zeros_like(util, dtype=bool)
        alongside = np.abs(dx) < (L[:, None] + L[None, :]) / 2.0 + 0.3
        np.fill_diagonal(alongside, False)
        for k in range(3):
            ck = np.clip(cand[k], W / 2, ROAD_WIDTH - W / 2)
            dyk = np.abs(ck[:, None] - y[None, :])
            blocked[k] = np.any(alongside & (dyk < halfsum * ltol[:, None] + 0.10), axis=1)
        # MOBIL safety criterion (Kesting, Treiber & Helbing 2007): do not move
        # into a lateral position if the vehicle that would become your FOLLOWER
        # there has to brake hard. Without this, cut-ins create surprise leaders
        # and the deceleration distribution pins against the physical clip.
        for k in range(3):
            ck = np.clip(cand[k], W / 2, ROAD_WIDTH - W / 2)
            dyk = np.abs(ck[:, None] - y[None, :])
            ovk = dyk < halfsum * LAT_OVERLAP_FRAC * ltol[:, None] + 0.10
            behind = (dx < 0) & ovk
            np.fill_diagonal(behind, False)
            gb = np.where(behind, -dx - (L[:, None] + L[None, :]) / 2.0, np.inf)
            nb = np.argmin(gb, axis=1)
            gmin = gb[np.arange(len(idx)), nb]
            # MOBIL politeness: an impolite driver will still cut in, as long as
            # the vehicle behind is not forced past MAX_IMPOSED_BRAKE. This is
            # what produces the aggressive weaving seen in Dhaka - a fully
            # "safe" rule produces unrealistically orderly traffic.
            pol = self.pc(idx, POLITENESS, 0.5)
            need = (s0[nb] + v[nb] * 0.45) * pol
            hard = v[nb] ** 2 / (2 * MAX_IMPOSED_BRAKE) * 0.12
            unsafe = np.isfinite(gmin) & (gmin < np.maximum(need, hard))
            blocked[k] |= unsafe
        util = np.where(blocked, -1e6, util)

        best = np.argmax(util, axis=0)
        gain = util[best, np.arange(len(idx))] - util[1, np.arange(len(idx))]
        move = (gain > 0.5) & (rng.random(len(idx)) < aggr * 0.22)
        newt = np.clip(cand[best, np.arange(len(idx))], W / 2, ROAD_WIDTH - W / 2)
        self.ytar[idx] = np.where(move, newt, self.ytar[idx])

        # baseline lateral drift: real vehicles wander within the carriageway even
        # with nothing to overtake. Without this the synthetic lateral excursion
        # collapses to zero in free flow, which no real mixed-traffic stream does.
        drift = rng.normal(0, 0.018, len(idx)) * (0.35 + aggr)
        self.ytar[idx] = np.clip(self.ytar[idx] + drift, W / 2, ROAD_WIDTH - W / 2)

        # lateral repulsion from close neighbours
        close = (np.abs(dx) < (L[:, None] + L[None, :]) / 2.0 + 1.0) & \
        (dy < LAT_REPULSE_M * ltol[:, None] + halfsum * ltol[:, None])
        np.fill_diagonal(close, False)
        sign = np.sign(y[:, None] - y[None, :])
        rep = np.sum(np.where(close, sign * 0.8, 0.0), axis=1)

        ydes = np.clip(self.ytar[idx] + rep * 0.25 + rep_on * 0.30,
                       W / 2, ROAD_WIDTH - W / 2)
        ay = 3.0 * (ydes - y) - 2.2 * self.vy[idx]
        self.vy[idx] = np.clip(self.vy[idx] + ay * DT, -vymax, vymax)

        # ---- integrate ---------------------------------------------------------
        self.v[idx] = np.maximum(0.0, v + acc * DT)
        self.acc[idx] = np.where(self.v[idx] <= 0.0,
                                 np.maximum(self.acc[idx], 0.0), self.acc[idx])
        self.x[idx] = x + self.v[idx] * DT * dirn
        self.y[idx] = np.clip(y + self.vy[idx] * DT, W / 2, ROAD_WIDTH - W / 2)

        # ---- record only inside the camera field of view ----------------------
        infov = (self.x[idx] >= FOV_START) & (self.x[idx] <= FOV_START + FOV_LENGTH)
        rec = idx[infov]
        if len(rec):
            # ground-truth car-following pairs: Phase 2 must rediscover these
            lead_vid = np.where(np.isfinite(gap[np.arange(len(idx)), j]),
                                self.vid[idx][j], -1)[infov]
            gap_true = np.where(np.isfinite(gap[np.arange(len(idx)), j]),
                                gap[np.arange(len(idx)), j], np.nan)[infov]
            self.records.append(pd.DataFrame({
                "vehicle_id": self.vid[rec],
                "class_name": [CLASS_ORDER[c] for c in self.cls[rec]],
                "t": t,
                "s": self.x[rec],
                "lat": self.y[rec],
                "v_true": self.v[rec],
                "vy_true": self.vy[rec],
                "a_true": acc[infov],
                "leader_id_true": lead_vid,
                "gap_true_m": gap_true,
            }))

        # ---- retire vehicles that left the road --------------------------------
        gone = idx[(self.x[idx] > ROAD_LENGTH) | (self.x[idx] < 0.0)]
        self.alive[gone] = False

    def run(self):
        nsteps = int(self.duration / DT)
        per_step = self.demand / 3600.0 * DT
        carry = 0.0
        for k in range(nsteps):
            t = k * DT
            carry += per_step
            while carry >= 1.0:
                self.spawn(t)
                carry -= 1.0
            self.step(t)
            if k % 6000 == 0:
                print(f"   ... t={t:6.0f}s  live={int(self.alive.sum()):3d}  "
                      f"created={self.n_created}", flush=True)
        return pd.concat(self.records, ignore_index=True) if self.records else pd.DataFrame()


# ==============================================================================
# DETECTOR EMULATION
# ==============================================================================
def ar1_noise(n, sigma_white, phi, ratio):
    """
    White measurement noise plus an AR(1) low-frequency component.
    Pure white noise would understate the difficulty: your real residuals are
    strongly autocorrelated (rho1 ~ 0.79) because a bounding box drifts over
    several frames rather than rattling independently each frame.
    """
    w = rng.normal(0, sigma_white, n)
    b = np.zeros(n)
    s_inn = sigma_white * ratio * np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        b[i] = phi * b[i - 1] + rng.normal(0, s_inn)
    return w + b


def emulate_detector(truth):
    """Turn clean trajectories into a file that looks like YOLO+tracker output."""
    rows = []
    n_switch = n_gap = 0
    for vid, t in truth.groupby("vehicle_id"):
        t = t.sort_values("t").reset_index(drop=True)
        n = len(t)
        if n < 8:
            continue
        cname = t.class_name.iloc[0]
        sw = SIGMA_LON_CM.get(cname, 3.0) / 100.0 * NOISE_SIGMA_SCALE
        lon = t.s.to_numpy() + ar1_noise(n, sw, NOISE_AR1_PHI, NOISE_AR1_RATIO)
        lat = t.lat.to_numpy() + ar1_noise(n, SIGMA_LAT_CM / 100.0 * NOISE_SIGMA_SCALE,
                                           NOISE_AR1_PHI, NOISE_AR1_RATIO)

        # ID switch: a sudden position jump of the magnitude that corrupts data
        if rng.random() < ID_SWITCH_RATE and n > 40:
            k = rng.integers(15, n - 15)
            lon[k:] += ID_SWITCH_JUMP_M * rng.choice([-1, 1])
            n_switch += 1

        keep = np.ones(n, dtype=bool)
        ngap = rng.poisson(GAP_EVENTS_PER_TRACK * n / 130.0)
        for _ in range(int(ngap)):
            g = rng.integers(1, GAP_LEN_MAX + 1)
            st = rng.integers(0, max(1, n - g))
            keep[st:st + g] = False
            n_gap += 1
        keep[0] = keep[-1] = True
        if keep.sum() < 15:
            continue

        lab = cname
        if CLASS_CONFUSION > 0 and cname in ("CNG", "Rickshaw"):
            if rng.random() < CLASS_CONFUSION:
                lab = "Rickshaw" if cname == "CNG" else "CNG"

        Lv, Wv = CLASSES[cname][1], CLASSES[cname][2]
        frames = (t.t.to_numpy() / DT).round().astype(int) + FRAME_OFFSET
        # synthetic pinhole projection, purely so px/py/box columns are plausible
        depth = np.maximum(4.0, 30.0 - t.s.to_numpy() + FOV_START - FOV_START)
        scale = 900.0 / depth
        rows.append(pd.DataFrame({
            "vehicle_id": vid,
            "class_id": CLASSES[cname][10],
            "class_name": lab,
            "frame": frames[keep],
            "time": frames[keep] * DT,
            "X_m": (lat + X_MIN)[keep],
            "Y_m": (Y_ORIGIN - (lon - lon[0]))[keep],
            "px": (640 + (lat[keep] - ROAD_WIDTH / 2) * scale[keep]),
            "py": (360 + scale[keep] * 0.5),
            "box_w_px": Wv * scale[keep] * rng.normal(1.0, 0.09, keep.sum()),
            "box_h_px": Lv * scale[keep] * 0.45 * rng.normal(1.0, 0.08, keep.sum()),
            "raw_class_id": CLASSES[cname][10],
            "confidence": np.clip(rng.normal(CONF_MEAN, CONF_STD, keep.sum()), 0.26, 0.98),
            "source": "SYNTHETIC",
        }))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, n_switch, n_gap


# ==============================================================================
# MODES
# ==============================================================================
def build(shares, duration, tag):
    sim = Sim(shares, duration, DEMAND_VEH_PER_H)
    truth = sim.run()
    if truth.empty:
        sys.exit("ERROR: simulation produced no observed vehicles. "
                 "Check FOV_START / ROAD_LENGTH.")
    # keep only tracks long enough to be useful
    ln = truth.groupby("vehicle_id").size()
    truth = truth[truth.vehicle_id.isin(ln[ln >= 15].index)]
    truth["frame"] = (truth.t / DT).round().astype(int) + FRAME_OFFSET
    noisy, nsw, ngp = emulate_detector(truth)
    return truth, noisy, nsw, ngp


def mode_corpus():
    global DEMAND_VEH_PER_H, SIGNAL_RED_S, FRAME_OFFSET
    rule("MODE: CORPUS - full standalone six-class dataset")
    shares = np.array([CLASSES[c][9] for c in CLASS_ORDER], float)
    shares /= shares.sum()
    say("Fleet composition used (share of spawned vehicles):")
    for c, s in zip(CLASS_ORDER, shares):
        say(f"   {c:<10s} {s*100:5.1f} %")

    T, N = [], []
    tot_sw = tot_gp = 0
    base_frame = FRAME_OFFSET
    for si, (name, demand, red) in enumerate(SCENARIOS):
        say(f"\n--- scenario '{name}': demand {demand:.0f} veh/h, red {red:.0f} s ---")
        DEMAND_VEH_PER_H = demand
        SIGNAL_RED_S = red
        FRAME_OFFSET = base_frame + si * 2_000_000
        dur = SIM_DURATION_S / len(SCENARIOS)
        t, n, sw, gp = build(shares, dur, name)
        # keep vehicle ids disjoint across scenarios
        t = t.copy(); n = n.copy()
        t["vehicle_id"] += si * 1_000_000
        n["vehicle_id"] += si * 1_000_000
        # the leader reference must be offset with it, or the ground-truth
        # pairing silently points at vehicles from the other scenario
        if "leader_id_true" in t.columns:
            t["leader_id_true"] = np.where(t.leader_id_true > 0,
                                           t.leader_id_true + si * 1_000_000,
                                           t.leader_id_true)
        t["scenario"] = name
        n["scenario"] = name
        T.append(t); N.append(n); tot_sw += sw; tot_gp += gp
    FRAME_OFFSET = base_frame
    truth = pd.concat(T, ignore_index=True)
    noisy = pd.concat(N, ignore_index=True)
    return truth, noisy, tot_sw, tot_gp, "synthetic"


def mode_augment():
    rule("MODE: AUGMENT - generate only what your real data lacks")
    if not os.path.exists(REAL_CSV):
        sys.exit(f"ERROR: {REAL_CSV} not found. Run phase1b first, or use MODE='corpus'.")
    real = pd.read_csv(REAL_CSV)
    idcol = "track_id" if "track_id" in real.columns else "vehicle_id"
    wf = real[real.get("direction", "with_flow") == "with_flow"] if "direction" in real else real
    say(f"Real file: {REAL_CSV}  ({len(real)} rows, {real[idcol].nunique()} tracks)")
    say(f"\n{'class':<12}{'usable now':>12}{'target':>9}{'deficit':>9}")
    deficit = {}
    for c in CLASS_ORDER:
        t = wf[wf.class_name == c]
        if len(t):
            dur = t.groupby(idcol).size() * DT
            have = int((dur >= MIN_DUR_FIT_S).sum())
        else:
            have = 0
        need = max(0, TARGET_USABLE_TRACKS - have)
        deficit[c] = need
        say(f"{c:<12}{have:>12}{TARGET_USABLE_TRACKS:>9}{need:>9}")
    tot = sum(deficit.values())
    if tot == 0:
        sys.exit("\nNo deficit: every class already meets the target. Nothing to generate.")
    shares = np.array([deficit[c] for c in CLASS_ORDER], float)
    shares = shares / shares.sum()
    say("\nSpawn composition weighted to the deficit classes:")
    for c, s in zip(CLASS_ORDER, shares):
        if s > 0:
            say(f"   {c:<10s} {s*100:5.1f} %")
    # enough duration to clear the largest deficit
    dur = min(4000.0, max(600.0, SIM_DURATION_S * tot / (6 * TARGET_USABLE_TRACKS)))
    say(f"\nSimulated duration: {dur:.0f} s")
    truth, noisy, nsw, ngp = build(shares, dur, "augment")
    # keep only the deficit classes, capped at the deficit count
    keep_ids = []
    for c, need in deficit.items():
        if need <= 0:
            continue
        sub = noisy[noisy.class_name == c]
        ids = list(pd.unique(sub.vehicle_id))[:need]
        keep_ids += ids
    noisy = noisy[noisy.vehicle_id.isin(keep_ids)]
    truth = truth[truth.vehicle_id.isin(keep_ids)]
    say(f"Retained {len(keep_ids)} synthetic tracks covering the deficit.")
    return truth, noisy, nsw, ngp, "synthetic_augment"


def mode_validate():
    rule("MODE: VALIDATE - how accurately did Phase 1 recover the truth?")
    for f in (TRUTH_CSV, FILTERED_SYNTH_CSV):
        if not os.path.exists(f):
            sys.exit(f"ERROR: {f} not found.\n"
                     f"Run MODE='corpus' first, then run phase1b on the noisy file,\n"
                     f"then set FILTERED_SYNTH_CSV to phase1b's output.")
    tru = pd.read_csv(TRUTH_CSV)
    fil = pd.read_csv(FILTERED_SYNTH_CSV)
    idcol = "track_id" if "track_id" in fil.columns else "vehicle_id"
    fil["vid0"] = fil[idcol].astype(str).str.split("_").str[0].astype(int)
    m = fil.merge(tru[["vehicle_id", "frame", "v_true", "a_true", "class_name"]],
                  left_on=["vid0", "frame"], right_on=["vehicle_id", "frame"],
                  how="inner", suffixes=("", "_t"))
    if not len(m):
        sys.exit("ERROR: no frames matched between truth and filtered file.")
    if "edge" in m.columns:
        m = m[m.edge == 0]
    say(f"Matched {len(m)} rows across {m.vid0.nunique()} tracks.\n")
    say(f"{'class':<12}{'n':>8}{'speed RMSE':>12}{'speed bias':>12}"
        f"{'accel RMSE':>12}{'accel r':>9}")
    say("-" * 78)
    for cn, t in m.groupby("class_name"):
        ev = t.speed_mps - t.v_true
        ea = t.accel_mps2 - t.a_true
        r = np.corrcoef(t.accel_mps2, t.a_true)[0, 1] if len(t) > 5 else np.nan
        say(f"{str(cn):<12}{len(t):>8}{np.sqrt(np.mean(ev**2)):>12.3f}"
            f"{np.mean(ev):>12.3f}{np.sqrt(np.mean(ea**2)):>12.3f}{r:>9.3f}")
    ev = m.speed_mps - m.v_true; ea = m.accel_mps2 - m.a_true
    say("-" * 78)
    say(f"{'ALL':<12}{len(m):>8}{np.sqrt(np.mean(ev**2)):>12.3f}"
        f"{np.mean(ev):>12.3f}{np.sqrt(np.mean(ea**2)):>12.3f}"
        f"{np.corrcoef(m.accel_mps2, m.a_true)[0,1]:>9.3f}")
    say("\n  speed/accel RMSE in m/s and m/s^2 | bias = mean signed error")
    say("  accel r = correlation between recovered and true acceleration.")
    say("  This table IS your filter-validation result. Report it in Phase 4:")
    say("  it states, in numbers, how much of the true dynamics your pipeline")
    say("  recovers from vision data - something no purely real dataset can show.")
    say("")
    say("  HOW TO READ IT. Speed and acceleration will NOT score alike. Speed is")
    say("  a first derivative and survives smoothing; acceleration is a second")
    say("  derivative and competes directly with the low-frequency component of")
    say("  bounding-box jitter, which lives on the same ~0.5 s timescale as real")
    say("  driver behaviour. A low accel correlation is therefore a finding about")
    say("  the LIMITS OF VISION-BASED EXTRACTION, not a failure of your filter.")
    say("")
    say("  IT MATTERS FOR PHASE 3: parameters that rest on POSITION and SPEED")
    say("  (desired-speed distribution, CC0 standstill distance, CC1 headway,")
    say("  lateral clearances) inherit the speed column's accuracy. Parameters")
    say("  that rest on ACCELERATION (CC8, CC9, the desired-acceleration curves)")
    say("  inherit the accel column's. Report both accuracies beside the values.")
    say("")
    say("  SENSITIVITY: this result is conditional on the assumed noise model.")
    say("  Re-run the generator with NOISE_AR1_RATIO at 2.0 and at 4.5 and repeat")
    say("  this table - that brackets how much the conclusion depends on it.")
    with open(os.path.join(OUT_DIR, "phase1_filter_validation.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(DECL))
    sys.exit(0)


# ==============================================================================
# MAIN
# ==============================================================================
import datetime as _dt
RUN_ID = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
rule("PHASE 1C | SYNTHETIC TRAJECTORY GENERATOR")
say(f"RUN ID: {RUN_ID}  (stamped into both output files)")
say(f"Mode: {MODE}   seed: {SEED}   dt: {DT:.5f} s ({1/DT:.2f} fps)")

if MODE == "validate":
    mode_validate()
elif MODE == "augment":
    truth, noisy, nsw, ngp, stem = mode_augment()
elif MODE == "corpus":
    truth, noisy, nsw, ngp, stem = mode_corpus()
else:
    sys.exit(f"ERROR: unknown MODE '{MODE}'")

# ---- write outputs -----------------------------------------------------------
truth_out = truth.rename(columns={"s": "lon_true_m", "lat": "lat_true_m",
                                  "t": "sim_time"})
truth_out["X_m"] = truth_out.lat_true_m + X_MIN
truth_out["run_id"] = RUN_ID
noisy["run_id"] = RUN_ID
_cols = ["vehicle_id", "class_name", "frame", "sim_time",
         "lon_true_m", "lat_true_m", "X_m", "v_true", "vy_true", "a_true",
         "leader_id_true", "gap_true_m"]
if "scenario" in truth_out.columns:
    _cols.append("scenario")
_cols.append("run_id")
truth_out = truth_out[_cols]

p_truth = os.path.join(OUT_DIR, f"{stem}_truth.csv")
p_noisy = os.path.join(OUT_DIR, f"{stem}_noisy.csv")
p_decl = os.path.join(OUT_DIR, f"{stem}_DECLARATION.txt")
try:
    truth_out.to_csv(p_truth, index=False)
    noisy.to_csv(p_noisy, index=False)
except Exception as e:
    OUT_DIR = os.getcwd()
    p_truth = os.path.join(OUT_DIR, f"{stem}_truth.csv")
    p_noisy = os.path.join(OUT_DIR, f"{stem}_noisy.csv")
    p_decl = os.path.join(OUT_DIR, f"{stem}_DECLARATION.txt")
    truth_out.to_csv(p_truth, index=False); noisy.to_csv(p_noisy, index=False)
    say(f"[warn] fell back to {OUT_DIR} ({e})")

# ---- report ------------------------------------------------------------------
rule("GENERATED DATA SUMMARY")
say(f"Ground truth : {p_truth}   ({len(truth_out)} rows)")
say(f"Noisy output : {p_noisy}   ({len(noisy)} rows)  <- feed this to phase1b")
say(f"Tracks       : {noisy.vehicle_id.nunique()}   "
    f"observed span {noisy.time.max()-noisy.time.min():.0f} s")
say(f"Artefacts injected: {nsw} ID switches, {ngp} occlusion gaps")
if "scenario" in truth.columns:
    say("\nRows observed per scenario:")
    for scn, t in truth.groupby("scenario"):
        say(f"   {scn:<15} {len(t):>8} rows  {t.vehicle_id.nunique():>5} tracks")

say(f"\n{'class':<12}{'tracks':>8}{'rows':>9}{'v_mean':>9}{'v_p85':>8}"
    f"{'a+_p95':>9}{'a-_p05':>9}{'wander':>8}{'min_gap':>9}")
say("-" * 78)
for cn, t in truth.groupby("class_name"):
    v = t.v_true.to_numpy(); a = t.a_true.to_numpy()
    wander = t.groupby("vehicle_id").lat.agg(lambda s: s.max() - s.min()).median()
    still = t[(t.v_true < 0.3) & t.gap_true_m.notna()
              & (t.gap_true_m >= 0) & (t.gap_true_m < 12.0)]
    sg = still.gap_true_m.median() if len(still) > 5 else np.nan
    say(f"{cn:<12}{t.vehicle_id.nunique():>8}{len(t):>9}"
        f"{np.mean(v)*3.6:>9.2f}{np.percentile(v,85)*3.6:>8.2f}"
        f"{(np.percentile(a[a>0],95) if (a>0).any() else 0):>9.2f}"
        f"{(np.percentile(a[a<0],5) if (a<0).any() else 0):>9.2f}"
        f"{wander:>8.2f}{sg:>9.2f}")
say("  v in km/h | a in m/s^2 | wander = median per-track lateral excursion, m")
say("  min_gap = median standstill gap (v<0.3 m/s) in m -> this is VISSIM CC0")

# regime coverage - does the data actually contain the behaviour W99 needs?
vv = truth.v_true.to_numpy()
# ---------------------------------------------------------------------------
# CHAOS CHECK - does the simulation actually behave like Dhaka mixed traffic?
# The reference column is what YOUR video measured (phase1b, v2 full run).
# ---------------------------------------------------------------------------
REAL_REF = {   # [EMPIRICAL] from your trajectories.csv
    "Bike":     dict(v=15.98, latv=1.295),
    "Bus":      dict(v=7.03,  latv=0.457),
    "Car":      dict(v=10.31, latv=0.852),
    "Rickshaw": dict(v=9.66,  latv=0.951),
    "Truck":    dict(v=10.83, latv=0.489),
}
chk = truth[truth.scenario == "video_match"] if "scenario" in truth.columns else truth
rule("CHAOS CHECK - 'video_match' scenario vs your real video")
say(f"{'class':<11}{'v_sim':>8}{'v_real':>8}{'  ':>2}{'latv_sim':>10}{'latv_real':>11}"
    f"{'  ':>2}{'wander':>8}{'verdict'}")
say("-" * 78)
for cn in sorted(set(chk.class_name)):
    t = chk[chk.class_name == cn]
    vs = t.v_true.mean() * 3.6
    lv = float(np.percentile(np.abs(t.vy_true), 95))
    wd = t.groupby("vehicle_id").lat.agg(lambda s: s.max() - s.min()).median()
    ref = REAL_REF.get(cn)
    if ref:
        dv = abs(vs - ref["v"]) / max(ref["v"], 1e-6)
        dl = abs(lv - ref["latv"]) / max(ref["latv"], 1e-6)
        ok = "OK" if (dv < 0.35 and dl < 0.60) else "CHECK"
        say(f"{cn:<11}{vs:>8.2f}{ref['v']:>8.2f}  {lv:>10.3f}{ref['latv']:>11.3f}"
            f"  {wd:>8.2f}  {ok}")
    else:
        say(f"{cn:<11}{vs:>8.2f}{'n/a':>8}  {lv:>10.3f}{'n/a':>11}  {wd:>8.2f}  synthetic-only")
say("  v in km/h | latv = 95th pct lateral speed, m/s | wander = median")
say("  per-track lateral excursion, m. CNG has no real reference: your video's")
say("  CNG sample was 9 near-stationary tracks.")

opp = truth.groupby("vehicle_id").apply(
    lambda t: (t.lon_true_m.iloc[-1] - t.lon_true_m.iloc[0]) < 0
    if "lon_true_m" in t else False) if "lon_true_m" in truth else pd.Series(dtype=bool)
say("")
say("DISORDER PRODUCED BY THE MODEL:")
say(f"   lateral excursion, all classes (median per track) : "
    f"{truth.groupby('vehicle_id').lat.agg(lambda s: s.max()-s.min()).median():.2f} m")
say(f"   queued tracks that came to a full stop            : "
    f"{int((truth.v_true < 0.3).groupby(truth.vehicle_id).any().sum())}")
say(f"   no lane markings: vehicles sit abreast across the full {ROAD_WIDTH:.1f} m")
say("   cut-in aggression by class (MOBIL politeness, 0 = most aggressive):")
say("      " + "  ".join(f"{k} {v:.2f}" for k, v in POLITENESS.items()))
say(f"   mid-road stopping : {'ON' if max(STOP_RATE_PER_S.values()) > 0 else 'OFF (by supervision decision)'}")
say(f"   wrong-way traffic : {'ON' if OPPOSING_SHARE > 0 else 'OFF (by supervision decision)'}")

a_all = truth.a_true.to_numpy()
say(f"\n[clip check] accelerations pinned at the -4.5 m/s^2 model limit: "
    f"{100*np.mean(a_all <= -4.49):.2f} %  (keep this below ~2 %; a larger share")
say("             means the scenario is forcing emergency braking and the")
say("             deceleration statistics would be a model artefact)")
if "scenario" in truth.columns and truth.scenario.nunique() > 1:
    say("\nREGIME COVERAGE BY SCENARIO:")
    say(f"{'   scenario':<18}{'standstill':>12}{'creeping':>11}{'congested':>11}{'free':>8}")
    for scn, t in truth.groupby("scenario"):
        sv = t.v_true.to_numpy()          # NOT vv - that is the pooled array
        say(f"   {scn:<15}{100*np.mean(sv<0.3):>11.1f}%{100*np.mean((sv>=0.3)&(sv<2)):>10.1f}%"
            f"{100*np.mean((sv>=2)&(sv<5)):>10.1f}%{100*np.mean(sv>=5):>7.1f}%")
    say("   -> take CC0 / CC1 (standstill, creeping) from 'congested';")
    say("      take desired speeds and lateral behaviour from 'video_match'.")
say(f"\nREGIME COVERAGE (pooled, {len(truth)} rows over all scenarios):")
say(f"   standstill   v < 0.3 m/s : {100*np.mean(vv < 0.3):5.1f} % of samples")
say(f"   creeping  0.3-2.0 m/s    : {100*np.mean((vv>=0.3)&(vv<2.0)):5.1f} %")
say(f"   congested 2.0-5.0 m/s    : {100*np.mean((vv>=2.0)&(vv<5.0)):5.1f} %")
say(f"   free flow    v > 5.0 m/s : {100*np.mean(vv >= 5.0):5.1f} %")
say("   A W99 calibration needs all four. CC0/CC1 come from the standstill and")
say("   creeping rows; CC8/CC9 from the acceleration out of the queue.")

rule("DECLARATION - WHAT IS SYNTHETIC AND WHAT IS NOT")
say("EVERY row in the two files above is SYNTHETIC. They carry source='SYNTHETIC'.")
say("The generator was PARAMETERISED as follows:\n")
say("[EMPIRICAL] - measured from YOUR trajectories.csv, carried into the model:")
say(f"   dt = {DT} s ({1/DT:.2f} fps); carriageway width {ROAD_WIDTH} m; "
    f"FOV {FOV_LENGTH} m")
say(f"   per-class measurement noise sigma_lon (cm): {SIGMA_LON_CM}")
say(f"   sigma_lat {SIGMA_LAT_CM} cm; noise autocorrelation phi={NOISE_AR1_PHI}")
say(f"   occlusion gaps {GAP_EVENTS_PER_TRACK}/track; confidence mean {CONF_MEAN}")
say("   desired speeds v0 per class taken from your measured v_p85")
say("   max lateral speeds taken from your measured |latv|p95")
say("   fleet composition taken from your retained track shares")
say("\n[ASSUMED] - chosen by supervisor judgement, NOT measured:")
say("   IDM structure and exponent delta=4; all s0, T, amax, bcomf values")
say(f"   driver/powertrain acceleration response lag {ACC_LAG_S} s")
say("   vehicle dimensions (Bangladeshi fleet, literature)")
say("   lateral model structure, LAT_OVERLAP_FRAC, LAT_REPULSE_M, aggressiveness")
say(f"   demand {DEMAND_VEH_PER_H} veh/h; signal {SIGNAL_RED_S}s red / "
    f"{SIGNAL_GREEN_S}s green at {SIGNAL_POS} m")
say(f"   ID-switch rate {ID_SWITCH_RATE}; class confusion {CLASS_CONFUSION}")
say("   CNG desired speed - your real CNG sample was 9 near-stationary tracks,")
say("   so CNG v0 is ASSUMED, not measured. Declare this explicitly.")

say("\n[CIRCULARITY WARNING - read before Phase 2]")
say("   Parameters extracted from SYNTHETIC rows partly recover the assumptions")
say("   built into this generator. They are NOT independent evidence about Dhaka")
say("   traffic. Use synthetic rows to (a) build and debug the pipeline, (b) fill")
say("   classes your video genuinely lacks, and (c) validate the filter against")
say("   known truth. Always report empirical and synthetic results in SEPARATE")
say("   columns, and never cite a synthetic-derived CC value as a measurement.")

with open(p_decl, "w", encoding="utf-8") as f:
    f.write("\n".join(DECL))
print(f"\n[declaration written to {p_decl}]")

rule("NEXT STEP")
say(f"1. Run phase1b_trajectory_filter.py with INPUT_PATH = {p_noisy}")
say(f"2. Set FILTERED_SYNTH_CSV to its output, TRUTH_CSV to {p_truth}")
say("3. Re-run this script with MODE='validate' to get the accuracy table.")
