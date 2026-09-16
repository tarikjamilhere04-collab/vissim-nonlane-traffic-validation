# vissim-nonlane-traffic-validation
Trajectory extraction, filtering and behaviour measurement pipeline for validating a PTV Vissim microsimulation against observed non-lane-based traffic in Dhaka.
# Validating PTV Vissim Against Observed Non-Lane-Based Traffic

Undergraduate thesis, Department of Civil Engineering, Islamic University
of Technology (IUT), Bangladesh. Supervised by [supervisor].

Microsimulation studies of South Asian traffic routinely apply lane-based
car-following models to traffic that does not use lanes. This repository
contains the pipeline used to test that assumption empirically: vehicle
trajectories are extracted from video of an urban arterial in Dhaka, four
behaviours are defined as computable quantities, and the same quantities
are measured on the field data and on the output of a PTV Vissim model
calibrated from that data.

**Study site:** Sonargaon Janapath, in front of Zamzam Tower, Uttara,
Dhaka — one carriageway of a four-lane divided road, 10.0 m usable width.

## Results

| Behaviour | Measure | Observed | Simulated |
|---|---|---|---|
| Lateral spreading | Normalised entropy of lateral position | 0.974 | 0.950 |
| Lateral packing | Vehicles abreast per 3 m (median) | 1.48 | 0.60 |
| Creeping | Still moving while leader gap < 2 m | 63.3% | 28.1% |
| Tailgating | Share of following time at T < 0.5 s | 13.3% | 1.9% |

Lateral spreading is reproduced. Lateral packing and close following are
not — tailgating is under-produced by roughly an order of magnitude at
every headway threshold tested (T < 0.5 s, 1.0 s and 2.0 s alike), which
points to a structural property of the Wiedemann car-following
formulation rather than to calibration error.

## Method

1. **Scale correction.** The camera homography compressed the lateral
   axis. The correction factor was derived from the measured carriageway
   width and then validated independently against measured vehicle
   widths — two unrelated quantities agreeing to within 5%.
2. **Outlier removal.** Single-frame detection jumps removed with a
   Hampel filter (3 MAD, scaled by 1.4826).
3. **Smoothing.** Differentiation noise estimated from the data itself;
   Kalman filtering with RTS smoothing selected over alternatives on a
   synthetic corpus with known ground truth (RMSE 0.23 m/s).
4. **Lane-free leader identification.** Lane-index methods cannot work
   without lanes, so leaders are identified by a lateral body-overlap
   rule, validated at 80.3% agreement with a gap RMSE of 0.46 m.
5. **Parameter fitting.** Wiedemann CC0 and CC1 fitted jointly by
   Theil-Sen regression over steady-following samples, with an explicit
   physical gate that rejects non-physical fits rather than accepting
   them.
6. **Comparison.** Each behaviour recomputed from the Vissim .fzp output
   using the same definitions applied to the field data.

## Repository structure

    scripts/    analysis pipeline, in phase order
    figures/    300 dpi figures used in the thesis
    data/       sampled trajectory data (see note below)
    docs/       thesis and supporting reports

Run the scripts in phase order — later phases read the outputs of
earlier ones.

## Data availability

The full trajectory dataset and the source video are not included: the
video shows identifiable vehicles and people, and the processed files
exceed GitHub's size limits. `data/` contains a sample sufficient to run
the pipeline. The full dataset is available on request.

## Requirements

Python 3.10 or later, with numpy, pandas, scipy and matplotlib.

    pip install -r requirements.txt

## Limitations

- The trajectory data covers 6.06 m of usable width while the Vissim
  model's approach is wider, so width-sensitive comparisons are
  normalised rather than compared directly.
- 2.8% of vehicle pairs yield a negative gap, indicating the assumed
  vehicle lengths are slightly too long. The data constrains length and
  lateral clearance only as a sum and cannot separate them.
- Only 4.5% of simulated samples are boxed in, against 19.8% of observed
  samples, so part of the creeping discrepancy is that the model rarely
  reaches the situation in which creeping occurs.

## Author

Md. Tarik Jamil
LinkedIn: (https://www.linkedin.com/in/md-tarik-jamil-8b739b315/)

## License

MIT (see LICENSE). Figures and thesis text are the author's own work.
