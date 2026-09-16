"""
speed_percentiles.py  (v2 -- uses your real 'time' column)
----------------------------------------------------------
Compute arbitrary percentiles of speed per vehicle class.
Speed is derived per vehicle from consecutive (X,Y) positions:
    v = sqrt(dX^2 + dY^2) / dt   (m/s)   ->  * 3.6  = km/h

Reads : S:\soscho\trajectories.csv
Writes: S:\Second\speed_percentiles_by_class.csv
"""

import os
import numpy as np
import pandas as pd

IN_CSV  = r"S:\soscho\trajectories.csv"
OUT_CSV = r"S:\Second\speed_percentiles_by_class.csv"

# --- MATCHED TO YOUR REAL CSV COLUMN NAMES ---
COL_ID  = "vehicle_id"
COL_CLS = "class_name"
COL_T   = "time"
COL_X   = "X_m"
COL_Y   = "Y_m"
# ---------------------------------------------

PCTS = [5, 10, 15, 25, 50, 75, 85, 90, 95]


def main():
    df = pd.read_csv(IN_CSV)
    df[COL_T] = df[COL_T].astype(float)
    df = df.sort_values([COL_ID, COL_T])

    # per-row speed (m/s) via forward difference within each vehicle
    g = df.groupby(COL_ID)
    df["dx"] = g[COL_X].diff()
    df["dy"] = g[COL_Y].diff()
    df["dt"] = g[COL_T].diff()

    df = df[df["dt"] > 0].copy()
    df["speed_kmph"] = np.sqrt(df["dx"]**2 + df["dy"]**2) / df["dt"] * 3.6

    rows = []
    for cls, sub in df.groupby(COL_CLS):
        v = sub["speed_kmph"].dropna().to_numpy()
        if len(v) == 0:
            continue
        row = {"class_name": cls, "n_samples": len(v)}
        for p in PCTS:
            row[f"P{p}"] = float(np.percentile(v, p))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("class_name")
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()