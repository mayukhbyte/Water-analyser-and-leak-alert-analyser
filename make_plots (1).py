"""
make_plots.py - regenerates all PNG plots from meter_readings.csv using Matplotlib.
Keep this file in the same folder as water_analyser.py, meter_readings.csv
and ground_truth_leaks.csv, then run:  python make_plots.py
"""
import os
import pandas as pd
from water_analyser import (load_meters, missing_sweep, plot_overview,
                            plot_building, plot_missing)

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "meter_readings.csv"), parse_dates=["date"])
truth = pd.read_csv(os.path.join(HERE, "ground_truth_leaks.csv"))
meters = load_meters(df)

# 1. campus overview (7-day moving mean for every building)
plot_overview(meters, os.path.join(HERE, "plot_overview.png"))

# 2. one trend + anomaly-marker plot per building
for name, m in meters.items():
    t = truth[truth.building == name][["leak_start", "leak_end"]].values
    fname = f"plot_{name.replace(' ', '_').lower()}.png"
    plot_building(m, t, os.path.join(HERE, fname))
    print("saved", fname)

# 3. false alarms vs missing-reading rate
plot_missing(missing_sweep(df, truth), os.path.join(HERE, "plot_missing_false_alarms.png"))
print("done")
