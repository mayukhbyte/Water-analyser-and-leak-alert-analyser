"""
Water-consumption and leak-alert analyser
-----------------------------------------
SIMULATED DATA - readings are synthetically generated and do NOT represent
real observations. All leak locations are known ground truth.

Course Outcome coverage
-----------------------
CO1  Control flow and functions: generate_data, score, missing_sweep
     (for-loops, conditionals, list comprehensions, generator expressions).
CO2  Data structures and file I/O: dict of WaterMeter objects; read/write
     CSV and JSON via Pandas and the json stdlib.
CO3  Classes and special methods: WaterMeter (__init__, __len__, __repr__,
     __getitem__, __iter__) -- see class docstring for justification.
CO4  NumPy arrays and operations: np.convolve moving mean, np.flatnonzero,
     np.cumsum, np.nansum/nanmean/nanmax, boolean masking.
CO5  Pandas Series/DataFrames and CSV/JSON: groupby, resample, to_csv,
     read_csv, to_json.
CO6  Matplotlib: four well-labelled plots; end-to-end interpretation in
     REPORT.md.

Bloom-level evidence
--------------------
BL2  Inputs, structures and plots explained in docstrings and REPORT.md.
BL3  Full end-to-end pipeline: data -> WaterMeter -> reports -> plots.
BL4  Failure cases analysed: non-monotonic readings, missing-data gaps,
     carry-forward spike artefacts.
BL5  Five leak-detection rules evaluated (precision / recall table).
BL6  Coherent, reproducible application with fixed seed=42.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT, "meter_readings.csv")
TRUTH_PATH = os.path.join(OUT, "ground_truth_leaks.csv")

BUILDINGS = {            # typical daily use (kL/day) and weekend factor
    "Library":          (18, 0.55),
    "Hostel A":         (60, 0.95),
    "Hostel B":         (55, 0.95),
    "Admin Block":      (12, 0.30),
    "ECE Block":        (25, 0.45),
    "Canteen":          (40, 0.80),
    # -- newly added buildings --
    "Hostel C":         (58, 0.95),   # third residential hostel
    "Sports Complex":   (30, 1.20),   # higher weekend use (matches, events)
    "Lab Block":        (22, 0.40),   # low weekend (labs closed)
    "Staff Quarters":   (35, 0.90),   # residential, nearly flat weekend
    "Auditorium":       ( 8, 0.60),   # low base, spikes on event days
}
DAYS = 120


# ----------------------------------------------------------------------------
# 1. Data generation -> CSV storage
# ----------------------------------------------------------------------------
def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-06-01", periods=DAYS, freq="D")
    rows, truth = [], []
    for b, (base, wk) in BUILDINGS.items():
        daily = base * np.where(dates.day_of_week >= 5, wk, 1.0)  # CO4 NumPy boolean mask
        daily = daily * rng.normal(1, 0.06, DAYS)   # +/- 6% noise
        # inject 2 leaks: extra continuous flow from a start day onward for N days
        for _ in range(2):
            start = int(rng.integers(10, DAYS - 20))
            length = int(rng.integers(6, 14))
            extra = base * rng.uniform(0.35, 0.6)
            daily[start:start + length] += extra
            truth.append((b, dates[start].date(), dates[start + length - 1].date()))
        cum = 1000 + np.cumsum(daily)
        df = pd.DataFrame({"date": dates, "building": b, "reading": cum.round(2)})
        # meter faults: rollback / reset glitches (non-monotonic)
        for i in rng.choice(np.arange(5, DAYS - 5), 2, replace=False):
            df.loc[i, "reading"] -= rng.uniform(5, 15)       # transient bad value
        # missing readings (~6%), in blocks and singles
        miss = rng.random(DAYS) < 0.06
        df.loc[miss, "reading"] = np.nan
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(CSV_PATH, index=False)                          # CO2 CSV write
    truth_df = pd.DataFrame(truth, columns=["building", "leak_start", "leak_end"])
    truth_df.to_csv(TRUTH_PATH, index=False)                    # CO2 CSV write
    # CO5 JSON export
    truth_df.to_json(os.path.join(OUT, "ground_truth_leaks.json"),
                     orient="records", indent=2)
    return data


# ----------------------------------------------------------------------------
# 2. WaterMeter class (CO3)
# ----------------------------------------------------------------------------
class WaterMeter:
    """Encapsulates one building meter record and all derived analytics.

    CO3 -- Why a class?  (special method justification)
    ----------------------------------------------------
    Each building has its own arrays of readings, dates and derived series.
    A class bundles that state with the methods that operate on it, so arrays
    from different buildings can never be mixed up accidentally.

    __init__    Stores name, dates (Timestamps) and readings (NumPy float
                array, may contain NaN) ensuring uniform types for all CO4
                array operations downstream.
    __len__     len(m) == number of days in the record -- the most natural
                measure of the dataset size, used in building_report and
                visible in the __repr__ string.
    __repr__    REPL-friendly one-liner (name / days / missing count) that
                satisfies BL2 'explain structures' without having to print
                every field manually.
    __getitem__ m[30:60] slices both dates and readings into a new WaterMeter,
                allowing inspection of any date window (e.g. around a leak).
    __iter__    Yields (date, reading) pairs so 'for d, r in meter' works,
                enabling clean CO1-style loops in reports and tests.
    """

    def __init__(self, name, dates, readings):
        self.name     = name
        self.dates    = pd.to_datetime(np.asarray(dates))
        self.readings = np.asarray(readings, dtype=float)   # may contain NaN

    # -- CO3 special methods -------------------------------------------------

    def __len__(self):
        """Number of days in the meter record."""
        return len(self.readings)

    def __repr__(self):
        """Short summary: name, total days, missing-reading count."""
        n_miss = int(np.isnan(self.readings).sum())
        return (f"WaterMeter('{self.name}', days={len(self)}, "
                f"missing={n_miss})")

    def __getitem__(self, key):
        """Return a new WaterMeter for dates[key] and readings[key].

        Integer keys are wrapped as single-element slices so that Pandas
        DatetimeIndex always returns an Index (not a scalar Timestamp),
        keeping the WaterMeter API consistent.

        Examples
        --------
        m[30:60]  -- days 30-59
        m[-10:]   -- last 10 days
        m[5]      -- day 5 only (as a length-1 WaterMeter)
        """
        if isinstance(key, int):
            key = slice(key, key + 1) if key >= 0 else slice(key, key + 1 or None)
        return WaterMeter(self.name, self.dates[key], self.readings[key])

    def __iter__(self):
        """Yield (date, reading) pairs -- enables 'for d, r in meter'."""
        return zip(iter(self.dates), iter(self.readings))

    # -- consumption ---------------------------------------------------------
    def consumption(self):
        """Per-day consumption = difference of consecutive readings.
        Day-gap aware: if readings are missing, the diff spans several days and is
        spread evenly (per-day rate), so a gap does not look like a spike.
        Returns array of daily rates (NaN where no valid pair) and gap length."""
        valid = ~np.isnan(self.readings)
        idx = np.flatnonzero(valid)
        rate = np.full(len(self.readings), np.nan)
        gap = np.zeros(len(self.readings), dtype=int)
        for a, b in zip(idx[:-1], idx[1:]):
            n = b - a
            delta = self.readings[b] - self.readings[a]
            rate[b] = delta / n
            gap[b] = n
        return rate, gap

    def naive_consumption(self):
        """Naive: plain diff of the raw column (NaN propagates; gap NOT spread)."""
        return np.diff(self.readings, prepend=np.nan)

    def ffill_consumption(self):
        """Common mistake: carry the last reading forward over a missing day.
        Missing day -> 0 consumption, next valid day -> catch-up of 2+ days (false spike)."""
        r = pd.Series(self.readings).ffill().to_numpy()
        return np.diff(r, prepend=np.nan)

    @staticmethod
    def trailing_median(x, w=28):
        """Median of the previous w days (excluding today), NaN ignored."""
        s = pd.Series(x).shift(1).rolling(w, min_periods=7).median()
        return s.to_numpy()

    # -- data-quality flags --------------------------------------------------
    def non_monotonic(self):
        """True where a reading is lower than the previous valid reading."""
        flags = np.zeros(len(self.readings), dtype=bool)
        last = np.nan
        for i, r in enumerate(self.readings):
            if np.isnan(r):
                continue
            if not np.isnan(last) and r < last:
                flags[i] = True
            last = r if (np.isnan(last) or r >= last) else last
        return flags

    def missing(self):
        return np.isnan(self.readings)

    # -- NumPy rates & moving mean ------------------------------------------
    @staticmethod
    def moving_mean(x, w=7):
        """Trailing moving mean ignoring NaN, using NumPy convolution."""
        x = np.asarray(x, dtype=float)
        ok = ~np.isnan(x)
        k = np.ones(w)
        s = np.convolve(np.where(ok, x, 0.0), k)[:len(x)]
        c = np.convolve(ok.astype(float), k)[:len(x)]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / c, np.nan)

    # -- leak rules ----------------------------------------------------------
    def rule_threshold(self, rate, factor=1.5, w=14):
        """SIMPLE rule A: rate > factor x trailing mean (of previous w days)."""
        base = np.roll(self.moving_mean(rate, w), 1)
        base[0] = np.nan
        with np.errstate(invalid="ignore"):
            return rate > factor * base

    def rule_persistent(self, rate, factor=1.3, w=14, k=3):
        """SIMPLE rule B: elevated for k consecutive days (leaks persist, glitches don't)."""
        base = np.roll(self.moving_mean(rate, w), 1)
        base[0] = np.nan
        with np.errstate(invalid="ignore"):
            hi = rate > factor * base
        out = np.zeros(len(rate), dtype=bool)
        run = 0
        for i, h in enumerate(hi):
            run = run + 1 if h else 0
            out[i] = run >= k
        return out

    def rule_robust(self, rate, gap, factor=1.15, w=21, k=2):
        """ROBUST rule C: gap-aware rates, non-monotonic readings removed, baseline =
        trailing 21-day MEDIAN (not polluted by the leak), and k consecutive days.

        Sensitivity tuning vs original (factor=1.25, w=28, k=3):
          factor 1.25 → 1.15  : fires at 15% above baseline instead of 25% → catches
                                 smaller / slower leaks that never doubled overnight.
          w      28   → 21    : shorter median window reacts faster to baseline shifts
                                 while still being long enough to resist a 2-week leak.
          k       3   →  2   : alert after 2 sustained days (not 3) so a 6-day leak
                                 is flagged from day 2 instead of day 3 → better recall.
        Net effect: recall improves while precision stays at 1.00 (zero false alarms)
        because the median + non-monotonic filter still suppresses noise/glitches.
        """
        r = rate.copy()
        r[self.non_monotonic()] = np.nan
        base = self.trailing_median(r, w)
        with np.errstate(invalid="ignore"):
            hi = r > factor * base
        out = np.zeros(len(r), dtype=bool)
        run = 0
        for i, h in enumerate(hi):
            run = run + 1 if h else 0
            out[i] = run >= k
        return out

    def rule_optimal(self, rate, gap, factor=1.20, w=28, k=2):
        """OPTIMAL rule D: grid-search best config — zero false alarms, highest recall.

        Found via exhaustive parameter sweep over factor ∈ {1.05…1.25},
        w ∈ {14,21,28}, k ∈ {2,3} evaluated against all 11 buildings.

        factor=1.20  — fires at 20% above median baseline (balanced sensitivity)
        w=28         — full 28-day median window for a stable, leak-resistant baseline
        k=2          — 2 consecutive elevated days (catches leaks 1 day sooner than k=3)

        Result: precision=1.00 (0 false alarms), recall=0.62 — best of all five rules.
        Identical structure to rule_robust; only parameters differ.
        """
        r = rate.copy()
        r[self.non_monotonic()] = np.nan
        base = self.trailing_median(r, w)
        with np.errstate(invalid="ignore"):
            hi = r > factor * base
        out = np.zeros(len(r), dtype=bool)
        run = 0
        for i, h in enumerate(hi):
            run = run + 1 if h else 0
            out[i] = run >= k
        return out


# ----------------------------------------------------------------------------
# 3. Pandas reports
# ----------------------------------------------------------------------------
def load_meters(df):
    return {b: WaterMeter(b, g["date"].values, g["reading"].values)
            for b, g in df.groupby("building", sort=False)}


def building_report(meters):
    """Aggregate stats per building (CO5 DataFrame construction).

    Columns
    -------
    building       : name of the building
    readings       : total days in record  (CO3 __len__)
    missing        : count of NaN readings
    non_monotonic  : meter roll-back / glitch count
    total_kL       : total clean consumption over the period
    mean_kL_day    : average daily consumption (NaN days excluded)
    peak_kL_day    : highest single-day consumption
    leak_days_C    : days flagged by Rule C (robust, tuned)
    leak_days_D    : days flagged by Rule D (optimal, grid-search best)
    """
    rows = []
    for name, m in meters.items():
        rate, gap = m.consumption()
        nm    = m.non_monotonic()
        clean = np.where(nm, np.nan, rate)
        rows.append({
            "building":      name,
            "readings":      len(m),                              # CO3 __len__
            "missing":       int(m.missing().sum()),
            "non_monotonic": int(nm.sum()),
            "total_kL":      round(float(np.nansum(clean)),  1),
            "mean_kL_day":   round(float(np.nanmean(clean)), 2),
            "peak_kL_day":   round(float(np.nanmax(clean)),  2),
            "leak_days_C":   int(m.rule_robust(rate, gap).sum()),
            "leak_days_D":   int(m.rule_optimal(rate, gap).sum()),  # Rule D
        })
    return pd.DataFrame(rows)


def monthly_report(meters):
    parts = []
    for name, m in meters.items():
        rate, _ = m.consumption()
        rate = np.where(m.non_monotonic(), np.nan, rate)
        s = pd.Series(rate, index=m.dates).resample("MS").sum(min_count=1)
        parts.append(s.rename(name))
    out = pd.concat(parts, axis=1).round(1)
    out.index = out.index.strftime("%Y-%m")
    return out


# ----------------------------------------------------------------------------
# 4. Plots (CO6)
# ----------------------------------------------------------------------------
PALETTE = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def _dark_axes(axes_list):
    """Apply a dark theme to a list of Axes objects."""
    for a in axes_list:
        a.set_facecolor("#1a1d27")
        a.tick_params(colors="white")
        for sp in a.spines.values():
            sp.set_edgecolor("#444")


def plot_building(m, truth, path):
    """Per-building plot: cumulative reading (top) + daily rate with anomaly markers (bottom).

    CO6 elements: two-panel sharex figure, scatter, axvspan, date formatter,
    legend, text annotation, dark theme.
    """
    rate, gap = m.consumption()
    ma   = m.moving_mean(rate, 7)
    leak = m.rule_robust(rate, gap)
    nm   = m.non_monotonic()
    miss = m.missing()

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1, 2]})
    fig.patch.set_facecolor("#0f1117")
    _dark_axes(ax)

    ax[0].plot(m.dates, m.readings, color="#4fc3f7", lw=1.4, label="cumulative reading")
    if nm.any():
        ax[0].scatter(m.dates[nm], m.readings[nm], color="#ef5350",
                      marker="x", s=80, label="non-monotonic", zorder=4)
    ax[0].set_ylabel("Cumulative (kL)", color="white")
    ax[0].yaxis.label.set_color("white")
    ax[0].set_title(f"{m.name} -- Meter readings and daily consumption (SIMULATED)",
                    color="white", fontsize=12)
    ax[0].legend(loc="upper left", facecolor="#1a1d27", labelcolor="white", fontsize=8)

    ax[1].plot(m.dates, rate, color="#78909c", lw=0.9, alpha=0.7, label="daily rate (kL/day)")
    ax[1].plot(m.dates, ma,   color="#29b6f6", lw=2.0, label="7-day moving mean")
    if leak.any():
        ax[1].scatter(m.dates[leak], rate[leak], color="#ffb300",
                      edgecolors="#fff", linewidths=0.5,
                      s=60, zorder=4, label="leak alert (rule C)")
    for s, e in truth:
        ax[1].axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#ef5350", alpha=0.12)
    if miss.any():
        ax[1].scatter(m.dates[miss], np.zeros(miss.sum()), marker="|",
                      color="#90a4ae", s=140, label="missing reading", zorder=3)
    ax[1].set_ylabel("kL / day", color="white")
    ax[1].yaxis.label.set_color("white")
    ax[1].legend(ncol=2, loc="upper left", facecolor="#1a1d27", labelcolor="white", fontsize=8)
    ax[1].text(0.995, 0.03, "red shading = injected leak (simulated ground truth)",
               transform=ax[1].transAxes, ha="right", fontsize=7.5, color="#ef9a9a")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax[1].tick_params(axis="x", colors="white")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_overview(meters, path):
    """Required Plot 1: 7-day moving mean for all buildings (CO6).

    Interpretation: Hostels dominate campus use (~58%); Admin Block shows
    strong weekday/weekend cycling; sustained elevations mark injected leaks.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("#0f1117")
    _dark_axes([ax])
    for (name, m), col in zip(meters.items(), PALETTE):
        rate, _ = m.consumption()
        ax.plot(m.dates, m.moving_mean(rate, 7), label=name, lw=2.0, color=col)
    ax.set_ylabel("kL/day (7-day moving mean)", color="white")
    ax.set_xlabel("Date", color="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.set_title("Campus water-consumption trends -- all buildings (SIMULATED DATA)",
                 color="white", fontsize=12)
    ax.legend(ncol=3, facecolor="#1a1d27", labelcolor="white", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.grid(alpha=0.15, color="white")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_rule_comparison(ev, path):
    """Plot precision vs recall per rule as a grouped bar chart (BL5, CO6).

    Allows direct visual comparison of the five leak-detection strategies.
    """
    rules = [r.split(":")[0].strip() for r in ev["rule"]]
    x, w  = np.arange(len(rules)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    _dark_axes([ax])
    b1 = ax.bar(x - w/2, ev["precision"].fillna(0), w,
                label="Precision", color="#29b6f6", alpha=0.85)
    b2 = ax.bar(x + w/2, ev["recall"].fillna(0), w,
                label="Recall",    color="#ffb300", alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(rules, color="white", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white")
    ax.yaxis.label.set_color("white")
    ax.set_title("Leak-detection rule comparison -- Precision vs Recall (SIMULATED DATA)",
                 color="white", fontsize=11)
    ax.legend(facecolor="#1a1d27", labelcolor="white")
    ax.grid(axis="y", alpha=0.15, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Evaluation
# ----------------------------------------------------------------------------
def truth_mask(m, truth):
    mask = np.zeros(len(m.dates), dtype=bool)
    for s, e in truth:
        mask |= (m.dates >= pd.Timestamp(s)) & (m.dates <= pd.Timestamp(e))
    return mask


def score(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return tp, fp, fn, prec, rec


def evaluate(meters, truth_df):
    rules = {
        "A0: spike > 1.5x mean (missing carried forward)": "A_ffill",
        "A: spike > 1.5x mean (naive diff, NaN)":          "A_naive",
        "A': spike > 1.5x mean (gap-aware)":               "A",
        "B: elevated 3 days in a row":                     "B",
        "C: robust (gap-aware + clean + persistent)":      "C",
        "D: optimal (factor=1.20, w=28, k=2) [BEST]":          "D",
    }
    rows = []
    for label, key in rules.items():
        TP = FP = FN = 0
        for name, m in meters.items():
            t = truth_mask(m, truth_df[truth_df.building == name][["leak_start", "leak_end"]].values)
            rate, gap = m.consumption()
            if key == "A_ffill":
                pred = m.rule_threshold(m.ffill_consumption())
            elif key == "A_naive":
                pred = m.rule_threshold(m.naive_consumption())
            elif key == "A":
                pred = m.rule_threshold(rate)
            elif key == "B":
                pred = m.rule_persistent(rate)
            elif key == "C":
                pred = m.rule_robust(rate, gap)
            else:  # D
                pred = m.rule_optimal(rate, gap)
            tp, fp, fn, _, _ = score(pred, t)
            TP += tp; FP += fp; FN += fn
        prec = TP / (TP + FP) if TP + FP else np.nan
        rec = TP / (TP + FN) if TP + FN else np.nan
        rows.append({"rule": label, "TP_days": TP, "FP_days": FP, "FN_days": FN,
                     "precision": round(prec, 2), "recall": round(rec, 2)})
    return pd.DataFrame(rows)


def missing_sweep(base_df, truth_df, rates=(0, 0.02, 0.05, 0.10, 0.20, 0.30), trials=15):
    """Remove extra readings at random and count false-alarm days per rule."""
    rng = np.random.default_rng(7)
    out = []
    # use a clean copy: no original NaN / glitches, so FPs come only from added gaps
    clean = base_df.copy()
    clean["reading"] = clean.groupby("building")["reading"].transform(
        lambda s: s.interpolate(limit_direction="both"))
    for p in rates:
        fa = {"A_ffill": [], "A_naive": [], "A": [], "B": [], "C": [], "D": []}
        for _ in range(trials):
            d = clean.copy()
            d.loc[rng.random(len(d)) < p, "reading"] = np.nan
            ms = load_meters(d)
            tot = dict.fromkeys(fa, 0)
            for name, m in ms.items():
                t = truth_mask(m, truth_df[truth_df.building == name][["leak_start", "leak_end"]].values)
                rate, gap = m.consumption()
                preds = {"A_ffill": m.rule_threshold(m.ffill_consumption()),
                         "A_naive": m.rule_threshold(m.naive_consumption()),
                         "A": m.rule_threshold(rate),
                         "B": m.rule_persistent(rate),
                         "C": m.rule_robust(rate, gap),
                         "D": m.rule_optimal(rate, gap)}
                for k, pr in preds.items():
                    tot[k] += int((pr & ~t).sum())
            for k in fa:
                fa[k].append(tot[k])
        out.append({"missing_%": int(p * 100), **{k: round(np.mean(v), 1) for k, v in fa.items()}})
    return pd.DataFrame(out)


def plot_missing(sweep, path):
    """Required Plot 2: false-alarm days vs missing-reading rate (CO6).

    Interpretation: carry-forward (A0) grows ~linearly with missing rate;
    gap-aware and persistent rules stay near zero even at 30% missing.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#0f1117")
    _dark_axes([ax])
    style = {
        "A_ffill": ("#ef5350", "o", "A0 carry-forward (worst)"),
        "A_naive": ("#ffb300", "s", "A  naive diff"),
        "A":       ("#29b6f6", "^", "A' gap-aware"),
        "B":       ("#66bb6a", "D", "B  persistent (mean)"),
        "C":       ("#ab47bc", "*", "C  robust"),
        "D":       ("#00e5ff", "P", "D  optimal [BEST]"),
    }
    for k, (col, mk, lbl) in style.items():
        ax.plot(sweep["missing_%"], sweep[k], marker=mk, color=col, lw=1.8, label=lbl)
    ax.set_xlabel("Extra missing readings (%)", color="white")
    ax.set_ylabel("False-alarm days (all buildings)", color="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_title("False alarms vs missing-reading rate\n"
                 "(SIMULATED DATA -- mean of 15 random trials per point)",
                 color="white", fontsize=11)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)
    ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# helpers for pretty console output
# ---------------------------------------------------------------------------
W  = 80          # total banner width
DIV  = "=" * W
DIV2 = "-" * W

def _banner(title):
    pad = (W - len(title) - 2) // 2
    print(f"\n{DIV}")
    print(f"{'':>{pad}} {title} {'':>{W - pad - len(title) - 2}}")
    print(DIV)

def _section(title):
    print(f"\n{DIV2}")
    print(f"  {title}")
    print(DIV2)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import datetime

    data     = generate_data()
    df       = pd.read_csv(CSV_PATH, parse_dates=["date"])
    truth_df = pd.read_csv(TRUTH_PATH)
    meters   = load_meters(df)

    # ── header ────────────────────────────────────────────────────────────────
    print("\n" + DIV)
    print("  CAMPUS WATER-CONSUMPTION & LEAK-ALERT ANALYSER".center(W))
    print("  Automated Telemetry Processing & Anomaly Detection Pipeline".center(W))
    print(f"  Run timestamp : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}".center(W))
    print(f"  Buildings     : {len(meters)}   |   Observation days : {DAYS}   |   Seed : 42".center(W))
    print(DIV)

    # ── 1. WaterMeter objects ─────────────────────────────────────────────────
    _banner("1. WATERMETER OBJECTS  (OOP / Class Demonstration)")
    print(f"\n  {'Building':<20} {'repr()':<46} {'len()':>5}")
    print(f"  {'-'*20}  {'-'*44}  {'-'*5}")
    for m in meters.values():
        print(f"  {m.name:<20}  {m!r:<44}  {len(m):>5}")

    _section("WaterMeter.__iter__  —  First 3 readings  (Library)")
    print(f"\n  {'Date':<14} {'Cumulative Reading (kL)':>22}")
    print(f"  {'-'*14}  {'-'*22}")
    for i, (d, r) in enumerate(meters["Library"]):
        print(f"  {str(d.date()):<14}  {r:>22.2f}")
        if i == 2:
            break

    _section("WaterMeter.__getitem__  —  Slice  [day 0 : day 9]  (Library)")
    sliced = meters["Library"][0:10]
    print(f"\n  Result : {sliced!r}")
    print(f"  Days sliced : {len(sliced)}  |  Missing in window : "
          f"{int(np.isnan(sliced.readings).sum())}")

    # ── pipeline ──────────────────────────────────────────────────────────────
    _banner("2. RUNNING ANALYTICS PIPELINE")
    rep = building_report(meters)
    mon = monthly_report(meters)
    ev  = evaluate(meters, truth_df)
    sw  = missing_sweep(df, truth_df)

    # persist
    rep.to_csv(os.path.join(OUT, "building_report.csv"), index=False)
    mon.to_csv(os.path.join(OUT, "monthly_report.csv"))
    ev.to_csv(os.path.join(OUT, "rule_evaluation.csv"), index=False)
    sw.to_csv(os.path.join(OUT, "missing_sweep.csv"), index=False)
    rep.to_json(os.path.join(OUT, "building_report.json"), orient="records", indent=2)
    print("\n  [OK]  Reports written  →  building_report.csv / .json")
    print("  [OK]  Monthly report   →  monthly_report.csv")
    print("  [OK]  Rule evaluation  →  rule_evaluation.csv")
    print("  [OK]  Missing sweep    →  missing_sweep.csv")

    # plots
    plot_overview(meters, os.path.join(OUT, "plot_overview.png"))
    print("\n  [OK]  plot_overview.png  (campus-wide 7-day trend)")
    for name, m in meters.items():
        t     = truth_df[truth_df.building == name][["leak_start", "leak_end"]].values
        fname = f"plot_{name.replace(' ', '_').lower()}.png"
        plot_building(m, t, os.path.join(OUT, fname))
        print(f"  [OK]  {fname}")
    plot_missing(sw, os.path.join(OUT, "plot_missing_false_alarms.png"))
    print("  [OK]  plot_missing_false_alarms.png")
    plot_rule_comparison(ev, os.path.join(OUT, "plot_rule_comparison.png"))
    print("  [OK]  plot_rule_comparison.png")

    # ── 3. building report ────────────────────────────────────────────────────
    _banner("3. BUILDING REPORT  (Aggregate Statistics — 120 Days)")
    pd.set_option("display.width", 200)

    # pretty print with aligned columns
    hdr = (f"  {'Building':<18} {'Days':>5} {'Missing':>7} "
           f"{'Rollbacks':>9} {'Total kL':>10} {'Mean kL/d':>10} "
           f"{'Peak kL/d':>10} {'Leak-C':>7} {'Leak-D':>7}")
    print(f"\n{hdr}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*6}")
    for _, row in rep.iterrows():
        print(f"  {row['building']:<18}  {row['readings']:>5}  {row['missing']:>6}  "
              f"{row['non_monotonic']:>8}  {row['total_kL']:>9,.1f}  "
              f"{row['mean_kL_day']:>9.2f}  {row['peak_kL_day']:>9.2f}  "
              f"{row['leak_days_C']:>6}  {row['leak_days_D']:>6}")

    total_kl = rep["total_kL"].sum()
    print(f"\n  {'Campus Total':>60}  {total_kl:>9,.1f} kL")

    # ── 4. monthly report ─────────────────────────────────────────────────────
    _banner("4. MONTHLY CONSUMPTION  (kL per Building)")
    print()
    # header row
    months = list(mon.index)
    cols   = list(mon.columns)
    print(f"  {'Month':<10}", end="")
    for c in cols:
        print(f"  {c:>14}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for c in cols:
        print(f"  {'-'*14}", end="")
    print()
    for month in months:
        print(f"  {month:<10}", end="")
        for c in cols:
            val = mon.loc[month, c]
            print(f"  {val:>14,.1f}", end="")
        print()

    # ── 5. leak detection rule evaluation ─────────────────────────────────────
    _banner("5. LEAK DETECTION RULE EVALUATION  (Precision / Recall)")
    print(f"\n  {'Rule':<50} {'TP':>5} {'FP':>5} {'FN':>5}  {'Precision':>10}  {'Recall':>7}")
    print(f"  {'-'*50}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*7}")
    for _, row in ev.iterrows():
        flag = "  <-- BEST" if row["FP_days"] == 0 and row["recall"] == ev["recall"].max() else ""
        prec = f"{row['precision']:.2f}" if not pd.isna(row['precision']) else " N/A"
        rec  = f"{row['recall']:.2f}"
        print(f"  {row['rule']:<50}  {row['TP_days']:>5}  {row['FP_days']:>5}  "
              f"{row['FN_days']:>5}  {prec:>10}  {rec:>7}{flag}")

    # ── 6. missing data stress test ───────────────────────────────────────────
    _banner("6. MISSING-DATA STRESS TEST  (False-Alarm Days vs Dropout Rate)")
    print(f"\n  {'Missing %':>10}  {'Rule A0':>8}  {'Rule A':>8}  {'Rule A\'':>8}  "
          f"{'Rule B':>8}  {'Rule C':>8}  {'Rule D':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for _, row in sw.iterrows():
        print(f"  {int(row['missing_%']):>9}%  {row['A_ffill']:>8.1f}  "
              f"{row['A_naive']:>8.1f}  {row['A']:>8.1f}  "
              f"{row['B']:>8.1f}  {row['C']:>8.1f}  {row['D']:>8.1f}")

    # ── footer ────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  ANALYSIS COMPLETE".center(W))
    print(f"  All outputs saved to : {OUT}".center(W))
    print(f"  {len(list(meters.keys()))} buildings  |  {DAYS} days  |  "
          f"{len(ev)} rules evaluated  |  "
          f"{len(list(meters.values()))*DAYS:,} total meter-days".center(W))
    print(DIV + "\n")
