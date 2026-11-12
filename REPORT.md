# Water-Consumption and Leak-Alert Analyser

**Goal:** analyse daily meter readings for several campus buildings.
**Files:** `water_analyser.py` (all code), `meter_readings.csv` (stored readings), `ground_truth_leaks.csv`, report CSVs, plots.
Run with `python water_analyser.py` (needs NumPy, Pandas, Matplotlib).

## 1. Data and storage

Six buildings (Library, Hostel A, Hostel B, Admin Block, ECE Block, Canteen) with 120 days of **cumulative** meter readings (1 Jun to 28 Sep 2026), stored in `meter_readings.csv` (`date, building, reading`). The data is synthetic, so true leaks are known and the rules can be scored:

- weekday/weekend usage pattern plus about 6% random noise per building
- 2 injected leaks per building (extra continuous flow for 6 to 13 days)
- 2 transient bad readings per building (reading drops, i.e. non-monotonic)
- about 6% of readings missing

## 2. Design

**`WaterMeter` class** (one per building)

| Method | Purpose |
|---|---|
| `consumption()` | daily consumption from cumulative readings; across a gap of *n* days the difference is spread as a per-day rate |
| `non_monotonic()` | flags any reading lower than the previous valid reading |
| `missing()` | flags NaN readings |
| `moving_mean()` | NumPy-convolution trailing moving mean that ignores NaN |
| `trailing_median()` | median of the previous 28 days |
| `rule_threshold / rule_persistent / rule_robust` | the leak rules compared below |

**NumPy** handles rates, gap lengths and moving means. **Pandas** handles loading/grouping (`building_report.csv`, `monthly_report.csv`) and the evaluation tables. **Matplotlib** produces one plot per building (cumulative line with non-monotonic markers, daily rate, 7-day mean, leak alerts, missing-reading ticks, true leak shading), plus a campus overview and a false-alarm chart.

## 3. Results

**Building report**

| Building | Missing | Non-monotonic | Mean kL/day | Peak kL/day | Leak-alert days (rule C) |
|---|---|---|---|---|---|
| Library | 7 | 1 | 16.84 | 28.16 | 0 |
| Hostel A | 10 | 0 | 65.56 | 125.43 | 6 |
| Hostel B | 9 | 0 | 60.27 | 121.21 | 8 |
| Admin Block | 4 | 0 | 10.51 | 26.83 | 6 |
| ECE Block | 9 | 1 | 23.99 | 41.30 | 7 |
| Canteen | 6 | 0 | 40.73 | 68.54 | 2 |

The two hostels account for about 58% of campus water use and show the largest leak peaks (about 2x normal flow).

## 4. Analyse and evaluate: simple leak rules

| Rule | Idea | TP days | FP days | FN days | Precision | Recall |
|---|---|---|---|---|---|---|
| A0 | day > 1.5x trailing 14-day mean, missing reading carried forward | 30 | **36** | 73 | 0.45 | 0.29 |
| A | same, plain diff (NaN) | 31 | 3 | 72 | 0.91 | 0.30 |
| A' | same, gap-aware rates | 31 | 3 | 72 | 0.91 | 0.30 |
| B | > 1.3x mean for 3 days in a row | 29 | 0 | 74 | 1.00 | 0.28 |
| C | gap-aware, non-monotonic removed, **28-day median** baseline, 3 days in a row | 46 | 0 | 57 | 1.00 | 0.45 |

(Counts are alert-days over all buildings; there are 103 true leak days.)

**Observations**

1. **Single-day threshold rules (A)** react in one day but fire on noise and on weekday/weekend swings, and they false-alarm on any one-off spike (3 FP days).
2. **Persistence (B)** removes those false alarms completely (precision 1.00), but needs 3 days, so the first days of every leak are missed. Recall is low because the 14-day *mean* baseline is contaminated by the leak itself: after a few days the baseline rises and the leak stops looking abnormal.
3. **Robust rule (C)** keeps precision at 1.00 and lifts recall from about 0.29 to 0.45, mainly because the median baseline ignores a leak lasting under 14 days. Remaining misses are the first two days of every leak (the price of persistence) and weak leaks close to the 1.25x threshold.
4. **Trade-off:** sensitivity versus false alarms. A water utility would normally accept a 2 to 3 day detection delay for near-zero false alarms, since each false alarm costs a technician visit.

## 5. False alarms caused by missing readings

I took a gap-free version of the data (so any false alarms come only from the missing readings I add), removed 0 to 30% of readings at random (15 trials each), and counted false-alarm days across all buildings.

| Missing % | A0 carry-forward | A naive (NaN) | A gap-aware | B persistent | C robust |
|---|---|---|---|---|---|
| 0 | 5.0 | 5.0 | 5.0 | 1.0 | 1.0 |
| 2 | 13.5 | 5.0 | 4.9 | 1.0 | 0.7 |
| 5 | 29.3 | 5.1 | 4.7 | 1.1 | 0.8 |
| 10 | 50.9 | 5.1 | 4.3 | 0.8 | 0.5 |
| 20 | 88.9 | 6.0 | 4.3 | 0.7 | 0.1 |
| 30 | 117.3 | 6.7 | 3.8 | 0.8 | 0.2 |

**Why gaps cause false alarms**

- If a missing day is treated as "no change" (reading carried forward), that day shows **0 kL** and the next valid day shows the **combined consumption of 2 or more days**. That catch-up value looks like a leak spike (about 2x normal). The false alarms grow roughly linearly with the missing rate: about 117 FP days at 30% missing, versus 5 at 0%.
- If gaps are left as NaN, no spike is produced, but the rate is lost for that day and the next, so the rule has less data. It does not add false alarms, though it also cannot detect a leak starting in the gap.
- Spreading the difference over the gap length (gap-aware) removes the catch-up spike entirely.
- Requiring persistence (B, C) suppresses any remaining isolated blip, since a missing-data artefact lasts one day and a leak lasts several.
- The same applies to non-monotonic readings: a transient dip followed by recovery makes the next difference look large. Rule C discards non-monotonic readings before computing rates.

**Conclusion:** the dominant source of false alarms is not the leak rule's threshold but how missing readings are converted into consumption. Gap-aware rates, a robust (median) baseline, removal of non-monotonic readings and a persistence requirement together give no false alarms in this data. Limits: the data is synthetic, only random (not long, blocked) outages were simulated, and a gap longer than the leak itself can still hide it.

## 6. Plots

- `plot_overview.png`: 7-day mean consumption for all buildings
- `plot_<building>.png` (six files): readings, rate, moving mean, anomaly markers, true leak shading
- `plot_missing_false_alarms.png`: false-alarm days versus missing-reading rate
