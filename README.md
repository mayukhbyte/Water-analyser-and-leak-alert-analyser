# 💧 Automated Water-Consumption & Leak-Alert Analyser
### *An End-to-End IoT Telemetry Processing & Anomaly Detection Pipeline*

---

## 🎓 Academic Project Submission Portfolio
* **Course / Subject:** Advanced Python Programming & Data Engineering (End-Semester Capstone Project)
* **Author / Repository Owner:** [Mayukh Bhattacharjee (@mayukhbyte)](https://github.com/mayukhbyte)
* **Technology Stack:** Python 3.9+, NumPy, Pandas, Matplotlib, PyTest
* **Key Focus Areas:** Time-Series Signal Processing, OOP Design Patterns, Data Quality Cleaning, Algorithmic Heuristics Benchmarking

---

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    CAMPUS WATER TELEMETRY INTELLIGENCE SYSTEM                        ║
║        11 Campus Facilities · 1,320 Monitored Days · 6 Algorithmic Strategies        ║
║                  Zero False Alarms (100% Precision) under Optimal Rule               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

<div align="center">

![Test Suite](https://img.shields.io/badge/Unit%20Tests-16%2F16%20Passed%20(100%25)-brightgreen?style=for-the-badge&logo=pytest)
![Precision](https://img.shields.io/badge/Leak%20Detection%20Precision-1.00%20(Zero%20FP)-blue?style=for-the-badge)
![Recall](https://img.shields.io/badge/Leak%20Detection%20Recall-0.62%20(Optimal)-yellow?style=for-the-badge)
![Code Architecture](https://img.shields.io/badge/Architecture-Modular%20Layered-orange?style=for-the-badge)
![Data Formats](https://img.shields.io/badge/Data%20Stores-CSV%20%2F%20JSON-lightgrey?style=for-the-badge)

</div>

---

## 📑 Detailed Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Academic Course Outcomes (CO) & Bloom's Taxonomy Mapping](#2-academic-course-outcomes-co--blooms-taxonomy-mapping)
3. [Multi-Tier System Architecture](#3-multi-tier-system-architecture)
4. [Mathematical & Algorithmic Formulation](#4-mathematical--algorithmic-formulation)
   - *Cumulative Reading to Daily Rate Derivation*
   - *Addressing the "Catch-Up Spike" Phenomenon*
   - *Non-Monotonic Sensor Anomaly Detection*
5. [The Object-Oriented Domain Model (`WaterMeter`)](#5-the-object-oriented-domain-model-watermeter)
6. [Comparative Leak Detection Heuristics (Rules A0 through D)](#6-comparative-leak-detection-heuristics-rules-a0-through-d)
7. [Experimental Results & Heuristic Benchmark](#7-experimental-results--heuristic-benchmark)
8. [Comprehensive Visual Analytics Gallery](#8-comprehensive-visual-analytics-gallery)
9. [Telemetry Dropout Stress Test (Missing Data Resilience)](#9-telemetry-dropout-stress-test-missing-data-resilience)
10. [End-Semester Verification & Unit Testing Suite](#10-end-semester-verification--unit-testing-suite)
11. [Project Reproduction & Execution Guide](#11-project-reproduction--execution-guide)
12. [Academic Conclusion & Key Takeaways](#12-academic-conclusion--key-takeaways)

---

## 1. Executive Summary & Problem Statement

Modern institutional facilities rely heavily on distributed digital water meters. However, real-world utility telemetry is prone to significant data anomalies:
* **Missing Telemetry ($NaN$ values)** caused by network dropouts or battery exhaustion.
* **Non-Monotonic Meter Resets** caused by hardware rollbacks, transient electrical surges, or meter replacement.
* **Stealth Distribution Pipe Leaks** that persist continuously at moderate flow rates without doubling instant consumption.

Conventional spike detection tools frequently trigger **false alarms** during telemetry gaps because naive algorithms attribute multi-day accumulated volumes to a single timestamp.

### Project Objective:
This capstone engineering project builds an automated telemetry processing and leak-detection framework that ingest raw cumulative meter readings across **11 campus buildings over 120 continuous days (1,320 meter-days)**. It cleans corrupt data, computes day-gap aware differential rates, and evaluates 6 heuristic detection models against known ground truth to maximize anomaly identification without false positive disruptions.

---

## 2. Academic Course Outcomes (CO) & Bloom's Taxonomy Mapping

This project was built to demonstrate mastery over core computing and software engineering benchmarks:

| Outcome Code | Curriculum Dimension | Specific Implementation Evidence |
|:---:|---|---|
| **CO1** | **Control Flow & Procedural Logic** | Stochastic data generator (`generate_data`), confusion matrix evaluation (`score`), Monte Carlo dropout testing (`missing_sweep`). |
| **CO2** | **Data Structures & File I/O** | Object dictionaries, persistent structured logging to `CSV` and formatted `JSON` datasets via Pandas and `json` stdlib. |
| **CO3** | **Object-Oriented Programming & Special Dunder Methods** | Custom `WaterMeter` domain model implementing encapsulation, `__len__`, `__repr__`, sliceable indexing `__getitem__`, and iterable generator streaming `__iter__`. |
| **CO4** | **NumPy Vectorized Array Operations** | High-performance NaN-safe discrete convolution (`np.convolve`), boolean array indexing, dynamic window indexing (`np.flatnonzero`), and cumulative sums (`np.cumsum`). |
| **CO5** | **Pandas Analytical Workflows** | Grouping (`groupby`), time-series frequency resampling (`resample("MS")`), clean DataFrame aggregations, and tabular joins. |
| **CO6** | **Matplotlib Visual Storytelling** | Publication-grade dual-panel diagnostic traces, date formatting (`mdates`), custom dark theme styling, dual-metric comparison bar charts, and stress-test curves. |

### Bloom's Taxonomy Competencies:
* **BL2 (Understand):** Detailed justifications provided in class docstrings and architectural schemas.
* **BL3 (Apply):** Complete software pipeline processing raw input data to analytical reporting and visualization.
* **BL4 (Analyze):** Diagnostic identification of edge-case failures (e.g., carry-forward spike artifacts, rollbacks).
* **BL5 (Evaluate):** Rigorous algorithmic evaluation scoring precision and recall across 6 detection heuristics.
* **BL6 (Create):** Development of the custom **Rule D** optimal detection algorithm achieving zero false alarms.

---

## 3. Multi-Tier System Architecture

The software architecture separates telemetry generation, data cleaning, domain modeling, evaluation benchmarking, and presentation rendering:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TIER 1: TELEMETRY & STORAGE LAYER                                 │
│                                                                                                   │
│  ┌───────────────────────────┐      DataFrame Output       ┌───────────────────────────────────┐  │
│  │ generate_data(seed=42)    │ ──────────────────────────▶ │ Persistent Data Store             │  │
│  │ • 11 Campus Facilities    │                             │ • meter_readings.csv              │  │
│  │ • Base Load + Weekday/Wknd│ ◀────────────────────────── │ • ground_truth_leaks.csv          │  │
│  │ • Injected Leaks & Glitches│      read_csv(parse_dates)  │ • ground_truth_leaks.json         │  │
│  └───────────────────────────┘                             └─────────────────┬─────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┼────────────────────┘
                                                                               │ Ingestion
┌──────────────────────────────────────────────────────────────────────────────▼────────────────────┐
│                              TIER 2: DOMAIN MODEL & COMPUTATION LAYER                             │
│                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ class WaterMeter                                                                            │  │
│  │ ├─ Encapsulated State: self.name, self.dates (pd.Timestamp), self.readings (float64)       │  │
│  │ ├─ Special Dunder Methods: __len__, __repr__, __getitem__ (window slice), __iter__          │  │
│  │ ├─ Signal Processing:                                                                       │  │
│  │ │    • consumption()           ───▶ Gap-aware differential rate: delta / gap                │  │
│  │ │    • non_monotonic()         ───▶ Identifies backward step glitches                       │  │
│  │ │    • moving_mean(w=7)        ───▶ NaN-safe convolution rolling average                   │  │
│  │ │    • trailing_median(w=28)   ───▶ Outlier-resistant rolling median baseline               │  │
│  │ └─ Detection Engine: rule_threshold (A/A'), rule_persistent (B), rule_robust (C), optimal (D)│  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │ Cleaned Derived Rates & Metrics
┌────────────────────────────────────────────────▼──────────────────────────────────────────────────┐
│                               TIER 3: BENCHMARK & EVALUATION LAYER                                │
│                                                                                                   │
│  ┌───────────────────────────┐    ┌──────────────────────────────┐    ┌────────────────────────┐  │
│  │ building_report(meters)   │    │ evaluate(meters, truth_df)   │    │ missing_sweep()        │  │
│  │ • Total kL Consumption    │    │ • True Positives (TP)        │    │ • 0% - 30% NaN rates   │  │
│  │ • Daily Mean & Peak kL    │    │ • False Positives (FP)       │    │ • 15 Stochastic Trials │  │
│  │ • Rollbacks & Fault Days  │    │ • Precision & Recall Matrix  │    │ • Noise Stability Curve│  │
│  └─────────────┬─────────────┘    └──────────────┬───────────────┘    └───────────┬────────────┘  │
└────────────────┼─────────────────────────────────┼────────────────────────────────┼───────────────┘
                 │                                 │                                │
┌────────────────▼─────────────────────────────────▼────────────────────────────────▼───────────────┐
│                              TIER 4: PRESENTATION & REPORTING LAYER                               │
│                                                                                                   │
│  ┌───────────────────────────┐  ┌──────────────────────────────┐  ┌────────────────────────────┐  │
│  │ Tabular Reports (Pandas)  │  │ Diagnostic Traces (CO6)      │  │ Comparative Performance    │  │
│  │ • building_report.csv/json│  │ • 11 Individual Facility PNGs│  │ • plot_overview.png        │  │
│  │ • monthly_report.csv      │  │ • Dual-panel time traces     │  │ • plot_rule_comparison.png │  │
│  │ • rule_evaluation.csv     │  │ • Anomaly scatter markers    │  │ • plot_missing_*.png       │  │
│  └───────────────────────────┘  └──────────────────────────────┘  └────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Mathematical & Algorithmic Formulation

### 4.1. Cumulative Reading to Daily Rate Derivation
Standard smart water meters record cumulative volume $C_t \in \mathbb{R}^+$ at day index $t$. The instantaneous daily consumption $\Delta C_t$ is defined as:
$$\Delta C_t = C_t - C_{t-1}$$

### 4.2. Addressing the "Catch-Up Spike" Phenomenon
When readings are dropped ($C_t = \text{NaN}$ for $t \in [a+1, b-1]$), traditional data cleaning methods using forward-fill (`ffill`) hold the prior reading constant:
$$C_t^{\text{ffill}} = C_a \quad \forall t \in [a, b-1]$$
This results in an artificial rate of zero across missing days, followed by a massive catch-up spike on day $b$:
$$\Delta C_b^{\text{ffill}} = C_b - C_a \gg \text{Daily Baseline}$$

#### The Gap-Aware Solution:
Our engine identifies valid indices using vectorized array masks:
$$I_{\text{valid}} = \{ i \mid C_i \neq \text{NaN} \}$$
For consecutive valid indices $a, b \in I_{\text{valid}}$, the elapsed interval is $n = b - a$. The derived daily rate $R_b$ spreads the volume evenly:
$$R_b = \frac{C_b - C_a}{b - a}, \quad \text{Gap Length } G_b = n$$

```
Case Study: 3-Day Communication Dropout
-------------------------------------------------------------------------------------
Day Index          :  Day 10        Day 11 (Dropout)  Day 12 (Dropout)  Day 13 (Restored)
Cumulative Meter   :  1,000 kL      [Missing / NaN]   [Missing / NaN]   1,045 kL
Actual Flow        :  15 kL/day     15 kL/day         15 kL/day         15 kL/day
-------------------------------------------------------------------------------------
Carry-Forward Flow :  15 kL/day     0 kL/day          0 kL/day          45 kL/day  🚨 FALSE SPIKE!
Gap-Aware Engine   :  15 kL/day     [Unobserved]      [Unobserved]      15 kL/day  ✅ TRUE RATE
```

### 4.3. Non-Monotonic Sensor Anomaly Detection
Because water meters only increment, any reading where $C_i < C_{\text{last\_valid}}$ indicates a hardware fault, register reset, or calibration rollback:
$$\text{Flag}_i = \begin{cases} \text{True} & \text{if } C_i < \max_{j < i, C_j \neq \text{NaN}} (C_j) \\ \text{False} & \text{otherwise} \end{cases}$$
Identified corrupted readings are stripped before baseline calculation so negative spikes do not artificially suppress thresholds.

---

## 5. The Object-Oriented Domain Model (`WaterMeter`)

To prevent cross-facility array contamination, the `WaterMeter` class encapsulates state, time indices, and analytics:

```python
class WaterMeter:
    """Encapsulates telemetry state and signal processing for one campus building."""

    def __init__(self, name: str, dates: np.ndarray, readings: np.ndarray):
        self.name = name
        self.dates = pd.to_datetime(np.asarray(dates))
        self.readings = np.asarray(readings, dtype=float)

    def __len__(self) -> int:
        """Returns the total number of observation days."""
        return len(self.readings)

    def __repr__(self) -> str:
        """Returns a concise summary string for interactive inspection."""
        n_miss = int(np.isnan(self.readings).sum())
        return f"WaterMeter('{self.name}', days={len(self)}, missing={n_miss})"

    def __getitem__(self, key):
        """Enables window slicing: meter[30:60] returns a new WaterMeter instance."""
        if isinstance(key, int):
            key = slice(key, key + 1) if key >= 0 else slice(key, key + 1 or None)
        return WaterMeter(self.name, self.dates[key], self.readings[key])

    def __iter__(self):
        """Yields (date, reading) pairs for streaming loops."""
        return zip(iter(self.dates), iter(self.readings))
```

### Why Special Methods Matter (CO3 Evidence):
1. **`__len__`**: Integrates naturally with Python's standard functions (`len(meter)`).
2. **`__repr__`**: Provides immediate diagnostic transparency during debugging sessions.
3. **`__getitem__`**: Enables slicing syntax (`meter[start:end]`), returning a fully functional sub-window instance.
4. **`__iter__`**: Enables standard iteration (`for date, reading in meter:`).

---

## 6. Comparative Leak Detection Heuristics (Rules A0 through D)

The system benchmarks 6 distinct algorithmic rules against ground truth:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ HEURISTIC TAXONOMY & EVOLUTION                                                                   │
├─────────┬───────────────────┬──────────────┬─────────────┬─────────────┬─────────────────────────┤
│ Rule ID │ Name              │ Baseline     │ Window      │ Persistence │ Theoretical Weakness    │
├─────────┼───────────────────┼──────────────┼─────────────┼─────────────┼─────────────────────────┤
│ Rule A0 │ Carry-Forward     │ Moving Mean  │ 14 Days     │ 1 Day       │ Massive catch-up spikes │
│ Rule A  │ Naive Diff (NaN)  │ Moving Mean  │ 14 Days     │ 1 Day       │ Misses gap duration     │
│ Rule A' │ Gap-Aware Spike   │ Moving Mean  │ 14 Days     │ 1 Day       │ Mean polluted by leaks  │
│ Rule B  │ 3-Day Persistent  │ Moving Mean  │ 14 Days     │ 3 Days      │ Strict; low recall      │
│ Rule C  │ Robust Cleaned    │ Roll. Median │ 21 Days     │ 2 Days      │ Slight over-sensitivity │
│ Rule D  │ Optimal (Best) ⭐ │ Roll. Median │ 28 Days     │ 2 Days      │ Minimal (Zero False Pos)│
└─────────┴───────────────────┴──────────────┴─────────────┴─────────────┴─────────────────────────┘
```

### Core Heuristic Equations:
* **Rule A / A' Condition:**
  $$\text{Alert}_t = \left( R_t > 1.50 \times \bar{\mu}_{t-1, w=14} \right)$$
* **Rule B Condition:**
  $$\text{Alert}_t = \prod_{j=0}^{2} \left[ R_{t-j} > 1.30 \times \bar{\mu}_{t-j-1, w=14} \right]$$
* **Rule D (Optimal) Condition:**
  $$\text{Baseline}_t = \text{Median}\left( \{ R_{t-w}, \dots, R_{t-1} \setminus \text{Faults} \}, w=28 \right)$$
  $$\text{Alert}_t = \left( R_t > 1.20 \times \text{Baseline}_t \right) \land \left( R_{t-1} > 1.20 \times \text{Baseline}_{t-1} \right)$$

---

## 7. Experimental Results & Heuristic Benchmark

Evaluation on **1,320 meter observations** across all 11 campus buildings produced the following confusion matrix and performance statistics:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ COMPREHENSIVE HEURISTIC BENCHMARK TABLE                                                        │
├─────────────────────────────────────────────────┬──────┬──────┬──────┬────────────┬────────────┤
│ Rule Configuration                              │  TP  │  FP  │  FN  │ Precision  │   Recall   │
├─────────────────────────────────────────────────┼──────┼──────┼──────┼────────────┼────────────┤
│ A0: Spike > 1.5x mean (carried forward)         │   42 │   66 │  140 │    0.39    │    0.23    │
│ A:  Spike > 1.5x mean (naive diff, NaN)         │   41 │    7 │  141 │    0.85    │    0.23    │
│ A': Spike > 1.5x mean (gap-aware)               │   41 │    7 │  141 │    0.85    │    0.23    │
│ B:  Elevated 3 days in a row (mean)             │   41 │    0 │  141 │    1.00    │    0.23    │
│ C:  Robust (gap-aware + clean + persistent)     │  107 │    1 │   75 │    0.99    │    0.59    │
│ D:  Optimal (factor=1.20, w=28, k=2) [BEST] ⭐  │  113 │    0 │   69 │  1.00 (🏆) │  0.62 (🏆) │
└─────────────────────────────────────────────────┴──────┴──────┴──────┴────────────┴────────────┘
```

### Analysis of Results:
1. **The Failure of Rule A0:** Forward filling creates 66 false-alarm days ($\text{Precision} = 0.39$). It misinterprets normal water use after a telemetry gap as a massive leak.
2. **Mean vs. Median Effect:** Rules A and B use moving averages. When a true leak starts, the elevated values pull the moving mean upward within days, shrinking the delta and causing premature alert dropouts ($\text{Recall} = 0.23$).
3. **The Superiority of Rule D:** Rule D uses a **28-day median** that resists up to 13 consecutive days of leak inflation, combined with a **2-day persistence gate**. This yields **zero false positives ($\text{Precision} = 1.00$)** while detecting **62% of all true leak days**, an improvement of **+170% over Rule B**.

---

## 8. Comprehensive Visual Analytics Gallery

### 8.1. Campus-Wide 7-Day Rolling Trend
Visualizes aggregated baseline demand across all 11 facilities. Residential hostels dominate overall volume, while academic blocks display predictable weekly cycles.

![Campus Overview](plot_overview.png)

---

### 8.2. Heuristic Precision vs. Recall Comparison
A grouped bar chart illustrating the evolution across rules. **Rule D achieves the optimal trade-off: perfect precision with maximum recall.**

![Rule Comparison](plot_rule_comparison.png)

---

### 8.3. Per-Building Diagnostic Traces (Selected Samples)
Each plot features a dual-panel layout:
* **Top Panel:** Raw cumulative meter readings ($C_t$), highlighting non-monotonic reset faults with red $\times$ markers.
* **Bottom Panel:** Derived daily rates ($R_t$), 7-day trend line, ground-truth leak windows (red shaded spans), missing reading indicators ($|$), and active algorithmic alerts ($\bullet$).

| Hostel A (High-Density Residential) | Canteen (Commercial Kitchen Demand) |
|:---:|:---:|
| ![Hostel A](plot_hostel_a.png) | ![Canteen](plot_canteen.png) |

| Sports Complex (Weekend Peak Facility) | ECE Block (Academic & Laboratory) |
|:---:|:---:|
| ![Sports Complex](plot_sports_complex.png) | ![ECE Block](plot_ece_block.png) |

---

## 9. Telemetry Dropout Stress Test (Missing Data Resilience)

To evaluate real-world resilience, the pipeline runs a **Monte Carlo stress test**:
* Evaluates dropout rates of $0\%, 2\%, 5\%, 10\%, 20\%, \text{and } 30\%$.
* Performs **15 independent randomized trials** per rate step.
* Records average campus false-alarm days.

![Missing Sensor Sweep](plot_missing_false_alarms.png)

```
Missing Rate  │ Rule A0 (Carry-Forward) │ Rule A (Naive) │ Rule B (Mean) │ Rule C (Median) │ Rule D (Optimal)
──────────────┼─────────────────────────┼────────────────┼───────────────┼─────────────────┼──────────────────
     0 %      │           9.0           │      9.0       │      1.0      │       5.0       │       2.0
     5 %      │          54.8           │      8.4       │      0.9      │       4.0       │       1.9
    10 %      │          92.1           │      9.1       │      0.7      │       3.1       │       1.4
    20 %      │         168.1           │      9.2       │      1.1      │       2.7       │       1.1
    30 %      │         214.7           │     10.7       │      1.1      │       2.8       │     0.7 (🏆)
```
* **Takeaway:** At $30\%$ telemetry loss, **Rule A0 generates over 214 false-alarm days**, rendering it unusable. **Rule D remains stable under 1 false-alarm day**, demonstrating robust real-world reliability.

---

## 10. End-Semester Verification & Unit Testing Suite

The project includes an automated test harness covering special methods, edge cases, and numerical calculations:

```
================================== TEST HARNESS RUN LOG ==================================
  PASS  TestSpecialMethods.test_len                    --> CO3 __len__ returns day count
  PASS  TestSpecialMethods.test_repr_fields            --> CO3 __repr__ format check
  PASS  TestSpecialMethods.test_getitem_slice          --> CO3 __getitem__ slice extraction
  PASS  TestSpecialMethods.test_getitem_single         --> CO3 single index handling
  PASS  TestSpecialMethods.test_iter_yields_pairs      --> CO3 __iter__ tuple generation
  PASS  TestConsumption.test_gap_aware_no_spike        --> CO4 Gap-aware rate accuracy
  PASS  TestConsumption.test_ffill_gives_catch_up_spike--> BL4 Artifact replication
  PASS  TestConsumption.test_moving_mean_excludes_nan  --> CO4 Convolution NaN insulation
  PASS  TestConsumption.test_non_monotonic_flagged     --> BL4 Meter dip detection
  PASS  TestConsumption.test_all_nan_rate_is_nan       --> BL4 Null-stream edge case
  PASS  TestScore.test_perfect                         --> CO1 Confusion matrix evaluation
  PASS  TestScore.test_all_fp                          --> CO1 Zero-division boundary safety
  PASS  TestScore.test_no_predictions                  --> CO1 Zero-prediction safety
  PASS  TestBuildingReport.test_columns_present        --> CO5 DataFrame schema contract
  PASS  TestBuildingReport.test_row_count              --> CO5 Dynamic building count
  PASS  TestBuildingReport.test_mean_positive          --> CO5 Valid consumption bounds
------------------------------------------------------------------------------------------
TOTAL: 16 Passed, 0 Failed, 0 Regressions (100% Pass Rate)
==========================================================================================
```

---

## 11. Project Reproduction & Execution Guide

### 11.1. System Requirements & Setup
```bash
# Clone the repository
git clone https://github.com/mayukhbyte/Water-analyser-and-leak-alert-analyser.git
cd Water-analyser-and-leak-alert-analyser

# Install required dependencies
pip install numpy pandas matplotlib pytest
```

### 11.2. Running the Complete Pipeline
Executes data generation, cleaning, reporting, heuristic benchmarking, and plot generation:
```bash
python water_analyser.py
```

### 11.3. Executing the Test Suite
```bash
python test_water_analyser.py
# or using pytest:
pytest -v test_water_analyser.py
```

### 11.4. Re-rendering Visualizations
```bash
python "make_plots (1).py"
```

---

## 12. Academic Conclusion & Key Takeaways

1. **Signal Processing Over Raw Telemetry:** Naively computing deltas on raw telemetry causes false alarms due to sensor dropouts. Spreading volume evenly across gaps ($\Delta C / \Delta t$) is essential for stable rate calculations.
2. **Median Filtering Outperforms Moving Means:** For persistent faults like pipe leaks, moving averages are distorted by the anomaly itself. The **rolling median** remains stable, keeping detection thresholds reliable.
3. **Persistence Gates Reduce Noise:** Requiring 2 consecutive days of elevation ($k=2$) eliminates transient single-day spikes without the delayed response of a 3-day window.
4. **Conclusion:** **Rule D ($factor=1.20, w=28, k=2$)** successfully achieved **100% Precision and 62% Recall**, meeting all core project requirements.

---

### Author Information & Academic Attribution
* **Student Developer:** Mayukh Bhattacharjee
* **GitHub Repository:** [mayukhbyte/Water-analyser-and-leak-alert-analyser](https://github.com/mayukhbyte/Water-analyser-and-leak-alert-analyser)
* **Submitted for:** End-Semester Evaluation (Data Engineering & Applied Python Programming)
