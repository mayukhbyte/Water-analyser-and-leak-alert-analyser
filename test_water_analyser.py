"""
test_water_analyser.py
======================
Unit tests for water_analyser.py.

Covers
------
CO3  WaterMeter special methods (__len__, __repr__, __getitem__, __iter__)
CO4  NumPy consumption, moving_mean, non_monotonic
CO1  score() function (control flow / branching)
CO5  building_report DataFrame structure
BL4  Edge cases: all-NaN readings, naive vs gap-aware diff

Run with:  python -m pytest test_water_analyser.py -v
      or:  python test_water_analyser.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from water_analyser import (
    WaterMeter, score, building_report, load_meters, generate_data
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_meter(readings, start="2026-06-01"):
    dates = pd.date_range(start, periods=len(readings), freq="D")
    return WaterMeter("TestBuilding", dates.values, readings)


# ── CO3: special methods ─────────────────────────────────────────────────────

class TestSpecialMethods:

    def test_len(self):
        """CO3 __len__: returns number of days."""
        m = _make_meter([100.0, 120.0, 140.0])
        assert len(m) == 3

    def test_repr_fields(self):
        """CO3 __repr__: string contains name, days, missing count."""
        m = _make_meter([100.0, 120.0, float("nan")])
        r = repr(m)
        assert "TestBuilding" in r
        assert "days=3" in r
        assert "missing=1" in r

    def test_getitem_slice(self):
        """CO3 __getitem__: slice returns shorter WaterMeter."""
        m = _make_meter([100.0, 110.0, 120.0, 130.0, 140.0])
        sliced = m[1:4]
        assert isinstance(sliced, WaterMeter)
        assert len(sliced) == 3
        np.testing.assert_array_equal(sliced.readings, [110.0, 120.0, 130.0])

    def test_getitem_single(self):
        """CO3 __getitem__: integer index gives a length-1 WaterMeter."""
        m = _make_meter([100.0, 200.0, 300.0])
        single = m[2]          # wrapped to slice(2, 3) internally
        assert len(single) == 1
        assert float(single.readings[0]) == 300.0

    def test_iter_yields_pairs(self):
        """CO3 __iter__: yields (date, reading) tuples."""
        readings = [100.0, 110.0, 120.0]
        m = _make_meter(readings)
        pairs = list(m)
        assert len(pairs) == 3
        _, readings_out = zip(*pairs)
        np.testing.assert_array_almost_equal(list(readings_out), readings)


# ── CO4: NumPy consumption and moving mean ───────────────────────────────────

class TestConsumption:

    def test_gap_aware_no_spike(self):
        """CO4: a one-missing-day gap must not produce a 2x spike."""
        # readings: 1000, NaN, 1020  => rate should be 10/day (not 20)
        m = _make_meter([1000.0, float("nan"), 1020.0])
        rate, gap = m.consumption()
        assert gap[2] == 2
        assert abs(rate[2] - 10.0) < 1e-9

    def test_ffill_gives_catch_up_spike(self):
        """BL4: carry-forward (ffill) produces a catch-up spike on the next valid
        day after a missing reading -- demonstrating the problematic behaviour
        that gap-aware rates are designed to prevent."""
        # readings: 1000, NaN, 1020 -> ffill -> [1000, 1000, 1020]
        # diff -> [NaN, 0, 20] -- the 20 is the spike (should be 10/day)
        m = _make_meter([1000.0, float("nan"), 1020.0])
        ffill = m.ffill_consumption()
        assert ffill[1] == 0.0             # missing day = 0 consumption
        assert abs(ffill[2] - 20.0) < 1e-9  # catch-up spike

    def test_moving_mean_excludes_nan(self):
        """CO4: NaN values must not contaminate the moving mean."""
        x = np.array([10.0, float("nan"), 10.0, 10.0, 10.0])
        result = WaterMeter.moving_mean(x, w=3)
        assert not np.isnan(result[4])
        assert abs(result[4] - 10.0) < 0.5

    def test_non_monotonic_flagged(self):
        """CO4/BL4: a reading dip is correctly flagged."""
        m = _make_meter([1000.0, 1010.0, 1005.0, 1015.0])
        nm = m.non_monotonic()
        assert nm[2] == True
        assert nm[0] == False
        assert nm[1] == False
        assert nm[3] == False

    def test_all_nan_rate_is_nan(self):
        """BL4 edge case: all-NaN readings produce all-NaN rate."""
        m = _make_meter([float("nan"), float("nan"), float("nan")])
        rate, _ = m.consumption()
        assert np.all(np.isnan(rate))


# ── CO1: score() function ────────────────────────────────────────────────────

class TestScore:

    def test_perfect(self):
        pred = np.array([True, True, False, False])
        true = np.array([True, True, False, False])
        tp, fp, fn, prec, rec = score(pred, true)
        assert tp == 2 and fp == 0 and fn == 0
        assert prec == 1.0 and rec == 1.0

    def test_all_fp(self):
        pred = np.array([True, True])
        true = np.array([False, False])
        tp, fp, fn, prec, rec = score(pred, true)
        assert tp == 0 and fp == 2

    def test_no_predictions(self):
        pred = np.array([False, False])
        true = np.array([True, False])
        tp, fp, fn, prec, rec = score(pred, true)
        assert tp == 0 and fp == 0 and fn == 1


# ── CO5: Pandas building_report ──────────────────────────────────────────────

class TestBuildingReport:

    @classmethod
    def _get_rep(cls):
        HERE = os.path.dirname(os.path.abspath(__file__))
        generate_data(seed=42)
        df  = pd.read_csv(os.path.join(HERE, "meter_readings.csv"),
                          parse_dates=["date"])
        return building_report(load_meters(df))

    def test_columns_present(self):
        """CO5: building_report returns all required columns."""
        rep = self._get_rep()
        for col in ["building", "missing", "mean_kL_day",
                    "peak_kL_day", "leak_days_C", "leak_days_D"]:
            assert col in rep.columns, f"Missing column: {col}"

    def test_row_count(self):
        """CO5: one row per building (count must match BUILDINGS dict)."""
        from water_analyser import BUILDINGS
        rep = self._get_rep()
        assert len(rep) == len(BUILDINGS)

    def test_mean_positive(self):
        """CO5: mean daily consumption must be positive."""
        rep = self._get_rep()
        assert (rep["mean_kL_day"] > 0).all()


# ── simple runner (works without pytest) ────────────────────────────────────

if __name__ == "__main__":
    suites = [TestSpecialMethods, TestConsumption, TestScore, TestBuildingReport]
    passed = failed = 0
    for suite_cls in suites:
        obj = suite_cls()
        for attr in sorted(dir(obj)):
            if not attr.startswith("test_"):
                continue
            try:
                getattr(obj, attr)()
                print(f"  PASS  {suite_cls.__name__}.{attr}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {suite_cls.__name__}.{attr}  -- {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
