"""Tests for forecast evaluation metrics with known analytical values."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.evaluate import directional_accuracy, mae, mape, rmse


class TestRMSE:
    def test_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        actual = np.array([2.0, 4.0, 6.0])
        pred = np.array([1.0, 3.0, 5.0])
        # All errors = 1.0, RMSE = sqrt(1) = 1.0
        assert rmse(actual, pred) == pytest.approx(1.0)

    def test_asymmetric_errors(self):
        actual = np.array([10.0, 20.0])
        pred = np.array([7.0, 20.0])
        # errors: [3, 0], MSE = 9/2 = 4.5, RMSE = sqrt(4.5)
        assert rmse(actual, pred) == pytest.approx(np.sqrt(4.5))

    def test_returns_float(self):
        assert isinstance(rmse(np.array([1.0]), np.array([2.0])), float)


class TestMAE:
    def test_perfect_forecast(self):
        y = np.array([5.0, 10.0, 15.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        actual = np.array([3.0, 6.0, 9.0])
        pred = np.array([2.0, 5.0, 7.0])
        # errors: [1, 1, 2], MAE = 4/3
        assert mae(actual, pred) == pytest.approx(4.0 / 3.0)

    def test_symmetric_property(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 1.0, 2.0])
        assert mae(actual, pred) == mae(pred, actual)

    def test_returns_float(self):
        assert isinstance(mae(np.array([1.0]), np.array([2.0])), float)


class TestMAPE:
    def test_perfect_forecast(self):
        y = np.array([100.0, 200.0, 300.0])
        assert mape(y, y) == pytest.approx(0.0, abs=1e-4)

    def test_known_value_ten_percent(self):
        actual = np.array([100.0, 200.0])
        pred = np.array([110.0, 220.0])
        # Both 10% off → MAPE ≈ 10.0 (subject to epsilon)
        result = mape(actual, pred)
        assert result == pytest.approx(10.0, rel=0.01)

    def test_returns_percentage_scale(self):
        actual = np.array([1000.0, 2000.0])
        pred = np.array([1100.0, 2100.0])
        result = mape(actual, pred)
        # Should be a percentage, not a fraction
        assert result > 1.0

    def test_returns_float(self):
        assert isinstance(mape(np.array([100.0]), np.array([110.0])), float)


class TestDirectionalAccuracy:
    def test_perfect_direction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 1.5, 2.5, 3.5])
        assert directional_accuracy(actual, pred) == pytest.approx(1.0)

    def test_all_wrong_direction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])   # always up
        pred = np.array([4.0, 3.0, 2.0, 1.0])       # always down
        assert directional_accuracy(actual, pred) == pytest.approx(0.0)

    def test_half_correct(self):
        actual = np.array([1.0, 2.0, 1.0, 2.0])   # up, down, up
        pred = np.array([1.0, 2.0, 2.0, 1.0])      # up, up, down  → 1 of 3 match
        result = directional_accuracy(actual, pred)
        assert result == pytest.approx(1.0 / 3.0, rel=0.01)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            directional_accuracy(np.array([1.0]), np.array([2.0]))

    def test_returns_float(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 1.5, 2.5])
        assert isinstance(directional_accuracy(actual, pred), float)
