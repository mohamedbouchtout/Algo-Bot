"""Integration tests for the retrain trigger."""

import pytest

from strategy.ai_analysis.retrain_trigger import RetrainTrigger
from tests.conftest import requires_network


@pytest.mark.integration
class TestRetrainTrigger:
    """Integration tests for the retrain trigger logic."""

    def test_regime_shift_vix_spike(self):
        trigger = RetrainTrigger(vix_change_pct=0.50, spy_drop_pct=0.05)
        trigger.snapshot_from_bars(vix_close=20.0, spy_close=450.0)

        assert trigger.check_regime_shift_from_bars(current_vix=35.0, current_spy=450.0) is True

    def test_regime_shift_spy_crash(self):
        trigger = RetrainTrigger(vix_change_pct=0.50, spy_drop_pct=0.05)
        trigger.snapshot_from_bars(vix_close=20.0, spy_close=450.0)

        assert trigger.check_regime_shift_from_bars(current_vix=20.0, current_spy=420.0) is True

    def test_no_regime_shift_stable(self):
        trigger = RetrainTrigger(vix_change_pct=0.50, spy_drop_pct=0.05)
        trigger.snapshot_from_bars(vix_close=20.0, spy_close=450.0)

        assert trigger.check_regime_shift_from_bars(current_vix=22.0, current_spy=445.0) is False

    def test_accuracy_degradation_triggers(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=10)

        preds = [0, 1, 2, 0, 1, 2, 0, 1, 0, 0]
        actuals = [2, 2, 0, 1, 0, 1, 2, 0, 1, 2]
        assert trigger.check_accuracy(preds, actuals) is True

    def test_accuracy_good_no_trigger(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=10)

        preds = [0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
        actuals = [0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
        assert trigger.check_accuracy(preds, actuals) is False

    def test_accuracy_insufficient_data_no_trigger(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=20)

        preds = [0, 1, 2]
        actuals = [2, 2, 0]
        assert trigger.check_accuracy(preds, actuals) is False

    def test_no_snapshot_no_regime_shift(self):
        trigger = RetrainTrigger()
        assert trigger.check_regime_shift_from_bars(30.0, 400.0) is False

    @requires_network
    def test_live_snapshot(self):
        trigger = RetrainTrigger()
        trigger.snapshot_market()

        assert trigger._train_vix is not None or trigger._train_spy is not None
