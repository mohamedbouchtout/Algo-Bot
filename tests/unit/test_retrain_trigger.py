"""Unit tests for the retrain trigger."""

from strategy.ai_analysis.retrain_trigger import RetrainTrigger


class TestRetrainTrigger:
    def test_no_trigger_without_snapshot(self):
        trigger = RetrainTrigger()
        assert trigger.check_regime_shift_from_bars(30.0, 400.0) is False

    def test_vix_spike_triggers(self):
        trigger = RetrainTrigger(vix_change_pct=0.50)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(35.0, 500.0) is True

    def test_vix_stable_no_trigger(self):
        trigger = RetrainTrigger(vix_change_pct=0.50)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(25.0, 500.0) is False

    def test_spy_drop_triggers(self):
        trigger = RetrainTrigger(spy_drop_pct=0.05)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(20.0, 470.0) is True

    def test_spy_stable_no_trigger(self):
        trigger = RetrainTrigger(spy_drop_pct=0.05)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(20.0, 490.0) is False

    def test_spy_rise_no_trigger(self):
        trigger = RetrainTrigger(spy_drop_pct=0.05)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(20.0, 550.0) is False

    def test_accuracy_below_threshold_triggers(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=10)
        preds = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
        actuals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
        assert trigger.check_accuracy(preds, actuals) is True

    def test_accuracy_above_threshold_no_trigger(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=10)
        preds = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        actuals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert trigger.check_accuracy(preds, actuals) is False

    def test_accuracy_not_enough_data(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=20)
        preds = [0, 0, 0, 0, 0]
        actuals = [1, 1, 1, 1, 1]
        assert trigger.check_accuracy(preds, actuals) is False

    def test_snapshot_from_bars(self):
        trigger = RetrainTrigger()
        trigger.snapshot_from_bars(25.0, 450.0)
        assert trigger._train_vix == 25.0
        assert trigger._train_spy == 450.0
