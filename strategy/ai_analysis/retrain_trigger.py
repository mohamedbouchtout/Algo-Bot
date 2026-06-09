"""
Monitors market conditions and model accuracy to trigger early retraining.

Two triggers:

1. **Regime shift**: VIX or SPY moves significantly from the levels seen at
   last training time, indicating the market environment has changed enough
   that the model's learned patterns may no longer apply.

2. **Accuracy degradation**: Walk-forward validation accuracy on recent data
   drops below a threshold, indicating the model is making poor predictions
   regardless of whether the market "looks" different.

Either trigger firing means the model should retrain before the regular
weekend schedule.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class RetrainTrigger:
    def __init__(
        self,
        vix_change_pct: float = 0.50,
        spy_drop_pct: float = 0.05,
        min_accuracy: float = 0.40,
        lookback_bars: int = 20,
    ):
        """
        Parameters
        ----------
        vix_change_pct  : trigger if VIX moves more than this fraction from
                          the level at last training (e.g. 0.50 = 50% change).
        spy_drop_pct    : trigger if SPY drops more than this fraction from
                          the level at last training (e.g. 0.05 = 5% drop).
        min_accuracy    : trigger if walk-forward accuracy on recent predictions
                          falls below this threshold.
        lookback_bars   : number of recent trading days to evaluate for
                          accuracy checks.
        """
        self.vix_change_pct = vix_change_pct
        self.spy_drop_pct = spy_drop_pct
        self.min_accuracy = min_accuracy
        self.lookback_bars = lookback_bars

        # Snapshot of market levels at last training time
        self._train_vix: float | None = None
        self._train_spy: float | None = None

    def snapshot_market(self) -> None:
        """Capture current VIX and SPY levels. Call after each training run."""
        try:
            vix = yf.Ticker('^VIX').fast_info.get('lastPrice', None)
            spy = yf.Ticker('SPY').fast_info.get('lastPrice', None)
            self._train_vix = float(vix) if vix else None
            self._train_spy = float(spy) if spy else None
            logger.info(f'Retrain trigger snapshot: VIX={self._train_vix}, SPY={self._train_spy}')
        except Exception as e:
            logger.warning(f'Failed to snapshot market for retrain trigger: {e}')

    def snapshot_from_bars(self, vix_close: float, spy_close: float) -> None:
        """Set snapshot from known values (used by backtest instead of live fetch)."""
        self._train_vix = vix_close
        self._train_spy = spy_close

    def check_regime_shift(self) -> bool:
        """
        Check if VIX or SPY has moved enough from the training snapshot
        to warrant an early retrain. Returns True if triggered.
        """
        if self._train_vix is None or self._train_spy is None:
            return False

        try:
            current_vix = yf.Ticker('^VIX').fast_info.get('lastPrice', None)
            current_spy = yf.Ticker('SPY').fast_info.get('lastPrice', None)

            if current_vix is None or current_spy is None:
                return False

            current_vix = float(current_vix)
            current_spy = float(current_spy)

            vix_change = abs(current_vix - self._train_vix) / self._train_vix
            spy_change = (self._train_spy - current_spy) / self._train_spy

            if vix_change >= self.vix_change_pct:
                logger.info(f'Regime shift trigger: VIX moved {vix_change:.1%} (from {self._train_vix:.1f} to {current_vix:.1f})')
                return True

            if spy_change >= self.spy_drop_pct:
                logger.info(f'Regime shift trigger: SPY dropped {spy_change:.1%} (from {self._train_spy:.1f} to {current_spy:.1f})')
                return True

            return False

        except Exception as e:
            logger.warning(f'Regime shift check failed: {e}')
            return False

    def check_regime_shift_from_bars(self, current_vix: float, current_spy: float) -> bool:
        """
        Same as check_regime_shift but with explicit values (for backtest).
        Returns True if triggered.
        """
        if self._train_vix is None or self._train_spy is None:
            return False

        vix_change = abs(current_vix - self._train_vix) / self._train_vix
        spy_change = (self._train_spy - current_spy) / self._train_spy

        if vix_change >= self.vix_change_pct:
            logger.info(f'Regime shift trigger: VIX moved {vix_change:.1%} (from {self._train_vix:.1f} to {current_vix:.1f})')
            return True

        if spy_change >= self.spy_drop_pct:
            logger.info(f'Regime shift trigger: SPY dropped {spy_change:.1%} (from {self._train_spy:.1f} to {current_spy:.1f})')
            return True

        return False

    def check_accuracy(self, recent_predictions: list[int], recent_actuals: list[int]) -> bool:
        """
        Check if model accuracy on recent predictions is below threshold.
        Returns True if retrain is needed.

        Parameters
        ----------
        recent_predictions : predicted class labels for recent bars.
        recent_actuals     : actual class labels for recent bars.
        """
        if len(recent_predictions) < self.lookback_bars:
            return False

        preds = np.array(recent_predictions[-self.lookback_bars :])
        actuals = np.array(recent_actuals[-self.lookback_bars :])
        accuracy = (preds == actuals).mean()

        if accuracy < self.min_accuracy:
            logger.info(f'Accuracy trigger: recent accuracy {accuracy:.1%} below threshold {self.min_accuracy:.1%}')
            return True

        return False

    def should_retrain(
        self,
        recent_predictions: list[int] | None = None,
        recent_actuals: list[int] | None = None,
    ) -> bool:
        """
        Combined check: returns True if either regime shift or accuracy
        degradation triggers fire.
        """
        if self.check_regime_shift():
            return True

        if recent_predictions and recent_actuals:
            if self.check_accuracy(recent_predictions, recent_actuals):
                return True

        return False
