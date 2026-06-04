"""Unit tests for RiskManager — pure logic, no IB connection required."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from execution.risk_manager import RiskManager

PARAMS = {
    'strategy_retest_200ma': {
        'ATR': 1.5,
    },
    'risk_management': {
        'risk_per_trade_pct': 0.03,
        'max_investment_pct': 0.70,
        'max_positions': 10,
        'max_position_pct': 0.20,
    },
}


@pytest.fixture
def rm():
    return RiskManager(PARAMS)


class TestCalculatePositionSize:
    def test_basic_sizing(self, rm):
        # $100k account, entry $50, stop $48 -> $2 risk per share
        # risk budget = 100000 * 0.03 = $3000, shares = 3000 / 2 = 1500
        # max cost cap = 100000 * 0.20 = $20000, shares by cost = 20000 / 50 = 400
        # min(1500, 400) = 400
        shares = rm.calculate_position_size(100_000, 50.0, 48.0)
        assert shares == 400

    def test_risk_limited(self, rm):
        # $100k account, entry $10, stop $9 -> $1 risk per share
        # risk budget = $3000, shares = 3000
        # max cost cap = $20000, shares by cost = 2000
        # min(3000, 2000) = 2000
        shares = rm.calculate_position_size(100_000, 10.0, 9.0)
        assert shares == 2000

    def test_zero_entry_price(self, rm):
        assert rm.calculate_position_size(100_000, 0, 48.0) == 0

    def test_zero_stop_price(self, rm):
        assert rm.calculate_position_size(100_000, 50.0, 0) == 0

    def test_same_entry_and_stop(self, rm):
        assert rm.calculate_position_size(100_000, 50.0, 50.0) == 0

    def test_small_account(self, rm):
        # $1000 account, entry $100, stop $95 -> $5 risk
        # risk budget = $30, shares = 6
        # max cost cap = $200, shares by cost = 2
        shares = rm.calculate_position_size(1_000, 100.0, 95.0)
        assert shares == 2

    def test_negative_entry(self, rm):
        assert rm.calculate_position_size(100_000, -10.0, 48.0) == 0

    def test_short_position_sizing(self, rm):
        # Short: entry $50, stop $53 -> $3 risk per share
        # risk budget = $3000, shares = 1000
        # max cost cap = $20000, shares by cost = 400
        shares = rm.calculate_position_size(100_000, 50.0, 53.0)
        assert shares == 400


class TestCanTakeTrade:
    def test_within_limits(self, rm):
        assert rm.can_take_trade(100_000, 50_000, 5) is True

    def test_max_investment_reached(self, rm):
        # invested >= 70% of account
        assert rm.can_take_trade(100_000, 70_000, 5) is False

    def test_max_positions_reached(self, rm):
        assert rm.can_take_trade(100_000, 50_000, 10) is False

    def test_both_limits_ok(self, rm):
        assert rm.can_take_trade(100_000, 69_999, 9) is True

    def test_zero_account_value(self, rm):
        assert rm.can_take_trade(0, 0, 0) is False


class TestValidateTradeSize:
    def test_fits_in_cash(self, rm):
        assert rm.validate_trade_size(100, 50.0, 10_000) is True

    def test_exceeds_cash(self, rm):
        assert rm.validate_trade_size(100, 50.0, 4_000) is False

    def test_exact_cash(self, rm):
        assert rm.validate_trade_size(100, 50.0, 5_000) is True


class TestGetStopLossPct:
    def _make_df(self, n=30):
        close = pd.Series(np.linspace(100, 110, n))
        high = close + 1.0
        low = close - 1.0
        return pd.DataFrame({'close': close, 'high': high, 'low': low})

    @patch('execution.risk_manager.yf')
    def test_normal_vix(self, mock_yf, rm):
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 20.0}
        df = self._make_df()
        sl = rm.get_stop_loss_pct(df)
        last_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        expected = last_range * PARAMS['strategy_retest_200ma']['ATR']
        assert abs(sl - expected) < 1e-6

    @patch('execution.risk_manager.yf')
    def test_high_vix_wider_stop(self, mock_yf, rm):
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 30.0}
        df = self._make_df()
        sl = rm.get_stop_loss_pct(df)
        last_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        expected = last_range * (PARAMS['strategy_retest_200ma']['ATR'] + 0.5)
        assert abs(sl - expected) < 1e-6

    @patch('execution.risk_manager.yf')
    def test_low_vix_tighter_stop(self, mock_yf, rm):
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 12.0}
        df = self._make_df()
        sl = rm.get_stop_loss_pct(df)
        last_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        expected = last_range * (PARAMS['strategy_retest_200ma']['ATR'] - 0.5)
        assert abs(sl - expected) < 1e-6

    @patch('execution.risk_manager.yf')
    def test_vix_fetch_failure_defaults(self, mock_yf, rm):
        mock_yf.Ticker.side_effect = Exception('network error')
        df = self._make_df()
        sl = rm.get_stop_loss_pct(df)
        # Should default to VIX=20 (normal range)
        last_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        expected = last_range * PARAMS['strategy_retest_200ma']['ATR']
        assert abs(sl - expected) < 1e-6
