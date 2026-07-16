"""Integration tests for the 200-MA strategy pipeline."""

import pytest

from data_fetch.historical_data import StockDataFetcher
from strategy.retest_200ma.indicators import TrendIndicator
from tests.conftest import CONFIG, PARAMS, make_synthetic_bars, requires_network


@pytest.mark.integration
class TestRetest200MAPipeline:
    """Exercise TrendIndicator on synthetic data."""

    def test_insufficient_data_returns_none(self):
        df = make_synthetic_bars(100)
        indicator = TrendIndicator(df, CONFIG, PARAMS)
        signal = indicator.detect_breakout_and_retest()
        assert signal is None

    def test_runs_on_sufficient_data(self):
        df = make_synthetic_bars(500)
        indicator = TrendIndicator(df, CONFIG, PARAMS)
        signal = indicator.detect_breakout_and_retest()
        if signal is not None:
            assert signal['strategy_type'] == '200ma_retest'
            assert signal['type'] in ('LONG', 'SHORT')
            assert signal['entry'] > 0
            assert signal['stop'] > 0
            assert signal['target'] > 0

    @requires_network
    def test_runs_on_real_data(self):
        fetcher = StockDataFetcher()
        df = fetcher.get_historical_data('AAPL', lookback_days=365)
        if df is None or len(df) < 250:
            pytest.skip('Insufficient AAPL data')

        indicator = TrendIndicator(df, CONFIG, PARAMS)
        signal = indicator.detect_breakout_and_retest()

        if signal is not None:
            assert signal['symbol'] == 'AAPL'
            assert 'entry' in signal
            assert 'stop' in signal
            assert 'target' in signal
