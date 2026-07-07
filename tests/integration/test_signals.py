"""Integration tests for signal construction and risk manager flows."""

from unittest.mock import patch

import pytest

from data_fetch.historical_data import StockDataFetcher
from execution.risk_manager import RiskManager
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from tests.conftest import PARAMS, make_synthetic_bars


@pytest.mark.integration
class TestSignalConstruction:
    """Test AIAnalyzer.construct_signal builds valid trade signals."""

    @patch('execution.risk_manager.yf')
    def test_long_signal(self, mock_yf):
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 20.0}
        df = make_synthetic_bars(300)
        fetcher = StockDataFetcher()
        analyzer = AIAnalyzer(stock_data=fetcher, params=PARAMS)

        signal = analyzer.construct_signal(df, PARAMS, 'LONG', 0.85)

        assert signal is not None
        assert signal['strategy_type'] == 'ai_analysis'
        assert signal['type'] == 'LONG'
        assert signal['entry'] > 0
        assert signal['stop'] < signal['entry']
        assert signal['target'] > signal['entry']
        assert signal['confidence'] == 0.85

    @patch('execution.risk_manager.yf')
    def test_short_signal(self, mock_yf):
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 20.0}
        df = make_synthetic_bars(300)
        fetcher = StockDataFetcher()
        analyzer = AIAnalyzer(stock_data=fetcher, params=PARAMS)

        signal = analyzer.construct_signal(df, PARAMS, 'SHORT', 0.90)

        assert signal is not None
        assert signal['type'] == 'SHORT'
        assert signal['stop'] > signal['entry']
        assert signal['target'] < signal['entry']

    @patch('execution.risk_manager.yf')
    def test_signal_risk_reward(self, mock_yf):
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 20.0}
        df = make_synthetic_bars(300)
        fetcher = StockDataFetcher()
        analyzer = AIAnalyzer(stock_data=fetcher, params=PARAMS)

        signal = analyzer.construct_signal(df, PARAMS, 'LONG', 0.80)

        assert signal['risk'] > 0
        expected_reward = signal['risk'] * PARAMS['ai_analyzer']['risk_reward_ratio']
        assert abs(signal['reward'] - expected_reward) < 0.01


@pytest.mark.integration
class TestRiskManagerIntegration:
    """Test RiskManager with realistic signal flows."""

    @patch('execution.risk_manager.yf')
    def test_position_sizing_with_signal(self, mock_yf):
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 20.0}

        rm = RiskManager(PARAMS)
        df = make_synthetic_bars(300)

        sl_pct = rm.get_stop_loss_pct(df)
        entry = float(df['close'].iloc[-1])
        stop = entry * (1 - sl_pct)

        shares = rm.calculate_position_size(100_000, entry, stop)
        assert shares > 0

        can_trade = rm.can_take_trade(100_000, 30_000, 3)
        assert can_trade is True

        trade_cost = shares * entry
        valid = rm.validate_trade_size(shares, entry, trade_cost + 1000)
        assert valid is True

    @patch('execution.risk_manager.yf')
    def test_full_signal_to_sizing_flow(self, mock_yf):
        """Simulate: fetch data -> AI predict -> construct signal -> size position."""
        mock_yf.Ticker.return_value.fast_info = {'lastPrice': 18.0}

        bars = {t: make_synthetic_bars(400, symbol=t) for t in ['FL_A', 'FL_B']}
        fetcher = StockDataFetcher()
        fetcher.get_historical_data = lambda sym, _days: bars.get(sym)

        analyzer = AIAnalyzer(
            stock_data=fetcher,
            cnn_epochs=3,
            params=PARAMS,
            model_type='lstm',
        )
        analyzer.train(list(bars.keys()), val_split=0.2)

        prediction = analyzer.predict('FL_A')
        assert prediction is not None

        df = bars['FL_A']
        signal = analyzer.construct_signal(df, PARAMS, prediction['class'], prediction['probs'][prediction['class']])

        if signal['type'] in ('LONG', 'SHORT'):
            rm = RiskManager(PARAMS)
            shares = rm.calculate_position_size(100_000, signal['entry'], signal['stop'])
            assert shares >= 0
