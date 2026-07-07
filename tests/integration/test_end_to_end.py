"""End-to-end integration tests with real yfinance data."""

import pytest

from data_fetch.historical_data import StockDataFetcher
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from tests.conftest import PARAMS, requires_network


@pytest.mark.integration
class TestEndToEndRealData:
    """Full pipeline with live yfinance data — the closest to production
    without needing IB."""

    @requires_network
    def test_fetch_build_features_train_predict(self):
        fetcher = StockDataFetcher()
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        bars = {}
        for t in tickers:
            df = fetcher.get_historical_data(t, lookback_days=1825)
            if df is not None and len(df) >= 250:
                bars[t] = df

        if len(bars) < 2:
            pytest.skip('Could not fetch enough tickers')

        fetcher.get_historical_data = lambda sym, _days: bars.get(sym)

        analyzer = AIAnalyzer(
            stock_data=fetcher,
            cnn_epochs=5,
            params=PARAMS,
            model_type='lstm',
        )
        analyzer.train(list(bars.keys()), val_split=0.2)

        for sym in bars:
            result = analyzer.predict(sym)
            assert result is not None, f'Prediction failed for {sym}'
            assert result['class'] in ('SHORT', 'FLAT', 'LONG')
            assert abs(sum(result['probs'].values()) - 1.0) < 1e-4
