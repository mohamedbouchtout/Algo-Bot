"""Integration tests for the data fetching pipeline."""

import numpy as np
import pytest

from data_fetch.historical_data import StockDataFetcher
from tests.conftest import requires_network


@pytest.mark.integration
class TestDataFetchingPipeline:
    """Verify the StockDataFetcher returns well-formed DataFrames."""

    @requires_network
    def test_fetch_single_ticker(self):
        fetcher = StockDataFetcher()
        df = fetcher.get_historical_data('AAPL', lookback_days=365)

        assert df is not None
        assert len(df) > 200
        for col in ('open', 'high', 'low', 'close', 'volume', 'date', 'symbol'):
            assert col in df.columns, f'Missing column: {col}'
        assert (df['symbol'] == 'AAPL').all()
        assert df['close'].dtype in (np.float64, np.float32)
        assert (df['high'] >= df['low']).all()

    @requires_network
    def test_fetch_invalid_ticker_returns_none(self):
        fetcher = StockDataFetcher()
        df = fetcher.get_historical_data('ZZZZZZZNOTREAL', lookback_days=365)
        assert df is None or len(df) == 0

    @requires_network
    def test_lookback_period_mapping(self):
        fetcher = StockDataFetcher()
        df_short = fetcher.get_historical_data('MSFT', lookback_days=30)
        df_long = fetcher.get_historical_data('MSFT', lookback_days=730)

        assert df_short is not None and df_long is not None
        assert len(df_long) > len(df_short)
