"""
Gets stock historical data using yfinance (no rate limits).
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

# Set up logging
logger = logging.getLogger()


class StockDataFetcher:
    def __init__(self, ib=None, config=None, params=None):
        self.ib = ib
        self.config = config
        self.params = params

    def get_historical_data(self, symbol: str, lookback_days: int) -> pd.DataFrame | None:
        """Fetch historical daily data for a stock via yfinance."""
        try:
            period = self._lookback_to_period(lookback_days)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                return None

            df = df.rename(
                columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                }
            )

            df = df[['open', 'high', 'low', 'close', 'volume']]
            df['date'] = df.index
            df = df.reset_index(drop=True)
            df['symbol'] = symbol

            return df

        except Exception as e:
            logger.warning(f'Failed to get data for {symbol}: {e}')
            return None

    @staticmethod
    def _lookback_to_period(lookback_days: int) -> str:
        if lookback_days <= 5:
            return '5d'
        elif lookback_days <= 30:
            return '1mo'
        elif lookback_days <= 90:
            return '3mo'
        elif lookback_days <= 180:
            return '6mo'
        elif lookback_days <= 365:
            return '1y'
        elif lookback_days <= 730:
            return '2y'
        elif lookback_days <= 1825:
            return '5y'
        else:
            return '10y'
