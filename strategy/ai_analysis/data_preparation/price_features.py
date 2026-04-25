"""
Price-based feature extraction for stock data.

Produces continuous features derived from OHLC columns (open, high, low, close)
that describe short term price behaviour in a scale-free way so different
tickers can be combined into a single training set.
"""

import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class PriceFeatureExtractor:
    """Extract scale-free price features from an OHLC DataFrame."""

    # Columns this extractor adds to the output (in order)
    FEATURE_NAMES: List[str] = [
        'log_return_1d',
        'log_return_5d',
        'close_vs_ma20',      # (close / ma20) - 1
        'close_vs_ma50',
        'close_vs_ma200',
        'daily_range_pct',    # (high - low) / close
        'close_to_high_pct',  # (high - close) / close
        'close_to_low_pct',   # (close - low) / close
    ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : DataFrame with columns 'open','high','low','close' (as returned by
             ib_insync util.df on reqHistoricalData bars).

        Returns
        -------
        DataFrame of the same index with the FEATURE_NAMES columns. NaN rows
        (from the rolling windows) are left in place, the FeatureBuilder drops
        them once all extractors have been combined.
        """
        required = {'close', 'high', 'low'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"PriceFeatureExtractor missing columns: {missing}")

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)

        out = pd.DataFrame(index=df.index)

        # Returns are already scale-free
        out['log_return_1d'] = np.log(close / close.shift(1))
        out['log_return_5d'] = np.log(close / close.shift(5))

        # Position relative to moving averages (scale free)
        for w, name in [(20, 'close_vs_ma20'), (50, 'close_vs_ma50'), (200, 'close_vs_ma200')]:
            ma = close.rolling(window=w, min_periods=w).mean()
            out[name] = (close / ma) - 1.0

        # Intraday range / position within the day
        out['daily_range_pct'] = (high - low) / close
        out['close_to_high_pct'] = (high - close) / close
        out['close_to_low_pct'] = (close - low) / close

        return out[self.FEATURE_NAMES]
