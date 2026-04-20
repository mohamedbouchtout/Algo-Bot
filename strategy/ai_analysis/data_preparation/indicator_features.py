"""
Technical indicator feature extraction.

All indicators are expressed on a fixed / scale-free range so they can be
safely pooled across tickers and later binarised for the RBM.
"""

import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class IndicatorFeatureExtractor:
    """Classic technical indicators, all scale-free."""

    FEATURE_NAMES: List[str] = [
        'rsi14',             # 0..100
        'macd_hist_norm',    # MACD histogram / close  (scale free)
        'bb_position',       # position within Bollinger Bands, 0..1
        'atr14_pct',         # ATR(14) / close
        'ma200_slope_pct',   # 20-day slope of 200MA divided by close
    ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {'close', 'high', 'low'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"IndicatorFeatureExtractor missing columns: {missing}")

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)

        out = pd.DataFrame(index=df.index)

        # --- RSI(14) --------------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        # Wilder's smoothing via EMA with alpha = 1/14
        avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out['rsi14'] = 100.0 - (100.0 / (1.0 + rs))

        # --- MACD histogram (normalised by price) ---------------------------
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        out['macd_hist_norm'] = (macd_line - signal_line) / close

        # --- Bollinger Band position (0 = lower, 1 = upper) -----------------
        ma20 = close.rolling(window=20, min_periods=20).mean()
        std20 = close.rolling(window=20, min_periods=20).std()
        upper = ma20 + 2.0 * std20
        lower = ma20 - 2.0 * std20
        width = (upper - lower).replace(0, np.nan)
        out['bb_position'] = (close - lower) / width

        # --- ATR(14) as a percentage of price -------------------------------
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
        out['atr14_pct'] = atr14 / close

        # --- 200MA slope (scale-free) ---------------------------------------
        ma200 = close.rolling(window=200, min_periods=200).mean()
        out['ma200_slope_pct'] = (ma200 - ma200.shift(20)) / (20.0 * close)

        return out[self.FEATURE_NAMES]
