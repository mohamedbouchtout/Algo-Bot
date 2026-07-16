"""
Market regime feature extraction.

Provides market-wide context features (VIX, SPY trend) that are the same
for every ticker on a given day. These help the model understand whether
the broader market is in a risk-on or risk-off regime.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketFeatureExtractor:
    FEATURE_NAMES: list[str] = [
        'vix_normalized',
        'spy_50d_return',
        'spy_vs_ma200',
    ]

    def __init__(self, market_data: pd.DataFrame | None = None):
        self._market_data = market_data

    def _fetch_market_data(self) -> pd.DataFrame:
        if self._market_data is not None:
            return self._market_data

        try:
            vix = yf.Ticker('^VIX').history(period='10y', auto_adjust=True)
            spy = yf.Ticker('SPY').history(period='10y', auto_adjust=True)

            if vix.empty or spy.empty:
                logger.warning('Market data fetch returned empty, using defaults')
                return pd.DataFrame()

            vix_close = vix['Close'].rename('vix_close')
            spy_close = spy['Close'].rename('spy_close')

            market = pd.DataFrame(
                {
                    'vix_close': vix_close,
                    'spy_close': spy_close,
                }
            )
            market.index = market.index.tz_localize(None)
            market = market.ffill()
            self._market_data = market
            return market

        except Exception as e:
            logger.warning(f'Failed to fetch market data: {e}')
            return pd.DataFrame()

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        market = self._fetch_market_data()

        if market.empty or 'date' not in df.columns:
            out['vix_normalized'] = 1.0
            out['spy_50d_return'] = 0.0
            out['spy_vs_ma200'] = 0.0
            return out[self.FEATURE_NAMES]

        dates = pd.to_datetime(df['date']).dt.tz_localize(None)

        market_sorted = market.sort_index()
        spy_ma200 = market_sorted['spy_close'].rolling(window=200, min_periods=1).mean()
        spy_return_50d = np.log(market_sorted['spy_close'] / market_sorted['spy_close'].shift(50))

        vix_vals = []
        spy_ret_vals = []
        spy_ma_vals = []

        for d in dates:
            mask = market_sorted.index <= d
            if not mask.any():
                vix_vals.append(1.0)
                spy_ret_vals.append(0.0)
                spy_ma_vals.append(0.0)
                continue

            idx = market_sorted.index[mask][-1]
            vix_vals.append(market_sorted.loc[idx, 'vix_close'] / 20.0)
            spy_ret_vals.append(spy_return_50d.loc[idx] if pd.notna(spy_return_50d.loc[idx]) else 0.0)
            ma_val = spy_ma200.loc[idx]
            spy_val = market_sorted.loc[idx, 'spy_close']
            spy_ma_vals.append((spy_val / ma_val - 1.0) if pd.notna(ma_val) and ma_val > 0 else 0.0)

        out['vix_normalized'] = vix_vals
        out['spy_50d_return'] = spy_ret_vals
        out['spy_vs_ma200'] = spy_ma_vals

        return out[self.FEATURE_NAMES]
