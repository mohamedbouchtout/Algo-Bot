"""Shared fixtures and helpers for the test suite."""

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

collect_ignore = [
    'legacy/test_ai_analysis.py',
    'legacy/test_ai_backtest.py',
    'legacy/test_retest_200ma.py',
    'legacy/test_position_manager.py',
    'legacy/run_tests.py',
    'legacy/run_predictions.py',
]

PARAMS = {
    'strategy_retest_200ma': {
        'ma_period': 200,
        'ma_slope_period': 20,
        'min_uptrend_slope': -0.01,
        'max_downtrend_slope': 0.01,
        'risk_reward_ratio': 2.0,
        'stop_loss_pct': 0.03,
        'lookback_days': 250,
        'min_breakout_volume': 1.7,
        'min_breakout_strength': 0.7,
        'min_bounce_strength': 0.02,
        'max_retest_volume_ratio': 0.5,
        'max_retest_volume_absolute': 0.8,
        'max_days_since_retest': 3,
        'retest_distance': 0.005,
        'ATR': 1.5,
    },
    'ai_analyzer': {
        'confidence_threshold': 0.75,
        'risk_reward_ratio': 2.0,
        'stop_loss_pct': 0.03,
        'lookback_days': 1825,
        'ATR': 1.5,
    },
    'risk_management': {
        'risk_per_trade_pct': 0.03,
        'max_investment_pct': 0.70,
        'max_positions': 10,
        'max_position_pct': 0.20,
    },
}

CONFIG = {
    'ib': {'host': '127.0.0.1', 'ports': [4002], 'client_id': 1},
    'logging': {'level': 'INFO', 'console': False},
}


def _have_network() -> bool:
    """Return True if we can reach yfinance."""
    try:
        df = yf.Ticker('SPY').history(period='5d', auto_adjust=True, timeout=5)
        return df is not None and not df.empty
    except Exception:
        return False


requires_network = pytest.mark.skipif(not _have_network(), reason='No network / yfinance unavailable')


def make_synthetic_bars(n: int = 500, symbol: str = 'TEST') -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame that mimics real market data."""
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    low = np.maximum(low, 0.5)
    open_ = np.clip(close + rng.randn(n) * 0.3, low, high)
    volume = rng.randint(500_000, 5_000_000, n).astype(float)
    dates = pd.bdate_range(end='2026-06-30', periods=n)

    return pd.DataFrame(
        {
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'date': dates,
            'symbol': symbol,
        }
    )


@pytest.fixture
def synthetic_bars():
    """Fixture providing synthetic OHLCV bars for unit tests."""
    rng = np.random.RandomState(42)
    n = 300
    close = 100.0 * np.cumprod(1 + rng.randn(n) * 0.02)
    high = close * (1 + np.abs(rng.randn(n)) * 0.01)
    low = close * (1 - np.abs(rng.randn(n)) * 0.01)
    open_ = np.clip(close * (1 + rng.randn(n) * 0.005), low, high)
    volume = rng.randint(1_000_000, 10_000_000, n).astype(float)
    dates = pd.bdate_range(start='2023-01-01', periods=n)
    return pd.DataFrame(
        {
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'date': dates,
            'symbol': 'SYNTH',
        }
    )
