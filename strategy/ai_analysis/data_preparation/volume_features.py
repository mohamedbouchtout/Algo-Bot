"""
Volume based feature extraction for stock data.

Raw share volumes differ by orders of magnitude between tickers, so everything
here is expressed as a ratio or a normalised delta to keep samples comparable
across names.
"""

import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class VolumeFeatureExtractor:
    """Extract scale-free volume features from a bar DataFrame."""

    FEATURE_NAMES: List[str] = [
        'volume_ratio_20',   # volume / 20d avg volume
        'volume_ratio_50',   # volume / 50d avg volume
        'volume_log_change', # log(volume / volume.shift(1))
        'obv_slope_20',      # slope of OBV over 20 days, normalised by 20d avg volume
    ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"VolumeFeatureExtractor missing columns: {missing}")

        close = df['close'].astype(float)
        volume = df['volume'].astype(float).replace(0, np.nan)

        out = pd.DataFrame(index=df.index)

        avg20 = volume.rolling(window=20, min_periods=20).mean()
        avg50 = volume.rolling(window=50, min_periods=50).mean()
        out['volume_ratio_20'] = volume / avg20
        out['volume_ratio_50'] = volume / avg50
        out['volume_log_change'] = np.log(volume / volume.shift(1))

        # On-Balance Volume: cumulative signed volume
        sign = np.sign(close.diff()).fillna(0.0)
        obv = (sign * volume.fillna(0.0)).cumsum()
        # Slope of OBV over the last 20 days, normalised so it's comparable
        # across tickers
        obv_slope = (obv - obv.shift(20)) / 20.0
        out['obv_slope_20'] = obv_slope / avg20

        return out[self.FEATURE_NAMES]
