"""
Class that has helper functions to determine if a stock is in an uptrend or downtrend based on moving average slope
"""

import logging
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger()

class TrendValidator:
    def __init__(self, df: pd.DataFrame, config, params):
        self.df = df.copy()
        self.config = config
        self.params = params

    def calculate_ma(self, period: int) -> pd.Series:
        """Calculate moving average"""
        return self.df['close'].rolling(window=period).mean()
    
    def calculate_ma_slope(self, ma_values: np.ndarray, current_idx: int) -> float:
        """
        Calculate the slope of the moving average
        Returns: Percentage change over the slope period
        Positive = uptrend, Negative = downtrend, ~0 = flat
        """
        if current_idx < self.params["strategy_retest_200ma"]["ma_slope_period"]:
            return 0.0
        
        # Compare current MA to MA N days ago
        ma_current = ma_values[current_idx]
        ma_past = ma_values[current_idx - self.params["strategy_retest_200ma"]["ma_slope_period"]]
        
        if ma_past == 0:
            return 0.0
        
        # Calculate percentage change
        slope = (ma_current - ma_past) / ma_past
        
        return slope
    
    def is_ma_trending_up_or_flat(self, ma_values: np.ndarray, idx: int) -> bool:
        """Check if MA is trending upward or flat (good for longs)"""
        slope = self.calculate_ma_slope(ma_values, idx)
        
        # Slope should be >= min_uptrend_slope (which is slightly negative, allowing flat)
        return slope >= self.params["strategy_retest_200ma"]["min_uptrend_slope"]
    
    def is_ma_trending_down_or_flat(self, ma_values: np.ndarray, idx: int) -> bool:
        """Check if MA is trending downward or flat (good for shorts)"""
        slope = self.calculate_ma_slope(ma_values, idx)
        
        # Slope should be <= max_downtrend_slope (which is slightly positive, allowing flat)
        return slope <= self.params["strategy_retest_200ma"]["max_downtrend_slope"]
