"""
Class that calls helper classes to determine if a stock is in an uptrend or downtrend based on moving average slope
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from strategy.retest_200ma.validators import TrendValidator
from strategy.retest_200ma.trend_detector import TrendDetector

# Setup logging
logger = logging.getLogger()

class TrendIndicator:
    def __init__(self, df: pd.DataFrame, config, params):
        self.df = df.copy()
        self.config = config
        self.params = params

    def detect_breakout_and_retest(self) -> Optional[Dict]:
        """
        Detect breakout and retest pattern
        Returns: Dict with signal info or None
        """
        if len(self.df) < self.params["strategy_retest_200ma"]["ma_period"] + 20:
            return None
        
        # Calculate 200 MA
        trend_validator = TrendValidator(self.df, self.config, self.params)
        self.df['ma200'] = trend_validator.calculate_ma(self.params["strategy_retest_200ma"]["ma_period"])
        
        # Need at least 20 days after MA is calculated
        recent_data = self.df.tail(30).copy()
        
        if recent_data['ma200'].isna().any():
            return None
        
        # Look for breakout and retest pattern
        signal = self.analyze_pattern(recent_data)
        
        return signal
    
    def analyze_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze for breakout and retest pattern
        Pattern:
        1. Price crosses above/below 200 MA (breakout)
        2. Price comes back to test 200 MA (retest)
        3. Price bounces off 200 MA in breakout direction
        """
        DF = df.reset_index(drop=True)
        
        # LONG SETUP: Breakout above, retest, bounce up
        trend_detector = TrendDetector(self.df, self.config, self.params)
        long_signal = trend_detector.detect_long_pattern(DF)
        if long_signal:
            return long_signal
        
        # SHORT SETUP: Breakout below, retest, bounce down
        short_signal = trend_detector.detect_short_pattern(DF)
        if short_signal:
            return short_signal
        
        return None
