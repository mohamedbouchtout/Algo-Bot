"""
Test script to verify breakout/retest pattern detection
Scans a few stocks and shows signals WITHOUT placing orders
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict

# Setup logging
logger = logging.getLogger(__name__)

class BreakoutRetestDetector:
    def __init__(self, df: pd.DataFrame, risk_reward_ratio: float = 2.0):
        self.ma_period = 200  # 200-day moving average
        self.risk_reward_ratio = risk_reward_ratio
        self.df = df.copy()
        self.min_breakout_volume = 2.0
        self.max_retest_volume_ratio = 0.5
        self.max_retest_volume_absolute = 0.8
        self.retest_distance = 0.005  # 0.5%
        self.min_bounce_strength = 0.02  # 2%
        self.min_breakout_strength = 0.7
        self.lookback_days = 10
        self.max_days_since_retest = 3
        self.ma_slope_period = 20 
        self.min_uptrend_slope = -0.001 
        self.max_downtrend_slope = 0.001 
    
    def calculate_ma(self, period: int) -> pd.Series:
        """Calculate moving average"""
        return self.df['close'].rolling(window=period).mean()
    
    def calculate_ma_slope(self, ma_values: np.ndarray, current_idx: int) -> float:
        """
        Calculate the slope of the moving average
        Returns: Percentage change over the slope period
        Positive = uptrend, Negative = downtrend, ~0 = flat
        """
        if current_idx < self.ma_slope_period:
            return 0.0
        
        # Compare current MA to MA N days ago
        ma_current = ma_values[current_idx]
        ma_past = ma_values[current_idx - self.ma_slope_period]
        
        if ma_past == 0:
            return 0.0
        
        # Calculate percentage change
        slope = (ma_current - ma_past) / ma_past
        
        return slope
    
    def is_ma_trending_up_or_flat(self, ma_values: np.ndarray, idx: int) -> bool:
        """Check if MA is trending upward or flat (good for longs)"""
        slope = self.calculate_ma_slope(ma_values, idx)
        
        # Slope should be >= min_uptrend_slope (which is slightly negative, allowing flat)
        return slope >= self.min_uptrend_slope
    
    def is_ma_trending_down_or_flat(self, ma_values: np.ndarray, idx: int) -> bool:
        """Check if MA is trending downward or flat (good for shorts)"""
        slope = self.calculate_ma_slope(ma_values, idx)
        
        # Slope should be <= max_downtrend_slope (which is slightly positive, allowing flat)
        return slope <= self.max_downtrend_slope
    
    def detect_breakout_and_retest(self) -> Optional[Dict]:
        """
        Detect breakout and retest pattern
        Returns: Dict with signal info or None
        """
        if len(self.df) < self.ma_period + 20:
            return None
        
        # Calculate 200 MA
        self.df['ma200'] = self.calculate_ma(self.ma_period)
        
        # Need at least 20 days after MA is calculated
        recent_data = self.df.tail(30).copy()
        
        if recent_data['ma200'].isna().any():
            return None
        
        # Look for breakout and retest pattern
        signal = self._analyze_pattern(recent_data)
        
        return signal
    
    def _analyze_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze for breakout and retest pattern
        Pattern:
        1. Price crosses above/below 200 MA (breakout)
        2. Price comes back to test 200 MA (retest)
        3. Price bounces off 200 MA in breakout direction
        """
        DF = df.reset_index(drop=True)
        
        # LONG SETUP: Breakout above, retest, bounce up
        long_signal = self._detect_long_pattern(DF)
        if long_signal:
            return long_signal
        
        # SHORT SETUP: Breakout below, retest, bounce down
        short_signal = self._detect_short_pattern(DF)
        if short_signal:
            return short_signal
        
        return None
    
    def _detect_long_pattern(self, DF: pd.DataFrame) -> Optional[Dict]:
        """Detect bullish breakout and retest"""
        ma200 = DF['ma200'].values
        close = DF['close'].values
        high = DF['high'].values
        low = DF['low'].values
        volume = DF['volume'].values
        
        # Calculate average volume for this stock (last 50 days)
        avg_volume = volume[-50:].mean() if len(volume) >= 50 else volume.mean()

        # Look in recent 10 days for the pattern
        for i in range(len(DF) - 5, max(len(DF) - self.lookback_days, 0), -1):
            # Step 1: Find breakout above MA200
            if i < 5:
                continue
                
            # For LONG, MA should be flat or trending upward
            if not self.is_ma_trending_up_or_flat(ma200, i):
                ma_slope = self.calculate_ma_slope(ma200, i)
                logging.debug(
                    f"Skipping long pattern at day {i} - MA trending down "
                    f"(slope: {ma_slope:.4f})"
                )
                continue

            # Require price well below MA before breakout
            avg_price_before = close[i-5:i].mean()
            avg_ma_before = ma200[i-5:i].mean()
            was_significantly_below = avg_price_before < avg_ma_before * 0.98
            
            crossed_above = close[i] > ma200[i] and close[i-1] <= ma200[i-1]
            
            if not (was_significantly_below and crossed_above):
                continue
            
            # Check if breakout had HIGH volume (strong buying)
            breakout_volume = volume[i]
            volume_ratio = breakout_volume / avg_volume
            
            # Breakout should have at least 1.5x average volume
            if volume_ratio < self.min_breakout_volume:
                logging.debug(f"Breakout volume too low: {volume_ratio:.2f}x average")
                continue

            # Breakout strength
            breakout_strength = (close[i] - low[i]) / (high[i] - low[i]) if (high[i] - low[i]) > 0 else 0
            if breakout_strength < self.min_breakout_strength:
                continue

            breakout_idx = i
            
            # Step 2: Look for retest in next few days
            for j in range(i+2, min(i+8, len(DF))):
                # MA trend at retest should still be bullish/flat
                if not self.is_ma_trending_up_or_flat(ma200, j):
                    continue

                # Price came back down near MA200 (within 1%)
                retest_distance = abs(low[j] - ma200[j]) / ma200[j]
                
                if retest_distance < self.retest_distance:  # Within 1.0% of MA200
                    # Check if retest has LOW volume (weak selling)
                    retest_volume = volume[j]
                    retest_volume_ratio = retest_volume / avg_volume
                    
                    # Retest should have LOWER volume than breakout (weak selling pressure)
                    # Ideally below average volume (< 1.0x) or at least less than breakout
                    if retest_volume_ratio > volume_ratio * self.max_retest_volume_ratio or retest_volume_ratio > self.max_retest_volume_absolute:
                        logging.debug(f"Retest volume too high: {retest_volume_ratio:.2f}x vs breakout {volume_ratio:.2f}x")
                        continue

                    # Step 3: Check if price bounced up (current price above retest low)
                    if j < len(DF) - 1:
                        bounce_strength = (close[-1] - low[j]) / low[j]
                    
                    if bounce_strength < self.min_bounce_strength:
                        continue
                    
                    # Upward momentum
                    if len(close) >= 2:
                        recent_momentum = close[-1] > close[-2]
                        if not recent_momentum:
                            continue
                    
                    entry_price = close[-1]
                    stop_loss = low[j] * 0.99
                    risk = entry_price - stop_loss
                    
                    if risk <= 0 or risk / entry_price > 0.05:
                        continue
                    
                    target_price = entry_price + (risk * self.risk_reward_ratio)
                    
                    # Recent retest
                    days_since_retest = len(DF) - 1 - j
                    if days_since_retest > self.max_days_since_retest:
                        continue
                    
                    # Calculate MA slope for logging
                    ma_slope = self.calculate_ma_slope(ma200, len(DF) - 1)
                    
                    return {
                        'type': 'LONG',
                        'symbol': DF['symbol'].iloc[0],
                        'entry': entry_price,
                        'stop': stop_loss,
                        'target': target_price,
                        'risk': risk,
                        'reward': risk * self.risk_reward_ratio,
                        'breakout_date': DF.iloc[breakout_idx]['date'],
                        'retest_date': DF.iloc[j]['date'],
                        'current_date': DF.iloc[-1]['date'],
                        'breakout_volume_ratio': volume_ratio,
                        'retest_volume_ratio': retest_volume_ratio,
                        'avg_volume': avg_volume,
                        'bounce_strength': bounce_strength,
                        'breakout_strength': breakout_strength,
                        'ma_slope': ma_slope,
                        'ma_slope_pct': ma_slope * 100  # For easier reading
                    }
        
        return None
    
    def _detect_short_pattern(self, DF: pd.DataFrame) -> Optional[Dict]:
        """Detect bearish breakout and retest"""
        ma200 = DF['ma200'].values
        close = DF['close'].values
        high = DF['high'].values
        low = DF['low'].values
        volume = DF['volume'].values
        
        # Calculate average volume for this stock (last 50 days)
        avg_volume = volume[-50:].mean() if len(volume) >= 50 else volume.mean()

        # Look in recent 20 days for the pattern
        for i in range(len(DF) - 5, max(len(DF) - self.lookback_days, 0), -1):
            # Step 1: Find breakout below MA200
            if i < 5:
                continue
                
            # Check MA trend BEFORE checking breakdown
            if not self.is_ma_trending_down_or_flat(ma200, i):
                ma_slope = self.calculate_ma_slope(ma200, i)
                logging.debug(
                    f"Skipping short pattern at day {i} - MA trending up "
                    f"(slope: {ma_slope:.4f})"
                )
                continue

            # Require price well above MA before breakdown
            avg_price_before = close[i-5:i].mean()
            avg_ma_before = ma200[i-5:i].mean()
            was_significantly_above = avg_price_before > avg_ma_before * 1.02
            
            crossed_below = close[i] < ma200[i] and close[i-1] >= ma200[i-1]
            
            if not (was_significantly_above and crossed_below):
                continue
            
            # Check if breakout had HIGH volume (strong buying)
            breakout_volume = volume[i]
            volume_ratio = breakout_volume / avg_volume
            
            # Breakout should have at least 1.5x average volume
            if volume_ratio < self.min_breakout_volume:
                logging.debug(f"Breakout volume too low: {volume_ratio:.2f}x average")
                continue

            # Breakdown strength
            breakdown_strength = (high[i] - close[i]) / (high[i] - low[i]) if (high[i] - low[i]) > 0 else 0
            if breakdown_strength < self.min_breakout_strength:
                continue

            breakout_idx = i
            
            # Step 2: Look for retest in next few days
            for j in range(i+2, min(i+8, len(DF))):
                # MA trend at retest should still be bearish/flat
                if not self.is_ma_trending_down_or_flat(ma200, j):
                    continue

                # Price came back up near MA200 (within 1%)
                retest_distance = abs(high[j] - ma200[j]) / ma200[j]
                
                if retest_distance < self.retest_distance:  # Within 1.0% of MA200
                    # Check if retest has LOW volume (weak buying)
                    retest_volume = volume[j]
                    retest_volume_ratio = retest_volume / avg_volume
                    
                    # Retest should have LOWER volume than breakdown (weak buying pressure)
                    if retest_volume_ratio > volume_ratio * self.max_retest_volume_ratio or retest_volume_ratio > self.max_retest_volume_absolute:
                        logging.debug(f"Retest volume too high: {retest_volume_ratio:.2f}x vs breakdown {volume_ratio:.2f}x")
                        continue

                    # Step 3: Check if price bounced down (current price below retest high)
                    if j < len(DF) - 1:
                        bounced = close[-1] < low[j]
                        
                        if bounced:
                            bounce_strength = (high[j] - close[-1]) / high[j]
                    
                        if bounce_strength < self.min_bounce_strength:
                            continue
                        
                        # Downward momentum
                        if len(close) >= 2:
                            recent_momentum = close[-1] < close[-2]
                            if not recent_momentum:
                                continue
                        
                        entry_price = close[-1]
                        stop_loss = high[j] * 1.01
                        risk = stop_loss - entry_price
                        
                        if risk <= 0 or risk / entry_price > 0.05:
                            continue
                        
                        target_price = entry_price - (risk * self.risk_reward_ratio)
                        
                        # Recent retest
                        days_since_retest = len(DF) - 1 - j
                        if days_since_retest > self.max_days_since_retest:
                            continue
                        
                        # Calculate MA slope
                        ma_slope = self.calculate_ma_slope(ma200, len(DF) - 1)
                        
                        return {
                            'type': 'SHORT',
                            'symbol': DF['symbol'].iloc[0],
                            'entry': entry_price,
                            'stop': stop_loss,
                            'target': target_price,
                            'risk': risk,
                            'reward': risk * self.risk_reward_ratio,
                            'breakout_date': DF.iloc[breakout_idx]['date'],
                            'retest_date': DF.iloc[j]['date'],
                            'current_date': DF.iloc[-1]['date'],
                            'breakdown_volume_ratio': volume_ratio,
                            'retest_volume_ratio': retest_volume_ratio,
                            'avg_volume': avg_volume,
                            'bounce_strength': bounce_strength,
                            'breakdown_strength': breakdown_strength,
                            'ma_slope': ma_slope,
                            'ma_slope_pct': ma_slope * 100
                        }
        
        return None
