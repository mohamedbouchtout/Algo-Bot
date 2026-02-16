"""
Test script to verify breakout/retest pattern detection
Scans a few stocks and shows signals WITHOUT placing orders
"""

import pandas as pd
import logging
from typing import Optional, Dict

# Setup logging
logger = logging.getLogger(__name__)

class BreakoutRetestDetector:
    def __init__(self, df: pd.DataFrame, risk_reward_ratio: float = 2.0):
        self.ma_period = 200  # 200-day moving average
        self.risk_reward_ratio = risk_reward_ratio
        self.df = df.copy()
    
    def calculate_ma(self, period: int) -> pd.Series:
        """Calculate moving average"""
        return self.df['close'].rolling(window=period).mean()
    
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
    
    def _analyze_pattern(self) -> Optional[Dict]:
        """
        Analyze for breakout and retest pattern
        Pattern:
        1. Price crosses above/below 200 MA (breakout)
        2. Price comes back to test 200 MA (retest)
        3. Price bounces off 200 MA in breakout direction
        """
        DF = self.df.reset_index(drop=True)
        
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
        
        # Look in recent 20 days for the pattern
        for i in range(len(DF) - 5, max(len(DF) - 20, 0), -1):
            # Step 1: Find breakout above MA200
            if i < 5:
                continue
                
            # Price was below MA200, then crossed above
            below_ma = close[i-3:i].mean() < ma200[i-3:i].mean()
            crossed_above = close[i] > ma200[i] and close[i-1] <= ma200[i-1]
            
            if not (below_ma and crossed_above):
                continue
            
            breakout_idx = i
            
            # Step 2: Look for retest in next few days
            for j in range(i+1, min(i+10, len(DF))):
                # Price came back down near MA200 (within 2%)
                retest_distance = abs(low[j] - ma200[j]) / ma200[j]
                
                if retest_distance < 0.02:  # Within 2% of MA200
                    # Step 3: Check if price bounced up (current price above retest low)
                    if j < len(DF) - 1:
                        bounced = close[-1] > low[j]
                        
                        if bounced:
                            # Calculate entry, stop, and target
                            entry_price = close[-1]
                            stop_loss = low[j]  # Below retest low
                            risk = entry_price - stop_loss
                            
                            if risk <= 0:
                                continue
                            
                            target_price = entry_price + (risk * self.risk_reward_ratio)
                            
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
                                'current_date': DF.iloc[-1]['date']
                            }
        
        return None
    
    def _detect_short_pattern(self, DF: pd.DataFrame) -> Optional[Dict]:
        """Detect bearish breakout and retest"""
        ma200 = DF['ma200'].values
        close = DF['close'].values
        high = DF['high'].values
        low = DF['low'].values
        
        # Look in recent 20 days for the pattern
        for i in range(len(DF) - 5, max(len(DF) - 20, 0), -1):
            # Step 1: Find breakout below MA200
            if i < 5:
                continue
                
            # Price was above MA200, then crossed below
            above_ma = close[i-3:i].mean() > ma200[i-3:i].mean()
            crossed_below = close[i] < ma200[i] and close[i-1] >= ma200[i-1]
            
            if not (above_ma and crossed_below):
                continue
            
            breakout_idx = i
            
            # Step 2: Look for retest in next few days
            for j in range(i+1, min(i+10, len(DF))):
                # Price came back up near MA200 (within 2%)
                retest_distance = abs(high[j] - ma200[j]) / ma200[j]
                
                if retest_distance < 0.02:  # Within 2% of MA200
                    # Step 3: Check if price bounced down (current price below retest high)
                    if j < len(DF) - 1:
                        bounced = close[-1] < high[j]
                        
                        if bounced:
                            # Calculate entry, stop, and target
                            entry_price = close[-1]
                            stop_loss = high[j]  # Above retest high
                            risk = stop_loss - entry_price
                            
                            if risk <= 0:
                                continue
                            
                            target_price = entry_price - (risk * self.risk_reward_ratio)
                            
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
                                'current_date': DF.iloc[-1]['date']
                            }
        
        return None
