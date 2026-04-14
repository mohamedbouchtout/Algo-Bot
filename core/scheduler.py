"""
Determine when to run the trading strategy based on market conditions and time of day.
"""

import logging
from datetime import datetime, time
import pandas_market_calendars as mcal

# Setup logging
logger = logging.getLogger()

class Scheduler:
    def __init__(self):
        pass

    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM EST)"""
        now = datetime.now()
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
        # Check if weekend
        if now.weekday() >= 5 or schedule.empty:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
