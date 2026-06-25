"""
Determine when to run the trading strategy based on market conditions and time of day.
"""

import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

# Setup logging
logger = logging.getLogger()

ET = ZoneInfo('America/New_York')  # Eastern Timezone for market hours


class Scheduler:
    def __init__(self):
        pass

    def is_weekend(self) -> bool:
        """Check if the current day is Saturday or Sunday."""
        return datetime.now(ET).weekday() >= 5  # 5 = Saturday, 6 = Sunday

    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM EST)"""
        now = datetime.now(ET)
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
        # Check if weekend
        if now.weekday() >= 5 or schedule.empty:  # 5 = Saturday, 6 = Sunday
            return False

        # Market hours to trade on: 9:30 AM - 3:30 PM EST
        market_open = time(9, 30)
        market_close = time(15, 30)
        current_time = now.time()

        return market_open <= current_time < market_close
