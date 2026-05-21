"""
Gets stock historical data from IB and stock list
"""

import logging
from ib_insync import *
from typing import Dict, List, Optional
import pandas as pd

# Setup logging
logger = logging.getLogger()

class StockDataFetcher:
    def __init__(self, ib, config, params):
        self.ib = ib
        self.config = config
        self.params = params

    def get_historical_data(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Fetch historical daily data for a stock"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            import math
            if lookback_days > 365:
                duration = f'{math.ceil(lookback_days / 365)} Y'
            else:
                duration = f'{lookback_days} D'

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                return None
            
            # Convert to DataFrame
            df = util.df(bars)
            df['symbol'] = symbol
            return df
            
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            return None
