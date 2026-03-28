"""
Gets stock historical data from IB and stock list
"""

import logging
from ib_insync import *
from typing import Dict, List, Optional
import os
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self, ib, config, params):
        self.ib = ib
        self.config = config
        self.params = params

    def get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical daily data for a stock"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f'{self.params["strategy_retest_200ma"]["lookback_days"]} D',
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
            logging.warning(f"Failed to get data for {symbol}: {e}")
            return None