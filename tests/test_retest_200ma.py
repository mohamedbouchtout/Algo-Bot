"""
Regression test for 200ma strategy logic
"""

from ib_insync import *
import logging
import os
import json
import subprocess
from datetime import datetime
from strategy.retest_200ma.indicators import TrendIndicator
from data_fetch.historical_data import StockDataFetcher
from data_fetch.stock_fetcher import StockTickerFetcher
from core.connection import ConnectionManager
from utils.alerts import AlertManager
from execution.position_manager import PositionManager

# Setup logging
now = datetime.now()
os.makedirs('data/test_logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/test_logs/test_retest_200ma_{now.month}-{now.day}-{now.year}_{now.hour}-{now.minute}.log'),
        logging.StreamHandler()
    ]
)

class TestRetest200MA:
    def __init__(self):
        self.ib = IB()
        self.config = self.load_config()
        self.params = self.load_params()
        self.stock_data_fetcher = StockDataFetcher(self.ib, self.config, self.params)
        self.stock_fetcher = StockTickerFetcher()
        self.alert_manager = AlertManager(self.config, self.params)
        self.position_manager = PositionManager(self.ib, self.alert_manager, self.config, self.params)

    def load_params(self):
        """Load parameters from JSON file"""
        try:
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'config/trading_params.json')
            with open(file_path, 'r') as file:
                params = json.load(file)
            return params
        except Exception as e:
            logging.error(f"Failed to load parameters: {e}")
            raise

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            # 1. Ask Git for the current branch name
            # 'git rev-parse --abbrev-ref HEAD' is the standard way to get the branch name
            branch_byte = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
            branch = branch_byte.decode('utf-8').strip()
        except Exception as e:
            # Fallback if Git isn't installed or this isn't a repo
            logging.warning(f"Could not detect Git branch ({e}). Defaulting to 'dev'.")
            branch = 'develop'

        # 2. Set environment based on branch
        # If we are on 'main' or 'master', use prod. Otherwise, use dev.
        env = 'prod' if branch in ['bot/production', 'origin/bot/production'] else 'dev'
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'config/{env}.json')
        
        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
            return config
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

    def test_retest_200ma(self):
        """Test the 200 MA breakout and retest logic on historical data"""

        connection_manager = ConnectionManager(self.ib, self.position_manager, self.alert_manager, self.config, self.params)
        if not connection_manager.connect():
            logging.error("Failed to connect. Exiting.")
            return

        signals = []
        try:
            for ticker in self.stock_fetcher.stock_list:
                logging.info(f"Testing {ticker}...")
                df = self.stock_data_fetcher.get_historical_data(ticker)
                
                indicator_200ma = TrendIndicator(df, self.config, self.params)
                signal = indicator_200ma.detect_breakout_and_retest()
                
                if signal:
                    logging.info(f"Signal detected for {ticker}: {signal}")
                    signals.append((ticker, signal))
                else:
                    logging.info(f"No signal for {ticker}.")

        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        except Exception as e:
            logging.error(f"Bot error: {e}")
        finally:
            logging.info(f"Total signals detected: {len(signals)}")
            connection_manager.disconnect()