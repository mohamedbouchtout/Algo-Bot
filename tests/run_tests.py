"""
Root module for running all tests. This can be expanded to include more test classes as needed.
"""

from ib_insync import *
import os
import json
import subprocess
from tests.test_retest_200ma import TestRetest200MA
from tests.test_ai_analysis import TestAIanalysis
from core.connection import ConnectionManager
from utils.alerts import AlertManager
from execution.position_manager import PositionManager
from data_fetch.historical_data import StockDataFetcher
from data_fetch.stock_fetcher import StockTickerFetcher
from utils.logger import setup_logger

class RunTests:
    def __init__(self):
        self.ib = IB()
        self.params = self.load_params()
        self.config = self.load_config()
        self.stock_data_fetcher = StockDataFetcher(self.ib, self.config, self.params)
        self.stock_fetcher = StockTickerFetcher()
        self.alert_manager = AlertManager(self.config, self.params)
        self.position_manager = PositionManager(self.ib, self.alert_manager, self.config, self.params)
        self.connection_manager = ConnectionManager(self.ib, self.position_manager, self.alert_manager, self.config, self.params)
        self.logger = setup_logger(self.config, 'test_logs', 'tests.log')

    def load_params(self):
        """Load parameters from JSON file"""
        try:
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'config/trading_params.json')
            with open(file_path, 'r') as file:
                params = json.load(file)
            return params
        except Exception as e:
            self.logger.error(f"Failed to load parameters: {e}")
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
            self.logger.warning(f"Could not detect Git branch ({e}). Defaulting to 'dev'.")
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
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def run(self):
        """Runs all the test classes"""

        # AI analysis test class
        test_ai_analysis = TestAIanalysis(self.ib, self.config, self.params, self.stock_data_fetcher, self.stock_fetcher, self.connection_manager)
        test_ai_analysis.train_modules()
        test_ai_analysis.predictions()

        # 200 MA retest test class
        test_bot = TestRetest200MA(self.ib, self.config, self.params, self.stock_data_fetcher, self.stock_fetcher, self.connection_manager)
        test_bot.test_retest_200ma()