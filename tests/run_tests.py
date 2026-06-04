"""
Root module for running all tests. This can be expanded to include more test classes as needed.
"""

import json
import os

from ib_insync import *

from core.connection import ConnectionManager
from data_fetch.historical_data import StockDataFetcher
from data_fetch.stock_fetcher import StockTickerFetcher
from execution.position_manager import PositionManager
from tests.test_ai_analysis import TestAIanalysis
from tests.test_retest_200ma import TestRetest200MA
from utils.alerts import AlertManager
from utils.logger import setup_logger


class RunTests:
    def __init__(self):
        self.ib = IB()
        self.params = self.load_params()
        self.config = self.load_config()
        self.logger = setup_logger(self.config, 'test_logs', 'tests.log')
        self.stock_data_fetcher = StockDataFetcher(self.ib, self.config, self.params)
        self.stock_fetcher = StockTickerFetcher()
        self.alert_manager = AlertManager(self.config, self.params)
        self.position_manager = PositionManager(self.ib, self.alert_manager, self.config, self.params)
        self.connection_manager = ConnectionManager(self.ib, self.position_manager, self.alert_manager, self.config, self.params)

    def load_params(self):
        """Load parameters from JSON file"""
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/trading_params.json')
        with open(file_path) as file:
            params = json.load(file)
        return params

    def load_config(self):
        """Load configuration from JSON file"""
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/config.json')
        with open(file_path) as file:
            config = json.load(file)

        # If running in a container, override host to use Docker DNS
        running_in_container = os.getenv('RUNNING_IN_CONTAINER', '').lower() == 'true'
        if not running_in_container:
            try:
                running_in_container = os.path.exists('/.dockerenv')
            except Exception:
                running_in_container = False
        if not running_in_container:
            try:
                with open('/proc/1/cgroup') as cgroup_file:
                    running_in_container = 'docker' in cgroup_file.read() or 'kubepods' in cgroup_file.read()
            except Exception:
                running_in_container = running_in_container

        if running_in_container:
            config['ib']['host'] = os.getenv('IB_HOST_CONTAINER', 'host.docker.internal')

        return config

    def run(self):
        """Runs all the test classes"""
        if not self.connection_manager.connect():
            self.logger.error('Failed to connect. Exiting.')
            return

        try:
            # AI analysis test class
            test_ai_analysis = TestAIanalysis(self.ib, self.config, self.params, self.stock_data_fetcher, self.stock_fetcher, self.connection_manager)
            test_ai_analysis.train_modules()
            test_ai_analysis.predictions()

            # 200 MA retest test class
            test_bot = TestRetest200MA(self.ib, self.config, self.params, self.stock_data_fetcher, self.stock_fetcher, self.connection_manager)
            test_bot.test_retest_200ma()
        except KeyboardInterrupt:
            self.logger.info('Bot stopped by user')
            self.alert_manager.alert_bot_stopped()
        except Exception as e:
            self.logger.error(f'Bot error: {e}')
            self.alert_manager.alert_error(str(e), 'Unexpected bot error thrown.')
        finally:
            self.connection_manager.disconnect()
