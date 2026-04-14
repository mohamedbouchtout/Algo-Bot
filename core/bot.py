"""
Main TradingBot orchestrator
Coordinates all modules
"""

import subprocess
from datetime import datetime
import os
import json
from ib_insync import *
from core.connection import ConnectionManager
from core.scheduler import Scheduler
from data_fetch.stock_fetcher import StockTickerFetcher
from data_fetch.historical_data import StockDataFetcher
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from utils.git_manager import GitManager
from utils.alerts import AlertManager
from utils.logger import setup_logger

# # Setup logging
# now = datetime.now()
# os.makedirs('data/bot_logs', exist_ok=True)
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f'data/bot_logs/trading_bot_{now.month}-{now.day}-{now.year}_{now.hour}-{now.minute}.log'),
#         logging.StreamHandler()
#     ]
# )

class TradingBot:
    def __init__(self):
        self.ib = IB()
        self.config = self.load_config()
        self.params = self.load_params()
        self.logger = setup_logger(self.config)

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

    def run(self):
        """Main bot loop"""
        self.logger.info("Starting trading bot...")
        
        stock_fetcher = StockTickerFetcher()
        stock_data = StockDataFetcher(self.ib, self.config, self.params)
        scheduler = Scheduler()
        alert_manager = AlertManager(self.config, self.params)
        position_manager = PositionManager(self.ib, alert_manager, self.config, self.params)
        connection_manager = ConnectionManager(self.ib, position_manager, alert_manager, self.config, self.params)
        order_manager = OrderManager(self.ib, stock_data, position_manager, alert_manager, self.config, self.params)
        git_manager = GitManager(self.ib, connection_manager, self.config, self.params)

        if not connection_manager.connect():
            self.logger.error("Failed to connect. Exiting.")
            return
         
        last_git_check = datetime.now()

        try:
            while True:
                if scheduler.is_market_hours() and connection_manager.ensure_connected():
                    self.logger.info(f"Market is open. Scanning for signals...")
                    
                    # Scan and execute new signals
                    order_manager.scan_stocks(stock_fetcher.stock_list)
                    
                    # Monitor existing positions
                    position_manager.monitor_positions()
                    
                    # Git operations
                    last_git_check = git_manager.git(last_git_check)  # Update last_git_check if it was changed

                    # Wait before next scan
                    self.logger.info(f"Waiting {self.params['timing']['scan_interval']} seconds until next scan...")
                    self.ib.sleep(self.params['timing']['scan_interval'])

                elif not scheduler.is_market_hours() and not connection_manager.ensure_connected():
                    self.logger.warning("Cannot connect to IB and market is closed - will retry in 30 minutes")
                    last_git_check = git_manager.git(last_git_check)  # Update last_git_check if it was changed
                    self.ib.sleep(1800)  # 30 minutes
                elif not connection_manager.ensure_connected():
                    self.logger.warning("Cannot connect to IB - will retry in 1 minute")
                    last_git_check = git_manager.git(last_git_check)  # Update last_git_check if it was changed
                    self.ib.sleep(60)  # 1 minute
                else:
                    self.logger.info(f"Market is closed. Next check in 30 minutes...")
                    last_git_check = git_manager.git(last_git_check)  # Update last_git_check if it was changed
                    self.ib.sleep(1800)  # 30 minutes
                    
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
            alert_manager.alert_bot_stopped()
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
            alert_manager.alert_error(str(e), "Unexpected bot error thrown.")
        finally:
            connection_manager.disconnect()