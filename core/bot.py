"""
Main TradingBot orchestrator
Coordinates all modules
"""

from datetime import datetime, timedelta
import os
import json
from ib_insync import *
from typing import Dict, List, Optional
from core.connection import ConnectionManager
from core.scheduler import Scheduler
from data_fetch.stock_fetcher import StockTickerFetcher
from data_fetch.historical_data import StockDataFetcher
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from utils.git_manager import GitManager
from utils.alerts import AlertManager
from utils.logger import setup_logger
from strategy.ai_analysis.ai_analyzer import AIAnalyzer

class TradingBot:
    TRAIN_INTERVAL = timedelta(days=6)

    def __init__(self):
        self.ib = IB()
        self.config = self.load_config()
        self.params = self.load_params()
        self.logger = setup_logger(self.config, 'bot_logs', 'trading_bot.log')
        self.last_train_time: datetime | None = None

    def load_config(self):
        """Load configuration from JSON file"""
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/config.json')
        with open(file_path, 'r') as file:
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
                with open('/proc/1/cgroup', 'r') as cgroup_file:
                    running_in_container = 'docker' in cgroup_file.read() or 'kubepods' in cgroup_file.read()
            except Exception:
                running_in_container = running_in_container

        if running_in_container:
            config['ib']['host'] = os.getenv('IB_HOST_CONTAINER', 'host.docker.internal')

        return config

    def load_params(self):
        """Load parameters from JSON file"""
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/trading_params.json')
        with open(file_path, 'r') as file:
            params = json.load(file)
        return params

    def should_retrain(self, scheduler: Scheduler) -> bool:
        # Cold start, no model exists, train regardless of day
        if self.last_train_time is None:
            return True

        # Model exists, only refresh on weekends, and only if it's stale
        if not scheduler.is_weekend():
            return False
        return datetime.now() - self.last_train_time >= self.TRAIN_INTERVAL

    def train_modules(self, ai_analyzers: Dict[str, AIAnalyzer], stock_fetcher: StockTickerFetcher):
        """Run a full pooled retrain of the RBM + CNN on the current ticker universe."""
        try:
            self.logger.info("Starting AI training...")

            for ai_analyzer in ai_analyzers.values():
                ai_analyzer.reset_dataset()

            added = 0
            for sector, industries in stock_fetcher.categorized_stocks.items():
                addedPerSector = 0
                for industry, tickers in industries.items():
                    for ticker in tickers:
                        if ai_analyzers[sector].add_ticker(ticker):
                            added += 1
                            addedPerSector += 1
                self.logger.info(f"Added {addedPerSector} tickers for {sector} sector")
            
            for ai_analyzer in ai_analyzers.values():
                ai_analyzer.finalize_training(val_split=0.2)
            
            self.last_train_time = datetime.now()
            self.logger.info(f"AI training finished: {added} tickers")
        except Exception as e:
            self.logger.error(f"AI training failed; bot will continue with previous model: {e}")

    def sectored_ai_objects(self, stock_data: StockDataFetcher, stock_fetcher: StockTickerFetcher) -> Dict[str, AIAnalyzer]:
        """Create separate AI analyzers for each sector"""
        sector_analyzers = {}
        for sector in stock_fetcher.categorized_stocks:
            analyzer = AIAnalyzer(stock_data, params=self.params)
            sector_analyzers[sector] = analyzer
        return sector_analyzers

    def run(self):
        """Main bot loop"""
        self.logger.info("Starting trading bot...")
        
        stock_fetcher = StockTickerFetcher()
        stock_data = StockDataFetcher(self.ib, self.config, self.params)
        scheduler = Scheduler()
        ai_analyzers = self.sectored_ai_objects(stock_data, stock_fetcher)
        alert_manager = AlertManager(self.config, self.params)
        position_manager = PositionManager(self.ib, alert_manager, self.config, self.params)
        connection_manager = ConnectionManager(self.ib, position_manager, alert_manager, self.config, self.params)
        order_manager = OrderManager(self.ib, stock_data, position_manager, alert_manager, ai_analyzers, self.config, self.params)
        git_manager = GitManager(self.ib, connection_manager, self.config, self.params)

        try:
            # Check for updates before starting the bot
            last_git_check = datetime.now()
            last_git_check = git_manager.git(last_git_check, force_check=True)

            max_connect_retries = 10
            connect_attempts = 0
            while not connection_manager.connect():
                connect_attempts += 1
                if connect_attempts >= max_connect_retries:
                    self.logger.error(f"Failed to connect after {max_connect_retries} attempts, exiting")
                    alert_manager.alert_error("Connection Failed", f"Could not connect to IB after {max_connect_retries} attempts")
                    return
                self.logger.warning(f"Cannot connect to IB - will retry in 1 minute (attempt {connect_attempts}/{max_connect_retries})")
                self.ib.sleep(60)
        
            # Cold-start training if no model exists
            if self.should_retrain(scheduler):
                self.train_modules(ai_analyzers, stock_fetcher)

            while True:
                market_open = scheduler.is_market_hours()
                connected = connection_manager.ensure_connected()

                if market_open and connected:
                    self.logger.info(f"Market is open. Scanning for signals...")

                    order_manager.scan_stocks(stock_fetcher.categorized_stocks)
                    position_manager.monitor_positions()
                    last_git_check = git_manager.git(last_git_check)

                    self.logger.info(f"Waiting {self.params['timing']['scan_interval']} seconds until next scan...")
                    self.ib.sleep(self.params['timing']['scan_interval'])

                elif market_open and not connected:
                    self.logger.warning("Cannot connect to IB - will retry in 1 minute")
                    last_git_check = git_manager.git(last_git_check)
                    self.ib.sleep(60)

                else:
                    if not connected:
                        self.logger.warning("Market is closed and IB disconnected. Next check in 10 minutes...")
                    else:
                        self.logger.info(f"Market is closed. Next check in 10 minutes...")

                    last_git_check = git_manager.git(last_git_check)

                    if self.should_retrain(scheduler):
                        self.train_modules(ai_analyzers, stock_fetcher)

                    self.ib.sleep(600)
                    
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
            alert_manager.alert_bot_stopped()
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
            alert_manager.alert_error(str(e), "Unexpected bot error thrown.")
        finally:
            connection_manager.disconnect() 