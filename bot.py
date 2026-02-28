"""
Breakout & Retest Trading Bot
Scans S&P 500 and NASDAQ stocks for 200 MA breakout/retest patterns
Enters long/short positions with 2:1 risk/reward ratio
"""

import sys
import _200ma_retest_detection as _200ma
import fetch_stocks as fs
from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, time
import time as time_module
import logging
from typing import List, Dict, Optional, Tuple
import os
import pandas_market_calendars as mcal
import subprocess
import json

# Setup logging
now = datetime.now()
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_bot_{now.month}-{now.day}-{now.year}_{now.hour}-{now.minute}.log'),
        logging.StreamHandler()
    ]
)

class BreakoutTradingBot:
    def __init__(self, host='127.0.0.1', port=9000, client_id=1):
        """Initialize the trading bot"""
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Trading parameters
        self.ma_period = 200  # 200-day moving average
        self.risk_reward_ratio = 2.0  # 2:1 reward to risk
        self.lookback_days = 250  # Days to fetch for analysis
        self.scan_interval = 1200  # Scan every 20 minutes during market hours
        
        # Position tracking
        self.active_positions = {}  # symbol -> position info
        self.monitored_stocks = {}  # symbol -> analysis data
        
        # Stock universe (S&P 500 + NASDAQ 100 major stocks)
        self.stock_universe = self._get_stock_universe()
    
    def _get_stock_universe(self) -> List[str]:
        """
        Get list of S&P 500 and major NASDAQ stocks
        For production, you'd fetch this from a data source
        Here's a starter list of liquid stocks
        """
        # Major S&P 500 and NASDAQ stocks (expanded list)
        # stocks
        stocks = []
        with open('stocks.txt', 'r') as f:
            for line in f:
                stocks.append(line.strip())  # strip() removes whitespace/newlines
        
        return stocks
    
    def connect(self):
        """Connect to Interactive Brokers"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.ib.reqMarketDataType(3)  # Use delayed data (free)
            logging.info(f"Connected to IB at {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        self.ib.disconnect()
        logging.info("Disconnected from IB")
    
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
    
    def get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical daily data for a stock"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f'{self.lookback_days} D',
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
    
    def calculate_position_size(self, risk_per_trade: float, risk_amount: float) -> int:
        """
        Calculate position size based on risk
        risk_per_trade: dollar amount willing to risk (e.g., $1000)
        risk_amount: price difference between entry and stop
        """
        if risk_amount <= 0:
            return 0
        
        shares = int(risk_per_trade / risk_amount)
        return max(shares, 1)  # At least 1 share
    
    def place_order(self, signal: Dict, shares: int):
        """Place order based on signal"""
        try:
            symbol = signal['symbol']
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Create bracket order
            bracket = self.ib.bracketOrder(
                action='BUY' if signal['type'] == 'LONG' else 'SELL',
                quantity=shares,
                limitPrice=signal['entry'],
                takeProfitPrice=signal['target'],
                stopLossPrice=signal['stop']
            )
            
            # Place the bracket order
            for order in bracket:
                order.tif = 'DAY'  # Day order
                order.outsideRth = False  # Don't allow outside regular trading hours
                trade = self.ib.placeOrder(contract, order)
                logging.info(f"Order placed: {trade}")
            
            # Track position
            self.active_positions[symbol] = {
                'signal': signal,
                'shares': shares,
                'entry_time': datetime.now()
            }

            # Add position info to JSON file for backup
            file_path = 'positions.json'
            new_entry = {
                'type': signal['type'],
                'symbol': symbol,
                'entry': signal['entry'],
                'stop': signal['stop'],
                'target': signal['target'],
                'risk': signal['risk'],
                'reward': signal['risk'] * self.risk_reward_ratio,
                'breakout_date': signal['breakout_date'].strftime('%Y-%m-%d'),
                'retest_date': signal['retest_date'].strftime('%Y-%m-%d'),
                'current_date': signal['current_date'].strftime('%Y-%m-%d'),
                'breakout_volume_ratio': signal['breakout_volume_ratio'],
                'retest_volume_ratio': signal['retest_volume_ratio'],
                'avg_volume': signal['avg_volume'],
                'shares': shares,
                'entry_time': datetime.now()
            }

            # 1. Check if file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    # Load existing data
                    data = json.load(file)
            else:
                # Initialize with an empty list if file doesn't exist
                data = {}

            # 2. Add/Append the new data
            data[symbol] = new_entry

            # 3. Write it back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            
            logging.info(f"Entered {signal['type']} position in {symbol}: "
                        f"{shares} shares @ ${signal['entry']:.2f}, "
                        f"Stop: ${signal['stop']:.2f}, Target: ${signal['target']:.2f}")
            
        except Exception as e:
            logging.error(f"Failed to place order for {signal['symbol']}: {e}")
    
    def scan_stocks(self):
        """Scan all stocks for trading signals"""
        logging.info(f"Scanning {len(self.stock_universe)} stocks...")
        
        signals_found = []
        
        for symbol in self.stock_universe:
            # Skip if we already have a position
            if symbol in self.active_positions:
                continue
            
            # Get historical data
            df = self.get_historical_data(symbol)
            
            if df is None or len(df) < self.ma_period:
                continue
            
            # Detect 200 MA pattern
            MA200 = _200ma.BreakoutRetestDetector(df, self.risk_reward_ratio)
            signal = MA200.detect_breakout_and_retest()
            
            if signal:
                signals_found.append(signal)
                logging.info(f"Signal found: {signal['type']} {symbol} @ ${signal['entry']:.2f}, "
                        f"Breakout Vol: {signal['breakout_volume_ratio']:.2f}x, "
                        f"Retest Vol: {signal['retest_volume_ratio']:.2f}x")
            
            # Small delay to avoid rate limiting
            time_module.sleep(0.5)
        
        return signals_found
    
    def execute_signals(self, signals: List[Dict]):
        """Execute trading signals"""
        if not signals:
            return
        
        # Get account info to determine position sizing
        account_summary = self.ib.accountSummary()
        net_liq = 0
        
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                net_liq = float(item.value)
                break
        
        # Risk 1% of account per trade
        risk_per_trade = net_liq * 0.01
        
        logging.info(f"Account value: ${net_liq:.2f}, Risk per trade: ${risk_per_trade:.2f}")
        
        for signal in signals:
            shares = self.calculate_position_size(risk_per_trade, signal['risk'])
            
            if shares > 0:
                self.place_order(signal, shares)
            else:
                logging.warning(f"Position size too small for {signal['symbol']}")
    
    def monitor_positions(self):
        """Monitor and manage active positions"""
        # Get current positions from IB
        ib_positions = self.ib.positions()
        
        # Create set of symbols with actual positions (non-zero quantity)
        ib_symbols = {pos.contract.symbol for pos in ib_positions if pos.position != 0}
        
        # Remove closed positions from our tracking
        closed_positions = []
        for symbol in list(self.active_positions.keys()):
            if symbol not in ib_symbols:
                closed_positions.append(symbol)
                position_info = self.active_positions[symbol]
                
                # Log the closed position
                logging.info(
                    f"Position closed: {symbol} ({position_info['signal']['type']}) - "
                    f"Removing from active positions"
                )
                
                # Remove from tracking
                del self.active_positions[symbol]

                # Also remove from JSON file
                try:
                    with open('positions.json', 'r') as file:
                        data = json.load(file)
                    
                    if symbol in data:
                        del data[symbol]
                    else:
                        logging.warning(f"{symbol} not found in positions.json for removal, but it was in active_positions. This could cause a logical error.")
                        
                    with open('positions.json', 'w') as file:
                        json.dump(data, file, indent=4)
                except Exception as e:
                    logging.error(f"Failed to update positions.json: {e}")
        
        # Check if program started with existing positions (e.g., from previous run)
        if len(self.active_positions) == 0 and len(ib_symbols) > 0:
            logging.info("Adding existing positions to tracking from IB data...")
            for pos in ib_positions:
                if pos.position != 0:
                    symbol = pos.contract.symbol

                    # Track position
                    with open('positions.json', 'r') as file:
                        data = json.load(file)

                    signal = {
                        'type': data[symbol]['type'],
                        'symbol': symbol,
                        'entry': data[symbol]['entry'],
                        'stop': data[symbol]['stop'],
                        'target': data[symbol]['target'],
                        'risk': data[symbol]['risk'],
                        'reward': data[symbol]['reward'],
                        'breakout_date': data[symbol]['breakout_date'],
                        'retest_date': data[symbol]['retest_date'],
                        'current_date': data[symbol]['current_date'],
                        'breakout_volume_ratio': data[symbol]['breakout_volume_ratio'],
                        'retest_volume_ratio': data[symbol]['retest_volume_ratio']
                    }

                    self.active_positions[symbol] = {
                        'signal': signal,
                        'quantity': data[symbol]['quantity'],
                        'avg_cost': data[symbol]['avg_cost']
                    }

        # Log summary
        if closed_positions:
            logging.info(f"Removed {len(closed_positions)} closed positions: {closed_positions}")
        
        # Log currently active positions
        if self.active_positions:
            logging.info(f"Active positions: {len(self.active_positions)} stocks")
            for symbol, info in self.active_positions.items():
                # Find the position in IB data for P&L info
                ib_pos = next((p for p in ib_positions if p.contract.symbol == symbol), None)
                if ib_pos:
                    logging.info(
                        f"  {symbol}: {info['signal']['type']}, "
                        f"Qty: {ib_pos.position}, "
                        f"Avg Cost: ${ib_pos.avgCost:.2f}, "
                        f"Current: ${ib_pos.marketPrice:.2f}, "
                        f"P&L: ${ib_pos.unrealizedPNL:.2f}"
                    )
        else:
            logging.info("No active positions")
    
    def run(self):
        """Main bot loop"""
        logging.info("Starting trading bot...")
        
        if not self.connect():
            logging.error("Failed to connect. Exiting.")
            return
        
        try:
            while True:
                if self.is_market_hours():
                    logging.info(f"Market is open. Scanning for signals...")
                    
                    # Monitor existing positions
                    self.monitor_positions()

                    # Scan for new signals
                    signals = self.scan_stocks()
                    
                    # Execute signals
                    if signals:
                        self.execute_signals(signals)
                    
                    # Monitor existing positions
                    self.monitor_positions()
                    
                    # Check for updates
                    if self.check_for_updates():
                        logging.info("Updating and restarting bot to apply new changes...")
                        if self.pull_updates():
                            self.restart_bot()
                    
                    # Auto-commit logs
                    self.git_commit_and_push("Auto-commit: Updated trading bot with new logs")

                    # Wait before next scan
                    logging.info(f"Waiting {self.scan_interval} seconds until next scan...")
                    time_module.sleep(self.scan_interval)
                    
                else:
                    # Market closed - wait 15 minutes and check again
                    logging.info(f"Market is closed. Next check in 15 minutes...")
                    time_module.sleep(900)  # 15 minutes

                self.git_commit_and_push("Auto-commit: Updated trading bot with new logs")
                    
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        except Exception as e:
            logging.error(f"Bot error: {e}")
        finally:
            self.disconnect()

    def check_for_updates(self) -> bool:
        """Check if remote has new commits"""
        try:
            # Fetch latest from remote
            subprocess.run(['git', 'fetch'], check=True, capture_output=True)
            
            # Compare local and remote
            result = subprocess.run(
                ['git', 'rev-list', 'HEAD...origin/main', '--count'],
                capture_output=True,
                text=True,
                check=True
            )
            
            commits_behind = int(result.stdout.strip())
            
            if commits_behind > 0:
                logging.info(f"{commits_behind} new commit(s) available")
                return True
            
            return False
            
        except Exception as e:
            logging.warning(f"Could not check for updates: {e}")
            return False
    
    def pull_updates(self) -> bool:
        """Pull latest changes from git"""
        try:
            logging.info("Pulling latest changes...")
            subprocess.run(['git', 'pull'], check=True)
            logging.info("Updates pulled successfully")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to pull updates: {e}")
            return False
    
    def restart_bot(self):
        """Restart the bot to apply updates"""
        logging.info("Restarting bot to apply updates...")
        
        # Disconnect cleanly
        self.disconnect()
        
        # Restart the Python script
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    def git_commit_and_push(self, message=None):
        """Commit and push changes to git"""
        try:
            if message is None:
                message = f"Auto-commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Commit with message
            subprocess.run(['git', 'commit', '-m', message], check=True)
            
            # Push to remote
            subprocess.run(['git', 'push'], check=True)
            
            logging.info(f"Successfully pushed: {message}")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Git error: {e}")
            return False

if __name__ == "__main__":
    # Fetch stock lists and save to file
    fs.save_stock_list(True, False)

    # Create and run the bot
    bot = BreakoutTradingBot(
        host='127.0.0.1',
        port=9000,  # Paper trading port
        client_id=1
    )

    bot.run()

    # Check for updates
    if bot.check_for_updates():
        logging.info("Updating and restarting bot to apply new changes...")
        if bot.pull_updates():
            bot.restart_bot()

    bot.git_commit_and_push("Auto-commit: Updated trading bot with new logs")