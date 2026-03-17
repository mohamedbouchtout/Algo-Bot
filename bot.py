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
        self.git_commit_interval = 3600  # Git commit every hour max
        
        # Position tracking
        self.active_positions = {}  # symbol -> position info
        self.monitored_stocks = {}  # symbol -> analysis data
        
        # Stock universe (S&P 500 + NASDAQ 100 major stocks)
        self.stock_universe = self._get_stock_universe()
    
    def _get_stock_universe(self, file_name = 'stocks.txt') -> List[str]:
        """
        Get list of S&P 500 and major NASDAQ stocks
        For production, you'd fetch this from a data source
        Here's a starter list of liquid stocks
        """
        # Major S&P 500 and NASDAQ stocks (expanded list)
        # stocks
        stocks = []
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(file_path, 'r') as f:
            for line in f:
                stocks.append(line.strip())  # strip() removes whitespace/newlines
        
        return stocks
    
    def connect(self):
        """Connect to Interactive Brokers"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.ib.reqMarketDataType(3)  # Use delayed data (free)
            logging.info(f"Connected to IB at {self.host}:{self.port}")

            # Load existing positions from JSON on startup
            self._load_positions_from_json()

            return True
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        self.ib.disconnect()
        logging.info("Disconnected from IB")
    
    def ensure_connected(self) -> bool:
        """Ensure IB connection is active, reconnect if needed"""
        if not self.ib.isConnected():
            logging.warning("IB connection lost. Reconnecting...")
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.ib.reqMarketDataType(3)
                logging.info("Reconnected to IB")
                return True
            except Exception as e:
                logging.error(f"Reconnection failed: {e}")
                return False
        return True

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
            entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.active_positions[symbol] = {
                'signal': signal,
                'shares': shares,
                'entry_time': entry_time
            }

            # Save to JSON with proper serialization
            self._save_position_to_json(symbol, signal, shares, entry_time)
            
            logging.info(f"Entered {signal['type']} position in {symbol}: "
                        f"{shares} shares @ ${signal['entry']:.2f}, "
                        f"Stop: ${signal['stop']:.2f}, Target: ${signal['target']:.2f}")
            
        except Exception as e:
            logging.error(f"Failed to place order for {signal['symbol']}: {e}")
    
    def _save_position_to_json(self, symbol: str, signal: Dict, shares: int, entry_time: str):
        """Save position to JSON file"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'positions.json')
            
            # Load existing data
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
            else:
                data = {}
            
            # Create entry with proper datetime serialization
            new_entry = {
                'type': signal['type'],
                'symbol': symbol,
                'entry': signal['entry'],
                'stop': signal['stop'],
                'target': signal['target'],
                'risk': signal['risk'],
                'reward': signal['risk'] * self.risk_reward_ratio,
                'breakout_date': signal['breakout_date'].strftime('%Y-%m-%d') if hasattr(signal['breakout_date'], 'strftime') else str(signal['breakout_date']),
                'retest_date': signal['retest_date'].strftime('%Y-%m-%d') if hasattr(signal['retest_date'], 'strftime') else str(signal['retest_date']),
                'current_date': signal['current_date'].strftime('%Y-%m-%d') if hasattr(signal['current_date'], 'strftime') else str(signal['current_date']),
                'breakout_volume_ratio': signal.get('breakout_volume_ratio', 0),
                'retest_volume_ratio': signal.get('retest_volume_ratio', 0),
                'avg_volume': signal.get('avg_volume', 0),
                'bounce_strength': signal.get('bounce_strength', 0),
                'breakdown_strength': signal.get('breakdown_strength', 0),
                'ma_slope': signal.get('ma_slope', 0),
                'ma_slope_pct': signal.get('ma_slope_pct', 0),
                'shares': shares,
                'entry_time': entry_time
            }
            
            # Add to data
            data[symbol] = new_entry
            
            # Write back to file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            
            logging.debug(f"Saved position {symbol} to positions.json")
            
        except Exception as e:
            logging.error(f"Failed to save position to JSON: {e}")

    def _load_positions_from_json(self):
        """Load positions from JSON file on startup"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'positions.json')
            
            if not os.path.exists(file_path):
                logging.info("No positions.json file found - starting fresh")
                return
            
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            if not data:
                logging.info("positions.json is empty - starting fresh")
                return
            
            # Get actual IB positions to verify
            ib_positions = self.ib.positions()
            ib_symbols = {pos.contract.symbol for pos in ib_positions if pos.position != 0}
            
            # Load positions from JSON
            loaded_count = 0
            symbols_to_remove = []
            for symbol, position_data in data.items():
                # Only load if position actually exists in IB
                if symbol in ib_symbols:
                    # Reconstruct signal dict
                    signal = {
                        'type': position_data['type'],
                        'symbol': symbol,
                        'entry': position_data['entry'],
                        'stop': position_data['stop'],
                        'target': position_data['target'],
                        'risk': position_data['risk'],
                        'reward': position_data['reward'],
                        'breakout_date': position_data['breakout_date'],
                        'retest_date': position_data['retest_date'],
                        'current_date': position_data['current_date'],
                        'breakout_volume_ratio': position_data.get('breakout_volume_ratio', 0),
                        'retest_volume_ratio': position_data.get('retest_volume_ratio', 0),
                        'avg_volume': position_data.get('avg_volume', 0),
                        'bounce_strength': position_data.get('bounce_strength', 0),
                        'breakdown_strength': position_data.get('breakdown_strength', 0),
                        'ma_slope': position_data.get('ma_slope', 0),
                        'ma_slope_pct': position_data.get('ma_slope_pct', 0)
                    }
                    
                    self.active_positions[symbol] = {
                        'signal': signal,
                        'shares': position_data['shares'],
                        'entry_time': position_data['entry_time']
                    }
                    
                    loaded_count += 1
                    logging.info(f"Loaded position from JSON: {symbol} ({position_data['type']})")
                else:
                    logging.warning(f"Position {symbol} in JSON but not in IB - removing from JSON")
                    symbols_to_remove.append(symbol)
            
            logging.info(f"Loaded {loaded_count} positions from positions.json")

            # Now safe to remove
            if symbols_to_remove:
                for symbol in symbols_to_remove:
                    del data[symbol]
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
            
        except Exception as e:
            logging.error(f"Failed to load positions from JSON: {e}")

    def _remove_position_from_json(self, symbol: str):
        """Remove position from JSON file"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'positions.json')
            
            if not os.path.exists(file_path):
                return
            
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            if symbol in data:
                del data[symbol]
                
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
                
                logging.debug(f"Removed position {symbol} from positions.json")
            else:
                logging.warning(f"{symbol} not found in positions.json for removal")
            
        except Exception as e:
            logging.error(f"Failed to remove position from JSON: {e}")

    def scan_stocks(self):
        """Scan all stocks for trading signals"""
        logging.info(f"Scanning {len(self.stock_universe)} stocks...")
        
        #signals_found = []
        
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
                logging.info(f"Signal found: {signal['type']} {symbol} @ ${signal['entry']:.2f}, "
                        f"Breakout Vol: {signal['breakout_volume_ratio']:.2f}x, "
                        f"Retest Vol: {signal['retest_volume_ratio']:.2f}x")
                
                self.execute_signal(signal)  # Execute immediately for each signal
            
            # Small delay to avoid rate limiting
            time_module.sleep(0.5)
        
        #return signals_found
    
    def execute_signal(self, signal: Dict):
        """Execute trading signals"""
        # if not signals:
        #     return
        
        # Get account info to determine position sizing
        account_summary = self.ib.accountSummary()
        net_liq = 0
        cash_balance = 0
        
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                net_liq = float(item.value)
            elif item.tag == 'TotalCashValue':
                cash_balance = float(item.value)
        
        # Calculate current invested amount
        current_positions = self.ib.positions()
        invested_amount = sum(
            abs(pos.position * pos.marketPrice) 
            for pos in current_positions 
            if pos.position != 0
        )

        # Cash reserve requirement (30% must stay in cash)
        max_investment_allowed = net_liq * 0.70  # Can invest up to 70%
        available_to_invest = max_investment_allowed - invested_amount
        invested_pct = (invested_amount / net_liq * 100) if net_liq > 0 else 0
        
        logging.info(f"Account Summary:")
        logging.info(f"  Net Liquidation: ${net_liq:,.2f}")
        logging.info(f"  Cash Balance: ${cash_balance:,.2f}")
        logging.info(f"  Currently Invested: ${invested_amount:,.2f} ({invested_pct:.1f}%)")
        logging.info(f"  Max Investment Allowed (70%): ${max_investment_allowed:,.2f}")
        logging.info(f"  Available to Invest: ${available_to_invest:,.2f}")
        
        # Check if we're already at max investment
        if available_to_invest <= 0:
            logging.warning(
                f"Cannot take new trades - already at max investment "
                f"(${invested_amount:,.2f} / ${max_investment_allowed:,.2f})"
            )
            return

        # Risk 1% of account per trade
        risk_per_trade = net_liq * 0.01
        
        logging.info(f"Account value: ${net_liq:.2f}, Risk per trade: ${risk_per_trade:.2f}")
        
        # Get position size based on risk
        shares = self.calculate_position_size(risk_per_trade, signal['risk'])

        if shares > 0:
            # Calculate trade cost
            trade_cost = shares * signal['entry']
            
            # Check if this trade would exceed available cash
            if trade_cost > available_to_invest:
                logging.warning(
                    f"Skipping {signal['symbol']} - trade cost ${trade_cost:,.2f} "
                    f"exceeds available investment ${available_to_invest:,.2f} - (would violate 30% cash reserve)"
                )
                return

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

                # Remove from JSON
                self._remove_position_from_json(symbol)
        
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
        
        last_git_commit = datetime.now()

        try:
            while True:
                if self.is_market_hours() and self.ensure_connected():
                    logging.info(f"Market is open. Scanning for signals...")
                    
                    # Scan and execute new signals
                    self.scan_stocks()
                    
                    # Monitor existing positions
                    self.monitor_positions()
                    
                    # Git operations
                    self.git(last_git_commit)

                    # Wait before next scan
                    logging.info(f"Waiting {self.scan_interval} seconds until next scan...")
                    time_module.sleep(self.scan_interval)

                elif not self.is_market_hours() and not self.ensure_connected():
                    logging.warning("Cannot connect to IB and market is closed - will retry in 15 minutes")
                    self.git(last_git_commit, self.git_commit_interval)  # Git operations even if market is closed and connection fails
                    time_module.sleep(900)  # Wait 15 minutes before retrying connection
                elif not self.ensure_connected():
                    logging.warning("Cannot connect to IB - will retry in 15 minutes")
                    self.git(last_git_commit, self.git_commit_interval)  # Git operations even if market is closed and connection fails
                    time_module.sleep(900)  # Wait 15 minutes before retrying connection
                else:
                    logging.info(f"Market is closed. Next check in 15 minutes...")
                    self.git(last_git_commit, self.git_commit_interval)  # Git operations even if market is closed and connection fails
                    time_module.sleep(900)  # 15 minutes
                    
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        except Exception as e:
            logging.error(f"Bot error: {e}")
        finally:
            self.disconnect()

    def git(self,last_git_commit: datetime):
        """Commit and push changes to git if interval has passed"""
        # Only commit once per hour
        if (datetime.now() - last_git_commit).total_seconds() > self.git_commit_interval:
            self.git_commit_and_push("Auto-commit: Trading bot update")
            last_git_commit = datetime.now()
        
        # Check for updates only once per day and after market close
        if datetime.now().hour == 16 and datetime.now().minute < 20:  # After market close
            if self.check_for_updates():
                logging.info("Updating and restarting bot to apply new changes...")
                if self.pull_updates():
                    self.restart_bot()

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
            
            # Check if there are changes to commit
            status = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if not status.stdout.strip():
                logging.debug("No changes to commit")
                return True

            # Commit with message
            subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True)
            
            # Push to remote
            subprocess.run(['git', 'push'], check=True)
            
            logging.info(f"Successfully pushed: {message}")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Git error: {e}")
            return False

if __name__ == "__main__":
    # Fetch stock lists and save to file
    fs.save_stock_list()

    # Create and run the bot
    bot = BreakoutTradingBot(
        host='127.0.0.1',
        port=9000,  # Paper trading port
        client_id=1
    )

    bot.run()