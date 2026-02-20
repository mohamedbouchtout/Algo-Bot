"""
Breakout & Retest Trading Bot
Scans S&P 500 and NASDAQ stocks for 200 MA breakout/retest patterns
Enters long/short positions with 2:1 risk/reward ratio
"""

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
        self.scan_interval = 300  # Scan every 5 minutes during market hours
        
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
            
            # Create bracket order (entry + stop loss + take profit)
            if signal['type'] == 'LONG':
                # Market order to enter long
                parent_order = MarketOrder('BUY', shares)
                
                # Stop loss
                stop_loss_order = StopOrder('SELL', shares, signal['stop'])
                
                # Take profit
                take_profit_order = LimitOrder('SELL', shares, signal['target'])
                
            else:  # SHORT
                # Market order to enter short
                parent_order = MarketOrder('SELL', shares)
                
                # Stop loss (buy back at higher price)
                stop_loss_order = StopOrder('BUY', shares, signal['stop'])
                
                # Take profit (buy back at lower price)
                take_profit_order = LimitOrder('BUY', shares, signal['target'])
            
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
                trade = self.ib.placeOrder(contract, order)
                logging.info(f"Order placed: {trade}")
            
            # Track position
            self.active_positions[symbol] = {
                'signal': signal,
                'shares': shares,
                'entry_time': datetime.now()
            }
            
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
                logging.info(f"Signal found: {signal['type']} {symbol} @ ${signal['entry']:.2f}")
            
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
        positions = self.ib.positions()
        
        # Update active positions tracking
        # In a real system, you'd check if stops/targets were hit
        # IB bracket orders handle this automatically
        
        for position in positions:
            symbol = position.contract.symbol
            if symbol in self.active_positions:
                logging.info(f"Active position: {symbol}, Qty: {position.position}")
    
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
                    
                    # Scan for new signals
                    signals = self.scan_stocks()
                    
                    # Execute signals
                    if signals:
                        self.execute_signals(signals)
                    
                    # Monitor existing positions
                    self.monitor_positions()
                    
                    # Wait before next scan
                    logging.info(f"Waiting {self.scan_interval} seconds until next scan...")
                    time_module.sleep(self.scan_interval)
                    
                else:
                    # Market closed - wait 15 minutes and check again
                    logging.info(f"Market is closed. Next check in 15 minutes...")
                    time_module.sleep(900)  # 15 minutes
                    
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        except Exception as e:
            logging.error(f"Bot error: {e}")
        finally:
            self.disconnect()

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
    # Create and run the bot
    bot = BreakoutTradingBot(
        host='127.0.0.1',
        port=9000,  # Paper trading port
        client_id=1
    )

    fs.save_stock_list()
    bot.run()
    bot.git_commit_and_push("Updated trading bot with new logs")