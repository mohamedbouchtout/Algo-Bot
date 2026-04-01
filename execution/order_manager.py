"""
Place and manage orders
"""

import logging
from ib_insync import *
from typing import Dict
from datetime import datetime
import time as time_module
from execution.risk_manager import RiskManager
from execution.position_manager import PositionManager
from data_fetch.historical_data import StockDataFetcher
from strategy.retest_200ma.indicators import TrendIndicator
from utils.alerts import AlertManager

# Setup logging
logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, ib, position_manager: PositionManager, alert_manager: AlertManager, config, params):
        self.ib = ib
        self.position_manager = position_manager
        self.config = config
        self.params = params
        self.alert_manager = alert_manager

    def scan_stocks(self, stock_list: list[str]):
        """Scan all stocks for trading signals"""
        logging.info(f"Scanning {len(stock_list)} stocks...")

        for symbol in stock_list:
            # Skip if we already have a position
            if symbol in self.position_manager.active_positions:
                continue
            
            # Get historical data
            stock_data = StockDataFetcher(self.ib, self.config, self.params)
            df = stock_data.get_historical_data(symbol)
            
            if df is None or len(df) < self.params['strategy_retest_200ma']['ma_period']:
                continue
            
            # Detect 200 MA pattern
            indicator_200ma = TrendIndicator(df, self.config, self.params)
            signal = indicator_200ma.detect_breakout_and_retest()
            
            if signal:
                logging.info(f"Signal found: {signal['type']} {symbol} @ ${signal['entry']:.2f}, "
                        f"Breakout Vol: {signal['breakout_volume_ratio']:.2f}x, "
                        f"Retest Vol: {signal['retest_volume_ratio']:.2f}x")
                
                self.execute_signal(signal)  # Execute immediately for each signal
            
            # Small delay to avoid rate limiting
            time_module.sleep(0.5)

    def execute_signal(self, signal: Dict):
        """Execute trading signals"""
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
        
        # Get position size based on risk
        risk_manager = RiskManager(self.config, self.params)
        shares = risk_manager.calculate_position_size(net_liq, signal['entry'], signal['stop'])

        if shares > 0:
            # Calculate trade cost
            trade_cost = shares * signal['entry']

            # Check if this trade would exceed available cash
            if risk_manager.can_take_trade(net_liq, invested_amount + trade_cost, len(current_positions)) is False:
                return
            
            self.place_order(signal, shares)
        else:
            logging.warning(f"Position size too small for {signal['symbol']}")

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
            self.position_manager.active_positions[symbol] = {
                'signal': signal,
                'shares': shares,
                'entry_time': entry_time
            }

            # Save to JSON with proper serialization
            self.position_manager.add_position(symbol, signal, shares, entry_time)
            
            logging.info(f"Entered {signal['type']} position in {symbol}: "
                        f"{shares} shares @ ${signal['entry']:.2f}, "
                        f"Stop: ${signal['stop']:.2f}, Target: ${signal['target']:.2f}")
            
            # Send email alert for new trade entry
            self.alert_manager.alert_trade_entry(signal)

        except Exception as e:
            logging.error(f"Failed to place order for {signal['symbol']}: {e}")