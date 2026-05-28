"""
Place and manage orders
"""

import logging
from ib_insync import *
from typing import Dict
import time
from datetime import datetime
from execution.risk_manager import RiskManager
from execution.position_manager import PositionManager
from data_fetch.historical_data import StockDataFetcher
from strategy.retest_200ma.indicators import TrendIndicator
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from utils.alerts import AlertManager

# Setup logging
logger = logging.getLogger()

class OrderManager:
    def __init__(self, ib, stock_data: StockDataFetcher, position_manager: PositionManager, alert_manager: AlertManager, ai_analyzers: Dict[str, AIAnalyzer], config, params):
        self.ib = ib
        self.stock_data = stock_data
        self.position_manager = position_manager
        self.alert_manager = alert_manager
        self.ai_analyzers = ai_analyzers
        self.config = config
        self.params = params

    def scan_stocks(self, categorized_stocks: Dict[str, Dict[str, list]]):
        """Scan all stocks for trading signals"""
        logger.info("Scanning all stocks...")

        for sector, industries in categorized_stocks.items():
            for industry, tickers in industries.items():
                for ticker in tickers:
                    # Skip if we already have a position
                    if ticker in self.position_manager.active_positions:
                        continue
                    
                    # Get historical data
                    logger.info(f"Testing {ticker}...")
                    df = self.stock_data.get_historical_data(ticker, self.params['strategy_retest_200ma']['lookback_days'])
                    
                    if df is None or len(df) < self.params['strategy_retest_200ma']['ma_period']:
                        continue

                    # AI predictions
                    try:
                        prediction = self.ai_analyzers[sector].predict(ticker)
                        if prediction is not None and 'probs' in prediction and prediction['class'] in prediction['probs']:
                            class_type = prediction['class']
                            if prediction['probs'][class_type] > self.params['ai_analyzer']['confidence_threshold'] and class_type in ['LONG', 'SHORT']:
                                logger.info(f"AI prediction for {ticker}: {class_type} with confidence {prediction['probs'][class_type]:.2f}")

                                ai_signal = self.ai_analyzers[sector].construct_signal(df, self.params, class_type, prediction['probs'][class_type])
                                if ai_signal:
                                    self.execute_signal(ai_signal)  # Execute immediately for each signal
                                    logger.info("Waiting 1 minute after executing signal...")
                                    self.ib.sleep(60)  # Small delay to avoid rate limiting
                                    continue
                    except RuntimeError as e:
                        logger.warning(f"AI prediction failed for {ticker} (not trained): {e}")
                    except KeyError as e:
                        logger.warning(f"AI prediction data error for {ticker}: {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected AI error for {ticker}: {e}")

                    # Detect 200 MA pattern
                    indicator_200ma = TrendIndicator(df, self.config, self.params)
                    signal = indicator_200ma.detect_breakout_and_retest()

                    if signal:
                        logger.info(f"Signal found: {signal['type']} {ticker} @ ${signal['entry']:.2f}, "
                                f"Breakout Vol: {signal['breakout_volume_ratio']:.2f}x, "
                                f"Retest Vol: {signal['retest_volume_ratio']:.2f}x")
                        
                        self.execute_signal(signal)  # Execute immediately for each signal
                        logger.info("Waiting 1 minute after executing signal...")
                        self.ib.sleep(60)  # Small delay to avoid rate limiting
                    else:
                        logger.info(f"No signal found for {ticker}")

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
        current_positions = [p for p in self.ib.portfolio() if p.position != 0]
        invested_amount = sum(
            abs(item.position * item.marketPrice)
            for item in self.ib.portfolio()
            if item.position != 0
        )

        max_investment_allowed = net_liq * self.params['risk_management']['max_investment_pct']
        available_to_invest = max_investment_allowed - invested_amount
        invested_pct = (invested_amount / net_liq * 100) if net_liq > 0 else 0
        
        logger.info(f"Account Summary:")
        logger.info(f"  Net Liquidation: ${net_liq:,.2f}")
        logger.info(f"  Cash Balance: ${cash_balance:,.2f}")
        logger.info(f"  Currently Invested: ${invested_amount:,.2f} ({invested_pct:.1f}%)")
        logger.info(f"  Max Investment Allowed ({self.params['risk_management']['max_investment_pct']*100:.0f}%): ${max_investment_allowed:,.2f}")
        logger.info(f"  Available to Invest: ${available_to_invest:,.2f}")
        
        # Check if we're already at max investment
        if available_to_invest <= 0:
            logger.warning(
                f"Cannot take new trades - already at max investment "
                f"(${invested_amount:,.2f} / ${max_investment_allowed:,.2f})"
            )
            return
        
        # Get position size based on risk
        risk_manager = RiskManager(self.params)
        shares = risk_manager.calculate_position_size(net_liq, signal['entry'], signal['stop'])

        if shares > 0:
            # Calculate trade cost
            trade_cost = shares * signal['entry']

            # Check if this trade would exceed available cash
            if risk_manager.can_take_trade(net_liq, invested_amount + trade_cost, len(current_positions)) is False:
                return
            
            self.place_order(signal, shares)
        else:
            logger.warning(f"Position size too small for {signal['symbol']}")

    def place_order(self, signal: Dict, shares: int):
        """Place order based on signal, waiting for the entry to actually fill
        before persisting the position or alerting the user."""
        try:
            symbol = signal['symbol']
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # MARKET parent ensures we actually get filled. Children stay as the
            # stop/target prices from the signal.
            action = 'BUY' if signal['type'] == 'LONG' else 'SELL'
            parent = MarketOrder(action, shares, tif='DAY', outsideRth=False, transmit=False)
            parent_trade = self.ib.placeOrder(contract, parent)

            take_profit = LimitOrder(
                'SELL' if action == 'BUY' else 'BUY',
                shares,
                signal['target'],
                tif='GTC',
                outsideRth=True,
                parentId=parent_trade.order.orderId,            # filled in below
                transmit=False,
            )
            stop_loss = StopOrder(
                'SELL' if action == 'BUY' else 'BUY',
                shares,
                signal['stop'],
                tif='GTC',
                outsideRth=True,
                parentId=parent_trade.order.orderId,
                transmit=True,         # last child triggers all three
            )

            # Submit children orders
            tp_trade = self.ib.placeOrder(contract, take_profit)
            sl_trade = self.ib.placeOrder(contract, stop_loss)

            # Wait up to 600 s for the parent to fill.  Yields to the event loop.
            deadline = time.time() + 600
            while parent_trade.orderStatus.status not in ('Filled', 'Cancelled', 'ApiCancelled', 'Inactive'):
                self.ib.waitOnUpdate(timeout=1)
                if time.time() > deadline:
                    break

            filled_qty = parent_trade.orderStatus.filled
            status = parent_trade.orderStatus.status
            fill_price = parent_trade.orderStatus.avgFillPrice
            if status != 'Filled' or fill_price <= 0:
                logger.warning(
                    f"{symbol} parent order did NOT fill (status={status}). "
                    f"Cancelling bracket and skipping persistence."
                )
                for trade in [parent_trade, tp_trade, sl_trade]:
                    try:
                        self.ib.cancelOrder(trade.order)
                    except Exception as cancel_err:
                        logger.error(f"Failed to cancel order: {cancel_err}")

                # Close out any partial filled orders
                if filled_qty > 0:
                    logger.warning(
                        f"{symbol} partial fill of {filled_qty} shares at ${fill_price:.2f}. "
                        f"Attempting to cancel remaining and skip persistence."
                    )
                    try:
                        close_action = 'SELL' if action == 'BUY' else 'BUY'
                        close_order = MarketOrder(close_action, filled_qty, tif='DAY', transmit=True)
                        self.ib.placeOrder(contract, close_order)
                    except Exception as cancel_err:
                        logger.error(f"Failed to cancel remaining order: {cancel_err}")
                        
                return

            # Use the actual fill price, not the stale signal price
            signal['entry'] = float(fill_price)

            entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.position_manager.active_positions[symbol] = {
                'signal': signal,
                'shares': shares,
                'entry_time': entry_time,
            }
            self.position_manager.add_position(symbol, signal, shares, entry_time)

            logger.info(
                f"FILLED {signal['type']} {symbol}: {shares} sh @ ${fill_price:.2f}, "
                f"Stop: ${signal['stop']:.2f}, Target: ${signal['target']:.2f}"
            )
            self.alert_manager.alert_trade_entry(signal)

        except Exception as e:
            logger.error(f"Failed to place order for {signal['symbol']}: {e}")
