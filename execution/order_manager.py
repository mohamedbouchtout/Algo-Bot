"""
Place and manage orders
"""

import logging
import time
from datetime import datetime
from typing import Dict

from ib_insync import LimitOrder, MarketOrder, Stock, StopOrder

from data_fetch.historical_data import StockDataFetcher
from execution.position_manager import PositionManager
from execution.risk_manager import RiskManager
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from strategy.retest_200ma.indicators import TrendIndicator
from utils.alerts import AlertManager

# Setup logging
logger = logging.getLogger()


class OrderManager:
    def __init__(
        self,
        ib,
        stock_data: StockDataFetcher,
        position_manager: PositionManager,
        alert_manager: AlertManager,
        ai_analyzers: dict[str, AIAnalyzer],
        config,
        params,
    ):
        self.ib = ib
        self.stock_data = stock_data
        self.position_manager = position_manager
        self.alert_manager = alert_manager
        self.ai_analyzers = ai_analyzers
        self.config = config
        self.params = params

    def scan_stocks(self, categorized_stocks: dict[str, dict[str, list]]):
        """Scan all stocks for trading signals"""
        logger.info('Scanning all stocks...')

        for sector, industries in categorized_stocks.items():
            for industry, tickers in industries.items():
                for ticker in tickers:
                    # Skip if we already have a position
                    if ticker in self.position_manager.active_positions:
                        continue

                    # Get historical data
                    logger.info(f'Testing {ticker}...')
                    df = self.stock_data.get_historical_data(ticker, self.params['strategy_retest_200ma']['lookback_days'])

                    if df is None or len(df) < self.params['strategy_retest_200ma']['ma_period']:
                        continue

                    # AI predictions
                    try:
                        prediction = self.ai_analyzers[sector].predict(ticker)
                        if prediction is not None and 'probs' in prediction and prediction['class'] in prediction['probs']:
                            class_type = prediction['class']
                            if prediction['probs'][class_type] > self.params['ai_analyzer']['confidence_threshold'] and class_type in [
                                'LONG',
                                'SHORT',
                            ]:
                                logger.info(f'AI prediction for {ticker}: {class_type} with confidence {prediction["probs"][class_type]:.2f}')

                                ai_signal = self.ai_analyzers[sector].construct_signal(df, self.params, class_type, prediction['probs'][class_type])
                                if ai_signal:
                                    self.execute_signal(ai_signal)  # Execute immediately for each signal
                                    logger.info('Waiting 1 minute after executing signal...')
                                    self.ib.sleep(60)  # Small delay to avoid rate limiting
                                    continue
                    except RuntimeError as e:
                        logger.warning(f'AI prediction failed for {ticker} (not trained): {e}')
                    except KeyError as e:
                        logger.warning(f'AI prediction data error for {ticker}: {e}')
                    except Exception as e:
                        logger.warning(f'Unexpected AI error for {ticker}: {e}')

                    # Detect 200 MA pattern
                    indicator_200ma = TrendIndicator(df, self.config, self.params)
                    signal = indicator_200ma.detect_breakout_and_retest()

                    if signal:
                        logger.info(
                            f'Signal found: {signal["type"]} {ticker} @ ${signal["entry"]:.2f}, '
                            f'Breakout Vol: {signal["breakout_volume_ratio"]:.2f}x, '
                            f'Retest Vol: {signal["retest_volume_ratio"]:.2f}x'
                        )

                        self.execute_signal(signal)  # Execute immediately for each signal
                        logger.info('Waiting 1 minute after executing signal...')
                        self.ib.sleep(60)  # Small delay to avoid rate limiting
                    else:
                        logger.info(f'No signal found for {ticker}')

    def execute_signal(self, signal: dict):
        """Execute trading signals"""
        # Get account info to determine position sizing
        account_summary = self.ib.accountSummary()
        net_liq = 0.0
        cash_balance = 0.0

        for item in account_summary:
            if item.tag == 'NetLiquidation':
                net_liq = float(item.value)
            elif item.tag == 'TotalCashValue':
                cash_balance = float(item.value)

        # Calculate current invested amount
        current_positions = [p for p in self.ib.portfolio() if p.position != 0]
        invested_amount = sum(abs(item.position * item.marketPrice) for item in self.ib.portfolio() if item.position != 0)

        max_investment_allowed = net_liq * self.params['risk_management']['max_investment_pct']
        available_to_invest = max_investment_allowed - invested_amount
        invested_pct = (invested_amount / net_liq * 100) if net_liq > 0 else 0

        logger.info('Account Summary:')
        logger.info(f'  Net Liquidation: ${net_liq:,.2f}')
        logger.info(f'  Cash Balance: ${cash_balance:,.2f}')
        logger.info(f'  Currently Invested: ${invested_amount:,.2f} ({invested_pct:.1f}%)')
        logger.info(f'  Max Investment Allowed ({self.params["risk_management"]["max_investment_pct"] * 100:.0f}%): ${max_investment_allowed:,.2f}')
        logger.info(f'  Available to Invest: ${available_to_invest:,.2f}')

        # Check if we're already at max investment
        if available_to_invest <= 0:
            logger.warning(f'Cannot take new trades - already at max investment (${invested_amount:,.2f} / ${max_investment_allowed:,.2f})')
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
            logger.warning(f'Position size too small for {signal["symbol"]}')

    def _recalculate_bracket_prices(self, signal: dict, fill_price: float) -> tuple[float, float]:
        """Recalculate stop/target relative to the actual fill price.

        The signal's stop and target were computed from yesterday's close.
        This preserves the same percentage distances so the risk/reward
        ratio stays intact regardless of overnight gaps or intraday drift.
        """
        original_entry = signal['entry']
        if original_entry <= 0:
            return signal['stop'], signal['target']

        if signal['type'] == 'LONG':
            stop_pct = (original_entry - signal['stop']) / original_entry
            target_pct = (signal['target'] - original_entry) / original_entry
            new_stop = fill_price * (1 - stop_pct)
            new_target = fill_price * (1 + target_pct)
        else:
            stop_pct = (signal['stop'] - original_entry) / original_entry
            target_pct = (original_entry - signal['target']) / original_entry
            new_stop = fill_price * (1 + stop_pct)
            new_target = fill_price * (1 - target_pct)

        if fill_price < 1:
            return round(new_stop, 4), round(new_target, 4)
        return round(new_stop, 2), round(new_target, 2)

    def place_order(self, signal: dict, shares: int):
        """Check the live price against the signal entry, then place a market
        order and attach bracket children using the actual fill price."""
        try:
            symbol = signal['symbol']
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Check live price before risking any capital.
            # Max allowed deviation scales with the signal's stop distance
            # (already VIX/volatility-adjusted), so volatile stocks get more
            # room and stable stocks get a tighter leash.
            stop_distance_pct = abs(signal['entry'] - signal['stop']) / signal['entry'] if signal['entry'] > 0 else 0
            max_deviation = max(stop_distance_pct, 0.005)

            [ticker] = self.ib.reqTickers(contract)
            self.ib.sleep(0.5)
            live_price = ticker.marketPrice()

            if live_price is None or live_price != live_price or live_price <= 0:
                logger.warning(f'{symbol} could not get a live quote, skipping trade.')
                return

            deviation = abs(live_price - signal['entry']) / signal['entry'] if signal['entry'] > 0 else 0
            if deviation > max_deviation:
                logger.warning(
                    f'{symbol} live price ${live_price:.2f} deviated {deviation:.1%} from signal entry '
                    f'${signal["entry"]:.2f} (max {max_deviation:.1%} based on stop distance). Skipping trade.'
                )
                return

            action = 'BUY' if signal['type'] == 'LONG' else 'SELL'

            parent = MarketOrder(action, shares, tif='DAY', outsideRth=False, transmit=True)
            parent_trade = self.ib.placeOrder(contract, parent)

            statuses = ['Filled', 'Cancelled', 'ApiCancelled', 'Inactive']
            deadline = time.time() + 600
            while parent_trade.orderStatus.status not in statuses:
                self.ib.waitOnUpdate(timeout=1)
                if time.time() > deadline:
                    break

            filled_qty = parent_trade.orderStatus.filled
            status = parent_trade.orderStatus.status
            fill_price = parent_trade.orderStatus.avgFillPrice
            if status != 'Filled' or fill_price <= 0:
                logger.warning(f'{symbol} parent order did NOT fill (status={status}). Cancelling and skipping persistence.')
                try:
                    self.ib.cancelOrder(parent_trade.order)
                except Exception as cancel_err:
                    logger.error(f'Failed to cancel order: {cancel_err}')

                if filled_qty > 0:
                    logger.warning(f'{symbol} partial fill of {filled_qty} shares at ${fill_price:.2f}. Closing out.')
                    try:
                        close_action = 'SELL' if action == 'BUY' else 'BUY'
                        close_order = MarketOrder(close_action, filled_qty, tif='DAY', transmit=True)
                        self.ib.placeOrder(contract, close_order)
                    except Exception as cancel_err:
                        logger.error(f'Failed to close partial fill: {cancel_err}')

                return

            # Recalculate stop/target from the actual fill price
            new_stop, new_target = self._recalculate_bracket_prices(signal, fill_price)

            logger.info(
                f'{symbol} filled @ ${fill_price:.2f} (signal entry ${signal["entry"]:.2f}). Stop: ${new_stop:.2f}, Target: ${new_target:.2f}'
            )

            signal['entry'] = float(fill_price)
            signal['stop'] = new_stop
            signal['target'] = new_target
            signal['risk'] = abs(fill_price - new_stop)
            signal['reward'] = abs(new_target - fill_price)

            child_action = 'SELL' if action == 'BUY' else 'BUY'
            take_profit = LimitOrder(
                child_action,
                shares,
                new_target,
                tif='GTC',
                outsideRth=True,
                parentId=parent_trade.order.orderId,
                transmit=False,
            )
            stop_loss = StopOrder(
                child_action,
                shares,
                new_stop,
                tif='GTC',
                outsideRth=True,
                parentId=parent_trade.order.orderId,
                transmit=True,
            )

            tp_trade = self.ib.placeOrder(contract, take_profit)
            sl_trade = self.ib.placeOrder(contract, stop_loss)

            deadline = time.time() + 600
            while tp_trade.orderStatus.status not in statuses or sl_trade.orderStatus.status not in statuses:
                self.ib.waitOnUpdate(timeout=1)
                if time.time() > deadline:
                    break

            filled_qty = parent_trade.orderStatus.filled
            status = parent_trade.orderStatus.status
            fill_price = parent_trade.orderStatus.avgFillPrice
            if status != 'Filled' or fill_price <= 0:
                logger.warning(f'{symbol} parent order did NOT fill (status={status}). Cancelling and skipping persistence.')
                for trade in [parent_trade, tp_trade, sl_trade]:
                    try:
                        self.ib.cancelOrder(trade.order)
                    except Exception as cancel_err:
                        logger.error(f'Failed to cancel order: {cancel_err}')

                if filled_qty > 0:
                    logger.warning(f'{symbol} partial fill of {filled_qty} shares at ${fill_price:.2f}. Closing out.')
                    try:
                        close_action = 'SELL' if action == 'BUY' else 'BUY'
                        close_order = MarketOrder(close_action, filled_qty, tif='DAY', transmit=True)
                        self.ib.placeOrder(contract, close_order)
                    except Exception as cancel_err:
                        logger.error(f'Failed to close partial fill: {cancel_err}')
                return

            entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.position_manager.active_positions[symbol] = {
                'signal': signal,
                'shares': shares,
                'entry_time': entry_time,
            }
            self.position_manager.add_position(symbol, signal, shares, entry_time)

            logger.info(f'FILLED {signal["type"]} {symbol}: {shares} sh @ ${fill_price:.2f}, Stop: ${new_stop:.2f}, Target: ${new_target:.2f}')
            self.alert_manager.alert_trade_entry(signal)

        except Exception as e:
            logger.error(f'Failed to place order for {signal["symbol"]}: {e}')
