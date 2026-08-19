"""
Risk management: position sizing, cash reserves, max positions
"""

import logging

import pandas as pd
import yfinance as yf

# Setup logging. Named rather than root so a caller that drives the risk
# checks in a tight loop (the backtest) can quiet the per-rejection warnings
# without silencing everything else. Records still propagate to root handlers.
logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, params):
        self.params = params

    def can_take_trade(self, account_value, invested_amount, num_positions):
        """Check if we can take a new trade"""
        # Check cash reserve
        if invested_amount >= account_value * self.params['risk_management']['max_investment_pct']:
            logger.warning(
                f'Cannot take new trade - cash reserve requirement not met (${invested_amount:,.2f} invested / ${account_value:,.2f} account value)'
            )
            return False

        # Check max positions
        if num_positions >= self.params['risk_management']['max_positions']:
            logger.warning(
                f'Cannot take new trade - max positions limit reached '
                f'({num_positions} positions / {self.params["risk_management"]["max_positions"]} max)'
            )
            return False

        return True

    def calculate_position_size(self, account_value, entry_price, stop_price):
        """Calculate shares based on risk, capped by max position cost"""
        risk_per_trade = account_value * self.params['risk_management']['risk_per_trade_pct']

        if entry_price <= 0 or stop_price <= 0:
            return 0

        per_share_risk = abs(entry_price - stop_price)
        if per_share_risk <= 0:
            return 0

        shares_by_risk = int(risk_per_trade / per_share_risk)

        max_position_pct = self.params['risk_management'].get('max_position_pct', 0.20)
        max_cost = account_value * max_position_pct
        shares_by_cost = int(max_cost / entry_price)

        shares = min(shares_by_risk, shares_by_cost)
        return max(shares, 0)

    def validate_trade_size(self, shares, entry_price, available_cash):
        """Check if trade fits within available cash"""
        trade_cost = shares * entry_price
        return trade_cost <= available_cash

    def get_stop_loss_pct(self, df: pd.DataFrame, vix: float | None = None) -> float:
        """Volatility-scaled stop distance as a fraction of price.

        `vix` lets a caller supply the VIX level to size against instead of the
        live quote. Live trading leaves it None and fetches the current value;
        the backtest passes the VIX close of the simulated day so historical
        signals are not sized against today's volatility.
        """
        vix_today = 20.0
        if vix is not None and vix == vix and vix > 0:
            vix_today = float(vix)
        else:
            try:
                vix_ticker = yf.Ticker('^VIX')
                vix_today = vix_ticker.fast_info['lastPrice']
            except Exception as e:
                logger.warning(f'Failed to fetch VIX data, defaulting to 20.0: {e}')

        # Calculate the daily range percentage
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        daily_range_pct = (high - low) / close

        sl = 0
        if vix_today > 25:
            sl = daily_range_pct.iloc[-1] * (self.params['strategy_retest_200ma']['ATR'] + 0.5)
        elif vix_today < 15:
            sl = daily_range_pct.iloc[-1] * (self.params['strategy_retest_200ma']['ATR'] - 0.5)
        else:
            sl = daily_range_pct.iloc[-1] * self.params['strategy_retest_200ma']['ATR']

        return sl
