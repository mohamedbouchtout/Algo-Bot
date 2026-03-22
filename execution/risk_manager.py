"""
Risk management: position sizing, cash reserves, max positions
"""

import logging

# Setup logging
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config, params):
        self.config = config
        self.params = params
    
    def can_take_trade(self, account_value, invested_amount, num_positions):
        """Check if we can take a new trade"""
        # Check cash reserve
        if invested_amount >= account_value * self.params['risk_management']['max_investment_pct']:
            logging.warning(
                f"Cannot take new trade - cash reserve requirement not met "
                f"(${invested_amount:,.2f} invested / ${account_value:,.2f} account value)"
            )
            return False
        
        # Check max positions
        if num_positions >= self.params['risk_management']['max_positions']:
            logging.warning(
                f"Cannot take new trade - max positions limit reached "
                f"({num_positions} positions / {self.params['risk_management']['max_positions']} max)"
            )
            return False
        
        return True
    
    def calculate_position_size(self, account_value, entry_price, stop_price):
        """Calculate shares based on risk"""
        risk_per_trade = account_value * self.params['risk_management']['risk_per_trade_pct']
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0:
            return 0
        
        shares = int(risk_per_trade / risk_per_share)
        return max(shares, 1)
    
    def validate_trade_size(self, shares, entry_price, available_cash):
        """Check if trade fits within available cash"""
        trade_cost = shares * entry_price
        return trade_cost <= available_cash