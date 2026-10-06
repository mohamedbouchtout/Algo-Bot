"""
Sharia compliance checks for stocks. This module is intended to be used as a filter for
stock selection that adheres to Islamic finance principles. It checks for compliance with
financial ratios and business activities that are considered permissible under Sharia law.
"""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)

PROHIBITED_INDUSTRIES = [
    'Beverages—Wineries & Distilleries',
    'gambling',
    'tobacco',
    'banks - diversified',
    'Banks—Diversified',
    'Mortgage Finance',
    'Credit Services',
    'Banks - Regional',
    'Insurance—Life',
    'Insurance—Property & Casualty',
    'Insurance—Reinsurance',
]


def is_sharia_compliant(stock_symbol: str) -> bool:
    """
    Check if a stock is Sharia-compliant based on its financial ratios and business activities.

    Parameters
    ----------
    stock_symbol : str
        The stock symbol to check for Sharia compliance.

    Returns
    -------
    bool
        True if the stock is Sharia-compliant, False otherwise.
    """
    try:
        # Fetch stock data using yfinance
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info

        # Example checks (these should be replaced with actual Sharia compliance criteria)
        # Check if the company is involved in prohibited industries (e.g., alcohol, gambling, etc.)
        industry = info.get('industry', '').lower()
        print(f'Checking {stock_symbol}: Industry - {industry}')
        if any(prohibited in industry for prohibited in PROHIBITED_INDUSTRIES):
            print(f'{stock_symbol} is not Sharia-compliant due to its industry: {industry}')
            return False

        # Check financial ratios (example thresholds, should be adjusted based on actual criteria)
        debt_to_equity = info.get('debtToEquity', 0)
        if debt_to_equity > 33:  # Example threshold for debt-to-equity ratio
            print(f'{stock_symbol} is not Sharia-compliant due to high debt-to-equity ratio: {debt_to_equity}')
            return False

        # Additional checks can be added here...

        print(f'{stock_symbol} is Sharia-compliant.')
        return True

    except Exception as e:
        print(f'Error checking Sharia compliance for {stock_symbol}: {e}')
        return False


if __name__ == '__main__':
    # Example usage
    stock_symbols = ['AAPL', 'TSLA', 'BAC', 'MO']  # Add more stock symbols as needed
    for symbol in stock_symbols:
        compliant = is_sharia_compliant(symbol)
        print(f'{symbol}: {"Compliant" if compliant else "Not Compliant"}')
        print('')
