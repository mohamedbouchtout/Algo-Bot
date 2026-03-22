"""
Gets stock tickers from all NASDAQ and S&P 500 stocks
"""

import logging
from ib_insync import *
import os
import requests
import json

# Setup logging
logger = logging.getLogger(__name__)

class StockTickerFetcher:
    def __init__(self):
        self.file_path = 'data/stock_list.txt'
        self.stock_list = self.save_stock_list()

    def get_sp500_tickers(self):
        """Fetch S&P 500 from GitHub dataset"""
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            logging.info("Fetching S&P 500 from GitHub...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse CSV
            lines = response.text.strip().split('\n')
            tickers = [line.split(',')[0] for line in lines[1:]]  # Skip header
            
            logging.info(f"S&P 500: {len(tickers)} stocks")
            return tickers
            
        except Exception as e:
            logging.error(f"Failed to fetch S&P 500: {e}")
            return []

    def get_nasdaq100_tickers(self):
        """Fetch NASDAQ-100 from GitHub dataset"""
        try:
            url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json"
            logging.info("Fetching NASDAQ from GitHub...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = json.loads(response.text)
            
            # Extract symbols from dictionaries and filter
            filtered_tickers = []
            
            for item in data:
                symbol = item.get('symbol', '')
                
                # Filter criteria to get major stocks (approximate NASDAQ-100)
                # 1. Symbol length <= 5 (exclude warrants, rights, units)
                # 2. No dots, hyphens (exclude special share classes)
                # 3. Has sector/industry (exclude SPACs and new listings)
                # 4. Not a warrant/right (no 'W', 'R', 'U' suffix)
                
                if (len(symbol) <= 5 and 
                    '.' not in symbol and 
                    '-' not in symbol and
                    not symbol.endswith('W') and  # Warrants
                    not symbol.endswith('R') and  # Rights
                    not symbol.endswith('U') and  # Units
                    item.get('sector') and        # Has sector
                    item.get('industry')):        # Has industry
                    
                    filtered_tickers.append(symbol)
            
            logging.info(f"Filtered NASDAQ: {len(filtered_tickers)} stocks")
            
            return filtered_tickers[:200]
            
        except Exception as e:
            logging.error(f"Failed to fetch NASDAQ: {e}")
            return []

    def get_stock_list(self, sp500=True, nasdaq=True):
        """Fetch S&P 500 and NASDAQ-100 tickers"""
        
        SP500 = self.get_sp500_tickers() if sp500 else []
        NASDAQ = self.get_nasdaq100_tickers() if nasdaq else []
        
        # Combine and deduplicate
        combined = sorted(list(set(SP500 + NASDAQ)))
        logging.info(f"Total unique tickers: {len(combined)}")

        return combined

    def save_stock_list(self, sp500=True, nasdaq=True):
        """Save comprehensive stock list to file"""
        
        logging.info("Generating stock list...")
        
        # Use comprehensive hardcoded list
        stocks = self.get_stock_list(sp500, nasdaq)
        
        logging.info(f"Using {len(stocks)} stocks from curated list")
        
        # Save to file
        file_path = os.path.join(os.path.dirname(__file__), self.file_path)
        with open(file_path, 'w') as f:
            for ticker in stocks:
                f.write(f"{ticker}\n")
        
        logging.info(f"Saved {len(stocks)} tickers to {self.file_path}")
        
        return stocks