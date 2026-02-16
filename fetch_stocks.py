"""
Improved stock list generator with multiple methods
"""

import pandas as pd
from io import StringIO

def get_hardcoded_stock_list():
    """
    Comprehensive list of major S&P 500 and NASDAQ stocks
    Updated for 2024/2025
    """
    stocks = [
        # Mega-cap Tech (FAANG+)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC',
        'NFLX', 'CRM', 'ORCL', 'ADBE', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'IBM', 'INTU',
        'NOW', 'PANW', 'SNPS', 'CDNS', 'AMAT', 'LRCX', 'KLAC', 'ASML', 'TSM', 'MU',
        'NXPI', 'ADI', 'MRVL', 'SHOP', 'SQ', 'PYPL', 'COIN', 'RBLX', 'U', 'ABNB',
        
        # Financial Services
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB',
        'PNC', 'TFC', 'COF', 'BK', 'STATE', 'STT', 'NTRS', 'CFG', 'FITB', 'HBAN',
        'RF', 'KEY', 'MTB', 'AIG', 'PRU', 'MET', 'ALL', 'TRV', 'PGR', 'CB',
        'AON', 'MMC', 'AJG', 'SPGI', 'MCO', 'ICE', 'CME', 'NDAQ', 'MKTX',
        
        # Payment Processors
        'V', 'MA', 'FIS', 'FISV', 'ADP', 'PAYX',
        
        # Healthcare & Pharma
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'AMGN', 'GILD',
        'CVS', 'BMY', 'MDT', 'DHR', 'ISRG', 'SYK', 'BSX', 'EW', 'ZTS', 'REGN',
        'VRTX', 'HUM', 'CI', 'ELV', 'CVS', 'MCK', 'COR', 'CAH', 'BIIB', 'MRNA',
        
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'TGT', 'DG',
        'ROST', 'ORLY', 'AZO', 'BBY', 'ULTA', 'CMG', 'YUM', 'DRI', 'MAR', 'HLT',
        'BKNG', 'ABNB', 'LVS', 'WYNN', 'MGM', 'F', 'GM', 'TSLA', 'RIVN', 'LCID',
        
        # Consumer Staples
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'MDLZ', 'PM', 'MO', 'CL', 'KMB',
        'GIS', 'K', 'HSY', 'SYY', 'KHC', 'CPB', 'CAG', 'MNST', 'KDP', 'STZ',
        'TAP', 'BF-B', 'EL', 'CLX', 'CHD',
        
        # Industrial
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'MMM', 'EMR',
        'ETN', 'ITW', 'PH', 'CMI', 'ROK', 'DOV', 'FDX', 'NSC', 'UNP', 'CSX',
        'ODFL', 'JBHT', 'CHRW', 'XPO', 'DAL', 'UAL', 'AAL', 'LUV', 'SAVE',
        
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
        'BKR', 'MRO', 'DVN', 'FANG', 'APA', 'HES', 'KMI', 'WMB', 'OKE', 'LNG',
        
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'NUE', 'STLD',
        'MLM', 'VMC', 'ALB', 'CE', 'FMC', 'PPG', 'IP', 'PKG', 'AMCR', 'AVY',
        
        # Real Estate & REITs
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'WELL', 'O', 'SPG', 'AVB',
        'EQR', 'VICI', 'VTR', 'ARE', 'CBRE', 'SBAC', 'ESS', 'MAA', 'INVH',
        
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
        'ES', 'PEG', 'PCG', 'AWK', 'AEE', 'CMS', 'DTE', 'PPL',
        
        # Communication Services
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'PARA',
        'WBD', 'EA', 'TTWO', 'ATVI', 'LYV', 'FOXA', 'FOX', 'NWSA', 'NWS', 'OMC',
        'IPG', 'MTCH', 'PINS', 'SNAP', 'SPOT', 'ZM', 'DOCU', 'DDOG', 'NET', 'TWLO',
        
        # Semiconductors & Hardware
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC',
        'MCHP', 'MRVL', 'SWKS', 'NXPI', 'ADI', 'ON', 'MPWR', 'ENTG', 'QRVO', 'WOLF',
        
        # Software & Cloud
        'MSFT', 'CRM', 'ORCL', 'ADBE', 'INTU', 'NOW', 'WDAY', 'TEAM', 'SNOW', 'DDOG',
        'ZS', 'CRWD', 'OKTA', 'PANW', 'FTNT', 'SPLK', 'VEEV', 'ANSS', 'CDNS', 'SNPS',
        'ADSK', 'ROP', 'TYL', 'GWRE', 'MANH',
        
        # E-commerce & Retail
        'AMZN', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR',
        'EBAY', 'ETSY', 'W', 'CHWY', 'CASY', 'BJ',
        
        # Biotech
        'GILD', 'AMGN', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'BNTX', 'ALNY', 'SGEN', 'BMRN',
        'NBIX', 'INCY', 'EXAS', 'TECH', 'ILMN', 'IONS', 'RARE', 'UTHR', 'SRPT',
        
        # Transportation & Logistics
        'UPS', 'FDX', 'XPO', 'ODFL', 'JBHT', 'CHRW', 'KNX', 'EXPD', 'LSTR',
        
        # Automotive
        'TSLA', 'F', 'GM', 'RIVN', 'LCID', 'TM', 'HMC', 'STLA',
        
        # Media & Entertainment
        'DIS', 'NFLX', 'PARA', 'WBD', 'LYV', 'SIRI', 'IMAX', 'ROKU', 'FUBO',
        
        # Food & Beverage
        'KO', 'PEP', 'MDLZ', 'MNST', 'KHC', 'GIS', 'K', 'HSY', 'CPB', 'CAG',
        
        # Restaurants
        'MCD', 'SBUX', 'YUM', 'DRI', 'CMG', 'DNKN', 'WEN', 'DPZ', 'QSR', 'JACK',
        
        # Apparel & Fashion
        'NKE', 'LULU', 'UAA', 'UA', 'VFC', 'HBI', 'PVH', 'RL', 'CPRI', 'TPR',
        
        # Travel & Leisure
        'BKNG', 'ABNB', 'EXPE', 'TRIP', 'MAR', 'HLT', 'H', 'RCL', 'CCL', 'NCLH',
        
        # Gaming
        'EA', 'TTWO', 'ATVI', 'RBLX', 'U', 'DKNG', 'PENN', 'LNW', 'CZR', 'MGM',
        
        # Cybersecurity
        'CRWD', 'PANW', 'ZS', 'FTNT', 'OKTA', 'CYBR', 'SAIL', 'S', 'TENB', 'QLYS',
        
        # Solar & Renewables
        'ENPH', 'SEDG', 'FSLR', 'RUN', 'NOVA', 'CSIQ', 'JKS', 'DQ', 'SPWR',
        
        # Electric Vehicles & Battery
        'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'QS', 'BLNK', 'CHPT',
        
        # Data & Analytics
        'PLTR', 'SNOW', 'DDOG', 'SPLK', 'ESTC', 'MDB', 'ZI', 'CFLT', 'NCNO',
        
        # Fintech
        'SQ', 'PYPL', 'COIN', 'AFRM', 'SOFI', 'UPST', 'LC', 'NU',
        
        # Aerospace & Defense
        'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TXT', 'HWM', 'HII', 'LDOS',
    ]
    
    # Remove duplicates and sort
    stocks = sorted(list(set(stocks)))
    return stocks

def save_stock_list(filename='stocks.txt'):
    """Save comprehensive stock list to file"""
    
    print("Generating stock list...")
    
    # Use comprehensive hardcoded list
    stocks = get_hardcoded_stock_list()
    
    print(f"Using {len(stocks)} stocks from curated list")
    
    # Save to file
    with open(filename, 'w') as f:
        for ticker in stocks:
            f.write(f"{ticker}\n")
    
    print(f"✅ Saved {len(stocks)} tickers to {filename}")
    
    # Print some stats
    print(f"\nStock list includes:")
    print(f"  - Major Tech companies")
    print(f"  - S&P 500 blue chips")
    print(f"  - NASDAQ-100 growth stocks")
    print(f"  - Various sectors: Finance, Healthcare, Energy, Consumer, etc.")
    
    return stocks

if __name__ == "__main__":
    print("="*80)
    print("Stock List Generator")
    print("="*80)
    print()
    
    stocks = save_stock_list()
    
    print(f"\nFirst 20 stocks: {stocks[:20]}")
    print(f"\n✅ Done! You can now run the trading bot.")
    print()
    print("The bot will monitor all these stocks for breakout/retest patterns.")