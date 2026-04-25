"""
Regression test for 200ma strategy logic
"""

import logging
from strategy.retest_200ma.indicators import TrendIndicator

# Setup logging
logger = logging.getLogger()

class TestRetest200MA:
    def __init__(self, ib, config, params, stock_data_fetcher, stock_fetcher, connection_manager):
        self.ib = ib
        self.config = config
        self.params = params
        self.stock_data_fetcher = stock_data_fetcher
        self.stock_fetcher = stock_fetcher
        self.connection_manager = connection_manager

    def test_retest_200ma(self):
        """Test the 200 MA breakout and retest logic on historical data"""

        if not self.connection_manager.connect():
            logging.error("Failed to connect. Exiting.")
            return

        signals = []
        try:
            for ticker in self.stock_fetcher.stock_list:
                logging.info(f"Testing {ticker}...")
                df = self.stock_data_fetcher.get_historical_data(ticker)
                
                indicator_200ma = TrendIndicator(df, self.config, self.params)
                signal = indicator_200ma.detect_breakout_and_retest()
                
                if signal:
                    logging.info(f"Signal detected for {ticker}: {signal}")
                    signals.append((ticker, signal))
                else:
                    logging.info(f"No signal for {ticker}.")

        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        except Exception as e:
            logging.error(f"Bot error: {e}")
        finally:
            logging.info(f"Total signals detected: {len(signals)}")
            self.connection_manager.disconnect()