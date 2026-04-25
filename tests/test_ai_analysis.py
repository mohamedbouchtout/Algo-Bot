"""
Tests the RBM + CNN module analysis feature
"""

import logging
import pprint
from strategy.ai_analysis.ai_analyzer import AIAnalyzer

# Setup logging
logger = logging.getLogger()

class TestAIanalysis:
    def __init__(self, ib, config, params, stock_data_fetcher, stock_fetcher, connection_manager):
        self.ib = ib
        self.config = config
        self.params = params
        self.stock_data_fetcher = stock_data_fetcher
        self.stock_fetcher = stock_fetcher
        self.connection_manager = connection_manager
        self.ai_analyzer = AIAnalyzer(self.stock_data)

    def train_modules(self):
        """Run a full pooled retrain of the RBM + CNN on the current ticker universe."""

        if not self.connection_manager.connect():
            logging.error("Failed to connect. Exiting.")
            return

        try:
            self.logger.info("Starting AI retrain...")
            self.ai_analyzer.reset_dataset()
            added = 0
            for ticker in self.stock_fetcher.stock_list:
                if self.ai_analyzer.add_ticker(ticker):
                    added += 1
                self.ib.sleep(1)

            if added < 2:
                self.logger.warning(f"Only {added} tickers accumulated, skipping training")
                return

            self.ai_analyzer.finalize_training(val_split=0.2)
            self.logger.info(f"AI retrain finished: {added} tickers")
        except Exception as e:
            self.logger.error(f"AI training failed: {e}")

    def predictions(self):
        """Prints the predictions for each stock from the training"""

        if not self.connection_manager.connect():
            logging.error("Failed to connect. Exiting.")
            return

        try:
            self.logger.info("Starting AI predictions...")
            for ticker in self.stock_fetcher.stock_list:
                result = self.ai_analyzer.predict(ticker)

                if result is not None:
                    self.logger.info(pprint.pformat(result, indent=4))

            self.logger.info(f"AI predictions finished")
        except Exception as e:
            self.logger.error(f"AI prediction failed: {e}")