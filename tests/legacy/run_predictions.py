"""
Train AI models per sector and print predictions for every ticker.

Usage:
    python -m tests.legacy.run_predictions
"""

import asyncio
import json
import logging
import os

asyncio.set_event_loop(asyncio.new_event_loop())

from data_fetch.historical_data import StockDataFetcher
from data_fetch.stock_fetcher import StockTickerFetcher
from strategy.ai_analysis.ai_analyzer import AIAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json(filename):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), filename)
    with open(path) as f:
        return json.load(f)


def main():
    params = load_json('config/trading_params.json')
    stock_fetcher = StockTickerFetcher()
    stock_data = StockDataFetcher()

    # Train one model per sector
    logger.info('Training AI models per sector...')
    sector_analyzers: dict[str, AIAnalyzer] = {}

    for sector, industries in stock_fetcher.categorized_stocks.items():
        analyzer = AIAnalyzer(stock_data, params=params)
        added = 0
        for industry, tickers in industries.items():
            for ticker in tickers:
                if analyzer.add_ticker(ticker):
                    added += 1

        if added < 2:
            logger.warning(f'Skipping {sector}: only {added} ticker(s)')
            continue

        try:
            analyzer.finalize_training(val_split=0.2)
            sector_analyzers[sector] = analyzer
            logger.info(f'Trained {sector} model on {added} tickers')
        except Exception as e:
            logger.error(f'Training failed for {sector}: {e}')

    logger.info(f'Training complete: {len(sector_analyzers)} sector models')

    # Predict for every ticker
    print()
    print('=' * 80)
    print(f'{"TICKER":<8} {"SECTOR":<25} {"CLASS":<7} {"SHORT":>8} {"FLAT":>8} {"LONG":>8}')
    print('-' * 80)

    for sector, industries in stock_fetcher.categorized_stocks.items():
        if sector not in sector_analyzers:
            continue
        analyzer = sector_analyzers[sector]

        for industry, tickers in industries.items():
            for ticker in tickers:
                try:
                    prediction = analyzer.predict(ticker)
                    if prediction is None:
                        continue
                    probs = prediction['probs']
                    print(
                        f'{ticker:<8} {sector:<25} {prediction["class"]:<7} '
                        f'{probs.get("SHORT", 0):>8.4f} {probs.get("FLAT", 0):>8.4f} {probs.get("LONG", 0):>8.4f}'
                    )
                except Exception as e:
                    logger.warning(f'{ticker}: prediction failed: {e}')

    print('=' * 80)


if __name__ == '__main__':
    main()
