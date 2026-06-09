"""
Backtest the AI analysis pipeline on historical data.

Simulates live trading by:
1. Training a separate AI model per sector (matching the bot's approach).
2. Walking forward through a held-out test period day by day.
3. Monitoring for regime shifts (VIX/SPY) and accuracy degradation to
   trigger mid-simulation retraining — same logic as the live bot.
4. Making predictions using the correct sector-specific model.
5. Simulating bracket-order trades (entry at close, stop/target from signal).
6. Reporting per-trade and overall performance statistics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data_fetch.historical_data import StockDataFetcher
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
from strategy.ai_analysis.retrain_trigger import RetrainTrigger

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    sector: str
    direction: str
    entry_date: str
    entry_price: float
    stop_price: float
    target_price: float
    confidence: float
    exit_date: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


class TestAIBacktest:
    CLASS_NAMES = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}

    def __init__(
        self,
        ib,
        config,
        params,
        stock_data_fetcher: StockDataFetcher,
        categorized_stocks: dict[str, dict[str, list]],
        train_ratio: float = 0.75,
        max_holding_days: int = 20,
    ):
        self.ib = ib
        self.config = config
        self.params = params
        self.stock_data_fetcher = stock_data_fetcher
        self.categorized_stocks = categorized_stocks
        self.train_ratio = train_ratio
        self.max_holding_days = max_holding_days

        self.trades: list[Trade] = []
        self.skipped_signals = 0
        self.retrain_count = 0
        self.retrain_trigger = RetrainTrigger()

        # Track recent predictions vs actuals for accuracy-based retraining
        self._recent_predictions: list[int] = []
        self._recent_actuals: list[int] = []

    def _fetch_market_history(self) -> pd.DataFrame:
        """Fetch VIX and SPY history for regime shift detection during backtest."""
        try:
            vix = yf.Ticker('^VIX').history(period='10y', auto_adjust=True)
            spy = yf.Ticker('SPY').history(period='10y', auto_adjust=True)
            if vix.empty or spy.empty:
                return pd.DataFrame()
            market = pd.DataFrame(
                {
                    'vix_close': vix['Close'],
                    'spy_close': spy['Close'],
                }
            )
            market.index = market.index.tz_localize(None)
            return market.ffill()
        except Exception as e:
            logger.warning(f'Failed to fetch market history for backtest: {e}')
            return pd.DataFrame()

    def _get_market_at_date(self, market_data: pd.DataFrame, date: pd.Timestamp) -> tuple[float, float]:
        """Look up VIX and SPY close for a given date."""
        if market_data.empty:
            return 20.0, 400.0
        mask = market_data.index <= date
        if not mask.any():
            return 20.0, 400.0
        row = market_data.loc[market_data.index[mask][-1]]
        return float(row['vix_close']), float(row['spy_close'])

    def _get_all_tickers(self) -> list[tuple[str, str]]:
        """Return (symbol, sector) pairs from categorized_stocks."""
        result = []
        for sector, industries in self.categorized_stocks.items():
            for industry, tickers in industries.items():
                for ticker in tickers:
                    result.append((ticker, sector))
        return result

    def run(self) -> dict:
        logger.info('=' * 60)
        logger.info('AI BACKTEST - starting (per-sector models)')
        logger.info('=' * 60)

        # 1. Fetch data for all tickers, track their sector
        ticker_sectors = self._get_all_tickers()
        all_bars: dict[str, pd.DataFrame] = {}
        symbol_to_sector: dict[str, str] = {}

        for sym, sector in ticker_sectors:
            if sym in all_bars:
                continue
            df = self.stock_data_fetcher.get_historical_data(sym, self.params['ai_analyzer']['lookback_days'])
            if df is not None and len(df) >= 300:
                all_bars[sym] = df.reset_index(drop=True)
                symbol_to_sector[sym] = sector
                logger.info(f'Fetched {len(df)} bars for {sym} ({sector})')
            else:
                logger.warning(f'Skipping {sym}: insufficient data')

        if len(all_bars) < 2:
            logger.error('Need at least 2 tickers with sufficient data')
            return {}

        # 2. Split train/test per ticker
        train_bars: dict[str, pd.DataFrame] = {}
        test_bars: dict[str, pd.DataFrame] = {}
        for sym, df in all_bars.items():
            split_idx = int(len(df) * self.train_ratio)
            train_bars[sym] = df.iloc[:split_idx].copy().reset_index(drop=True)
            test_bars[sym] = df.iloc[split_idx:].copy().reset_index(drop=True)
            logger.info(f'{sym}: train={len(train_bars[sym])} bars, test={len(test_bars[sym])} bars')

        # 3. Group tickers by sector for training
        sector_tickers: dict[str, list[str]] = {}
        for sym, sector in symbol_to_sector.items():
            sector_tickers.setdefault(sector, []).append(sym)

        # 4. Train one analyzer per sector
        logger.info('Training per-sector AI models...')
        sector_analyzers: dict[str, AIAnalyzer] = {}
        for sector, tickers in sector_tickers.items():
            if len(tickers) < 2:
                logger.warning(f'Skipping sector {sector}: only {len(tickers)} ticker(s)')
                continue

            sector_train = {sym: train_bars[sym] for sym in tickers}
            analyzer = self._train_sector_analyzer(sector, sector_train)
            if analyzer is not None:
                sector_analyzers[sector] = analyzer
                logger.info(f'Trained {sector} model on {len(tickers)} tickers')

        if not sector_analyzers:
            logger.error('No sectors produced trained models')
            return {}

        logger.info(f'Training complete: {len(sector_analyzers)} sector models')

        # 4b. Fetch market data for regime shift detection and snapshot initial levels
        market_data = self._fetch_market_history()
        first_test_dates = []
        for sym in test_bars:
            if 'date' in test_bars[sym].columns and len(test_bars[sym]) > 0:
                first_test_dates.append(pd.Timestamp(test_bars[sym]['date'].iloc[0]))
        if first_test_dates:
            train_end_date = min(first_test_dates)
            vix_at_train, spy_at_train = self._get_market_at_date(market_data, train_end_date)
            self.retrain_trigger.snapshot_from_bars(vix_at_train, spy_at_train)

        # 5. Walk-forward simulation using sector-specific models
        logger.info('Starting walk-forward simulation...')
        confidence_threshold = self.params['ai_analyzer']['confidence_threshold']
        regime_check_interval = 5  # Check for regime shift every N days

        for sym in all_bars:
            sector = symbol_to_sector[sym]
            if sector not in sector_analyzers:
                continue

            analyzer = sector_analyzers[sector]
            train_df = train_bars[sym]
            test_df = test_bars[sym]
            full_df = pd.concat([train_df, test_df], ignore_index=True)
            train_len = len(train_df)
            forward_horizon = analyzer.feature_builder.forward_horizon

            for day_offset in range(len(test_df)):
                current_idx = train_len + day_offset
                if current_idx + forward_horizon >= len(full_df):
                    break

                history_slice = full_df.iloc[: current_idx + 1].copy()
                if len(history_slice) < 250:
                    continue

                # Check regime shift periodically
                if day_offset % regime_check_interval == 0 and 'date' in history_slice.columns:
                    current_date = pd.Timestamp(history_slice['date'].iloc[-1])
                    current_vix, current_spy = self._get_market_at_date(market_data, current_date)
                    regime_shifted = self.retrain_trigger.check_regime_shift_from_bars(current_vix, current_spy)
                    accuracy_degraded = self.retrain_trigger.check_accuracy(self._recent_predictions, self._recent_actuals)

                    if regime_shifted or accuracy_degraded:
                        reason = 'regime shift' if regime_shifted else 'accuracy degradation'
                        logger.info(f'Mid-simulation retrain triggered ({reason}) at day {day_offset} for {sym}')
                        # Retrain using all data up to current point
                        expanded_train = {
                            s: full_df.iloc[: train_len + day_offset].copy().reset_index(drop=True)
                            for s in sector_tickers.get(sector, [])
                            if s in all_bars
                        }
                        new_analyzer = self._train_sector_analyzer(sector, expanded_train)
                        if new_analyzer is not None:
                            sector_analyzers[sector] = new_analyzer
                            analyzer = new_analyzer
                            self.retrain_count += 1
                            self.retrain_trigger.snapshot_from_bars(current_vix, current_spy)
                            self._recent_predictions.clear()
                            self._recent_actuals.clear()

                prediction = self._predict_on_slice(analyzer, history_slice)
                if prediction is None:
                    continue

                class_type = prediction['class']
                confidence = prediction['probs'].get(class_type, 0.0)

                # Track prediction vs actual for accuracy monitoring
                if current_idx + forward_horizon < len(full_df):
                    actual_return = (full_df['close'].iloc[current_idx + forward_horizon] / full_df['close'].iloc[current_idx]) - 1.0
                    if actual_return > 0.01:
                        actual_class = 2
                    elif actual_return < -0.01:
                        actual_class = 0
                    else:
                        actual_class = 1
                    self._recent_predictions.append(prediction['class_id'])
                    self._recent_actuals.append(actual_class)

                if class_type == 'FLAT' or confidence < confidence_threshold:
                    continue

                signal = analyzer.construct_signal(history_slice, self.params, class_type, confidence)
                if signal is None:
                    continue

                future_bars = full_df.iloc[current_idx + 1 :]
                trade = self._simulate_trade(signal, history_slice, future_bars, sector)
                if trade is not None:
                    self.trades.append(trade)

        # 6. Results
        results = self._compute_stats()
        self._print_report(results)
        return results

    def _train_sector_analyzer(self, sector: str, train_bars: dict[str, pd.DataFrame]) -> AIAnalyzer | None:
        try:
            feature_builder = FeatureBuilder(window_size=10, n_bits=4)
            analyzer = AIAnalyzer(
                stock_data=self.stock_data_fetcher,
                feature_builder=feature_builder,
                params=self.params,
            )

            for sym, df in train_bars.items():
                analyzer._bar_cache[sym] = df
                try:
                    feats = feature_builder.build_continuous_features(df)
                    analyzer._continuous_per_ticker.append(feats)
                    analyzer._kept_tickers.append(sym)
                except ValueError as e:
                    logger.warning(f'{sym}: feature extraction failed: {e}')

            if len(analyzer._kept_tickers) < 2:
                return None

            analyzer.finalize_training(val_split=0.2)
            return analyzer
        except Exception as e:
            logger.error(f'Training failed for sector {sector}: {e}')
            return None

    def _predict_on_slice(self, analyzer: AIAnalyzer, df_slice: pd.DataFrame) -> dict | None:
        try:
            _, cnn_x, _ = analyzer.feature_builder.build_windows(df_slice, include_labels=False, include_rbm=False)
            if len(cnn_x) == 0:
                return None

            cnn_last = cnn_x[-1:]
            preds, probs = analyzer._trainer.predict(cnn_last)

            cls = int(preds[0])
            return {
                'class': self.CLASS_NAMES[cls],
                'class_id': cls,
                'probs': {self.CLASS_NAMES[i]: float(p) for i, p in enumerate(probs[0])},
            }
        except Exception as e:
            logger.debug(f'Prediction failed on slice: {e}')
            return None

    def _simulate_trade(
        self,
        signal: dict,
        history: pd.DataFrame,
        future_bars: pd.DataFrame,
        sector: str,
    ) -> Trade | None:
        entry_price = signal['entry']
        stop_price = signal['stop']
        target_price = signal['target']
        direction = signal['type']
        symbol = signal['symbol']
        confidence = signal.get('confidence', 0.0)

        entry_date = str(history['date'].iloc[-1]) if 'date' in history.columns else 'N/A'

        if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
            self.skipped_signals += 1
            return None

        bars_to_check = future_bars.head(self.max_holding_days)
        if len(bars_to_check) == 0:
            self.skipped_signals += 1
            return None

        for i, (_, bar) in enumerate(bars_to_check.iterrows()):
            high = float(bar['high'])
            low = float(bar['low'])
            bar_date = str(bar['date']) if 'date' in bar else f'day+{i + 1}'

            if direction == 'LONG':
                if low <= stop_price:
                    return self._make_trade(
                        symbol,
                        sector,
                        direction,
                        entry_date,
                        entry_price,
                        stop_price,
                        target_price,
                        confidence,
                        bar_date,
                        stop_price,
                        'STOP_LOSS',
                    )
                if high >= target_price:
                    return self._make_trade(
                        symbol,
                        sector,
                        direction,
                        entry_date,
                        entry_price,
                        stop_price,
                        target_price,
                        confidence,
                        bar_date,
                        target_price,
                        'TAKE_PROFIT',
                    )
            elif direction == 'SHORT':
                if high >= stop_price:
                    return self._make_trade(
                        symbol,
                        sector,
                        direction,
                        entry_date,
                        entry_price,
                        stop_price,
                        target_price,
                        confidence,
                        bar_date,
                        stop_price,
                        'STOP_LOSS',
                    )
                if low <= target_price:
                    return self._make_trade(
                        symbol,
                        sector,
                        direction,
                        entry_date,
                        entry_price,
                        stop_price,
                        target_price,
                        confidence,
                        bar_date,
                        target_price,
                        'TAKE_PROFIT',
                    )

        last_bar = bars_to_check.iloc[-1]
        exit_price = float(last_bar['close'])
        exit_date = str(last_bar['date']) if 'date' in last_bar else f'day+{len(bars_to_check)}'
        return self._make_trade(
            symbol,
            sector,
            direction,
            entry_date,
            entry_price,
            stop_price,
            target_price,
            confidence,
            exit_date,
            exit_price,
            'MAX_HOLD',
        )

    def _make_trade(
        self,
        symbol,
        sector,
        direction,
        entry_date,
        entry_price,
        stop_price,
        target_price,
        confidence,
        exit_date,
        exit_price,
        exit_reason,
    ) -> Trade:
        if direction == 'LONG':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        pnl_pct = (pnl / entry_price) * 100 if entry_price > 0 else 0.0

        return Trade(
            symbol=symbol,
            sector=sector,
            direction=direction,
            entry_date=entry_date,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            confidence=confidence,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 4),
        )

    def _compute_stats(self) -> dict:
        if not self.trades:
            return {
                'total_trades': 0,
                'message': 'No trades were generated during the test period',
            }

        pnl_pcts = [t.pnl_pct for t in self.trades]
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_return = 1.0
        for pct in pnl_pcts:
            total_return *= 1 + pct / 100
        total_return_pct = (total_return - 1) * 100

        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
        profit_factor = (
            abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
        )

        exit_reasons: dict[str, int] = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Per-sector breakdown
        sectors = sorted(set(t.sector for t in self.trades))
        per_sector: dict[str, dict] = {}
        for sector in sectors:
            sector_trades = [t for t in self.trades if t.sector == sector]
            sector_wins = [t for t in sector_trades if t.pnl > 0]
            per_sector[sector] = {
                'total_trades': len(sector_trades),
                'win_rate': round(len(sector_wins) / len(sector_trades) * 100, 2) if sector_trades else 0.0,
                'avg_return_pct': round(np.mean([t.pnl_pct for t in sector_trades]), 4),
            }

        equity = [1.0]
        for pct in pnl_pcts:
            equity.append(equity[-1] * (1 + pct / 100))
        peak = equity[0]
        max_drawdown = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

        per_trade = []
        for t in self.trades:
            per_trade.append(
                {
                    'symbol': t.symbol,
                    'sector': t.sector,
                    'direction': t.direction,
                    'entry_date': t.entry_date,
                    'entry_price': t.entry_price,
                    'exit_date': t.exit_date,
                    'exit_price': t.exit_price,
                    'exit_reason': t.exit_reason,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'confidence': t.confidence,
                }
            )

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(len(wins) / len(self.trades) * 100, 2),
            'avg_win_pct': round(avg_win, 4),
            'avg_loss_pct': round(avg_loss, 4),
            'best_trade_pct': round(max(pnl_pcts), 4),
            'worst_trade_pct': round(min(pnl_pcts), 4),
            'avg_return_pct': round(np.mean(pnl_pcts), 4),
            'total_return_pct': round(total_return_pct, 4),
            'profit_factor': round(profit_factor, 4),
            'max_drawdown_pct': round(max_drawdown, 4),
            'exit_reasons': exit_reasons,
            'per_sector': per_sector,
            'skipped_signals': self.skipped_signals,
            'retrain_count': self.retrain_count,
            'trades': per_trade,
        }

    def _print_report(self, results: dict) -> None:
        logger.info('')
        logger.info('=' * 60)
        logger.info('BACKTEST RESULTS')
        logger.info('=' * 60)

        if results.get('total_trades', 0) == 0:
            logger.info(results.get('message', 'No trades'))
            return

        logger.info(f'Total Trades:      {results["total_trades"]}')
        logger.info(f'Winning Trades:    {results["winning_trades"]}')
        logger.info(f'Losing Trades:     {results["losing_trades"]}')
        logger.info(f'Win Rate:          {results["win_rate"]:.2f}%')
        logger.info(f'Avg Win:           {results["avg_win_pct"]:.4f}%')
        logger.info(f'Avg Loss:          {results["avg_loss_pct"]:.4f}%')
        logger.info(f'Best Trade:        {results["best_trade_pct"]:.4f}%')
        logger.info(f'Worst Trade:       {results["worst_trade_pct"]:.4f}%')
        logger.info(f'Avg Return/Trade:  {results["avg_return_pct"]:.4f}%')
        logger.info(f'Total Return:      {results["total_return_pct"]:.4f}%')
        logger.info(f'Profit Factor:     {results["profit_factor"]:.4f}')
        logger.info(f'Max Drawdown:      {results["max_drawdown_pct"]:.4f}%')
        logger.info(f'Skipped Signals:   {results["skipped_signals"]}')
        logger.info(f'Mid-Sim Retrains:  {results["retrain_count"]}')
        logger.info(f'Exit Reasons:      {results["exit_reasons"]}')

        logger.info('')
        logger.info('-' * 60)
        logger.info('PER-SECTOR BREAKDOWN')
        logger.info('-' * 60)
        for sector, stats in results.get('per_sector', {}).items():
            logger.info(f'  {sector}: {stats["total_trades"]} trades, win rate {stats["win_rate"]:.2f}%, avg return {stats["avg_return_pct"]:+.4f}%')

        logger.info('')
        logger.info('-' * 60)
        logger.info('INDIVIDUAL TRADES')
        logger.info('-' * 60)
        for t in results['trades']:
            result_tag = 'WIN ' if t['pnl'] > 0 else 'LOSS'
            logger.info(
                f'  {result_tag} | {t["symbol"]:>5} ({t["sector"]}) '
                f'{t["direction"]:>5} | '
                f'Entry: ${t["entry_price"]:.2f} @ {t["entry_date"]} | '
                f'Exit: ${t["exit_price"]:.2f} @ {t["exit_date"]} | '
                f'P&L: {t["pnl_pct"]:+.4f}% | {t["exit_reason"]} | '
                f'Conf: {t["confidence"]:.2f}'
            )

        logger.info('=' * 60)
