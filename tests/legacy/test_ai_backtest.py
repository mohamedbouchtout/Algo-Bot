"""
Backtest the AI analysis pipeline against history as if the bot had been
running live.

The simulation is driven by a **calendar**, not by ticker: for every trading
day in the test window it replays what `TradingBot.run()` would have done that
day, in the same order.

    1. Fill orders queued the previous session at today's open, applying the
       same price-deviation guard `OrderManager.place_order` applies to the
       live quote and recomputing the bracket from the fill price the way
       `OrderManager._recalculate_bracket_prices` does.
    2. Walk today's high/low against every open bracket to see whether the
       stop or the target would have been hit.
    3. Retrain the per-sector models when `TradingBot.should_retrain()` would
       have fired (weekly schedule, or a VIX/SPY regime shift), always on data
       no fresher than the previous close.
    4. Scan every symbol without an open position, predict with that sector's
       model, and size the resulting signal through the real `RiskManager`
       against the simulated account.
    5. Mark the book to today's close and record the equity curve.

Nothing in the loop is allowed to see a bar the live bot would not have had.
Features are precomputed once per symbol because every extractor is causal
(rolling / ewm / shift / cumsum, and market features look up `index <= date`),
so the value on row *i* is identical whether it was computed on the full frame
or on a slice ending at *i*. That turns an O(days x history) rebuild into a
slice lookup without changing a single number.

Run it standalone - no IB connection needed, all data comes from yfinance:

    python -m tests.legacy.test_ai_backtest --max-per-sector 5
"""

import argparse
import csv
import json
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from data_fetch.historical_data import StockDataFetcher
from execution.risk_manager import RiskManager
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
from strategy.ai_analysis.data_preparation.indicator_features import IndicatorFeatureExtractor
from strategy.ai_analysis.data_preparation.market_features import MarketFeatureExtractor
from strategy.ai_analysis.data_preparation.price_features import PriceFeatureExtractor
from strategy.ai_analysis.data_preparation.volume_features import VolumeFeatureExtractor
from strategy.ai_analysis.retrain_trigger import RetrainTrigger

logger = logging.getLogger(__name__)

# AIAnalyzer.add_ticker and predict() both refuse a symbol with fewer bars than
# this, so the simulation applies the same floor before it will trade a name.
MIN_BARS_FOR_PREDICTION = 250

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Trade:
    """One completed round trip."""

    symbol: str
    sector: str
    direction: str
    signal_date: str
    entry_date: str
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    confidence: float
    exit_date: str
    exit_price: float
    exit_reason: str
    hold_days: int
    gross_pnl: float
    commission: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    equity_after: float

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class OpenPosition:
    """A filled bracket the simulation is still carrying."""

    symbol: str
    sector: str
    direction: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    confidence: float
    entry_commission: float
    bars_held: int = 0


@dataclass
class PendingOrder:
    """A signal queued at the close, to be filled at the next open."""

    symbol: str
    sector: str
    direction: str
    signal_date: pd.Timestamp
    signal_entry: float
    stop_price: float
    target_price: float
    confidence: float
    shares: int


@dataclass
class PendingLabel:
    """A prediction waiting for its forward horizon to actually elapse."""

    symbol: str
    sector: str
    predicted: int
    origin_row: int
    resolve_row: int


@dataclass
class Funnel:
    """Where every candidate died, so an empty result set is explainable."""

    scans: int = 0
    predictions: int = 0
    flat: int = 0
    below_threshold: int = 0
    signals: int = 0
    invalid_stop: int = 0
    rejected_position_size: int = 0
    rejected_risk_limits: int = 0
    rejected_cash: int = 0
    rejected_price_deviation: int = 0
    skipped_already_held: int = 0
    filled: int = 0
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    _confidence_sum: float = 0.0


class TestAIBacktest:
    # Not a pytest case despite the name - it takes constructor arguments and
    # is driven by run_tests.py / the CLI below.
    __test__ = False

    CLASS_NAMES = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}

    # Mirrors TradingBot.TRAIN_INTERVAL (6 days) combined with
    # Scheduler.is_weekend(): the live bot only retrains on a weekend, so the
    # refreshed model first goes to work on the next trading week.
    TRAIN_INTERVAL_DAYS = 6

    def __init__(
        self,
        ib=None,
        config=None,
        params=None,
        stock_data_fetcher: StockDataFetcher | None = None,
        categorized_stocks: dict[str, dict[str, list]] | None = None,
        train_ratio: float = 0.75,
        starting_equity: float = 100_000.0,
        max_holding_days: int | None = None,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_bps: float = 2.0,
        epochs: int = 100,
        retrain: bool = True,
        use_accuracy_trigger: bool = False,
        max_tickers_per_sector: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Parameters
        ----------
        train_ratio          : fraction of the shared calendar used for the
                               initial fit; the rest is walked forward.
        starting_equity      : simulated NetLiquidation at the first bar.
        max_holding_days     : optional time stop. The live bot attaches GTC
                               brackets and has no time exit, so the default
                               (None) reproduces it; set an int to cap holds.
        commission_per_share : IB-style per-share commission, floored at
                               `min_commission` per side.
        slippage_bps         : slippage applied to market fills (entries and
                               stop exits). Target exits are limit orders and
                               fill at the limit.
        epochs               : LSTM epochs per fit (early stopping still applies).
        retrain              : replay the bot's mid-flight retraining. Turning
                               it off gives a much faster single-model run.
        use_accuracy_trigger : also retrain on accuracy degradation. The live
                               `TradingBot.should_retrain()` does NOT do this,
                               so it defaults off; accuracy is still reported.
        max_tickers_per_sector : cap the universe. The full S&P/NASDAQ list is
                               several hundred names and takes hours.
        """
        self.ib = ib
        self.config = config or {}
        self.params = params or {}
        self.stock_data_fetcher = stock_data_fetcher or StockDataFetcher()
        self.categorized_stocks = categorized_stocks or {}
        self.train_ratio = train_ratio
        self.starting_equity = float(starting_equity)
        self.max_holding_days = max_holding_days
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage = slippage_bps / 10_000.0
        self.epochs = epochs
        self.retrain = retrain
        self.use_accuracy_trigger = use_accuracy_trigger
        self.max_tickers_per_sector = max_tickers_per_sector
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data',
            'backtest',
        )

        self.risk_manager = RiskManager(self.params)
        self.retrain_trigger = RetrainTrigger()

        # Sharing one set of extractor instances keeps the market-data fetch
        # (VIX/SPY) to a single call for the whole run.
        self._extractors = [
            PriceFeatureExtractor(),
            VolumeFeatureExtractor(),
            IndicatorFeatureExtractor(),
            MarketFeatureExtractor(),
        ]
        self._feature_template = FeatureBuilder(window_size=10, n_bits=4, extractors=self._extractors)

        # Market history
        self._bars: dict[str, pd.DataFrame] = {}
        self._sector_of: dict[str, str] = {}
        self._sector_tickers: dict[str, list[str]] = {}
        self._date_pos: dict[str, dict[pd.Timestamp, int]] = {}
        self._feature_cols: list[str] = []
        self._feat_matrix: dict[str, np.ndarray] = {}
        self._valid_pos: dict[str, np.ndarray] = {}
        self._market: pd.DataFrame = pd.DataFrame()

        # Portfolio
        self.cash = self.starting_equity
        self.positions: dict[str, OpenPosition] = {}
        self.pending_orders: list[PendingOrder] = []
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

        # Model state
        self.sector_analyzers: dict[str, AIAnalyzer] = {}
        self.retrain_count = 0
        self.retrain_events: list[dict] = []
        self.last_train_date: pd.Timestamp | None = None
        self._pending_labels: list[PendingLabel] = []
        self._sector_accuracy: dict[str, deque[tuple[int, int]]] = {}
        self._accuracy_totals: dict[str, list[int]] = {}

        self.funnel = Funnel()

    # ------------------------------------------------------------------ data

    def _universe(self) -> list[tuple[str, str]]:
        """(symbol, sector) pairs, deduplicated, optionally capped per sector."""
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for sector, industries in self.categorized_stocks.items():
            per_sector = 0
            for _industry, tickers in industries.items():
                for ticker in tickers:
                    if ticker in seen:
                        continue
                    if self.max_tickers_per_sector is not None and per_sector >= self.max_tickers_per_sector:
                        break
                    seen.add(ticker)
                    pairs.append((ticker, sector))
                    per_sector += 1
        return pairs

    @staticmethod
    def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(df['date'])
        if getattr(dates.dt, 'tz', None) is not None:
            dates = dates.dt.tz_localize(None)
        df = df.copy()
        df['date'] = dates.dt.normalize()
        return df

    def _load_universe(self) -> bool:
        lookback = self.params['ai_analyzer']['lookback_days']
        for symbol, sector in self._universe():
            df = self.stock_data_fetcher.get_historical_data(symbol, lookback)
            if df is None or len(df) < MIN_BARS_FOR_PREDICTION + 50:
                logger.warning(f'Skipping {symbol}: insufficient data')
                continue
            df = self._normalize_dates(df).reset_index(drop=True)
            self._bars[symbol] = df
            self._sector_of[symbol] = sector
            self._sector_tickers.setdefault(sector, []).append(symbol)
            self._date_pos[symbol] = {d: i for i, d in enumerate(df['date'])}
            logger.info(f'Fetched {len(df)} bars for {symbol} ({sector})')

        # A sector needs two names for pooled training, same rule as the bot.
        for sector in list(self._sector_tickers):
            if len(self._sector_tickers[sector]) < 2:
                dropped = self._sector_tickers.pop(sector)
                logger.warning(f'Dropping sector {sector}: only {len(dropped)} ticker(s) with data')
                for symbol in dropped:
                    self._bars.pop(symbol, None)
                    self._sector_of.pop(symbol, None)
                    self._date_pos.pop(symbol, None)

        return len(self._bars) >= 2

    def _load_market_history(self) -> None:
        """VIX / SPY closes, used for regime detection and the benchmark."""
        try:
            vix = yf.Ticker('^VIX').history(period='10y', auto_adjust=True)
            spy = yf.Ticker('SPY').history(period='10y', auto_adjust=True)
            if vix.empty or spy.empty:
                logger.warning('Market history empty; regime detection disabled')
                return
            # ^VIX and SPY bars carry different intraday stamps, so the joined
            # frame interleaves VIX-only and SPY-only rows. Forward filling on
            # that interleaved index is what gives each row both closes; only
            # then is it safe to collapse to one row per date.
            market = pd.DataFrame({'vix_close': vix['Close'], 'spy_close': spy['Close']}).sort_index()
            if market.index.tz is not None:
                market.index = market.index.tz_localize(None)
            market = market.ffill()
            market.index = market.index.normalize()
            market = market[~market.index.duplicated(keep='last')]
            self._market = market.dropna()
        except Exception as e:
            logger.warning(f'Failed to fetch market history: {e}')

    def _market_at(self, date: pd.Timestamp) -> tuple[float, float]:
        """Last VIX / SPY close at or before `date`."""
        if self._market.empty:
            return 20.0, 400.0
        pos = int(np.searchsorted(self._market.index.values, np.datetime64(date), side='right')) - 1
        if pos < 0:
            return 20.0, 400.0
        return float(self._market['vix_close'].iloc[pos]), float(self._market['spy_close'].iloc[pos])

    def _precompute_features(self) -> None:
        """
        Build the continuous feature matrix once per symbol.

        Every extractor is causal, so row *i* of this matrix equals what
        `build_windows` would compute from a slice ending at *i*. `_valid_pos`
        records which rows survive the NaN drop, which is how `build_windows`
        compacts the frame before windowing.
        """
        for symbol, df in self._bars.items():
            feats = self._feature_template.build_continuous_features(df)
            feats = feats.replace([np.inf, -np.inf], np.nan)
            if not self._feature_cols:
                self._feature_cols = list(feats.columns)
            valid = feats.notna().all(axis=1).values
            self._feat_matrix[symbol] = feats[self._feature_cols].values.astype(np.float32)
            self._valid_pos[symbol] = np.flatnonzero(valid)
            logger.debug(f'{symbol}: {valid.sum()}/{len(df)} usable feature rows')

    # ------------------------------------------------------------- inference

    def _window_at(self, symbol: str, row: int, fb: FeatureBuilder) -> np.ndarray | None:
        """
        The exact vector `build_windows(df[:row + 1])[-1]` would return, without
        rebuilding the features.
        """
        valid = self._valid_pos[symbol]
        # Number of usable feature rows at or before `row`.
        n_valid = int(np.searchsorted(valid, row, side='right'))
        # build_windows bails out when the compacted frame is no longer than
        # the window, so mirror that instead of producing a phantom sample.
        if n_valid <= fb.window_size:
            return None
        rows = valid[n_valid - fb.window_size : n_valid]
        raw = self._feat_matrix[symbol][rows]
        if fb._feat_mean is not None and fb._feat_std is not None:
            raw = (raw - fb._feat_mean) / fb._feat_std
        return raw.reshape(-1)

    def _predict_sector(self, sector: str, symbols: list[str], date: pd.Timestamp) -> dict[str, dict]:
        """One batched forward pass for every scannable symbol in a sector."""
        analyzer = self.sector_analyzers.get(sector)
        if analyzer is None or analyzer._trainer is None:
            return {}

        fb = analyzer.feature_builder
        if list(fb.feature_names) != self._feature_cols:
            logger.error(f'{sector}: feature order drifted from the precomputed matrix, skipping')
            return {}

        windows, keys, rows = [], [], []
        for symbol in symbols:
            row = self._date_pos[symbol].get(date)
            if row is None or row + 1 < MIN_BARS_FOR_PREDICTION:
                continue
            window = self._window_at(symbol, row, fb)
            if window is None:
                continue
            windows.append(window)
            keys.append(symbol)
            rows.append(row)

        if not windows:
            return {}

        preds, probs = analyzer._trainer.predict(np.stack(windows))

        out: dict[str, dict] = {}
        for i, symbol in enumerate(keys):
            cls = int(preds[i])
            out[symbol] = {
                'symbol': symbol,
                'row': rows[i],
                'class': self.CLASS_NAMES[cls],
                'class_id': cls,
                'probs': {self.CLASS_NAMES[j]: float(p) for j, p in enumerate(probs[i])},
            }
        return out

    # -------------------------------------------------------------- training

    def _bars_through(self, symbol: str, date: pd.Timestamp) -> pd.DataFrame | None:
        """Bars up to and including `date` (or the last bar before it)."""
        df = self._bars[symbol]
        n = int(np.searchsorted(df['date'].values, np.datetime64(date), side='right'))
        if n < MIN_BARS_FOR_PREDICTION:
            return None
        return df.iloc[:n]

    def _train_sector(self, sector: str, as_of: pd.Timestamp) -> AIAnalyzer | None:
        """Fit one sector's model on everything known at `as_of`'s close."""
        try:
            fb = FeatureBuilder(window_size=10, n_bits=4, extractors=self._extractors)
            analyzer = AIAnalyzer(
                stock_data=self.stock_data_fetcher,
                feature_builder=fb,
                cnn_epochs=self.epochs,
                params=self.params,
            )

            for symbol in self._sector_tickers.get(sector, []):
                df = self._bars_through(symbol, as_of)
                if df is None:
                    continue
                # Reuse the precomputed rows rather than recomputing them; the
                # features are causal so truncating them is exact.
                feats = pd.DataFrame(self._feat_matrix[symbol][: len(df)], columns=self._feature_cols)
                analyzer._bar_cache[symbol] = df.reset_index(drop=True)
                analyzer._continuous_per_ticker.append(feats)
                analyzer._kept_tickers.append(symbol)

            if len(analyzer._kept_tickers) < 2:
                logger.warning(f'{sector}: fewer than 2 trainable tickers at {as_of.date()}')
                return None

            analyzer.finalize_training(val_split=0.2)
            return analyzer
        except Exception as e:
            logger.error(f'Training failed for sector {sector} at {as_of.date()}: {e}')
            return None

    def _train_all_sectors(self, as_of: pd.Timestamp, reason: str) -> None:
        """The backtest's stand-in for TradingBot.train_modules()."""
        logger.info(f'Training all sector models on data through {as_of.date()} ({reason})...')
        trained = 0
        for sector in self._sector_tickers:
            analyzer = self._train_sector(sector, as_of)
            if analyzer is not None:
                self.sector_analyzers[sector] = analyzer
                trained += 1
                logger.info(f'  {sector}: fitted on {len(analyzer._kept_tickers)} tickers')

        if trained:
            self.last_train_date = as_of
            vix, spy = self._market_at(as_of)
            self.retrain_trigger.snapshot_from_bars(vix, spy)
            # A fresh model has no track record; drop the old one's scores.
            for scores in self._sector_accuracy.values():
                scores.clear()
            if reason != 'initial fit':
                self.retrain_count += 1
                self.retrain_events.append({'date': str(as_of.date()), 'reason': reason, 'sectors': trained})
        logger.info(f'Training complete: {trained} sector model(s)')

    def _should_retrain(self, date: pd.Timestamp, prev_date: pd.Timestamp | None) -> str | None:
        """
        Replays TradingBot.should_retrain() on the simulated clock.

        Returns the trigger reason, or None. Everything is evaluated on the
        previous close, which is all the live bot would have had before the
        session opened.
        """
        if not self.retrain or prev_date is None or self.last_train_date is None:
            return None

        vix, spy = self._market_at(prev_date)
        if self.retrain_trigger.check_regime_shift_from_bars(vix, spy):
            return 'regime shift'

        if self.use_accuracy_trigger:
            for sector, scores in self._sector_accuracy.items():
                preds = [p for p, _ in scores]
                actuals = [a for _, a in scores]
                if self.retrain_trigger.check_accuracy(preds, actuals):
                    return f'accuracy degradation ({sector})'

        # The live bot only retrains on a weekend, so on a trading-day calendar
        # the model comes back refreshed on the first session of a new week.
        crossed_weekend = date.weekday() < prev_date.weekday()
        if crossed_weekend and (date - self.last_train_date).days >= self.TRAIN_INTERVAL_DAYS:
            return 'weekly schedule'

        return None

    # -------------------------------------------------------- label tracking

    def _queue_label(self, symbol: str, sector: str, predicted: int, row: int) -> None:
        horizon = self._feature_template.forward_horizon
        self._pending_labels.append(PendingLabel(symbol=symbol, sector=sector, predicted=predicted, origin_row=row, resolve_row=row + horizon))

    def _resolve_labels(self, date: pd.Timestamp) -> None:
        """
        Score predictions whose forward horizon has now actually elapsed.

        Doing this at prediction time (as the previous version did) fed the
        retrain trigger with returns that had not happened yet.
        """
        if not self._pending_labels:
            return

        fb = self._feature_template
        atr_col = self._feature_cols.index('atr14_pct') if 'atr14_pct' in self._feature_cols else None
        still_pending: list[PendingLabel] = []

        for item in self._pending_labels:
            row_today = self._date_pos[item.symbol].get(date)
            if row_today is None or row_today < item.resolve_row:
                still_pending.append(item)
                continue

            close = self._bars[item.symbol]['close'].values
            if item.resolve_row >= len(close):
                continue

            base = float(close[item.origin_row])
            if base <= 0:
                continue
            fwd_return = float(close[item.resolve_row]) / base - 1.0

            threshold = fb.label_threshold
            if fb.volatility_adjusted_labels and atr_col is not None:
                atr = float(self._feat_matrix[item.symbol][item.origin_row, atr_col])
                if not np.isfinite(atr) or atr <= 0:
                    atr = fb.label_threshold
                fwd_return = fwd_return / atr
                threshold = fb.volatility_threshold

            if fwd_return > threshold:
                actual = 2
            elif fwd_return < -threshold:
                actual = 0
            else:
                actual = 1

            self._sector_accuracy.setdefault(item.sector, deque(maxlen=200)).append((item.predicted, actual))
            totals = self._accuracy_totals.setdefault(item.sector, [0, 0])
            totals[0] += int(item.predicted == actual)
            totals[1] += 1

        self._pending_labels = still_pending

    # ------------------------------------------------------------- portfolio

    def _commission(self, shares: int) -> float:
        return round(max(self.min_commission, shares * self.commission_per_share), 4)

    def _price_at(self, symbol: str, date: pd.Timestamp, field: str) -> float | None:
        row = self._date_pos[symbol].get(date)
        if row is None:
            return None
        return float(self._bars[symbol][field].iloc[row])

    def _bar_at(self, symbol: str, date: pd.Timestamp) -> pd.Series | None:
        row = self._date_pos[symbol].get(date)
        if row is None:
            return None
        return self._bars[symbol].iloc[row]

    def _mark_to_market(self, date: pd.Timestamp) -> float:
        """cash + long market value - short market value, IB style."""
        equity = self.cash
        for position in self.positions.values():
            price = self._price_at(position.symbol, date, 'close')
            if price is None:
                price = position.entry_price
            if position.direction == 'LONG':
                equity += position.shares * price
            else:
                equity -= position.shares * price
        return equity

    def _invested_amount(self, date: pd.Timestamp) -> float:
        """Absolute exposure, mirroring the bot's sum over ib.portfolio()."""
        total = 0.0
        for position in self.positions.values():
            price = self._price_at(position.symbol, date, 'close') or position.entry_price
            total += abs(position.shares * price)
        return total

    @staticmethod
    def _recalculate_bracket(direction: str, signal_entry: float, stop: float, target: float, fill: float) -> tuple[float, float]:
        """Same percentage-preserving reprice as OrderManager._recalculate_bracket_prices."""
        if signal_entry <= 0:
            return stop, target
        if direction == 'LONG':
            stop_pct = (signal_entry - stop) / signal_entry
            target_pct = (target - signal_entry) / signal_entry
            new_stop = fill * (1 - stop_pct)
            new_target = fill * (1 + target_pct)
        else:
            stop_pct = (stop - signal_entry) / signal_entry
            target_pct = (signal_entry - target) / signal_entry
            new_stop = fill * (1 + stop_pct)
            new_target = fill * (1 - target_pct)
        if fill < 1:
            return round(new_stop, 4), round(new_target, 4)
        return round(new_stop, 2), round(new_target, 2)

    def _fill_pending_orders(self, date: pd.Timestamp) -> None:
        """Market-on-open fills for everything queued at the previous close."""
        orders, self.pending_orders = self.pending_orders, []

        for order in orders:
            bar = self._bar_at(order.symbol, date)
            if bar is None:
                continue  # symbol did not trade; the DAY order would have expired
            if order.symbol in self.positions:
                continue

            open_price = float(bar['open'])
            if open_price <= 0:
                continue

            fill = open_price * (1 + self.slippage) if order.direction == 'LONG' else open_price * (1 - self.slippage)

            # OrderManager refuses to chase a quote that has run away from the
            # signal, with the tolerance scaled by the stop distance.
            stop_distance_pct = abs(order.signal_entry - order.stop_price) / order.signal_entry if order.signal_entry > 0 else 0.0
            max_deviation = max(stop_distance_pct, 0.005)
            deviation = abs(fill - order.signal_entry) / order.signal_entry if order.signal_entry > 0 else 0.0
            if deviation > max_deviation:
                self.funnel.rejected_price_deviation += 1
                logger.debug(f'{order.symbol}: open ${fill:.2f} deviated {deviation:.1%} from signal, order skipped')
                continue

            stop_price, target_price = self._recalculate_bracket(order.direction, order.signal_entry, order.stop_price, order.target_price, fill)

            cost = order.shares * fill
            commission = self._commission(order.shares)
            if order.direction == 'LONG' and cost + commission > self.cash:
                self.funnel.rejected_cash += 1
                continue

            if order.direction == 'LONG':
                self.cash -= cost + commission
            else:
                self.cash += cost - commission

            self.positions[order.symbol] = OpenPosition(
                symbol=order.symbol,
                sector=order.sector,
                direction=order.direction,
                signal_date=order.signal_date,
                entry_date=date,
                entry_price=fill,
                shares=order.shares,
                stop_price=stop_price,
                target_price=target_price,
                confidence=order.confidence,
                entry_commission=commission,
            )
            self.funnel.filled += 1
            logger.debug(
                f'FILL {order.direction} {order.symbol} {order.shares}sh @ ${fill:.2f} '
                f'stop ${stop_price:.2f} target ${target_price:.2f} on {date.date()}'
            )

    def _process_exits(self, date: pd.Timestamp) -> None:
        """Walk today's bar against every live bracket."""
        for symbol in list(self.positions):
            position = self.positions[symbol]
            bar = self._bar_at(symbol, date)
            if bar is None:
                continue
            position.bars_held = 0 if date == position.entry_date else position.bars_held + 1

            open_p = float(bar['open'])
            high = float(bar['high'])
            low = float(bar['low'])

            exit_price: float | None = None
            reason = ''

            if position.direction == 'LONG':
                if open_p <= position.stop_price:
                    # Gapped through the stop: it becomes a market order at the open.
                    exit_price, reason = open_p * (1 - self.slippage), 'STOP_LOSS_GAP'
                elif open_p >= position.target_price:
                    exit_price, reason = open_p, 'TAKE_PROFIT_GAP'
                elif low <= position.stop_price:
                    # Stop and target inside the same bar: assume the stop went first.
                    exit_price, reason = position.stop_price * (1 - self.slippage), 'STOP_LOSS'
                elif high >= position.target_price:
                    exit_price, reason = position.target_price, 'TAKE_PROFIT'
            else:
                if open_p >= position.stop_price:
                    exit_price, reason = open_p * (1 + self.slippage), 'STOP_LOSS_GAP'
                elif open_p <= position.target_price:
                    exit_price, reason = open_p, 'TAKE_PROFIT_GAP'
                elif high >= position.stop_price:
                    exit_price, reason = position.stop_price * (1 + self.slippage), 'STOP_LOSS'
                elif low <= position.target_price:
                    exit_price, reason = position.target_price, 'TAKE_PROFIT'

            if exit_price is None and self.max_holding_days is not None and position.bars_held >= self.max_holding_days:
                exit_price, reason = float(bar['close']), 'MAX_HOLD'

            if exit_price is not None:
                self._close_position(position, date, exit_price, reason)

    def _close_position(self, position: OpenPosition, date: pd.Timestamp, exit_price: float, reason: str) -> None:
        commission = self._commission(position.shares)
        proceeds = position.shares * exit_price

        if position.direction == 'LONG':
            self.cash += proceeds - commission
            gross = (exit_price - position.entry_price) * position.shares
        else:
            self.cash -= proceeds + commission
            gross = (position.entry_price - exit_price) * position.shares

        total_commission = position.entry_commission + commission
        pnl = gross - total_commission
        cost_basis = position.entry_price * position.shares
        risk_per_share = abs(position.entry_price - position.stop_price)

        del self.positions[position.symbol]
        equity_after = self._mark_to_market(date)

        self.trades.append(
            Trade(
                symbol=position.symbol,
                sector=position.sector,
                direction=position.direction,
                signal_date=str(position.signal_date.date()),
                entry_date=str(position.entry_date.date()),
                entry_price=round(position.entry_price, 4),
                shares=position.shares,
                stop_price=round(position.stop_price, 4),
                target_price=round(position.target_price, 4),
                confidence=round(position.confidence, 4),
                exit_date=str(date.date()),
                exit_price=round(exit_price, 4),
                exit_reason=reason,
                hold_days=position.bars_held,
                gross_pnl=round(gross, 2),
                commission=round(total_commission, 2),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / cost_basis * 100, 4) if cost_basis else 0.0,
                r_multiple=round(gross / (risk_per_share * position.shares), 4) if risk_per_share > 0 else 0.0,
                equity_after=round(equity_after, 2),
            )
        )
        logger.debug(f'EXIT {position.symbol} {reason} @ ${exit_price:.2f} on {date.date()} pnl=${pnl:,.2f}')

    # -------------------------------------------------------------- scanning

    def _scan(self, date: pd.Timestamp, equity: float) -> None:
        """
        One pass of OrderManager.scan_stocks(), queueing fills for the next open.

        Risk checks run against the account as it stands at this close, and each
        accepted order reserves its capital so the same slot cannot be handed
        out twice within one scan.
        """
        confidence_threshold = self.params['ai_analyzer']['confidence_threshold']
        vix, _spy = self._market_at(date)

        reserved_positions = len(self.positions) + len(self.pending_orders)
        reserved_invested = self._invested_amount(date) + sum(o.shares * o.signal_entry for o in self.pending_orders)
        reserved_cash = self.cash - sum(o.shares * o.signal_entry for o in self.pending_orders if o.direction == 'LONG')

        for sector, symbols in self._sector_tickers.items():
            queued = {o.symbol for o in self.pending_orders}
            candidates = []
            for symbol in symbols:
                if symbol in self.positions or symbol in queued:
                    self.funnel.skipped_already_held += 1
                    continue
                candidates.append(symbol)
            if not candidates:
                continue

            predictions = self._predict_sector(sector, candidates, date)
            self.funnel.scans += len(candidates)
            self.funnel.predictions += len(predictions)

            for symbol, prediction in predictions.items():
                class_type = prediction['class']
                confidence = prediction['probs'].get(class_type, 0.0)
                self._queue_label(symbol, sector, prediction['class_id'], prediction['row'])

                # Tracked so a run that never trades can be told apart from a
                # run whose model simply never cleared the confidence gate.
                self.funnel.max_confidence = max(self.funnel.max_confidence, confidence)
                self.funnel._confidence_sum += confidence

                if class_type == 'FLAT':
                    self.funnel.flat += 1
                    continue
                if confidence <= confidence_threshold:
                    self.funnel.below_threshold += 1
                    continue

                analyzer = self.sector_analyzers[sector]
                row = prediction['row']
                # construct_signal only reads the last bar, so a tail slice is
                # equivalent to handing it the whole history.
                history = self._bars[symbol].iloc[max(0, row - 250) : row + 1]
                signal = analyzer.construct_signal(history, self.params, class_type, confidence, vix=vix)
                if signal is None:
                    continue
                self.funnel.signals += 1

                entry = float(signal['entry'])
                stop = float(signal['stop'])
                target = float(signal['target'])
                if entry <= 0 or stop <= 0 or target <= 0 or abs(entry - stop) <= 0:
                    self.funnel.invalid_stop += 1
                    continue

                shares = self.risk_manager.calculate_position_size(equity, entry, stop)
                if shares <= 0:
                    self.funnel.rejected_position_size += 1
                    continue

                cost = shares * entry
                if not self.risk_manager.can_take_trade(equity, reserved_invested + cost, reserved_positions):
                    self.funnel.rejected_risk_limits += 1
                    continue
                if class_type == 'LONG' and not self.risk_manager.validate_trade_size(shares, entry, reserved_cash):
                    self.funnel.rejected_cash += 1
                    continue

                self.pending_orders.append(
                    PendingOrder(
                        symbol=symbol,
                        sector=sector,
                        direction=class_type,
                        signal_date=date,
                        signal_entry=entry,
                        stop_price=stop,
                        target_price=target,
                        confidence=confidence,
                        shares=shares,
                    )
                )
                reserved_positions += 1
                reserved_invested += cost
                if class_type == 'LONG':
                    reserved_cash -= cost
                logger.info(
                    f'{date.date()} SIGNAL {class_type} {symbol} ({sector}) conf {confidence:.2f} '
                    f'entry ${entry:.2f} stop ${stop:.2f} target ${target:.2f} x{shares}sh'
                )

    # ------------------------------------------------------------------- run

    def run(self) -> dict:
        logger.info('=' * 78)
        logger.info('AI BACKTEST - replaying the live bot day by day')
        logger.info('=' * 78)

        if not self._load_universe():
            logger.error('Need at least 2 tickers with sufficient data')
            return {'total_trades': 0, 'message': 'Not enough tickers with usable history'}

        self._load_market_history()
        self._precompute_features()

        # One shared calendar so every sector trains and trades on the same
        # boundary, instead of a per-ticker index split.
        calendar = sorted({d for df in self._bars.values() for d in df['date']})
        split_at = int(len(calendar) * self.train_ratio)
        train_end = calendar[split_at - 1]
        test_dates = calendar[split_at:]
        if len(test_dates) < 30:
            logger.error(f'Test window is only {len(test_dates)} sessions; lower train_ratio or fetch more history')
            return {'total_trades': 0, 'message': 'Test window too short'}

        logger.info(
            f'Universe: {len(self._bars)} tickers in {len(self._sector_tickers)} sectors | '
            f'train {calendar[0].date()} -> {train_end.date()} | '
            f'test {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} sessions)'
        )
        logger.info(
            f'Account: ${self.starting_equity:,.0f} | risk/trade {self.params["risk_management"]["risk_per_trade_pct"]:.1%} | '
            f'max {self.params["risk_management"]["max_positions"]} positions | '
            f'commission ${self.commission_per_share}/sh (min ${self.min_commission}) | slippage {self.slippage * 1e4:.1f}bps'
        )

        self._train_all_sectors(train_end, 'initial fit')
        if not self.sector_analyzers:
            logger.error('No sector produced a trained model')
            return {'total_trades': 0, 'message': 'No sector models could be trained'}

        logger.info('Starting walk-forward simulation...')
        progress_every = max(1, len(test_dates) // 20)

        # RiskManager warns on every rejected trade, which is normal and
        # expected thousands of times here. The signal funnel counts them
        # instead, so quiet the per-rejection noise for the duration.
        risk_logger = logging.getLogger('execution.risk_manager')
        previous_level = risk_logger.level
        risk_logger.setLevel(logging.ERROR)
        try:
            self._simulate(test_dates, progress_every)
        finally:
            risk_logger.setLevel(previous_level)

        # Flat the book at the last close so nothing is left unaccounted for.
        final_date = test_dates[-1]
        for symbol in list(self.positions):
            price = self._price_at(symbol, final_date, 'close')
            if price is None:
                price = self.positions[symbol].entry_price
            self._close_position(self.positions[symbol], final_date, float(price), 'END_OF_TEST')
        if self.equity_curve:
            self.equity_curve[-1] = (final_date, self._mark_to_market(final_date))

        results = self._compute_stats(test_dates)
        self._print_report(results)
        self._save_results(results)
        return results

    def _simulate(self, test_dates: list[pd.Timestamp], progress_every: int) -> None:
        """The day-by-day replay itself."""
        prev_date: pd.Timestamp | None = None

        for i, date in enumerate(test_dates):
            # A retrain would have happened overnight, on data through the
            # previous close, and is live for today's scan.
            reason = self._should_retrain(date, prev_date)
            if reason is not None:
                logger.info(f'{date.date()}: retrain triggered ({reason})')
                self._train_all_sectors(prev_date, reason)

            self._fill_pending_orders(date)
            self._process_exits(date)
            self._resolve_labels(date)

            equity = self._mark_to_market(date)
            self.equity_curve.append((date, equity))

            self._scan(date, equity)

            if i % progress_every == 0:
                logger.info(f'  [{i + 1}/{len(test_dates)}] {date.date()} equity ${equity:,.0f} open {len(self.positions)} closed {len(self.trades)}')
            prev_date = date

    # ----------------------------------------------------------------- stats

    def _benchmark(self, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
        """SPY buy & hold over the same window, for context."""
        if self._market.empty:
            return None
        window = self._market.loc[(self._market.index >= start) & (self._market.index <= end), 'spy_close']
        if len(window) < 2:
            return None
        total = (float(window.iloc[-1]) / float(window.iloc[0]) - 1) * 100
        daily = window.pct_change(fill_method=None).dropna()
        peak, max_dd = float(window.iloc[0]), 0.0
        for value in window:
            peak = max(peak, float(value))
            max_dd = max(max_dd, (peak - float(value)) / peak * 100)
        return {
            'symbol': 'SPY',
            'total_return_pct': round(total, 4),
            'max_drawdown_pct': round(max_dd, 4),
            'sharpe': round(float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 4) if daily.std() > 0 else 0.0,
        }

    def _compute_stats(self, test_dates: list[pd.Timestamp]) -> dict:
        equity_values = np.array([e for _d, e in self.equity_curve], dtype=float)
        start_equity = self.starting_equity
        end_equity = float(equity_values[-1]) if len(equity_values) else start_equity
        total_return_pct = (end_equity / start_equity - 1) * 100

        years = max(len(test_dates) / TRADING_DAYS_PER_YEAR, 1e-9)
        cagr = ((end_equity / start_equity) ** (1 / years) - 1) * 100 if end_equity > 0 else -100.0

        daily_returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else np.array([])
        daily_returns = daily_returns[np.isfinite(daily_returns)]
        volatility = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100) if len(daily_returns) > 1 else 0.0
        sharpe = (
            float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if len(daily_returns) > 1 and daily_returns.std() > 0
            else 0.0
        )
        downside = daily_returns[daily_returns < 0]
        sortino = float(daily_returns.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 and downside.std() > 0 else 0.0

        peak, max_dd, dd_date = start_equity, 0.0, None
        for date, value in self.equity_curve:
            peak = max(peak, value)
            dd = (peak - value) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd, dd_date = dd, date

        accuracy = {
            sector: {
                'evaluated': totals[1],
                'accuracy_pct': round(totals[0] / totals[1] * 100, 2) if totals[1] else 0.0,
            }
            for sector, totals in sorted(self._accuracy_totals.items())
        }

        base = {
            'start_date': str(test_dates[0].date()),
            'end_date': str(test_dates[-1].date()),
            'sessions': len(test_dates),
            'universe_size': len(self._bars),
            'sectors': len(self._sector_tickers),
            'starting_equity': round(start_equity, 2),
            'ending_equity': round(end_equity, 2),
            'total_return_pct': round(total_return_pct, 4),
            'cagr_pct': round(cagr, 4),
            'annualized_volatility_pct': round(volatility, 4),
            'sharpe': round(sharpe, 4),
            'sortino': round(sortino, 4),
            'max_drawdown_pct': round(max_dd, 4),
            'max_drawdown_date': str(dd_date.date()) if dd_date is not None else None,
            'retrain_count': self.retrain_count,
            'retrain_events': self.retrain_events,
            'model_accuracy': accuracy,
            'signal_funnel': self._funnel_summary(),
            'benchmark': self._benchmark(test_dates[0], test_dates[-1]),
            'equity_curve': [{'date': str(d.date()), 'equity': round(v, 2)} for d, v in self.equity_curve],
        }

        if not self.trades:
            base.update({'total_trades': 0, 'message': 'No trades were generated during the test period', 'trades': []})
            return base

        wins = [t for t in self.trades if t.is_win]
        losses = [t for t in self.trades if not t.is_win]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = sum(t.pnl for t in losses)

        exit_reasons: dict[str, int] = {}
        for trade in self.trades:
            exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1

        per_sector: dict[str, dict] = {}
        for sector in sorted({t.sector for t in self.trades}):
            group = [t for t in self.trades if t.sector == sector]
            group_wins = [t for t in group if t.is_win]
            per_sector[sector] = {
                'trades': len(group),
                'win_rate_pct': round(len(group_wins) / len(group) * 100, 2),
                'net_pnl': round(sum(t.pnl for t in group), 2),
                'avg_return_pct': round(float(np.mean([t.pnl_pct for t in group])), 4),
            }

        per_direction: dict[str, dict] = {}
        for direction in sorted({t.direction for t in self.trades}):
            group = [t for t in self.trades if t.direction == direction]
            group_wins = [t for t in group if t.is_win]
            per_direction[direction] = {
                'trades': len(group),
                'win_rate_pct': round(len(group_wins) / len(group) * 100, 2),
                'net_pnl': round(sum(t.pnl for t in group), 2),
                'avg_r_multiple': round(float(np.mean([t.r_multiple for t in group])), 4),
            }

        exposure_days = sum(max(t.hold_days, 1) for t in self.trades)
        max_positions = self.params['risk_management']['max_positions']

        base.update(
            {
                'total_trades': len(self.trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate_pct': round(len(wins) / len(self.trades) * 100, 2),
                'net_pnl': round(sum(t.pnl for t in self.trades), 2),
                'total_commission': round(sum(t.commission for t in self.trades), 2),
                'avg_win_pct': round(float(np.mean([t.pnl_pct for t in wins])), 4) if wins else 0.0,
                'avg_loss_pct': round(float(np.mean([t.pnl_pct for t in losses])), 4) if losses else 0.0,
                'avg_win_usd': round(float(np.mean([t.pnl for t in wins])), 2) if wins else 0.0,
                'avg_loss_usd': round(float(np.mean([t.pnl for t in losses])), 2) if losses else 0.0,
                'best_trade_usd': round(max(t.pnl for t in self.trades), 2),
                'worst_trade_usd': round(min(t.pnl for t in self.trades), 2),
                'expectancy_usd': round(float(np.mean([t.pnl for t in self.trades])), 2),
                'avg_r_multiple': round(float(np.mean([t.r_multiple for t in self.trades])), 4),
                'profit_factor': round(gross_win / abs(gross_loss), 4) if gross_loss < 0 else float('inf'),
                'avg_hold_days': round(float(np.mean([t.hold_days for t in self.trades])), 2),
                'exposure_pct': round(exposure_days / (len(test_dates) * max_positions) * 100, 2),
                'exit_reasons': exit_reasons,
                'per_sector': per_sector,
                'per_direction': per_direction,
                'trades': [asdict(t) for t in self.trades],
            }
        )
        return base

    # ---------------------------------------------------------------- report

    def _print_report(self, results: dict) -> None:
        log = logger.info
        log('')
        log('=' * 78)
        log('AI BACKTEST RESULTS')
        log('=' * 78)
        log(
            f'Period:            {results["start_date"]} -> {results["end_date"]} '
            f'({results["sessions"]} sessions, {results["universe_size"]} tickers, {results["sectors"]} sectors)'
        )
        log(f'Starting Equity:   ${results["starting_equity"]:,.2f}')
        log(f'Ending Equity:     ${results["ending_equity"]:,.2f}')
        log(f'Total Return:      {results["total_return_pct"]:+.2f}%')
        log(f'CAGR:              {results["cagr_pct"]:+.2f}%')
        log(f'Annualized Vol:    {results["annualized_volatility_pct"]:.2f}%')
        log(f'Sharpe / Sortino:  {results["sharpe"]:.2f} / {results["sortino"]:.2f}')
        drawdown_when = f' (on {results["max_drawdown_date"]})' if results['max_drawdown_date'] else ''
        log(f'Max Drawdown:      {results["max_drawdown_pct"]:.2f}%{drawdown_when}')
        log(f'Mid-Sim Retrains:  {results["retrain_count"]}')

        benchmark = results.get('benchmark')
        if benchmark:
            log(
                f'Benchmark (SPY):   {benchmark["total_return_pct"]:+.2f}% total, '
                f'max DD {benchmark["max_drawdown_pct"]:.2f}%, Sharpe {benchmark["sharpe"]:.2f}'
            )

        if results.get('total_trades', 0) == 0:
            log('')
            log(results.get('message', 'No trades'))
            self._print_funnel(results)
            log('=' * 78)
            return

        log('')
        log('-' * 78)
        log('TRADE STATISTICS')
        log('-' * 78)
        log(f'Total Trades:      {results["total_trades"]}  (won {results["winning_trades"]}, lost {results["losing_trades"]})')
        log(f'Win Rate:          {results["win_rate_pct"]:.2f}%')
        log(f'Net P&L:           ${results["net_pnl"]:+,.2f}  (commissions ${results["total_commission"]:,.2f})')
        log(
            f'Avg Win / Loss:    ${results["avg_win_usd"]:+,.2f} ({results["avg_win_pct"]:+.2f}%) / '
            f'${results["avg_loss_usd"]:+,.2f} ({results["avg_loss_pct"]:+.2f}%)'
        )
        log(f'Best / Worst:      ${results["best_trade_usd"]:+,.2f} / ${results["worst_trade_usd"]:+,.2f}')
        log(f'Expectancy:        ${results["expectancy_usd"]:+,.2f} per trade  ({results["avg_r_multiple"]:+.2f}R)')
        log(f'Profit Factor:     {results["profit_factor"]:.2f}')
        log(f'Avg Hold:          {results["avg_hold_days"]:.1f} sessions   Exposure: {results["exposure_pct"]:.1f}% of capacity')
        log(f'Exit Reasons:      {results["exit_reasons"]}')

        log('')
        log('-' * 78)
        log('PER-SECTOR')
        log('-' * 78)
        for sector, stats in results['per_sector'].items():
            log(
                f'  {sector:<26} {stats["trades"]:>4} trades | win {stats["win_rate_pct"]:>6.2f}% | '
                f'net ${stats["net_pnl"]:>+11,.2f} | avg {stats["avg_return_pct"]:>+7.2f}%'
            )

        log('')
        log('-' * 78)
        log('PER-DIRECTION')
        log('-' * 78)
        for direction, stats in results['per_direction'].items():
            log(
                f'  {direction:<26} {stats["trades"]:>4} trades | win {stats["win_rate_pct"]:>6.2f}% | '
                f'net ${stats["net_pnl"]:>+11,.2f} | avg {stats["avg_r_multiple"]:>+6.2f}R'
            )

        accuracy = results.get('model_accuracy') or {}
        if accuracy:
            log('')
            log('-' * 78)
            log('LIVE MODEL ACCURACY (each prediction scored once its horizon elapsed)')
            log('-' * 78)
            for sector, stats in accuracy.items():
                log(f'  {sector:<26} {stats["accuracy_pct"]:>6.2f}% over {stats["evaluated"]} predictions')

        self._print_funnel(results)

        log('')
        log('-' * 78)
        log('INDIVIDUAL TRADES')
        log('-' * 78)
        for trade in results['trades']:
            tag = 'WIN ' if trade['pnl'] > 0 else 'LOSS'
            log(
                f'  {tag} | {trade["symbol"]:<6} {trade["direction"]:<5} | '
                f'in {trade["entry_date"]} ${trade["entry_price"]:>9,.2f} x{trade["shares"]:<6} | '
                f'out {trade["exit_date"]} ${trade["exit_price"]:>9,.2f} | '
                f'P&L ${trade["pnl"]:>+10,.2f} ({trade["pnl_pct"]:>+6.2f}%, {trade["r_multiple"]:>+5.2f}R) | '
                f'{trade["exit_reason"]:<15} | held {trade["hold_days"]:>3}d | conf {trade["confidence"]:.2f} | {trade["sector"]}'
            )
        log('=' * 78)

    def _funnel_summary(self) -> dict:
        summary = asdict(self.funnel)
        predictions = summary.pop('_confidence_sum', 0.0)
        summary['mean_confidence'] = round(predictions / self.funnel.predictions, 4) if self.funnel.predictions else 0.0
        summary['max_confidence'] = round(self.funnel.max_confidence, 4)
        summary['confidence_threshold'] = self.params['ai_analyzer']['confidence_threshold']
        return summary

    def _print_funnel(self, results: dict) -> None:
        funnel = results.get('signal_funnel') or {}
        if not funnel:
            return
        logger.info('')
        logger.info('-' * 78)
        logger.info('SIGNAL FUNNEL')
        logger.info('-' * 78)
        logger.info(f'  symbol-days scanned:        {funnel.get("scans", 0)}')
        logger.info(f'  predictions produced:       {funnel.get("predictions", 0)}')
        logger.info(f'    -> FLAT:                  {funnel.get("flat", 0)}')
        logger.info(f'    -> below confidence:      {funnel.get("below_threshold", 0)}')
        logger.info(f'  signals constructed:        {funnel.get("signals", 0)}')
        logger.info(f'  dropped, bad stop:          {funnel.get("invalid_stop", 0)}')
        logger.info(f'  dropped, size = 0:          {funnel.get("rejected_position_size", 0)}')
        logger.info(f'  dropped, risk limits:       {funnel.get("rejected_risk_limits", 0)}')
        logger.info(f'  dropped, cash:              {funnel.get("rejected_cash", 0)}')
        logger.info(f'  dropped, price deviation:   {funnel.get("rejected_price_deviation", 0)}')
        logger.info(f'  skipped, already held:      {funnel.get("skipped_already_held", 0)}')
        logger.info(f'  orders filled:              {funnel.get("filled", 0)}')
        logger.info(
            f'  top-class confidence:       mean {funnel.get("mean_confidence", 0.0):.3f}, '
            f'max {funnel.get("max_confidence", 0.0):.3f} (gate > {funnel.get("confidence_threshold", 0.0):.2f})'
        )

    def _save_results(self, results: dict) -> None:
        """Persist the full run so it can be compared against a later one."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            stamp = f'{results.get("start_date", "start")}_{results.get("end_date", "end")}'
            json_path = os.path.join(self.output_dir, f'ai_backtest_{stamp}.json')
            with open(json_path, 'w') as handle:
                json.dump(results, handle, indent=2, default=str)

            trades = results.get('trades') or []
            if trades:
                csv_path = os.path.join(self.output_dir, f'ai_backtest_trades_{stamp}.csv')
                with open(csv_path, 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(trades[0].keys()))
                    writer.writeheader()
                    writer.writerows(trades)
                logger.info(f'Saved results to {json_path} and {csv_path}')
            else:
                logger.info(f'Saved results to {json_path}')
        except Exception as e:
            logger.warning(f'Failed to save backtest results: {e}')


# ---------------------------------------------------------------------- CLI


def _load_json(name: str) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, 'config', name)) as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description='Backtest the AI pipeline as if the bot had been running live.')
    parser.add_argument('--tickers', help='Comma separated symbols, bypassing the sector fetcher')
    parser.add_argument('--max-per-sector', type=int, default=5, help='Cap tickers per sector (default 5; the full universe takes hours)')
    parser.add_argument('--train-ratio', type=float, default=0.75)
    parser.add_argument('--equity', type=float, default=100_000.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max-holding-days', type=int, default=None, help='Optional time stop; omitted means GTC brackets like the live bot')
    parser.add_argument('--no-retrain', action='store_true', help='Skip mid-simulation retraining (much faster)')
    parser.add_argument('--accuracy-trigger', action='store_true', help='Also retrain on accuracy degradation')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger('strategy.ai_analysis.lstm_trainer').setLevel(logging.WARNING)

    params = _load_json('trading_params.json')

    if args.tickers:
        symbols = [s.strip().upper() for s in args.tickers.split(',') if s.strip()]
        categorized = {'Backtest': {'All': symbols}}
    else:
        from data_fetch.stock_fetcher import StockTickerFetcher

        categorized = StockTickerFetcher().categorized_stocks

    backtest = TestAIBacktest(
        params=params,
        stock_data_fetcher=StockDataFetcher(),
        categorized_stocks=categorized,
        train_ratio=args.train_ratio,
        starting_equity=args.equity,
        epochs=args.epochs,
        max_holding_days=args.max_holding_days,
        retrain=not args.no_retrain,
        use_accuracy_trigger=args.accuracy_trigger,
        max_tickers_per_sector=args.max_per_sector,
    )
    return backtest.run()


if __name__ == '__main__':
    main()
