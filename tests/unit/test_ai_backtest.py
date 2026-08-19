"""
Unit tests for the AI backtest simulator.

These pin the properties that make the result trustworthy - no lookahead, a
balanced ledger, and bracket handling that matches OrderManager - using a stub
model so nothing here needs a network or a trained LSTM.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import PARAMS, make_synthetic_bars
from tests.legacy.test_ai_backtest import TestAIBacktest


class StubTrainer:
    """Emits a fixed class with a fixed confidence for every window."""

    def __init__(self, class_id: int = 2, confidence: float = 0.95):
        self.class_id = class_id
        self.confidence = confidence

    def predict(self, x):
        n = len(x)
        probs = np.full((n, 3), (1.0 - self.confidence) / 2.0, dtype=np.float32)
        probs[:, self.class_id] = self.confidence
        return np.full(n, self.class_id, dtype=np.int64), probs


class StubAnalyzer:
    """Stands in for a trained AIAnalyzer without touching torch."""

    def __init__(self, feature_builder, trainer):
        self.feature_builder = feature_builder
        self._trainer = trainer
        self._kept_tickers = ['AAA', 'BBB']

    def construct_signal(self, df, params, class_type, confidence, vix=None):
        entry = round(float(df['close'].iloc[-1]), 2)
        risk = round(entry * 0.03, 2)
        if class_type == 'LONG':
            stop, target = entry - risk, entry + 2 * risk
        else:
            stop, target = entry + risk, entry - 2 * risk
        return {
            'strategy_type': 'ai_analysis',
            'type': class_type,
            'symbol': df['symbol'].iloc[0],
            'entry': entry,
            'stop': round(stop, 2),
            'target': round(target, 2),
            'risk': risk,
            'reward': 2 * risk,
            'confidence': confidence,
        }


def _build(class_id: int = 2, **kwargs) -> TestAIBacktest:
    """A backtest wired to synthetic bars and a stub model, ready to simulate."""
    backtest = TestAIBacktest(
        params=PARAMS,
        categorized_stocks={'Synthetic': {'All': ['AAA', 'BBB']}},
        **kwargs,
    )

    for symbol, seed_offset in (('AAA', 0), ('BBB', 7)):
        bars = make_synthetic_bars(600, symbol=symbol)
        # Shift the second name so the two series are not identical.
        if seed_offset:
            bars['close'] = bars['close'] * 1.15 + seed_offset
            bars['high'] = bars['high'] * 1.15 + seed_offset
            bars['low'] = bars['low'] * 1.15 + seed_offset
            bars['open'] = bars['open'] * 1.15 + seed_offset
        bars = backtest._normalize_dates(bars).reset_index(drop=True)
        backtest._bars[symbol] = bars
        backtest._sector_of[symbol] = 'Synthetic'
        backtest._date_pos[symbol] = {d: i for i, d in enumerate(bars['date'])}
    backtest._sector_tickers['Synthetic'] = ['AAA', 'BBB']

    backtest._precompute_features()

    feature_builder = backtest._feature_template
    feature_builder.feature_names = list(backtest._feature_cols)
    feature_builder._feat_mean = np.zeros(len(backtest._feature_cols), dtype=np.float32)
    feature_builder._feat_std = np.ones(len(backtest._feature_cols), dtype=np.float32)
    backtest.sector_analyzers['Synthetic'] = StubAnalyzer(feature_builder, StubTrainer(class_id))
    return backtest


def _run(backtest: TestAIBacktest) -> tuple[dict, list]:
    calendar = sorted({d for df in backtest._bars.values() for d in df['date']})
    test_dates = calendar[400:]
    backtest._simulate(test_dates, progress_every=10_000)
    for symbol in list(backtest.positions):
        price = backtest._price_at(symbol, test_dates[-1], 'close')
        backtest._close_position(backtest.positions[symbol], test_dates[-1], float(price), 'END_OF_TEST')
    backtest.equity_curve[-1] = (test_dates[-1], backtest._mark_to_market(test_dates[-1]))
    return backtest._compute_stats(test_dates), test_dates


class TestNoLookahead:
    def test_entry_is_after_the_signal(self):
        backtest = _build(class_id=2)
        results, _ = _run(backtest)
        assert results['total_trades'] > 0
        for trade in results['trades']:
            assert trade['entry_date'] > trade['signal_date'], 'entry must fill after the signal bar closed'

    def test_exit_is_never_before_entry(self):
        backtest = _build(class_id=2)
        results, _ = _run(backtest)
        for trade in results['trades']:
            assert trade['exit_date'] >= trade['entry_date']

    def test_window_matches_a_full_rebuild(self):
        """The precomputed slice must equal what build_windows would return."""
        backtest = _build()
        feature_builder = backtest._feature_template
        df = backtest._bars['AAA']
        for row in (300, 420, 555):
            _, cnn_x, _ = feature_builder.build_windows(df.iloc[: row + 1], include_labels=False, include_rbm=False)
            fast = backtest._window_at('AAA', row, feature_builder)
            assert fast is not None
            assert np.allclose(cnn_x[-1], fast, atol=1e-6)

    def test_labels_are_only_scored_once_the_horizon_elapsed(self):
        backtest = _build()
        dates = list(backtest._bars['AAA']['date'])
        backtest._queue_label('AAA', 'Synthetic', 2, row=400)

        horizon = backtest._feature_template.forward_horizon
        backtest._resolve_labels(dates[400 + horizon - 1])
        assert backtest._accuracy_totals == {}, 'scored before the forward window closed'

        backtest._resolve_labels(dates[400 + horizon])
        assert backtest._accuracy_totals['Synthetic'][1] == 1


class TestLedger:
    def test_equity_reconciles_with_trade_pnl(self):
        backtest = _build(class_id=2)
        results, _ = _run(backtest)
        expected = results['starting_equity'] + results['net_pnl']
        # Per-trade P&L is rounded to the cent, so allow the rounding dust.
        assert results['ending_equity'] == pytest.approx(expected, abs=0.05 * results['total_trades'] + 0.01)

    def test_no_position_is_opened_twice(self):
        backtest = _build(class_id=2)
        _run(backtest)
        assert not backtest.positions
        # Overlapping round trips on one symbol must not share a date range.
        by_symbol: dict[str, list] = {}
        for trade in backtest.trades:
            by_symbol.setdefault(trade.symbol, []).append(trade)
        for trades in by_symbol.values():
            ordered = sorted(trades, key=lambda t: t.entry_date)
            for earlier, later in zip(ordered, ordered[1:]):
                assert later.entry_date >= earlier.exit_date

    def test_position_count_respects_the_risk_limit(self):
        backtest = _build(class_id=2)
        _run(backtest)
        assert backtest.funnel.filled > 0
        assert len(backtest.positions) <= PARAMS['risk_management']['max_positions']

    def test_shorts_are_simulated_too(self):
        backtest = _build(class_id=0)
        results, _ = _run(backtest)
        assert results['total_trades'] > 0
        assert set(results['per_direction']) == {'SHORT'}


class TestBracketMath:
    def test_long_bracket_keeps_its_percentages(self):
        stop, target = TestAIBacktest._recalculate_bracket('LONG', 100.0, 97.0, 106.0, 110.0)
        assert stop == pytest.approx(106.7, abs=0.01)
        assert target == pytest.approx(116.6, abs=0.01)

    def test_short_bracket_keeps_its_percentages(self):
        stop, target = TestAIBacktest._recalculate_bracket('SHORT', 100.0, 103.0, 94.0, 90.0)
        assert stop == pytest.approx(92.7, abs=0.01)
        assert target == pytest.approx(84.6, abs=0.01)

    def test_gap_through_the_stop_fills_at_the_open(self):
        backtest = _build()
        symbol = 'AAA'
        date = backtest._bars[symbol]['date'].iloc[500]
        bar = backtest._bars[symbol].iloc[500]
        from tests.legacy.test_ai_backtest import OpenPosition

        # Stop sits above the open, so the whole bar is already through it.
        backtest.positions[symbol] = OpenPosition(
            symbol=symbol,
            sector='Synthetic',
            direction='LONG',
            signal_date=date,
            entry_date=backtest._bars[symbol]['date'].iloc[499],
            entry_price=float(bar['open']) * 1.10,
            shares=10,
            stop_price=float(bar['open']) * 1.05,
            target_price=float(bar['open']) * 1.30,
            confidence=0.9,
            entry_commission=1.0,
        )
        backtest._process_exits(date)
        assert len(backtest.trades) == 1
        trade = backtest.trades[0]
        assert trade.exit_reason == 'STOP_LOSS_GAP'
        assert trade.exit_price == pytest.approx(float(bar['open']) * (1 - backtest.slippage), rel=1e-6)


class TestReporting:
    def test_report_covers_the_headline_numbers(self):
        backtest = _build(class_id=2)
        results, _ = _run(backtest)
        for key in (
            'total_return_pct',
            'cagr_pct',
            'sharpe',
            'max_drawdown_pct',
            'win_rate_pct',
            'profit_factor',
            'expectancy_usd',
            'per_sector',
            'per_direction',
            'signal_funnel',
            'equity_curve',
        ):
            assert key in results, f'{key} missing from the results'
        assert len(results['equity_curve']) > 0
        for trade in results['trades']:
            assert trade['exit_reason']
            assert isinstance(trade['pnl'], float)

    def test_empty_run_still_reports_a_funnel(self):
        backtest = _build(class_id=1)  # always FLAT, so nothing ever trades
        results, _ = _run(backtest)
        assert results['total_trades'] == 0
        assert results['signal_funnel']['flat'] > 0
        assert results['signal_funnel']['filled'] == 0
