"""Integration tests for the AI training and prediction pipeline."""

import numpy as np
import pytest

from data_fetch.historical_data import StockDataFetcher
from strategy.ai_analysis.ai_analyzer import AIAnalyzer
from strategy.ai_analysis.walk_forward import WalkForwardValidator
from tests.conftest import PARAMS, make_synthetic_bars


@pytest.mark.integration
class TestAITrainPredictPipeline:
    """End-to-end: synthetic data -> feature build -> train -> predict."""

    @pytest.fixture
    def trained_analyzer(self):
        """Build and train an AIAnalyzer on synthetic tickers."""
        tickers = ['SYN_A', 'SYN_B', 'SYN_C']
        bars = {t: make_synthetic_bars(400, symbol=t) for t in tickers}

        fetcher = StockDataFetcher()
        fetcher.get_historical_data = lambda sym, _days: bars.get(sym)

        analyzer = AIAnalyzer(
            stock_data=fetcher,
            cnn_epochs=3,
            params=PARAMS,
            model_type='lstm',
        )
        analyzer.train(tickers, val_split=0.2)
        return analyzer, bars

    def test_train_completes(self, trained_analyzer):
        analyzer, _ = trained_analyzer
        assert analyzer._trainer is not None
        assert analyzer._trainer.model is not None

    def test_predict_returns_valid_result(self, trained_analyzer):
        analyzer, bars = trained_analyzer
        result = analyzer.predict('SYN_A')

        assert result is not None
        assert result['symbol'] == 'SYN_A'
        assert result['class'] in ('SHORT', 'FLAT', 'LONG')
        assert result['class_id'] in (0, 1, 2)
        probs = result['probs']
        assert set(probs.keys()) == {'SHORT', 'FLAT', 'LONG'}
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_predict_unseen_ticker(self, trained_analyzer):
        analyzer, _ = trained_analyzer
        result = analyzer.predict('SYN_UNSEEN')
        assert result is None

    def test_add_ticker_dedup(self, trained_analyzer):
        analyzer, _ = trained_analyzer
        added = analyzer.add_ticker('SYN_A')
        assert added is False

    def test_reset_dataset(self, trained_analyzer):
        analyzer, _ = trained_analyzer
        analyzer.reset_dataset()
        assert len(analyzer._kept_tickers) == 0
        assert len(analyzer._bar_cache) == 0

    def test_cnn_model_type(self):
        tickers = ['SYN_X', 'SYN_Y']
        bars = {t: make_synthetic_bars(400, symbol=t) for t in tickers}

        fetcher = StockDataFetcher()
        fetcher.get_historical_data = lambda sym, _days: bars.get(sym)

        analyzer = AIAnalyzer(
            stock_data=fetcher,
            cnn_epochs=3,
            params=PARAMS,
            model_type='cnn',
        )
        analyzer.train(tickers, val_split=0.2)

        result = analyzer.predict('SYN_X')
        assert result is not None
        assert result['class'] in ('SHORT', 'FLAT', 'LONG')

    def test_invalid_model_type_raises(self):
        fetcher = StockDataFetcher()
        with pytest.raises(ValueError, match='model_type'):
            AIAnalyzer(stock_data=fetcher, params=PARAMS, model_type='transformer')


@pytest.mark.integration
class TestWalkForwardIntegration:
    """Walk-forward cross-validation on synthetic data through AIAnalyzer."""

    def test_walk_forward_produces_fold_metrics(self):
        tickers = ['WF_A', 'WF_B', 'WF_C']
        bars = {t: make_synthetic_bars(400, symbol=t) for t in tickers}

        fetcher = StockDataFetcher()
        fetcher.get_historical_data = lambda sym, _days: bars.get(sym)

        analyzer = AIAnalyzer(
            stock_data=fetcher,
            cnn_epochs=3,
            params=PARAMS,
            model_type='lstm',
        )
        for t in tickers:
            analyzer.add_ticker(t)

        results = analyzer.walk_forward_train(n_splits=3)

        assert 'folds' in results
        assert len(results['folds']) == 3
        for fold in results['folds']:
            assert 'accuracy' in fold
            assert 0.0 <= fold['accuracy'] <= 1.0
            assert fold['train_size'] > 0
            assert fold['val_size'] > 0
        assert 0.0 <= results['avg_accuracy'] <= 1.0
        assert results['total_samples'] > 0

        result = analyzer.predict('WF_A')
        assert result is not None

    def test_walk_forward_validator_splits(self):
        validator = WalkForwardValidator(n_splits=4, min_train_pct=0.5)
        splits = validator.split(1000)

        assert len(splits) == 4
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert train_idx[-1] < val_idx[0]

    def test_walk_forward_expanding_window(self):
        validator = WalkForwardValidator(n_splits=3, min_train_pct=0.5)
        splits = validator.split(600)

        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]
