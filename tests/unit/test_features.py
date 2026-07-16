"""Unit tests for volatility-adjusted labels and market features."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
from strategy.ai_analysis.data_preparation.indicator_features import IndicatorFeatureExtractor
from strategy.ai_analysis.data_preparation.market_features import MarketFeatureExtractor
from strategy.ai_analysis.data_preparation.price_features import PriceFeatureExtractor
from strategy.ai_analysis.data_preparation.volume_features import VolumeFeatureExtractor


class TestVolatilityAdjustedLabels:
    @patch('strategy.ai_analysis.data_preparation.market_features.yf')
    def test_fixed_vs_vol_adjusted_differ(self, mock_yf, synthetic_bars):
        mock_yf.Ticker.return_value.history.return_value = pd.DataFrame()

        extractors = [
            PriceFeatureExtractor(),
            VolumeFeatureExtractor(),
            IndicatorFeatureExtractor(),
        ]

        fb_fixed = FeatureBuilder(
            extractors=extractors,
            volatility_adjusted_labels=False,
        )
        fb_vol = FeatureBuilder(
            extractors=extractors,
            volatility_adjusted_labels=True,
            volatility_threshold=1.0,
        )

        feats = fb_fixed.build_continuous_features(synthetic_bars)
        fb_fixed.fit_bin_edges([feats])
        fb_vol.feature_names = fb_fixed.feature_names
        fb_vol.bin_edges = fb_fixed.bin_edges

        _, _, labels_fixed = fb_fixed.build_windows(synthetic_bars, include_labels=True, include_rbm=False)
        _, _, labels_vol = fb_vol.build_windows(synthetic_bars, include_labels=True, include_rbm=False)

        assert labels_fixed is not None
        assert labels_vol is not None
        assert not np.array_equal(labels_fixed, labels_vol)

    @patch('strategy.ai_analysis.data_preparation.market_features.yf')
    def test_zero_atr_fallback(self, mock_yf, synthetic_bars):
        mock_yf.Ticker.return_value.history.return_value = pd.DataFrame()

        extractors = [
            PriceFeatureExtractor(),
            VolumeFeatureExtractor(),
            IndicatorFeatureExtractor(),
        ]
        fb = FeatureBuilder(
            extractors=extractors,
            volatility_adjusted_labels=True,
        )
        feats = fb.build_continuous_features(synthetic_bars)
        fb.fit_bin_edges([feats])

        _, _, labels = fb.build_windows(synthetic_bars, include_labels=True, include_rbm=False)
        assert labels is not None
        assert len(labels) > 0
        assert not np.isnan(labels).any()


class TestMarketFeatureExtractor:
    def test_feature_names(self):
        assert len(MarketFeatureExtractor.FEATURE_NAMES) == 3

    def test_extract_with_empty_market_data(self, synthetic_bars):
        extractor = MarketFeatureExtractor(market_data=pd.DataFrame())
        result = extractor.extract(synthetic_bars)
        assert list(result.columns) == MarketFeatureExtractor.FEATURE_NAMES
        assert len(result) == len(synthetic_bars)
        assert (result['vix_normalized'] == 1.0).all()

    def test_extract_with_mock_market_data(self, synthetic_bars):
        dates = pd.bdate_range(start='2022-01-01', periods=600)
        market = pd.DataFrame(
            {
                'vix_close': np.full(600, 20.0),
                'spy_close': np.linspace(400, 500, 600),
            },
            index=dates,
        )

        extractor = MarketFeatureExtractor(market_data=market)
        result = extractor.extract(synthetic_bars)
        assert list(result.columns) == MarketFeatureExtractor.FEATURE_NAMES
        assert len(result) == len(synthetic_bars)
        assert (result['vix_normalized'] == 1.0).all()
