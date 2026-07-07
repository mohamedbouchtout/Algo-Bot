"""Integration tests for the feature extraction pipeline."""

import numpy as np
import pandas as pd
import pytest

from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
from strategy.ai_analysis.data_preparation.indicator_features import IndicatorFeatureExtractor
from strategy.ai_analysis.data_preparation.market_features import MarketFeatureExtractor
from strategy.ai_analysis.data_preparation.price_features import PriceFeatureExtractor
from strategy.ai_analysis.data_preparation.volume_features import VolumeFeatureExtractor
from tests.conftest import make_synthetic_bars, requires_network


@pytest.mark.integration
class TestFeaturePipeline:
    """Test the full feature extraction chain on synthetic data."""

    def test_price_features_shape(self):
        df = make_synthetic_bars(300)
        extractor = PriceFeatureExtractor()
        feats = extractor.extract(df)

        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(df)
        assert len(feats.columns) > 0
        assert not feats.iloc[50:].isnull().all(axis=1).any()

    def test_volume_features_shape(self):
        df = make_synthetic_bars(300)
        extractor = VolumeFeatureExtractor()
        feats = extractor.extract(df)

        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(df)

    def test_indicator_features_shape(self):
        df = make_synthetic_bars(300)
        extractor = IndicatorFeatureExtractor()
        feats = extractor.extract(df)

        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(df)

    @requires_network
    def test_market_features_shape(self):
        df = make_synthetic_bars(300)
        extractor = MarketFeatureExtractor()
        feats = extractor.extract(df)

        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(df)

    def test_feature_builder_continuous(self):
        df = make_synthetic_bars(300)
        fb = FeatureBuilder(window_size=10)
        feats = fb.build_continuous_features(df)

        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(df)
        assert len(feats.columns) >= 15

    def test_feature_builder_windows(self):
        df = make_synthetic_bars(300)
        fb = FeatureBuilder(window_size=10)

        feats_list = [fb.build_continuous_features(df)]
        fb.fit_bin_edges(feats_list)

        rbm_x, cnn_x, labels = fb.build_windows(df, include_labels=True)

        assert cnn_x.ndim == 2
        assert cnn_x.shape[0] > 0
        n_features = len(fb.feature_names)
        assert cnn_x.shape[1] == 10 * n_features
        assert labels is not None
        assert len(labels) == len(cnn_x)
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_feature_builder_no_labels(self):
        df = make_synthetic_bars(300)
        fb = FeatureBuilder(window_size=10)
        feats_list = [fb.build_continuous_features(df)]
        fb.fit_bin_edges(feats_list)

        _, cnn_x, labels = fb.build_windows(df, include_labels=False)
        assert cnn_x.shape[0] > 0
        assert labels is None

    def test_feature_builder_insufficient_data(self):
        df = make_synthetic_bars(12)
        fb = FeatureBuilder(window_size=10, forward_horizon=5)
        feats_list = [fb.build_continuous_features(make_synthetic_bars(300))]
        fb.fit_bin_edges(feats_list)

        _, cnn_x, labels = fb.build_windows(df, include_labels=True)
        assert len(cnn_x) == 0
