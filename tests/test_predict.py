"""
test_predict.py — Unit tests for the predictive maintenance pipeline.

Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataLoader:
    """Tests for data loading and preprocessing."""

    def test_load_train_data_shape(self):
        """Training data should have 28 columns (26 original + RUL)."""
        from src.data_loader import load_train_data
        df = load_train_data()
        assert df.shape[1] >= 27  # 26 original + RUL (27 or 28 depending on data source)
        assert 'RUL' in df.columns
        assert 'engine_id' in df.columns

    def test_load_train_data_engines(self):
        """Should have 100 engines."""
        from src.data_loader import load_train_data
        df = load_train_data()
        assert df['engine_id'].nunique() == 100

    def test_rul_is_zero_at_failure(self):
        """Last cycle of each engine should have RUL = 0."""
        from src.data_loader import load_train_data
        df = load_train_data()
        for eid in [1, 50, 100]:
            engine = df[df['engine_id'] == eid]
            last_row = engine.iloc[-1]
            assert last_row['RUL'] == 0, f"Engine {eid} RUL at last cycle should be 0"

    def test_rul_decreases_over_time(self):
        """RUL should decrease as cycle increases for each engine."""
        from src.data_loader import load_train_data
        df = load_train_data()
        engine_1 = df[df['engine_id'] == 1]
        # Check that RUL at cycle 1 > RUL at last cycle
        assert engine_1.iloc[0]['RUL'] > engine_1.iloc[-1]['RUL']

    def test_cap_rul(self):
        """Capping should limit RUL to specified value."""
        from src.data_loader import cap_rul
        df = pd.DataFrame({'RUL': [10, 50, 100, 200, 300]})

        capped = cap_rul(df, cap_value=125)
        assert capped['RUL'].max() == 125
        assert capped['RUL'].min() == 10
        assert list(capped['RUL']) == [10, 50, 100, 125, 125]

    def test_cap_rul_does_not_modify_original(self):
        """cap_rul should return a copy, not modify original."""
        from src.data_loader import cap_rul
        df = pd.DataFrame({'RUL': [200, 300]})
        _ = cap_rul(df, cap_value=125)
        assert df['RUL'].max() == 300  # original unchanged


class TestFeatures:
    """Tests for feature engineering."""

    def test_feature_count_increases(self):
        """Feature engineering should create more columns than original."""
        from src.data_loader import load_train_data, cap_rul
        from src.features import build_features, get_feature_columns

        df = load_train_data()
        df = cap_rul(df)
        # Use only 2 engines for speed
        small = df[df['engine_id'].isin([1, 2])]

        featured = build_features(small)
        feature_cols = get_feature_columns(featured)

        assert len(feature_cols) > 20, "Should have many engineered features"

    def test_no_rows_lost(self):
        """Feature engineering should not lose any rows."""
        from src.data_loader import load_train_data, cap_rul
        from src.features import build_features

        df = load_train_data()
        df = cap_rul(df)
        small = df[df['engine_id'] == 1]

        featured = build_features(small)
        assert len(featured) == len(small), "Row count should stay the same"

    def test_no_nans_after_features(self):
        """No NaN values should remain after feature engineering."""
        from src.data_loader import load_train_data, cap_rul
        from src.features import build_features

        df = load_train_data()
        df = cap_rul(df)
        small = df[df['engine_id'] == 1]

        featured = build_features(small)
        assert featured.isna().sum().sum() == 0, "No NaN values allowed"


class TestHealthStatus:
    """Tests for health status classification."""

    def test_healthy_status(self):
        from src.predict import RULPredictor
        predictor = RULPredictor.__new__(RULPredictor)
        result = predictor.get_health_status(100)
        assert result['status'] == 'HEALTHY'
        assert result['urgency'] == 1

    def test_warning_status(self):
        from src.predict import RULPredictor
        predictor = RULPredictor.__new__(RULPredictor)
        result = predictor.get_health_status(60)
        assert result['status'] == 'WARNING'
        assert result['urgency'] == 2

    def test_critical_status(self):
        from src.predict import RULPredictor
        predictor = RULPredictor.__new__(RULPredictor)
        result = predictor.get_health_status(25)
        assert result['status'] == 'CRITICAL'
        assert result['urgency'] == 3

    def test_danger_status(self):
        from src.predict import RULPredictor
        predictor = RULPredictor.__new__(RULPredictor)
        result = predictor.get_health_status(5)
        assert result['status'] == 'DANGER'
        assert result['urgency'] == 4

    def test_boundary_values(self):
        """Test exact boundary values."""
        from src.predict import RULPredictor
        predictor = RULPredictor.__new__(RULPredictor)

        assert predictor.get_health_status(80)['status'] == 'WARNING'   # 80 is NOT > 80
        assert predictor.get_health_status(81)['status'] == 'HEALTHY'
        assert predictor.get_health_status(40)['status'] == 'CRITICAL'  # 40 is NOT > 40
        assert predictor.get_health_status(15)['status'] == 'DANGER'    # 15 is NOT > 15
        assert predictor.get_health_status(0)['status'] == 'DANGER'


class TestModelPrediction:
    """Tests for the full prediction pipeline."""

    def test_model_files_exist(self):
        """Model and feature files should exist after training."""
        assert os.path.exists('models/best_model.pkl'), "Model file missing"
        assert os.path.exists('models/feature_columns.json'), "Feature file missing"

    def test_predictor_loads(self):
        """Predictor should load without errors."""
        from src.predict import RULPredictor
        predictor = RULPredictor()
        assert predictor.model is not None
        assert len(predictor.feature_columns) > 0

    def test_predictions_in_range(self):
        """All predictions should be between 0 and 125."""
        from src.data_loader import load_train_data, cap_rul
        from src.features import build_features, get_feature_columns
        from src.predict import RULPredictor

        df = load_train_data()
        df = cap_rul(df)
        small = df[df['engine_id'] == 1]
        featured = build_features(small)
        feature_cols = get_feature_columns(featured)

        predictor = RULPredictor()
        preds = predictor.predict(featured[feature_cols])

        assert np.all(preds >= 0), "Predictions should be >= 0"
        assert np.all(preds <= 125), "Predictions should be <= 125"
