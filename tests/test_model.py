import pytest
import numpy as np
from src.features.feature_engineering import FeatureEngineer
from src.models.online_models import OnlineCrashPredictor

def test_feature_engineering_insufficient_data():
    fe = FeatureEngineer()
    # Add fewer than 60 ticks
    for i in range(50):
        fe.add_tick(60000.0 + i, 100.0)
    features = fe.compute_features()
    # Should be None since we need at least 60 ticks to form features
    assert features is None

def test_feature_engineering_computation():
    fe = FeatureEngineer()
    # Add 100 ticks
    for i in range(100):
        fe.add_tick(60000.0 + (i * 10), 100.0)
    features = fe.compute_features()
    assert features is not None
    assert 'price_change_1min' in features
    assert 'volatility_5min' in features
    assert 'volume_spike' in features
    assert features['price_change_1min'] > 0

def test_online_model_incremental_learning():
    model = OnlineCrashPredictor()
    sample_features = {
        'price_change_1min': -0.01,
        'price_change_5min': -0.03,
        'price_change_15min': -0.04,
        'volatility_5min': 0.005,
        'volume_spike': 2.5,
        'price_vs_ema_10': -0.02
    }
    
    # Predict before learning
    prob_before = model.predict_proba_one(sample_features)
    assert 0.0 <= prob_before <= 1.0
    
    # Learn
    model.learn_one(sample_features, 1) # Class 1 (Crash)
    assert model.samples_processed == 1
    
    # Predict after learning
    prob_after = model.predict_proba_one(sample_features)
    assert 0.0 <= prob_after <= 1.0
