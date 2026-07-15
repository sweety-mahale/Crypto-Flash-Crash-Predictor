# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from src.features.feature_engineering import FeatureEngineer
from src.models.online_models import OnlineCrashPredictor

def test_feature_engineering_insufficient_data():
    import pandas as pd
    from datetime import datetime, timedelta
    fe = FeatureEngineer()
    # Fewer than 60 rows
    timestamps = [datetime.now() - timedelta(minutes=50-i) for i in range(50)]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [60000.0 + i for i in range(50)],
        'high': [60000.0 + i for i in range(50)],
        'low': [60000.0 + i for i in range(50)],
        'close': [60000.0 + i for i in range(50)],
        'volume': [100.0] * 50
    })
    df_feat = fe.create_all_features(df)
    # Should be empty since we need at least 60 records for rolling windows (like dist_from_high_60)
    assert df_feat.empty

def test_feature_engineering_computation():
    import pandas as pd
    from datetime import datetime, timedelta
    fe = FeatureEngineer()
    # 100 rows
    timestamps = [datetime.now() - timedelta(minutes=100-i) for i in range(100)]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [60000.0 + (i * 10) for i in range(100)],
        'high': [60000.0 + (i * 10) for i in range(100)],
        'low': [60000.0 + (i * 10) for i in range(100)],
        'close': [60000.0 + (i * 10) for i in range(100)],
        'volume': [100.0] * 100
    })
    df_feat = fe.create_all_features(df)
    assert not df_feat.empty
    latest_feat = df_feat.iloc[-1].to_dict()
    assert 'price_change_1min' in latest_feat
    assert 'volatility_5min' in latest_feat
    assert 'volume_spike_30' in latest_feat
    assert latest_feat['price_change_1min'] > 0

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
