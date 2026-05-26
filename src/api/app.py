import os
import json
import redis
import pickle
import asyncio
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Extra
from src.features.feature_engineering import FeatureEngineer
from src.models.online_models import OnlineCrashPredictor

app = FastAPI(title="Cryptocurrency Flash Crash Predictor API")

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

# Initialize global components
feature_engineer = FeatureEngineer()
model = OnlineCrashPredictor(model_type='arf', drift_detection=True)

# Load initial state if present in Redis
try:
    saved_model = redis_client.get("model:online_state")
    if saved_model:
        model = OnlineCrashPredictor.load_state(saved_model)
        print("Restored model state from Redis cache.")
except Exception as e:
    print(f"No model state found or failed to restore: {e}")

class PredictRequest(BaseModel):
    # Support arbitrary dynamic features
    class Config:
        extra = Extra.allow

@app.on_event("startup")
async def startup_event():
    # Start the background task processing Redis ticks
    asyncio.create_task(process_ticks_loop())

async def process_ticks_loop():
    """
    Consumer loop reading market ticks from Redis Stream,
    computing features, predicting crash risk, and learning labels incrementally.
    """
    print("Inference/Retraining background loop started.")
    # Local decoder redis client for reading stream fields
    r_text = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    # Ensure stream exists
    try:
        r_text.xgroup_create("stream:market", "group:ml", id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        # Group already exists
        pass

    while True:
        try:
            # Read from group
            streams = r_text.xreadgroup("group:ml", "consumer:ml", {"stream:market": ">"}, count=1, block=1000)
            if not streams:
                await asyncio.sleep(0.5)
                continue
                
            for stream_name, messages in streams:
                for msg_id, fields in messages:
                    price = float(fields['price'])
                    volume = float(fields['volume'])
                    
                    # Accumulate ticks for feature rolling windows
                    # Simple incremental bridge: build a small DataFrame for the features pipeline
                    # We store recent prices locally on feature_engineer.prices / volumes
                    if not hasattr(feature_engineer, 'prices'):
                        feature_engineer.prices = []
                        feature_engineer.volumes = []
                        feature_engineer.timestamps = []
                        
                    feature_engineer.prices.append(price)
                    feature_engineer.volumes.append(volume)
                    feature_engineer.timestamps.append(datetime.now())
                    
                    # Cap window to 1 hour (3600 seconds)
                    if len(feature_engineer.prices) > 3600:
                        feature_engineer.prices.pop(0)
                        feature_engineer.volumes.pop(0)
                        feature_engineer.timestamps.pop(0)
                    
                    # If we have enough data (at least 60 ticks), compile features
                    if len(feature_engineer.prices) >= 60:
                        df_temp = pd.DataFrame({
                            'timestamp': feature_engineer.timestamps,
                            'open': feature_engineer.prices,
                            'high': feature_engineer.prices,
                            'low': feature_engineer.prices,
                            'close': feature_engineer.prices,
                            'volume': feature_engineer.volumes
                        })
                        # Apply complete engineering pipeline
                        df_feat = feature_engineer.create_all_features(df_temp)
                        if not df_feat.empty:
                            latest_feat_row = df_feat.iloc[-1]
                            exclude_cols = ['timestamp', 'crash', 'open', 'high', 'low', 'close', 'volume', 'future_return']
                            features_dict = {col: latest_feat_row[col] for col in df_feat.columns if col not in exclude_cols}
                            
                            # Predict probability
                            risk = model.predict_proba_one(features_dict)
                            
                            # Save current prediction/features to Redis for monitoring dashboard
                            r_text.hset("metrics:latest", mapping={
                                "price": price,
                                "risk": risk,
                                "drift": int(model.drift_detected_flag),
                                "samples": model.samples_processed
                            })
                            
                            # Real-time label simulation: 1 if 5-minute price return goes below -3%.
                            y_true = 1 if features_dict.get('price_change_5min', 0.0) <= -0.03 else 0
                            model.learn_one(features_dict, y_true)
                            
                            # Periodic checkpoint (every 300 ticks)
                            if model.samples_processed % 300 == 0:
                                redis_client.set("model:online_state", model.save_state())
                            
                    # Acknowledge message
                    r_text.xack("stream:market", "group:ml", msg_id)
        except Exception as e:
            print(f"Error in background stream loop: {e}")
            await asyncio.sleep(1)

# Import pandas dynamically inside loop
import pandas as pd

@app.post("/predict")
def predict(payload: PredictRequest):
    x_dict = payload.model_dump()
    prob = model.predict_proba_one(x_dict)
    return {"crash_probability": prob}

@app.get("/metrics")
def get_metrics():
    r_text = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    latest = r_text.hgetall("metrics:latest")
    
    return {
        "status": "active",
        "samples_trained": model.samples_processed,
        "drift_detected": model.drift_detected_flag,
        "latest_features": latest
    }
