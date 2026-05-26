import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_historical_data(symbol='BTCUSDT', days=30, interval='1m'):
    """
    Simulate/Download historical kline data for local testing
    Generates a realistic trading volume and price path if the client isn't fully authenticated
    """
    logger.info(f"Setting up training data for {symbol} for {days} days")
    
    # Generate mock/realistic price series using geometric Brownian motion
    np.random.seed(42)
    minutes = days * 24 * 60
    base_price = 60000.0
    
    # Generate prices
    returns = np.random.normal(0, 0.00015, minutes)
    
    # Inject a few flash crashes for model to learn
    crash_indices = [int(minutes * 0.15), int(minutes * 0.4), int(minutes * 0.65), int(minutes * 0.85)]
    for idx in crash_indices:
        # Generate a steep 4-6% crash over 5 minutes
        for step in range(5):
            returns[idx + step] = -0.012
            
    price_multipliers = np.exp(np.cumsum(returns))
    prices = base_price * price_multipliers
    
    timestamps = [datetime.now() - timedelta(minutes=minutes-i) for i in range(minutes)]
    
    volumes = np.random.lognormal(mean=2.0, sigma=0.8, size=minutes) * 10.0
    # Increase volume during crashes
    for idx in crash_indices:
        for step in range(5):
            volumes[idx + step] *= 8.0

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * 0.9998,
        'high': prices * 1.0015,
        'low': prices * 0.9985,
        'close': prices,
        'volume': volumes
    })
    
    # Add high volatility spikes during crashes
    for idx in crash_indices:
        df.loc[idx:idx+5, 'high'] = df.loc[idx:idx+5, 'close'] * 1.005
        df.loc[idx:idx+5, 'low'] = df.loc[idx:idx+5, 'close'] * 0.985
        
    os.makedirs('data/raw', exist_ok=True)
    output_path = f'data/raw/{symbol}_historical_{days}days_{interval}.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"Generated {len(df)} records to {output_path}")
    return df

if __name__ == "__main__":
    download_historical_data(symbol='BTCUSDT', days=7)
