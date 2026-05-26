"""
Feature Engineering for Crypto Crash Prediction
Creates time-series, statistical, and technical features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for crash prediction"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_features(self, df):
        """
        Create all features
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        
        Returns:
            DataFrame with all features
        """
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        
        # 1. Price-based features
        df = self._create_price_features(df)
        
        # 2. Volume-based features
        df = self._create_volume_features(df)
        
        # 3. Volatility features
        df = self._create_volatility_features(df)
        
        # 4. Technical indicators
        df = self._create_technical_indicators(df)
        
        # 5. Time-based features
        df = self._create_time_features(df)
        
        # 6. Lag features
        df = self._create_lag_features(df)
        
        # Remove rows with NaN (from rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")
        
        # Capture feature names (excluding metadata & label)
        exclude_cols = ['timestamp', 'crash', 'open', 'high', 'low', 'close', 'volume', 'future_return']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df
    
    def _create_price_features(self, df):
        """Create price-based features"""
        logger.info("Creating price features...")
        
        # Price changes (returns)
        for window in [1, 5, 10, 15, 30, 60]:
            col_name = f'price_change_{window}min'
            df[col_name] = df['close'].pct_change(periods=window)
        
        # Price momentum (difference between short and long term changes)
        df['price_momentum_5_15'] = df['price_change_5min'] - df['price_change_15min']
        df['price_momentum_5_30'] = df['price_change_5min'] - df['price_change_30min']
        
        # Price acceleration (change in returns)
        df['price_acceleration'] = df['price_change_5min'] - df['price_change_5min'].shift(5)
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Distance from recent highs/lows
        df['dist_from_high_60'] = (df['close'] - df['high'].rolling(60).max()) / df['close']
        df['dist_from_low_60'] = (df['close'] - df['low'].rolling(60).min()) / df['close']
        
        return df
    
    def _create_volume_features(self, df):
        """Create volume-based features"""
        logger.info("Creating volume features...")
        
        # Volume moving averages
        for window in [10, 30, 60]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Volume spikes (current vs average)
        for window in [10, 30, 60]:
            col_name = f'volume_spike_{window}'
            df[col_name] = df['volume'] / df[f'volume_ma_{window}']
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=30).mean()
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change(periods=5)
        
        return df
    
    def _create_volatility_features(self, df):
        """Create volatility features"""
        logger.info("Creating volatility features...")
        
        # Rolling standard deviation of returns
        for window in [5, 10, 15, 30, 60]:
            col_name = f'volatility_{window}min'
            df[col_name] = df['close'].pct_change().rolling(window=window).std()
        
        # Volatility ratios
        df['volatility_ratio_5_15'] = df['volatility_5min'] / df['volatility_15min']
        df['volatility_ratio_5_30'] = df['volatility_5min'] / df['volatility_30min']
        
        # Volatility change
        df['volatility_change'] = df['volatility_15min'] - df['volatility_15min'].shift(15)
        
        return df
    
    def _create_technical_indicators(self, df):
        """Create technical indicators (EMA, RSI, etc.)"""
        logger.info("Creating technical indicators...")
        
        # Exponential Moving Averages
        for span in [10, 20, 50]:
            col_name = f'ema_{span}'
            df[col_name] = df['close'].ewm(span=span, adjust=False).mean()
            # EMA crossover signal
            df[f'price_vs_ema_{span}'] = (df['close'] - df[col_name]) / df['close']
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        
        # MACD
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _create_time_features(self, df):
        """Create time-based features"""
        logger.info("Creating time features...")
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding (important for hours)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _create_lag_features(self, df):
        """Create lagged features"""
        logger.info("Creating lag features...")
        
        # Lag important features
        lag_features = ['price_change_5min', 'volume_spike_30', 'volatility_15min']
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 5, 10]:
                    col_name = f'{feature}_lag_{lag}'
                    df[col_name] = df[feature].shift(lag)
        
        return df
    
    # Helper functions for technical indicators
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def get_feature_names(self):
        """Get list of all feature names"""
        return self.feature_names


# Main execution
if __name__ == "__main__":
    import os
    # Load processed data
    input_path = 'data/processed/btc_labeled.csv'
    if not os.path.exists(input_path):
        # Fallback to historical if processed labeled not generated
        input_path = 'data/raw/BTCUSDT_historical_7days_1m.csv'
        
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_all_features(df)
        
        # Save
        os.makedirs('data/processed', exist_ok=True)
        output_path = 'data/processed/btc_features.csv'
        df_features.to_csv(output_path, index=False)
        
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 70)
        print(f"Total samples: {len(df_features):,}")
        print(f"Total features: {len(feature_engineer.get_feature_names())}")
        print("=" * 70)
        print(f"\n✅ Features saved to: {output_path}")
    else:
        print(f"Input path {input_path} not found. Run download_historical.py first.")
