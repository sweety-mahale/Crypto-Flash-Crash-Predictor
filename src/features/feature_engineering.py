"""
Feature Engineering - Data-driven based on EDA findings:
1. Price windows: 3, 6, 12, 24 candles (skip 1-candle, weak)
2. Volatility: STRONG signal (1.77x), multiple windows
3. Hour features: STRONG signal (7.8x difference)
4. Volume: ONE feature only (weak signal, 1.13x)
5. Technical indicators: RSI, MACD, Bollinger
Total: ~25 focused features
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:

    def __init__(self):
        self.feature_names = []
        # Each candle = 5 minutes
        # EDA: best windows are longer (12-24 candles)
        self.price_windows  = [3, 6, 12, 24]   # 15, 30, 60, 120 min
        self.vol_windows    = [3, 6, 12]        # 15, 30, 60 min
        self.volume_window  = 12                # 1 volume feature only

    # ─────────────────────────────────────────────
    # MASTER METHOD
    # ─────────────────────────────────────────────
    def create_all_features(self, df):
        logger.info("Starting feature engineering...")
        df = df.copy()

        df = self._price_features(df)       # ~10 features
        df = self._volatility_features(df)  # ~5 features  ← STRONG
        df = self._volume_features(df)      # ~2 features  ← WEAK, keep minimal
        df = self._technical_indicators(df) # ~5 features
        df = self._time_features(df)        # ~6 features  ← STRONG
        df = self._lag_features(df)         # ~4 features

        before = len(df)
        df = df.dropna()
        logger.info(f"Dropped {before - len(df)} NaN rows (rolling warmup)")
        logger.info(f"Final: {len(df)} rows, {len(self.feature_names)} features")
        return df

    # ─────────────────────────────────────────────
    # 1. PRICE FEATURES
    # EDA: 24-candle window best (corr=0.059)
    # Skip 1-candle (corr=0.007, noise)
    # ─────────────────────────────────────────────
    def _price_features(self, df):
        # Price changes - only windows EDA confirmed useful
        for w in self.price_windows:
            name = f'price_change_{w}c'
            df[name] = df['close'].pct_change(w)
            self.feature_names.append(name)

        # Momentum: fast vs slow window
        # If short-term drops faster than long-term → crash signal
        df['price_momentum'] = df['price_change_3c'] - df['price_change_12c']
        df['price_acceleration'] = df['price_change_3c'] - df['price_change_3c'].shift(3)
        self.feature_names += ['price_momentum', 'price_acceleration']

        # Candle body size (high-low range / close)
        # Larger candle = more volatility = potential crash
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        self.feature_names.append('hl_spread')

        # Distance from recent high (how far has price fallen?)
        df['dist_from_high'] = (df['close'] - df['high'].rolling(12).max()) / df['close']
        self.feature_names.append('dist_from_high')

        logger.info(f"Price features: {len([f for f in self.feature_names if 'price' in f or 'hl_' in f or 'dist_' in f or 'momentum' in f or 'accel' in f])}")
        return df

    # ─────────────────────────────────────────────
    # 2. VOLATILITY FEATURES  ← STRONGEST SIGNAL
    # EDA: 1.77x higher before crash
    # This is our most important feature group
    # ─────────────────────────────────────────────
    def _volatility_features(self, df):
        returns = df['close'].pct_change()

        for w in self.vol_windows:
            name = f'volatility_{w}c'
            df[name] = returns.rolling(w).std()
            self.feature_names.append(name)

        # Volatility ratio: short/long
        # If recent volatility spikes vs historical → crash risk
        df['volatility_ratio'] = df['volatility_3c'] / (df['volatility_12c'] + 1e-8)
        self.feature_names.append('volatility_ratio')

        # Volatility change (is it accelerating?)
        df['volatility_change'] = df['volatility_3c'] - df['volatility_3c'].shift(3)
        self.feature_names.append('volatility_change')

        logger.info(f"Volatility features created: {len(self.vol_windows) + 2}")
        return df

    # ─────────────────────────────────────────────
    # 3. VOLUME FEATURES  ← WEAK SIGNAL
    # EDA: only 1.13x higher during crashes
    # Keep minimal - just 1 normalized feature
    # ─────────────────────────────────────────────
    def _volume_features(self, df):
        # Single volume spike feature
        vol_ma = df['volume'].rolling(self.volume_window).mean()
        df['volume_spike'] = df['volume'] / (vol_ma + 1e-8)
        self.feature_names.append('volume_spike')

        # Log volume (reduces skewness)
        df['volume_log'] = np.log1p(df['volume'])
        self.feature_names.append('volume_log')

        logger.info("Volume features created: 2 (minimal - EDA showed weak signal)")
        return df

    # ─────────────────────────────────────────────
    # 4. TECHNICAL INDICATORS
    # Standard ML features for price prediction
    # ─────────────────────────────────────────────
    def _technical_indicators(self, df):
        # RSI (overbought > 70, oversold < 30)
        df['rsi_14'] = self._rsi(df['close'], 14)
        self.feature_names.append('rsi_14')

        # MACD histogram (trend reversal signal)
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        df['macd_hist'] = macd - signal
        self.feature_names.append('macd_hist')

        # Bollinger Band position (0=lower band, 1=upper band)
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        self.feature_names += ['bb_position', 'bb_width']

        logger.info("Technical indicators created: 5 (RSI, MACD, BB x2)")
        return df

    # ─────────────────────────────────────────────
    # 5. TIME FEATURES  ← STRONG SIGNAL
    # EDA: 7.8x crash rate difference by hour!
    # 14:00 UTC (US open) = highest risk
    # 7:00 UTC (Asian) = lowest risk
    # ─────────────────────────────────────────────
    def _time_features(self, df):
        hour = df['timestamp'].dt.hour
        dow  = df['timestamp'].dt.dayofweek

        # Cyclical encoding (hour 23 and 0 are close, not far)
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin']  = np.sin(2 * np.pi * dow / 7)
        df['dow_cos']  = np.cos(2 * np.pi * dow / 7)
        self.feature_names += ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

        # EDA-derived binary features
        # 13-16 UTC = US market hours = 4-5% crash rate
        df['is_us_market'] = hour.between(13, 16).astype(int)
        # 5-9 UTC = Asian hours = 0.7-1% crash rate
        df['is_asian_hours'] = hour.between(5, 9).astype(int)
        self.feature_names += ['is_us_market', 'is_asian_hours']

        logger.info("Time features created: 6 (hour sin/cos, dow sin/cos, US/Asian binary)")
        return df

    # ─────────────────────────────────────────────
    # 6. LAG FEATURES
    # Recent history of key signals
    # Focus on volatility (strong) not volume (weak)
    # ─────────────────────────────────────────────
    def _lag_features(self, df):
        # Lag volatility (was it already high?)
        for lag in [1, 3]:
            name = f'volatility_3c_lag{lag}'
            df[name] = df['volatility_3c'].shift(lag)
            self.feature_names.append(name)

        # Lag price change (recent momentum direction)
        for lag in [1, 3]:
            name = f'price_change_6c_lag{lag}'
            df[name] = df['price_change_6c'].shift(lag)
            self.feature_names.append(name)

        logger.info("Lag features created: 4")
        return df

    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def get_feature_names(self):
        return self.feature_names


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs('data/features', exist_ok=True)

    df = pd.read_csv('data/processed/btc_labeled.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Apply fixes from EDA
    df = df.drop(columns=['num_trades','buy_volume',
                          'sell_volume','buy_sell_ratio'], errors='ignore')
    df['volume'] = df['volume_btc'].replace(0, np.nan).ffill().bfill()
    df = df.drop(columns=['volume_btc'], errors='ignore')

    # Relabel with EDA-informed threshold
    future = df['close'].pct_change(3).shift(-3)
    df['crash'] = (future <= -0.005).astype(int)
    df['future_return'] = future.fillna(0)

    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)

    # Save
    df_features.to_csv('data/features/btc_features.csv', index=False)

    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Rows:     {len(df_features):,}")
    print(f"Features: {len(fe.feature_names)}")
    print(f"Crashes:  {df_features['crash'].sum()} ({df_features['crash'].mean()*100:.2f}%)")
    print(f"\nFeature list:")
    for i, f in enumerate(fe.feature_names, 1):
        print(f"  {i:2d}. {f}")
    print(f"\n✅ Saved: data/features/btc_features.csv")