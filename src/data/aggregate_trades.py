"""
Aggregate raw trade data into OHLCV candles and merge with historical data
"""

import pandas as pd
import glob
import os
import logging
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_trades(raw_path='data/raw/'):
    """Load all collected trade CSV files"""
    all_files = glob.glob(os.path.join(raw_path, '*_2*.csv'))
    
    # Exclude historical file
    all_files = [f for f in all_files if 'historical' not in f]
    
    if not all_files:
        logger.warning("No live trade files found!")
        return None
    
    logger.info(f"Found {len(all_files)} trade files")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if not df.empty and len(df.columns) > 1:  # FIX: skip empty files
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
    
    if not dfs:
        return None
    
    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.sort_values('timestamp').drop_duplicates()
    df_raw = df_raw.set_index('timestamp')
    
    logger.info(f"Total trades loaded: {len(df_raw):,}")
    logger.info(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")
    
    return df_raw


def aggregate_to_ohlcv(df_raw, interval='1min'):
    """Aggregate tick data to OHLCV candles"""
    logger.info(f"Aggregating to {interval} candles...")
    
    # Real buy/sell split from is_buyer_maker
    df_raw['buy_volume'] = df_raw.apply(
        lambda x: x['quantity'] if not x['is_buyer_maker'] else 0, axis=1
    )
    df_raw['sell_volume'] = df_raw.apply(
        lambda x: x['quantity'] if x['is_buyer_maker'] else 0, axis=1
    )
    
    ohlcv = df_raw['price'].resample(interval).ohlc()
    volume = df_raw['quantity'].resample(interval).sum()
    num_trades = df_raw['quantity'].resample(interval).count()
    buy_vol = df_raw['buy_volume'].resample(interval).sum()
    sell_vol = df_raw['sell_volume'].resample(interval).sum()
    
    df_candles = ohlcv.copy()
    df_candles['volume'] = volume
    df_candles['num_trades'] = num_trades
    df_candles['buy_volume'] = buy_vol
    df_candles['sell_volume'] = sell_vol
    df_candles['buy_sell_ratio'] = (
        df_candles['buy_volume'] / (df_candles['sell_volume'] + 1e-8)
    )
    
    # FIX: Convert volume to BTC (live collector gives BTC quantity directly)
    # No conversion needed - quantity is already in BTC
    
    df_candles = df_candles.dropna()
    df_candles = df_candles.reset_index()
    
    logger.info(f"Generated {len(df_candles):,} candles")
    return df_candles


def download_historical(symbol='BTCUSDT', days=30):
    """
    Download historical OHLCV from Yahoo Finance
    Note: Yahoo gives 1m data for last 7 days only
          For >7 days it switches to 5m candles
    """
    
    yf_symbol = 'BTC-USD'
    
    # Yahoo Finance limits:
    # 1m interval → max 7 days
    # 5m interval → max 60 days  
    # 1h interval → max 730 days
    if days <= 7:
        interval = '1m'
    elif days <= 60:
        interval = '5m'
    else:
        interval = '1h'
    
    logger.info(f"Downloading {days} days at {interval} interval from Yahoo Finance...")
    
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        df = yf.download(
            yf_symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False
        )
        
        if df.empty:
            raise ValueError("No data returned from Yahoo Finance")
        
        # Fix MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        
        # Rename columns
        col_map = {
            df.columns[0]: 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=col_map)
        
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        # FIX: Convert USD volume to BTC
        df['volume_btc'] = df['volume'] / df['close']
        
        # FIX: Drop fake buy/sell columns
        # We don't have real data, so don't pretend we do
        df['num_trades'] = 0  # Unknown from Yahoo
        df['buy_volume'] = np.nan  # Unknown - mark as NaN not fake
        df['sell_volume'] = np.nan
        df['buy_sell_ratio'] = np.nan
        
        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close',
                 'volume', 'volume_btc', 'num_trades',
                 'buy_volume', 'sell_volume', 'buy_sell_ratio']]
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'volume_btc']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        logger.info(f"✅ Downloaded {len(df):,} candles ({interval} interval)")
        logger.info(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        logger.info(f"   Volume (BTC) mean: {df['volume_btc'].mean():.2f} BTC/candle")
        
        # FIX: Warn user clearly about data source
        logger.warning("⚠️  Using Yahoo Finance data. buy_sell_ratio = NaN (not available)")
        logger.warning("⚠️  For production: use VPN + Binance API for complete data")
        
        return df, interval  # FIX: return interval so label_crashes can use it
    
    except Exception as e:
        # FIX: Don't silently fall back to mock - raise error
        logger.error(f"❌ Failed to download from Yahoo Finance: {e}")
        logger.error("Please check your internet connection")
        raise  # Don't use mock data silently


def label_crashes(df, interval='1m', threshold=-0.03):
    """
    Label crash events - window adjusted based on candle interval
    
    Args:
        df: DataFrame with 'close' column
        interval: Candle interval ('1m', '5m', '1h')
        threshold: Price drop threshold
    """
    # FIX: Adjust window based on candle interval
    # Goal: detect crashes within 10-minute window
    window_map = {
        '1m':  10,   # 10 candles = 10 minutes
        '5m':  3,    # 3 candles = 15 minutes  
        '1h':  1,    # 1 candle = 1 hour (too coarse, but only option)
    }
    window = window_map.get(interval, 10)
    
    logger.info(f"Labeling crashes (interval={interval}, window={window}, threshold={threshold})...")
    
    df = df.copy()
    df['future_return'] = df['close'].pct_change(periods=window).shift(-window)
    df['crash'] = (df['future_return'] <= threshold).astype(int)
    df['crash'] = df['crash'].fillna(0).astype(int)
    df['future_return'] = df['future_return'].fillna(0.0)
    
    crash_rate = df['crash'].mean() * 100
    logger.info(f"Crash labeling complete.")
    logger.info(f"   Crashes found: {df['crash'].sum():,} ({crash_rate:.2f}%)")
    
    # Warn if crash rate is bad
    if crash_rate < 0.5:
        logger.warning("⚠️  Crash rate very low (<0.5%). Consider relaxing threshold.")
    elif crash_rate > 15:
        logger.warning("⚠️  Crash rate very high (>15%). Consider tightening threshold.")
    else:
        logger.info("✅ Crash rate looks healthy for model training")
    
    return df


def combine_and_save(df_historical, df_live, output_path):
    """Merge historical and live data"""
    logger.info("Combining historical and live data...")
    
    if df_historical is not None and not df_historical.empty:
        df_historical['timestamp'] = pd.to_datetime(
            df_historical['timestamp']).dt.tz_localize(None)
    
    if df_live is not None and not df_live.empty:
        df_live['timestamp'] = pd.to_datetime(
            df_live['timestamp']).dt.tz_localize(None)
    
    if df_live is not None and not df_live.empty:
        df_combined = pd.concat([df_historical, df_live], ignore_index=True)
    else:
        df_combined = df_historical
    
    df_combined = df_combined.drop_duplicates(subset='timestamp')
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    
    df_combined.to_csv(output_path, index=False)
    
    logger.info(f"✅ Combined dataset saved to {output_path}")
    logger.info(f"   Total candles: {len(df_combined):,}")
    logger.info(f"   Date range: {df_combined['timestamp'].min()} → {df_combined['timestamp'].max()}")
    
    return df_combined


if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Download historical data
    # 7 days = 1m candles (most granular, best for model)
    # 30 days = 5m candles (less granular but more data)
    df_historical, interval = download_historical(days=60)
    df_historical.to_csv('data/raw/BTCUSDT_historical.csv', index=False)
    
    # Step 2: Aggregate live trades (1m candles with real buy/sell data)
    df_raw = load_raw_trades()
    if df_raw is not None:
        df_live = aggregate_to_ohlcv(df_raw, interval='5min')
        df_live.to_csv('data/processed/btc_live_ohlcv.csv', index=False)
        logger.info(f"Live data: {len(df_live)} candles with REAL buy/sell ratio ✅")
    else:
        logger.info("No live trade data. Using historical only.")
        df_live = pd.DataFrame()
    
    # Step 3: Combine
    df_final = combine_and_save(
        df_historical,
        df_live,
        output_path='data/processed/btc_combined.csv'
    )
    
    # Step 4: Label crashes (interval-aware)
    df_labeled = label_crashes(df_final, interval=interval, threshold=-0.005)
    df_labeled.to_csv('data/processed/btc_labeled.csv', index=False)
    
    # Step 5: Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Candle interval:  {interval}")
    print(f"Total candles:    {len(df_labeled):,}")
    print(f"Date range:       {df_labeled['timestamp'].min()} → {df_labeled['timestamp'].max()}")
    print(f"Crashes labeled:  {df_labeled['crash'].sum():,} ({df_labeled['crash'].mean()*100:.2f}%)")
    print(f"Columns:          {list(df_labeled.columns)}")
    print(f"\nData sources:")
    print(f"  Historical:     Yahoo Finance ({interval} candles)")
    print(f"  Live:           Binance WebSocket ({'yes' if len(df_live) > 0 else 'no'})")
    print(f"\n✅ Ready for EDA → data/processed/btc_labeled.csv")
    print("="*60)
