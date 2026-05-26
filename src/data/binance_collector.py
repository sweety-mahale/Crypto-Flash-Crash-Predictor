"""
Binance WebSocket Data Collector
Collects real-time BTC price, volume, and order book data
"""

import json
import time
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance.websockets import BinanceSocketManager
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """Collects real-time crypto data from Binance"""
    
    def __init__(self, symbol='BTCUSDT', output_path='data/raw/'):
        """
        Args:
            symbol: Trading pair (default: BTCUSDT)
            output_path: Path to save collected data
        """
        self.symbol = symbol
        self.output_path = output_path
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize Binance client (no API key needed for public data)
        self.client = Client("", "")
        self.bm = BinanceSocketManager(self.client)
        
        # Data buffer
        self.data_buffer = []
        self.buffer_size = 1000  # Save to disk every 1000 records
        
        logger.info(f"Initialized collector for {symbol}")
    
    def process_message(self, msg):
        """Process incoming WebSocket message"""
        if msg.get('e') == 'error':
            logger.error(f"Error: {msg}")
            return
        
        if msg.get('e') == 'trade' or ('e' in msg and msg['e'] == 'trade'):
            # Extract trade data
            trade_data = {
                'timestamp': datetime.fromtimestamp(msg['T'] / 1000),
                'symbol': msg['s'],
                'price': float(msg['p']),
                'quantity': float(msg['q']),
                'is_buyer_maker': msg['m']  # True if sell order
            }
            
            self.data_buffer.append(trade_data)
            
            # Log every 100 trades
            if len(self.data_buffer) % 100 == 0:
                logger.info(f"Collected {len(self.data_buffer)} trades. Latest price: ${trade_data['price']}")
            
            # Save buffer when full
            if len(self.data_buffer) >= self.buffer_size:
                self.save_buffer()
    
    def save_buffer(self):
        """Save buffered data to CSV"""
        if not self.data_buffer:
            return
        
        df = pd.DataFrame(self.data_buffer)
        
        # Create filename with timestamp
        filename = os.path.join(self.output_path, f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Append to existing file or create new
        try:
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(self.data_buffer)} records to {filename}")
        
        # Clear buffer
        self.data_buffer = []
    
    def start(self):
        """Start collecting data"""
        logger.info(f"Starting data collection for {self.symbol}...")
        
        # Start trade socket
        conn_key = self.bm.start_trade_socket(self.symbol, self.process_message)
        self.bm.start()
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping data collection...")
            self.save_buffer()  # Save remaining data
            self.bm.stop_socket(conn_key)
            self.bm.close()
            logger.info("Data collection stopped")


if __name__ == "__main__":
    # Create collector
    collector = BinanceDataCollector(symbol='BTCUSDT')
    
    # Start collecting
    collector.start()
