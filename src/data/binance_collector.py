"""
Binance WebSocket Data Collector
Collects real-time BTC price, volume, and order book data
"""

import json
import time
import asyncio
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance import BinanceSocketManager
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
        """Save buffered data to CSV in a background thread to avoid blocking the queue"""
        if not self.data_buffer:
            return
            
        data_to_save = list(self.data_buffer)
        self.data_buffer = []
        
        def save_task(data):
            df = pd.DataFrame(data)
            filename = os.path.join(self.output_path, f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(filename, index=False)
            logger.info(f"Background thread saved {len(data)} records to {filename}")
            
        import threading
        threading.Thread(target=save_task, args=(data_to_save,), daemon=True).start()
    
    async def start_async(self):
        """Start collecting data asynchronously"""
        logger.info(f"Starting data collection for {self.symbol}...")
        
        while True:
            try:
                ts = self.bm.aggtrade_socket(self.symbol)
                async with ts as t_socket:
                    logger.info("WebSocket connected successfully.")
                    while True:
                        try:
                            # Wait for a message with a 15-second timeout to handle silent hangs
                            res = await asyncio.wait_for(t_socket.recv(), timeout=15.0)
                        except asyncio.TimeoutError:
                            logger.warning("No message received for 15 seconds. Triggering reconnection...")
                            break
                        
                        # If returned response is a dict, parse it
                        if isinstance(res, dict):
                            # Verify we have actual trade data fields before mapping
                            if res.get('T') is not None and res.get('p') is not None:
                                mapped_res = {
                                    'e': 'trade',
                                    'T': res.get('T'),
                                    's': res.get('s'),
                                    'p': res.get('p'),
                                    'q': res.get('q'),
                                    'm': res.get('m')
                                }
                                self.process_message(mapped_res)
                        elif isinstance(res, str):
                            try:
                                parsed = json.loads(res)
                                if parsed.get('T') is not None and parsed.get('p') is not None:
                                    mapped_res = {
                                        'e': 'trade',
                                        'T': parsed.get('T'),
                                        's': parsed.get('s'),
                                        'p': parsed.get('p'),
                                        'q': parsed.get('q'),
                                        'm': parsed.get('m')
                                    }
                                    self.process_message(mapped_res)
                            except Exception:
                                pass
            except asyncio.CancelledError:
                logger.info("Stopping data collection...")
                self.save_buffer()
                logger.info("Data collection stopped")
                break
            except Exception as e:
                logger.error(f"Error in connection loop: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

    def start(self):
        import asyncio
        try:
            asyncio.run(self.start_async())
        except KeyboardInterrupt:
            logger.info("Stopped by user. Saving data.")
            self.save_buffer()


if __name__ == "__main__":
    # Create collector
    collector = BinanceDataCollector(symbol='BTCUSDT')
    
    # Start collecting
    collector.start()
