import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
import websocket
import threading
from pathlib import Path

@dataclass
class OrderBookSnapshot:
    """Represents a single order book snapshot"""
    timestamp: int  # Unix timestamp in milliseconds
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'bids': [[p, q] for p, q in self.bids],
            'asks': [[p, q] for p, q in self.asks]
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            timestamp=data['timestamp'],
            symbol=data['symbol'],
            bids=[(p, q) for p, q in data['bids']],
            asks=[(p, q) for p, q in data['asks']]
        )

class OrderBookCache:
    """Manages local cache of order book data to avoid re-downloading"""
    
    def __init__(self, cache_dir: str = "./orderbook_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load the cache index mapping intervals to filenames"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save the cache index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_cache_key(self, symbol: str, start_ts: int, end_ts: int, interval_ms: int) -> str:
        """Generate a unique key for a given time interval"""
        return f"{symbol}_{start_ts}_{end_ts}_{interval_ms}"
    
    def get(self, symbol: str, start_ts: int, end_ts: int, interval_ms: int) -> Optional[List[OrderBookSnapshot]]:
        """Retrieve cached data if it exists"""
        key = self._get_cache_key(symbol, start_ts, end_ts, interval_ms)
        
        if key in self.index:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return [OrderBookSnapshot.from_dict(item) for item in data]
        return None
    
    def put(self, symbol: str, start_ts: int, end_ts: int, interval_ms: int, 
            snapshots: List[OrderBookSnapshot]):
        """Store data in cache"""
        key = self._get_cache_key(symbol, start_ts, end_ts, interval_ms)
        cache_file = self.cache_dir / f"{key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump([snap.to_dict() for snap in snapshots], f, indent=2)
        
        self.index[key] = str(cache_file)
        self._save_index()

class BinanceOrderBookCollector:
    """Collects sequential order book snapshots from Binance"""
    
    def __init__(self, cache_dir: str = "./orderbook_cache", 
                 num_levels: int = 10,
                 update_interval_ms: int = 1000):
        """
        Initialize the collector
        
        Args:
            cache_dir: Directory for caching data
            num_levels: Number of price levels per side (5, 10, or 20)
            update_interval_ms: Update interval (1000 ms for 1 second)
        """
        self.cache = OrderBookCache(cache_dir)
        self.num_levels = num_levels
        self.update_interval_ms = update_interval_ms
        self.base_url = "https://api.binance.com/api/v3"
        self.ws_url = f"wss://stream.binance.com:9443/ws"
        
    def get_order_book_snapshot_rest(self, symbol: str) -> OrderBookSnapshot:
        """Fetch a single order book snapshot using REST API"""
        response = requests.get(
            f"{self.base_url}/depth",
            params={
                'symbol': symbol.upper(),
                'limit': self.num_levels
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return OrderBookSnapshot(
            timestamp=int(time.time() * 1000),
            symbol=symbol,
            bids=[(float(price), float(qty)) for price, qty in data['bids']],
            asks=[(float(price), float(qty)) for price, qty in data['asks']]
        )
    
    def collect_historical(self, symbol: str, start_time: datetime, 
                          end_time: Optional[datetime] = None,
                          right_open: bool = True) -> List[OrderBookSnapshot]:
        """
        Collect historical order book snapshots via REST API polling
        
        Note: Binance doesn't provide historical order book data directly.
        This method polls REST API at fixed intervals which is rate-limited.
        For true historical data, you'd need to save WebSocket stream.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_time: Beginning of time interval
            end_time: End of time interval (defaults to None)
            right_open: If True, use current time as end time
        
        Returns:
            List of order book snapshots
        """
        if right_open or end_time is None:
            end_time = datetime.now()
        
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Check cache first
        cached = self.cache.get(symbol, start_ts, end_ts, self.update_interval_ms)
        if cached:
            print(f"Returning {len(cached)} cached snapshots")
            return cached
        
        print(f"Collecting {symbol} order books from {start_time} to {end_time}")
        
        snapshots = []
        current_time = start_time
        interval_seconds = self.update_interval_ms / 1000.0
        
        # Rate limiting awareness
        calls = 0
        start_collect = time.time()
        
        while current_time <= end_time:
            snapshot = self.get_order_book_snapshot_rest(symbol)
            snapshots.append(snapshot)
            
            calls += 1
            current_time += timedelta(seconds=interval_seconds)
            
            # Binance rate limit: 1200 weight per minute
            # Each depth query with limit=10 costs weight 2
            # Sleep briefly to respect rate limits
            if calls % 50 == 0:
                time.sleep(0.1)  # Small delay to avoid hitting rate limits
        
        elapsed = time.time() - start_collect
        print(f"Collected {len(snapshots)} snapshots in {elapsed:.2f}s")
        
        # Cache the results
        self.cache.put(symbol, start_ts, end_ts, self.update_interval_ms, snapshots)
        
        return snapshots

class RealTimeOrderBookStream:
    """Real-time order book stream using WebSocket with 1000ms updates"""
    
    def __init__(self, symbol: str, num_levels: int = 10, update_interval_ms: int = 1000):
        self.symbol = symbol.lower()
        self.num_levels = num_levels
        self.update_interval_ms = update_interval_ms
        self.snapshots: List[OrderBookSnapshot] = []
        self.current_snapshot: Optional[OrderBookSnapshot] = None
        self.is_running = False
        self.ws = None
        
    def get_stream_name(self) -> str:
        """Get the WebSocket stream name based on parameters"""
        # Use partial book stream with specified update interval
        return f"{self.symbol}@depth{self.num_levels}@{self.update_interval_ms}ms"
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        data = json.loads(message)
        
        snapshot = OrderBookSnapshot(
            timestamp=data['E'],  # Event time
            symbol=data['s'],
            bids=[(float(price), float(qty)) for price, qty in data['b']],
            asks=[(float(price), float(qty)) for price, qty in data['a']]
        )
        
        self.current_snapshot = snapshot
        self.snapshots.append(snapshot)
        
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.is_running = False
        
    def on_open(self, ws):
        print(f"Connected to {self.get_stream_name()}")
        
    def start(self, duration_seconds: Optional[int] = None):
        """
        Start collecting real-time order books
        
        Args:
            duration_seconds: How long to collect (None = run until stop() called)
        """
        self.is_running = True
        stream_name = self.get_stream_name()
        
        # Construct full WebSocket URL
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run WebSocket in separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
        if duration_seconds:
            time.sleep(duration_seconds)
            self.stop()
    
    def stop(self):
        """Stop collecting and close WebSocket"""
        if self.ws:
            self.ws.close()
        self.is_running = False
    
    def get_snapshots(self) -> List[OrderBookSnapshot]:
        """Return collected snapshots and clear internal buffer"""
        snapshots = self.snapshots.copy()
        self.snapshots.clear()
        return snapshots
    
    def get_latest(self) -> Optional[OrderBookSnapshot]:
        """Get the most recent snapshot"""
        return self.current_snapshot

class SequenceCollector:
    """
    Main class for collecting sequences of order books for a given time interval.
    Handles both historical (cached) and real-time data with right-open intervals.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", num_levels: int = 10, 
                 update_interval_ms: int = 1000):
        self.symbol = symbol
        self.num_levels = num_levels
        self.update_interval_ms = update_interval_ms
        self.collector = BinanceOrderBookCollector(
            num_levels=num_levels,
            update_interval_ms=update_interval_ms
        )
        self.real_time_stream: Optional[RealTimeOrderBookStream] = None
        
    def get_sequence(self, start_time: datetime, 
                    end_time: Optional[datetime] = None,
                    right_open: bool = True) -> List[OrderBookSnapshot]:
        """
        Get a sequence of order book snapshots for a given time interval.
        
        Args:
            start_time: Beginning of the interval (inclusive)
            end_time: End of the interval (inclusive if right_open=False)
            right_open: If True, uses current time as end_time (real-time)
                       If False, uses end_time as fixed end (historical)
        
        Returns:
            List of OrderBookSnapshot objects at 1000ms intervals
        
        Examples:
            # Get last 10 seconds of data (right-open, real-time)
            sequence = collector.get_sequence(
                start_time=datetime.now() - timedelta(seconds=10),
                right_open=True
            )
            
            # Get historical data from 1 hour ago to 30 minutes ago
            sequence = collector.get_sequence(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now() - timedelta(minutes=30),
                right_open=False
            )
        """
        
        if right_open:
            # Use current time as end time
            end_time = datetime.now()
            print(f"Right-open mode: collecting from {start_time} to NOW ({end_time})")
        else:
            if end_time is None:
                raise ValueError("end_time must be provided when right_open=False")
            print(f"Fixed interval mode: {start_time} to {end_time}")
        
        # For historical data (end_time is in the past), use cache/REST
        if end_time < datetime.now():
            return self.collector.collect_historical(
                self.symbol, start_time, end_time, right_open=False
            )
        
        # For intervals that include "now", we need both historical + real-time
        historical_end = end_time if end_time < datetime.now() else datetime.now()
        snapshots = []
        
        # Fetch historical part (before now)
        if historical_end > start_time:
            historical = self.collector.collect_historical(
                self.symbol, start_time, historical_end, right_open=False
            )
            snapshots.extend(historical)
        
        # If end_time is in the future or now, we need real-time data
        if end_time >= datetime.now():
            remaining_seconds = (end_time - datetime.now()).total_seconds()
            if remaining_seconds > 0:
                print(f"Waiting for real-time data: {remaining_seconds:.1f} seconds remaining")
                
                # Start real-time collection
                self.real_time_stream = RealTimeOrderBookStream(
                    self.symbol, self.num_levels, self.update_interval_ms
                )
                self.real_time_stream.start(duration_seconds=int(remaining_seconds) + 1)
                
                # Get real-time snapshots
                real_time_snapshots = self.real_time_stream.get_snapshots()
                snapshots.extend(real_time_snapshots)
                print(f"Added {len(real_time_snapshots)} real-time snapshots")
        
        return snapshots

# Example usage and helper functions
def print_order_book_summary(snapshots: List[OrderBookSnapshot], max_display: int = 5):
    """Pretty print order book summaries"""
    print(f"\n{'='*80}")
    print(f"Retrieved {len(snapshots)} order book snapshots")
    print(f"{'='*80}")
    
    for i, snap in enumerate(snapshots[:max_display]):
        dt = datetime.fromtimestamp(snap.timestamp / 1000)
        best_bid = snap.bids[0][0] if snap.bids else 0
        best_ask = snap.asks[0][0] if snap.asks else 0
        spread = best_ask - best_bid
        
        print(f"{i+1}. {dt.strftime('%H:%M:%S.%f')[:-3]} | "
              f"Best Bid: {best_bid:.2f} | Best Ask: {best_ask:.2f} | "
              f"Spread: {spread:.2f} ({spread/best_bid*100:.4f}%)")
    
    if len(snapshots) > max_display:
        print(f"... and {len(snapshots) - max_display} more")

def main():
    """Example usage demonstrating the collector"""
    
    # Initialize the collector with 1000ms updates
    collector = SequenceCollector(
        symbol="BTCUSDT",
        num_levels=10,
        update_interval_ms=1000
    )
    
    # Example 1: Get the last 5 seconds of data (right-open, real-time)
    print("\n📊 EXAMPLE 1: Right-open (last 5 seconds)")
    print("-" * 50)
    snapshots = collector.get_sequence(
        start_time=datetime.now() - timedelta(seconds=5),
        right_open=True
    )
    print_order_book_summary(snapshots)
    
    # Example 2: Get fixed historical interval (cached)
    print("\n📊 EXAMPLE 2: Fixed historical interval (will be cached)")
    print("-" * 50)
    snapshots2 = collector.get_sequence(
        start_time=datetime.now() - timedelta(seconds=30),
        end_time=datetime.now() - timedelta(seconds=20),
        right_open=False
    )
    print_order_book_summary(snapshots2)
    
    # Example 3: Run again to demonstrate caching
    print("\n📊 EXAMPLE 3: Same interval (should come from cache)")
    print("-" * 50)
    snapshots3 = collector.get_sequence(
        start_time=datetime.now() - timedelta(seconds=30),
        end_time=datetime.now() - timedelta(seconds=20),
        right_open=False
    )
    print_order_book_summary(snapshots3)

if __name__ == "__main__":
    main()