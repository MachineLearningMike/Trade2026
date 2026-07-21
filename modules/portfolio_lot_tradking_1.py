"""
Lot-Based Portfolio Management System - Complete Version
Tracks individual lots (batches) instead of averaged positions.
Preserves complete trade history for advanced strategies.
Includes: Email alerts, VaR calculations, Equity curves, Multiple report formats, Full test suite
"""

import threading
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict
import json
import pickle
from pathlib import Path
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= Configuration =============

class Config:
    """Central configuration for the portfolio system."""
    
    # Risk limits
    MAX_POSITION_SIZE_PCT = 0.30
    MAX_EXPOSURE_PCT = 0.70
    MAX_DRAWDOWN_PCT = 0.20
    MAX_CONCENTRATION_PCT = 0.30
    DAILY_LOSS_LIMIT = 0.05
    RISK_PER_TRADE = 0.01
    MAX_LOTS_PER_SYMBOL = 10
    
    # Performance
    RISK_FREE_RATE = 0.02
    SYNC_INTERVAL = 1.0
    SLIPPAGE_BUFFER = 0.001
    BASE_CURRENCY = "USD"
    INITIAL_EQUITY = 10000.0
    LOT_ALLOCATION_STRATEGY = "FIFO"  # FIFO, LIFO, HIFO, LOFO
    
    # Paths
    REPORT_DIR = Path("reports")
    STATE_FILE = "portfolio_state.pkl"
    
    # Email (set these in environment or config)
    EMAIL_ENABLED = False
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_USER = None
    EMAIL_PASSWORD = None
    EMAIL_RECIPIENTS = []
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}


# ============= Enums =============

class OrderSide(Enum): BUY = "buy"; SELL = "sell"
class OrderType(Enum): MARKET = "market"; LIMIT = "limit"; STOP_LOSS = "stop_loss"; TAKE_PROFIT = "take_profit"
class OrderStatus(Enum): PENDING = "pending"; SUBMITTED = "submitted"; OPEN = "open"; FILLED = "filled"; CANCELLED = "cancelled"; REJECTED = "rejected"
class OrderTimeInForce(Enum): GTC = "gtc"; IOC = "ioc"; FOK = "fok"; DAY = "day"
class RiskLevel(Enum): LOW = "low"; MEDIUM = "medium"; HIGH = "high"; CRITICAL = "critical"
class AlertSeverity(Enum): INFO = "info"; WARNING = "warning"; ERROR = "error"; CRITICAL = "critical"
class LotStatus(Enum): OPEN = "open"; PARTIALLY_CLOSED = "partially_closed"; CLOSED = "closed"
class ReportFormat(Enum): JSON = "json"; CSV = "csv"; HTML = "html"


# ============= Helper =============

def to_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (datetime, date)): return obj.isoformat()
    if isinstance(obj, Enum): return obj.value
    if hasattr(obj, 'to_dict'): return obj.to_dict()
    if isinstance(obj, (list, tuple)): return [to_serializable(i) for i in obj]
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    return obj


# ============= Exchange Interface =============

class ExchangeInterface(ABC):
    """Abstract interface for exchange adapters."""
    
    @abstractmethod
    def place_order(self, order: 'Order') -> Tuple[bool, Optional[str], Optional[Dict]]:
        pass
    
    @abstractmethod
    def cancel_order(self, exchange_order_id: str, symbol: str) -> bool:
        pass
    
    @abstractmethod
    def get_order_status(self, exchange_order_id: str, symbol: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        pass
    
    @abstractmethod
    def get_balances(self) -> Dict[str, Dict[str, float]]:
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_historical_ohlcv(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        """Get historical OHLCV data for risk calculations."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        pass


# ============= Data Classes =============

@dataclass
class Trade:
    """Represents a single executed trade."""
    trade_id: str
    exchange: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    order_id: str
    fee: float = 0.0
    fee_currency: str = ""
    commission_asset: str = ""
    commission_paid: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.side, str): self.side = OrderSide(self.side)
    
    @property
    def value(self) -> float: return self.quantity * self.price
    @property
    def is_buy(self) -> bool: return self.side == OrderSide.BUY
    @property
    def is_sell(self) -> bool: return self.side == OrderSide.SELL
    
    def to_dict(self) -> Dict: return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['side'] = OrderSide(data['side'])
        return cls(**data)


@dataclass
class Balance:
    """Currency balance on an exchange."""
    exchange: str
    currency: str
    total: float = 0.0
    locked: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def free(self) -> float: return self.total - self.locked
    
    def update(self, total: float, locked: float):
        self.total = total
        self.locked = locked
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict: return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data):
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        return cls(**data)


@dataclass
class Order:
    """Represents an order (active or historical)."""
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    internal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    client_order_id: Optional[str] = None
    trades: List[Trade] = field(default_factory=list)
    reject_reason: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.side, str): self.side = OrderSide(self.side)
        if isinstance(self.order_type, str): self.order_type = OrderType(self.order_type)
        if isinstance(self.time_in_force, str): self.time_in_force = OrderTimeInForce(self.time_in_force)
        if isinstance(self.status, str): self.status = OrderStatus(self.status)
        if self.remaining_quantity == 0 and self.quantity > 0: self.remaining_quantity = self.quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.OPEN]
    
    @property
    def is_done(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
    
    def update_from_exchange(self, exchange_data: Dict[str, Any]):
        if 'status' in exchange_data:
            self.status = OrderStatus(exchange_data['status'])
        if 'filled' in exchange_data:
            self.filled_quantity = float(exchange_data['filled'])
        if 'remaining' in exchange_data:
            self.remaining_quantity = float(exchange_data['remaining'])
        if 'avg_price' in exchange_data and exchange_data['avg_price'] is not None:
            self.average_fill_price = float(exchange_data['avg_price'])
        self.updated_at = datetime.now()
        if self.status == OrderStatus.FILLED:
            self.closed_at = datetime.now()
    
    def to_dict(self) -> Dict: return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data):
        for field in ['created_at', 'submitted_at', 'updated_at', 'closed_at']:
            if data.get(field): data[field] = datetime.fromisoformat(data[field])
        data['side'] = OrderSide(data['side'])
        data['order_type'] = OrderType(data['order_type'])
        data['time_in_force'] = OrderTimeInForce(data['time_in_force'])
        data['status'] = OrderStatus(data['status'])
        return cls(**data)


# ============= Lot System =============

@dataclass
class Lot:
    """
    A lot represents a batch of assets bought at a specific price and time.
    Preserves all entry information for advanced strategies.
    """
    lot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exchange: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    remaining_quantity: float = 0.0
    status: LotStatus = LotStatus.OPEN
    closed_quantity: float = 0.0
    closed_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    order_id: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.side, str): self.side = OrderSide(self.side)
        if isinstance(self.status, str): self.status = LotStatus(self.status)
        if self.remaining_quantity == 0 and self.quantity > 0:
            self.remaining_quantity = self.quantity
    
    @property
    def is_closed(self) -> bool:
        return self.status == LotStatus.CLOSED
    
    @property
    def closed_pnl_percentage(self) -> float:
        if self.entry_price == 0 or self.closed_quantity == 0:
            return 0.0
        return (self.realized_pnl / (self.closed_quantity * self.entry_price)) * 100
    
    @property
    def value_at_entry(self) -> float:
        return self.quantity * self.entry_price
    
    @property
    def remaining_value(self) -> float:
        return self.remaining_quantity * self.entry_price
    
    @property
    def holding_days(self) -> int:
        if self.closed_time:
            return (self.closed_time - self.entry_time).days
        return (datetime.now() - self.entry_time).days
    
    def close_partial(self, quantity: float, exit_price: float, fee: float = 0.0) -> float:
        """Close a portion of the lot and return realized PnL."""
        if quantity <= 0 or quantity > self.remaining_quantity:
            raise ValueError(f"Cannot close {quantity} from lot with {self.remaining_quantity} remaining")
        
        entry_cost = quantity * self.entry_price
        exit_value = quantity * exit_price
        pnl = exit_value - entry_cost - fee
        
        self.remaining_quantity -= quantity
        self.closed_quantity += quantity
        self.realized_pnl += pnl
        
        if self.remaining_quantity <= 0:
            self.status = LotStatus.CLOSED
            self.closed_time = datetime.now()
        else:
            self.status = LotStatus.PARTIALLY_CLOSED
        
        return pnl
    
    def to_dict(self) -> Dict:
        return {
            'lot_id': self.lot_id,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'remaining_quantity': self.remaining_quantity,
            'status': self.status.value,
            'closed_quantity': self.closed_quantity,
            'closed_time': self.closed_time.isoformat() if self.closed_time else None,
            'realized_pnl': self.realized_pnl,
            'order_id': self.order_id,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data):
        data['side'] = OrderSide(data['side'])
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        data['status'] = LotStatus(data['status'])
        if data.get('closed_time'):
            data['closed_time'] = datetime.fromisoformat(data['closed_time'])
        return cls(**data)


@dataclass
class PositionV2:
    """
    Enhanced position that tracks individual lots instead of averaging.
    """
    symbol: str
    exchange: str
    lots: List[Lot] = field(default_factory=list)
    last_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def total_quantity(self) -> float:
        return sum(lot.remaining_quantity for lot in self.lots if not lot.is_closed)
    
    @property
    def total_cost_basis(self) -> float:
        return sum(lot.remaining_quantity * lot.entry_price for lot in self.lots if not lot.is_closed)
    
    @property
    def average_cost(self) -> float:
        qty = self.total_quantity
        if qty == 0: return 0.0
        return self.total_cost_basis / qty
    
    @property
    def market_value(self) -> float:
        return self.total_quantity * self.last_price
    
    @property
    def unrealized_pnl(self) -> float:
        return sum(lot.remaining_quantity * (self.last_price - lot.entry_price) for lot in self.lots if not lot.is_closed)
    
    @property
    def realized_pnl(self) -> float:
        return sum(lot.realized_pnl for lot in self.lots if lot.is_closed)
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def lot_count(self) -> int:
        return len([l for l in self.lots if not l.is_closed])
    
    def get_lots_by_status(self, status: LotStatus) -> List[Lot]:
        return [l for l in self.lots if l.status == status]
    
    def get_lots_by_strategy(self, tag: str) -> List[Lot]:
        return [l for l in self.lots if tag in l.tags]
    
    def get_open_lots_sorted(self, strategy: str = "FIFO") -> List[Lot]:
        open_lots = [l for l in self.lots if not l.is_closed]
        if strategy == "FIFO":
            return sorted(open_lots, key=lambda l: l.entry_time)
        elif strategy == "LIFO":
            return sorted(open_lots, key=lambda l: l.entry_time, reverse=True)
        elif strategy == "HIFO":
            return sorted(open_lots, key=lambda l: l.entry_price, reverse=True)
        elif strategy == "LOFO":
            return sorted(open_lots, key=lambda l: l.entry_price)
        else:
            return open_lots
    
    def get_equity_curve(self) -> List[Dict]:
        """Build equity curve from lots."""
        curve = []
        equity = Config.INITIAL_EQUITY
        
        # Sort all events (buys and sells) by time
        events = []
        for lot in self.lots:
            events.append({
                'time': lot.entry_time,
                'type': 'buy',
                'lot': lot,
                'quantity': lot.quantity,
                'price': lot.entry_price
            })
            if lot.is_closed and lot.closed_time:
                events.append({
                    'time': lot.closed_time,
                    'type': 'sell',
                    'lot': lot,
                    'quantity': lot.closed_quantity,
                    'price': lot.entry_price + (lot.realized_pnl / lot.closed_quantity)
                })
        
        events.sort(key=lambda x: x['time'])
        
        for event in events:
            if event['type'] == 'buy':
                equity -= event['quantity'] * event['price']
            else:
                equity += event['quantity'] * event['price']
            curve.append({
                'time': event['time'].isoformat(),
                'equity': equity,
                'event': event['type'],
                'symbol': self.symbol
            })
        
        return curve
    
    def add_lot(self, lot: Lot):
        self.lots.append(lot)
        self.last_update = datetime.now()
    
    def close_lots(self, quantity: float, exit_price: float, fee: float = 0.0, 
                   strategy: str = "FIFO") -> List[Tuple[Lot, float]]:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if quantity > self.total_quantity:
            raise ValueError(f"Insufficient quantity: have {self.total_quantity}, want {quantity}")
        
        remaining = quantity
        results = []
        lots_to_close = self.get_open_lots_sorted(strategy)
        
        for lot in lots_to_close:
            if remaining <= 0:
                break
            close_qty = min(remaining, lot.remaining_quantity)
            pnl = lot.close_partial(close_qty, exit_price, fee * (close_qty / quantity))
            results.append((lot, pnl))
            remaining -= close_qty
        
        self.last_update = datetime.now()
        return results
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'lots': [l.to_dict() for l in self.lots],
            'last_price': self.last_price,
            'last_update': self.last_update.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data):
        data['lots'] = [Lot.from_dict(l) for l in data['lots']]
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        return cls(**data)


@dataclass
class RiskMetrics:
    """Risk metrics with VaR and CVaR calculations."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    exposure: float = 0.0
    concentration_ratio: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> Dict: return to_serializable(asdict(self))


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_return: float = 0.0
    monthly_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    
    def to_dict(self) -> Dict: return to_serializable(asdict(self))


@dataclass
class Alert:
    """Alert message."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.INFO
    category: str = ""
    title: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[Dict] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict: return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('acknowledged_at'): data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        data['severity'] = AlertSeverity(data['severity'])
        return cls(**data)


# ============= Mock Exchange =============

class MockExchange(ExchangeInterface):
    """Mock exchange for testing with OHLCV support."""
    
    def __init__(self, name: str):
        self.name = name
        self.orders = {}
        self.balances = {
            'USD': {'free': 50000, 'locked': 0, 'total': 50000},
            'BTC': {'free': 1.0, 'locked': 0, 'total': 1.0},
            'ETH': {'free': 20.0, 'locked': 0, 'total': 20.0}
        }
        self.order_counter = 0
        self.prices = {'BTC/USD': 50000.0, 'ETH/USD': 3000.0}
        self._historical_data = {}
        self._generate_historical_data()
    
    def _generate_historical_data(self):
        """Generate mock OHLCV data."""
        for symbol in ['BTC/USD', 'ETH/USD']:
            base_price = self.prices.get(symbol, 50000)
            data = []
            for i in range(100):
                day = datetime.now() - timedelta(days=i)
                noise = np.random.normal(0, 0.02) * base_price
                close = base_price * (1 + noise * 0.1)
                data.append({
                    'timestamp': day.isoformat(),
                    'open': close * (1 - 0.005),
                    'high': close * (1 + 0.01),
                    'low': close * (1 - 0.01),
                    'close': close,
                    'volume': 1000 + np.random.randint(0, 500)
                })
            self._historical_data[symbol] = data
    
    def place_order(self, order: Order):
        self.order_counter += 1
        order_id = f"{self.name}_order_{self.order_counter}"
        self.orders[order_id] = order
        order.exchange_order_id = order_id
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        if order.side == OrderSide.BUY:
            quote = order.symbol.split('/')[-1] if '/' in order.symbol else 'USD'
            cost = order.quantity * (order.price or self.prices.get(order.symbol, 0))
            self.balances[quote]['locked'] += cost
            self.balances[quote]['free'] -= cost
        else:
            base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            self.balances[base]['locked'] += order.quantity
            self.balances[base]['free'] -= order.quantity
        
        return True, order_id, {'id': order_id, 'status': 'submitted'}
    
    def cancel_order(self, exchange_order_id: str, symbol: str) -> bool:
        if exchange_order_id in self.orders:
            order = self.orders[exchange_order_id]
            if order.status != OrderStatus.FILLED:
                if order.side == OrderSide.BUY:
                    quote = symbol.split('/')[-1] if '/' in symbol else 'USD'
                    cost = order.quantity * (order.price or self.prices.get(symbol, 0))
                    self.balances[quote]['locked'] -= cost
                    self.balances[quote]['free'] += cost
                else:
                    base = symbol.split('/')[0] if '/' in symbol else symbol
                    self.balances[base]['locked'] -= order.quantity
                    self.balances[base]['free'] += order.quantity
                order.status = OrderStatus.CANCELLED
                order.closed_at = datetime.now()
                return True
        return False
    
    def fill_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status != OrderStatus.SUBMITTED:
                return False
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.remaining_quantity = 0
            order.average_fill_price = order.price or self.prices.get(order.symbol, 0)
            order.closed_at = datetime.now()
            
            if order.side == OrderSide.BUY:
                quote = order.symbol.split('/')[-1] if '/' in order.symbol else 'USD'
                cost = order.quantity * order.average_fill_price
                self.balances[quote]['total'] -= cost
                self.balances[quote]['locked'] -= cost
                base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
                if base not in self.balances:
                    self.balances[base] = {'free': 0, 'locked': 0, 'total': 0}
                self.balances[base]['total'] += order.quantity
                self.balances[base]['free'] += order.quantity
            else:
                base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
                self.balances[base]['total'] -= order.quantity
                self.balances[base]['locked'] -= order.quantity
                quote = order.symbol.split('/')[-1] if '/' in order.symbol else 'USD'
                self.balances[quote]['total'] += order.quantity * order.average_fill_price
                self.balances[quote]['free'] += order.quantity * order.average_fill_price
            return True
        return False
    
    def get_order_status(self, exchange_order_id: str, symbol: str) -> Optional[Dict]:
        if exchange_order_id in self.orders:
            o = self.orders[exchange_order_id]
            return {'id': exchange_order_id, 'status': o.status.value, 'filled': o.filled_quantity, 
                    'remaining': o.remaining_quantity, 'avg_price': o.average_fill_price}
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        return [{'id': oid, 'symbol': o.symbol, 'status': o.status.value} 
                for oid, o in self.orders.items() if o.is_active and (symbol is None or o.symbol == symbol)]
    
    def get_balances(self) -> Dict[str, Dict[str, float]]:
        return self.balances
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        return {'last': self.prices.get(symbol, 50000)}
    
    def get_historical_ohlcv(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        data = self._historical_data.get(symbol, [])
        return data[:limit]
    
    def get_positions(self) -> List[Dict]:
        return [{'symbol': f"{c}/USD", 'quantity': b['total'], 'price': self.prices.get(f"{c}/USD", 0)}
                for c, b in self.balances.items() if c not in ['USD', 'USDT', 'USDC'] and b['total'] > 0]
    
    def update_price(self, symbol: str, price: float):
        self.prices[symbol] = price


# ============= Alert System with Email =============

class AlertSystem:
    """Alert system with email notifications."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        self.thresholds = self.config.get('thresholds', {})
        
        # Email configuration
        self.email_enabled = self.config.get('email_enabled', Config.EMAIL_ENABLED)
        self.smtp_server = self.config.get('smtp_server', Config.SMTP_SERVER)
        self.smtp_port = self.config.get('smtp_port', Config.SMTP_PORT)
        self.email_user = self.config.get('email_user', Config.EMAIL_USER)
        self.email_password = self.config.get('email_password', Config.EMAIL_PASSWORD)
        self.email_recipients = self.config.get('email_recipients', Config.EMAIL_RECIPIENTS)
        
        if self.email_enabled and not all([self.smtp_server, self.email_user, self.email_password, self.email_recipients]):
            logger.warning("Email alerts enabled but missing configuration, disabling")
            self.email_enabled = False
        
        logger.info("AlertSystem initialized with email: %s", "enabled" if self.email_enabled else "disabled")
    
    def add_alert(self, severity: AlertSeverity, category: str, title: str, message: str, data: Dict = None):
        """Add a new alert and notify handlers."""
        alert = Alert(severity=severity, category=category, title=title, message=message, data=data)
        self.alerts.append(alert)
        
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)
        logger.log(log_level, f"[{severity.value}] {title}: {message}")
        
        # Call handlers
        for handler in list(self.alert_handlers):
            try: handler(alert)
            except Exception as e: logger.error(f"Alert handler error: {e}")
        
        # Send email for critical alerts
        if self.email_enabled and severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            self._send_email(alert)
    
    def register_handler(self, handler: Callable):
        self.alert_handlers.append(handler)
    
    def get_unacknowledged_alerts(self) -> List[Alert]:
        return [a for a in self.alerts if not a.acknowledged]
    
    def acknowledge_alert(self, alert_id: str):
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert {alert_id} acknowledged")
                break
    
    def _send_email(self, alert: Alert):
        """Send email alert for critical issues."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.email_recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert Details:
--------------
Severity: {alert.severity.value}
Category: {alert.category}
Time: {alert.timestamp}

Message: {alert.message}

Data: {alert.data}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def check_portfolio_alerts(self, portfolio, risk_manager):
        """Check for portfolio alerts based on thresholds."""
        summary = portfolio.get_portfolio_summary()
        total_equity = summary['total_equity']
        
        if risk_manager.current_drawdown > self.thresholds.get('max_drawdown', 0.15):
            self.add_alert(AlertSeverity.WARNING, "Risk", "High Drawdown", 
                          f"Drawdown {risk_manager.current_drawdown:.2%} exceeds threshold")
        
        positions = portfolio.get_all_positions()
        for pos in positions:
            if pos.market_value / max(total_equity, 1) > self.thresholds.get('max_concentration', 0.25):
                self.add_alert(AlertSeverity.WARNING, "Risk", "High Concentration", 
                              f"{pos.symbol} concentration exceeds limit")
        
        # Check daily loss
        if risk_manager.daily_pnl < -total_equity * self.thresholds.get('daily_loss_limit', 0.05):
            self.add_alert(AlertSeverity.ERROR, "Risk", "Daily Loss Limit", 
                          f"Daily loss ${risk_manager.daily_pnl:.2f} exceeds limit")


# ============= Risk Manager with VaR =============

class RiskManagerV2:
    """Risk manager with lot-level awareness and VaR calculations."""
    
    def __init__(self, portfolio: 'PortfolioManagerV2', config: Dict = None):
        self.portfolio = portfolio
        self.config = config or {}
        
        # Risk limits
        self.max_position_size_pct = self.config.get('max_position_size_pct', Config.MAX_POSITION_SIZE_PCT)
        self.max_exposure_pct = self.config.get('max_exposure_pct', Config.MAX_EXPOSURE_PCT)
        self.max_lot_size_pct = self.config.get('max_lot_size_pct', 0.10)
        self.max_lots_per_symbol = self.config.get('max_lots_per_symbol', Config.MAX_LOTS_PER_SYMBOL)
        self.daily_loss_limit = self.config.get('daily_loss_limit', Config.DAILY_LOSS_LIMIT)
        self.risk_free_rate = self.config.get('risk_free_rate', Config.RISK_FREE_RATE)
        
        # Tracking
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.last_reset_day = datetime.now().date()
        
        # Returns cache for VaR
        self._returns_cache = []
        self._cache_timestamp = None
    
    def check_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        """Check if order complies with risk limits."""
        total_equity = self.portfolio.get_total_equity()
        if total_equity <= 0:
            return False, "Total equity is zero or negative"
        
        position = self.portfolio.get_position(order.exchange, order.symbol)
        order_price = order.price or self.portfolio._last_prices.get(order.symbol, 0)
        if order_price <= 0:
            return False, f"Cannot determine price for {order.symbol}"
        order_value = order.quantity * order_price
        
        # Sell order checks
        if order.side == OrderSide.SELL:
            if position and order.quantity > position.total_quantity:
                return False, f"Insufficient position: have {position.total_quantity}, want {order.quantity}"
            return True, None
        
        # Buy order checks
        if order_value > total_equity * self.max_position_size_pct:
            return False, f"Position size ${order_value:.2f} exceeds limit ${total_equity * self.max_position_size_pct:.2f}"
        
        total_exposure = sum(p.market_value for p in self.portfolio.get_all_positions())
        if total_exposure + order_value > total_equity * self.max_exposure_pct:
            return False, f"Exposure would exceed ${total_equity * self.max_exposure_pct:.2f}"
        
        if position:
            if len(position.get_lots_by_status(LotStatus.OPEN)) >= self.max_lots_per_symbol:
                return False, f"Max lots ({self.max_lots_per_symbol}) reached for {order.symbol}"
        
        if self.daily_pnl < -total_equity * self.daily_loss_limit:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        if self.current_drawdown > Config.MAX_DRAWDOWN_PCT:
            return False, f"Max drawdown {self.current_drawdown:.2%} exceeded"
        
        return True, None
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk per trade."""
        total_equity = self.portfolio.get_total_equity()
        risk_amount = total_equity * self.config.get('risk_per_trade', Config.RISK_PER_TRADE)
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0
        return min(risk_amount / risk_per_unit, (total_equity * self.max_position_size_pct) / entry_price)
    
    def update_daily_pnl(self):
        """Update daily P&L tracking."""
        if datetime.now().date() > self.last_reset_day:
            self.daily_pnl = 0.0
            self.last_reset_day = datetime.now().date()
    
    def update_drawdown(self):
        """Update current drawdown."""
        current_equity = self.portfolio.get_total_equity()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk using historical method."""
        if not returns:
            return 0.0
        returns = np.array(returns)
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if not returns:
            return 0.0
        var = self.calculate_var(returns, confidence)
        returns_below_var = [r for r in returns if r <= var]
        if not returns_below_var:
            return var
        return np.mean(returns_below_var)
    
    def get_historical_returns(self, days: int = 30) -> List[float]:
        """Get historical returns from lot trades."""
        # Use cache
        if self._returns_cache and self._cache_timestamp:
            if (datetime.now() - self._cache_timestamp).seconds < 60:
                return self._returns_cache
        
        returns = []
        trades = self.portfolio.get_trades(limit=1000)
        if not trades:
            return []
        
        # Calculate daily returns from trades
        daily_pnl = defaultdict(float)
        for trade in trades:
            if trade.is_sell:
                day = trade.timestamp.date()
                position = self.portfolio.get_position(trade.exchange, trade.symbol)
                if position and position.average_cost > 0:
                    pnl = trade.quantity * (trade.price - position.average_cost) - trade.fee
                    daily_pnl[day] += pnl
        
        # Convert to daily returns
        equity = Config.INITIAL_EQUITY
        for day in sorted(daily_pnl.keys()):
            if equity > 0:
                returns.append(daily_pnl[day] / equity)
                equity += daily_pnl[day]
        
        self._returns_cache = returns
        self._cache_timestamp = datetime.now()
        return returns
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        metrics = RiskMetrics()
        
        # VaR calculations
        returns = self.get_historical_returns()
        if returns:
            metrics.var_95 = self.calculate_var(returns, 0.95)
            metrics.var_99 = self.calculate_var(returns, 0.99)
            metrics.cvar_95 = self.calculate_cvar(returns, 0.95)
            metrics.volatility = np.std(returns) * np.sqrt(252)
            
            risk_free_rate = self.risk_free_rate / 252
            excess_returns = np.array(returns) - risk_free_rate
            if np.std(returns) > 0:
                metrics.sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
        
        # Exposure metrics
        total_equity = self.portfolio.get_total_equity()
        total_exposure = sum(p.market_value for p in self.portfolio.get_all_positions())
        metrics.exposure = total_exposure / max(total_equity, 1)
        
        positions = self.portfolio.get_all_positions()
        if positions:
            largest = max(positions, key=lambda p: p.market_value)
            metrics.concentration_ratio = largest.market_value / max(total_equity, 1)
        
        metrics.max_drawdown = self.current_drawdown
        metrics.max_drawdown_percentage = self.current_drawdown * 100
        
        # Risk level
        if metrics.var_95 < -0.02 and metrics.exposure < 0.3:
            metrics.risk_level = RiskLevel.LOW
        elif metrics.var_95 < -0.05 and metrics.exposure < 0.6:
            metrics.risk_level = RiskLevel.MEDIUM
        elif metrics.var_95 < -0.10 and metrics.exposure < 0.8:
            metrics.risk_level = RiskLevel.HIGH
        else:
            metrics.risk_level = RiskLevel.CRITICAL
        
        return metrics


# ============= Performance Analyzer =============

class PerformanceAnalyzerV2:
    """Performance analyzer with lot-level metrics and equity curves."""
    
    def __init__(self, portfolio: 'PortfolioManagerV2', initial_equity: Optional[float] = None):
        self.portfolio = portfolio
        self._initial_equity = initial_equity
        self._equity_curve_cache = None
        self._cache_timestamp = None
    
    def get_lot_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed lot performance metrics."""
        lot_stats = self.portfolio.get_lot_statistics()
        open_lots = self.portfolio.get_open_lots()
        
        if open_lots:
            now = datetime.now()
            ages = [(now - lot.entry_time).days for lot in open_lots]
            avg_age = sum(ages) / len(ages) if ages else 0
            max_age = max(ages) if ages else 0
        else:
            avg_age = 0
            max_age = 0
        
        return {
            **lot_stats,
            'avg_lot_age_days': avg_age,
            'max_lot_age_days': max_age,
            'open_lot_value': sum(lot.remaining_quantity * lot.entry_price for lot in open_lots),
            'open_lot_market_value': sum(lot.remaining_quantity * self.portfolio._last_prices.get(lot.symbol, 0) 
                                         for lot in open_lots)
        }
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate trade statistics from ledger."""
        trades = self.portfolio.get_trades()
        if len(trades) < 2:
            return {}
        
        trade_pnls = []
        for trade in trades:
            if trade.is_sell:
                position = self.portfolio.get_position(trade.exchange, trade.symbol)
                if position and position.average_cost > 0:
                    pnl = trade.quantity * (trade.price - position.average_cost) - trade.fee
                    trade_pnls.append(pnl)
        
        if not trade_pnls:
            return {}
        
        winning = [p for p in trade_pnls if p > 0]
        losing = [p for p in trade_pnls if p < 0]
        
        # Calculate consecutive wins/losses
        max_wins = max_losses = current_wins = current_losses = 0
        for pnl in trade_pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return {
            'total_trades': len(trade_pnls),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(trade_pnls) * 100 if trade_pnls else 0,
            'average_win': np.mean(winning) if winning else 0,
            'average_loss': np.mean(losing) if losing else 0,
            'largest_win': max(winning) if winning else 0,
            'largest_loss': min(losing) if losing else 0,
            'profit_factor': abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else float('inf'),
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses
        }
    
    def calculate_returns(self) -> Dict[str, float]:
        """Calculate various return metrics."""
        trades = self.portfolio.get_trades()
        if not trades:
            return {'total_return': 0.0, 'annualized_return': 0.0, 'daily_return': 0.0, 'monthly_return': 0.0}
        
        initial_equity = self._get_initial_equity()
        current_equity = self.portfolio.get_total_equity()
        total_return = (current_equity - initial_equity) / initial_equity if initial_equity > 0 else 0
        
        days = max(1, (datetime.now() - trades[0].timestamp).days)
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else total_return
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'daily_return': total_return / max(days, 1),
            'monthly_return': total_return / max(days / 30, 1)
        }
    
    def calculate_risk_adjusted_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        trades = self.portfolio.get_trades()
        if len(trades) < 2:
            return {}
        
        returns = self._calculate_daily_returns()
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        risk_free_rate = Config.RISK_FREE_RATE / 252
        excess_returns = returns_array - risk_free_rate
        
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        
        downside_returns = returns_array[returns_array < 0]
        sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        
        max_dd = self.calculate_max_drawdown()
        calmar = np.mean(returns_array) * 252 / max_dd if max_dd > 0 else 0
        
        gains = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]
        omega = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf') if len(gains) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega
        }
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        equity_curve = self._get_equity_curve()
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
        return max_dd
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        metrics = PerformanceMetrics()
        
        returns = self.calculate_returns()
        metrics.total_return = returns.get('total_return', 0)
        metrics.annualized_return = returns.get('annualized_return', 0)
        metrics.daily_return = returns.get('daily_return', 0)
        metrics.monthly_return = returns.get('monthly_return', 0)
        
        trade_stats = self.calculate_trade_statistics()
        metrics.win_rate = trade_stats.get('win_rate', 0)
        metrics.profit_factor = trade_stats.get('profit_factor', 0)
        metrics.average_win = trade_stats.get('average_win', 0)
        metrics.average_loss = trade_stats.get('average_loss', 0)
        metrics.largest_win = trade_stats.get('largest_win', 0)
        metrics.largest_loss = trade_stats.get('largest_loss', 0)
        metrics.consecutive_wins = trade_stats.get('max_consecutive_wins', 0)
        metrics.consecutive_losses = trade_stats.get('max_consecutive_losses', 0)
        
        risk_metrics = self.calculate_risk_adjusted_metrics()
        metrics.sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
        metrics.sortino_ratio = risk_metrics.get('sortino_ratio', 0)
        metrics.calmar_ratio = risk_metrics.get('calmar_ratio', 0)
        metrics.omega_ratio = risk_metrics.get('omega_ratio', 0)
        
        return metrics
    
    def _get_initial_equity(self) -> float:
        """Get initial equity."""
        if self._initial_equity is not None:
            return self._initial_equity
        
        # Calculate from portfolio
        total_pnl = self.portfolio.calculate_realized_pnl() + self.portfolio.calculate_unrealized_pnl()
        current_equity = self.portfolio.get_total_equity()
        return max(current_equity - total_pnl, Config.INITIAL_EQUITY)
    
    def _calculate_daily_returns(self) -> List[float]:
        """Calculate daily returns from equity curve."""
        equity_curve = self._get_equity_curve()
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
        return returns
    
    def _get_equity_curve(self) -> List[float]:
        """Get equity curve with caching."""
        if self._equity_curve_cache and self._cache_timestamp:
            if (datetime.now() - self._cache_timestamp).seconds < 60:
                return self._equity_curve_cache
        
        curve = [self._get_initial_equity()]
        equity = curve[0]
        
        trades = self.portfolio.get_trades()
        for trade in sorted(trades, key=lambda x: x.timestamp):
            if trade.is_sell:
                position = self.portfolio.get_position(trade.exchange, trade.symbol)
                if position and position.average_cost > 0:
                    pnl = trade.quantity * (trade.price - position.average_cost) - trade.fee
                    equity += pnl
                    curve.append(equity)
        
        self._equity_curve_cache = curve
        self._cache_timestamp = datetime.now()
        return curve


# ============= Reporting Module =============

class ReportingModuleV2:
    """Reporting with multiple formats and lot-level reports."""
    
    def __init__(self, portfolio: 'PortfolioManagerV2', risk_manager: RiskManagerV2,
                 performance_analyzer: PerformanceAnalyzerV2, alert_system: AlertSystem):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.analyzer = performance_analyzer
        self.alerts = alert_system
        self.report_dir = Config.REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_lot_report(self, format: ReportFormat = ReportFormat.JSON) -> str:
        """Generate a detailed lot report."""
        summary = self.portfolio.get_portfolio_summary()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'lot_performance': self.analyzer.get_lot_performance_metrics(),
            'trade_statistics': self.analyzer.calculate_trade_statistics(),
            'risk_metrics': self.risk_manager.get_risk_metrics().to_dict(),
            'open_lots': [],
            'closed_lots': []
        }
        
        for pos in self.portfolio.get_all_positions():
            for lot in pos.lots:
                lot_data = {
                    'lot_id': lot.lot_id,
                    'exchange': lot.exchange,
                    'symbol': lot.symbol,
                    'quantity': lot.quantity,
                    'entry_price': lot.entry_price,
                    'entry_time': lot.entry_time.isoformat(),
                    'remaining_quantity': lot.remaining_quantity,
                    'status': lot.status.value,
                    'closed_quantity': lot.closed_quantity,
                    'closed_time': lot.closed_time.isoformat() if lot.closed_time else None,
                    'realized_pnl': lot.realized_pnl,
                    'tags': lot.tags,
                    'current_price': pos.last_price if pos else 0,
                    'unrealized_pnl': lot.remaining_quantity * (pos.last_price - lot.entry_price) if not lot.is_closed else 0
                }
                
                if lot.is_closed:
                    report_data['closed_lots'].append(lot_data)
                else:
                    report_data['open_lots'].append(lot_data)
        
        filename = self.report_dir / f"lot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self._save_report(report_data, filename, format)
    
    def generate_pnl_report(self, format: ReportFormat = ReportFormat.JSON) -> str:
        """Generate a P&L report."""
        summary = self.portfolio.get_portfolio_summary()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'performance_metrics': self.analyzer.get_performance_metrics().to_dict(),
            'returns': self.analyzer.calculate_returns(),
            'risk_adjusted': self.analyzer.calculate_risk_adjusted_metrics(),
            'equity_curve': self.analyzer._get_equity_curve(),
            'positions': [{
                'symbol': p.symbol,
                'exchange': p.exchange,
                'total_quantity': p.total_quantity,
                'average_cost': p.average_cost,
                'market_value': p.market_value,
                'unrealized_pnl': p.unrealized_pnl,
                'realized_pnl': p.realized_pnl,
                'lot_count': p.lot_count
            } for p in self.portfolio.get_all_positions()]
        }
        
        filename = self.report_dir / f"pnl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self._save_report(report_data, filename, format)
    
    def generate_risk_report(self, format: ReportFormat = ReportFormat.JSON) -> str:
        """Generate a risk report."""
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'risk_metrics': risk_metrics.to_dict(),
            'var_95': risk_metrics.var_95,
            'var_99': risk_metrics.var_99,
            'cvar_95': risk_metrics.cvar_95,
            'current_drawdown': self.risk_manager.current_drawdown,
            'daily_pnl': self.risk_manager.daily_pnl,
            'peak_equity': self.risk_manager.peak_equity,
            'exposure_by_symbol': [{
                'symbol': p.symbol,
                'exposure': p.market_value / max(self.portfolio.get_total_equity(), 1),
                'value': p.market_value
            } for p in self.portfolio.get_all_positions()]
        }
        
        filename = self.report_dir / f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self._save_report(report_data, filename, format)
    
    def generate_performance_report(self, format: ReportFormat = ReportFormat.JSON) -> str:
        """Generate a performance report."""
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': self.analyzer.get_performance_metrics().to_dict(),
            'trade_statistics': self.analyzer.calculate_trade_statistics(),
            'returns': self.analyzer.calculate_returns(),
            'risk_adjusted': self.analyzer.calculate_risk_adjusted_metrics(),
            'lot_performance': self.analyzer.get_lot_performance_metrics(),
            'equity_curve': self.analyzer._get_equity_curve()
        }
        
        filename = self.report_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self._save_report(report_data, filename, format)
    
    def generate_tax_report(self, year: int, format: ReportFormat = ReportFormat.JSON) -> str:
        """Generate a tax report with lot-level details."""
        report_data = {
            'tax_year': year,
            'generated_at': datetime.now().isoformat(),
            'closed_lots': []
        }
        
        for pos in self.portfolio.get_all_positions():
            for lot in pos.lots:
                if lot.is_closed and lot.closed_time and lot.closed_time.year == year:
                    report_data['closed_lots'].append({
                        'symbol': lot.symbol,
                        'exchange': lot.exchange,
                        'entry_date': lot.entry_time.isoformat(),
                        'exit_date': lot.closed_time.isoformat(),
                        'quantity': lot.closed_quantity,
                        'cost_basis': lot.closed_quantity * lot.entry_price,
                        'proceeds': lot.closed_quantity * lot.entry_price + lot.realized_pnl,
                        'realized_pnl': lot.realized_pnl,
                        'holding_days': (lot.closed_time - lot.entry_time).days,
                        'tags': lot.tags
                    })
        
        # Calculate totals
        report_data['summary'] = {
            'total_realized_pnl': sum(l['realized_pnl'] for l in report_data['closed_lots']),
            'total_lots': len(report_data['closed_lots']),
            'short_term_pnl': sum(l['realized_pnl'] for l in report_data['closed_lots'] if l['holding_days'] <= 365),
            'long_term_pnl': sum(l['realized_pnl'] for l in report_data['closed_lots'] if l['holding_days'] > 365)
        }
        
        filename = self.report_dir / f"tax_report_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self._save_report(report_data, filename, format)
    
    def _save_report(self, data: Dict, filename: Path, format: ReportFormat) -> str:
        """Save report in the specified format."""
        if format == ReportFormat.JSON:
            return self._save_json(data, filename)
        elif format == ReportFormat.CSV:
            return self._save_csv(data, filename)
        elif format == ReportFormat.HTML:
            return self._save_html(data, filename)
        return self._save_json(data, filename)
    
    def _save_json(self, data: Dict, filename: Path) -> str:
        """Save as JSON."""
        filepath = filename.with_suffix('.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON report saved to {filepath}")
        return str(filepath)
    
    def _save_csv(self, data: Dict, filename: Path) -> str:
        """Save as CSV."""
        filepath = filename.with_suffix('.csv')
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write summary
            writer.writerow(['Report Generated', data.get('generated_at', '')])
            writer.writerow([])
            
            # Write positions
            if 'summary' in data and 'positions' in data['summary']:
                writer.writerow(['Positions'])
                positions = data['summary']['positions']
                if positions:
                    writer.writerow(positions[0].keys())
                    for pos in positions:
                        writer.writerow(pos.values())
                writer.writerow([])
            
            # Write lots
            if 'open_lots' in data:
                writer.writerow(['Open Lots'])
                if data['open_lots']:
                    writer.writerow(data['open_lots'][0].keys())
                    for lot in data['open_lots']:
                        writer.writerow(lot.values())
                writer.writerow([])
            
            if 'closed_lots' in data:
                writer.writerow(['Closed Lots'])
                if data['closed_lots']:
                    writer.writerow(data['closed_lots'][0].keys())
                    for lot in data['closed_lots']:
                        writer.writerow(lot.values())
        
        logger.info(f"CSV report saved to {filepath}")
        return str(filepath)
    
    def _save_html(self, data: Dict, filename: Path) -> str:
        """Save as HTML."""
        filepath = filename.with_suffix('.html')
        
        html = self._generate_html_report(data)
        with open(filepath, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {filepath}")
        return str(filepath)
    
    def _generate_html_report(self, data: Dict) -> str:
        """Generate HTML report."""
        summary = data.get('summary', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #666; margin-top: 30px; }}
                h3 {{ color: #888; margin-top: 20px; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .summary-card {{ background: #e7f3fe; padding: 15px; border-radius: 8px; }}
                .summary-card .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
                .summary-card .value {{ font-size: 20px; font-weight: bold; color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .tag {{ display: inline-block; background: #e0e0e0; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin: 2px; }}
                .lot-open {{ background: #e8f5e9; }}
                .lot-closed {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Portfolio Report</h1>
                <p><strong>Generated:</strong> {data.get('generated_at', 'N/A')}</p>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="label">Total Equity</div>
                        <div class="value">${summary.get('total_equity', 0):.2f}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Total P&L</div>
                        <div class="value {'positive' if summary.get('total_pnl', 0) >= 0 else 'negative'}">
                            ${summary.get('total_pnl', 0):.2f}
                        </div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Open Lots</div>
                        <div class="value">{summary.get('lot_statistics', {}).get('open_lots', 0)}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Win Rate</div>
                        <div class="value">{summary.get('lot_statistics', {}).get('win_rate', 0):.1%}</div>
                    </div>
                </div>
        """
        
        # Open Lots
        if data.get('open_lots'):
            html += f"""
                <h2>Open Lots</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>Unrealized P&L</th>
                        <th>Tags</th>
                        <th>Holding Days</th>
                    </tr>
            """
            for lot in data['open_lots']:
                pnl = lot.get('unrealized_pnl', 0)
                html += f"""
                    <tr>
                        <td>{lot.get('symbol', '')}</td>
                        <td>{lot.get('remaining_quantity', 0):.4f}</td>
                        <td>${lot.get('entry_price', 0):.2f}</td>
                        <td>${lot.get('current_price', 0):.2f}</td>
                        <td class="{'positive' if pnl >= 0 else 'negative'}">${pnl:.2f}</td>
                        <td>{" ".join([f'<span class="tag">{tag}</span>' for tag in lot.get('tags', [])])}</td>
                        <td>{lot.get('holding_days', 0)}</td>
                    </tr>
                """
            html += "</table>"
        
        # Closed Lots
        if data.get('closed_lots'):
            html += f"""
                <h2>Closed Lots</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Entry Price</th>
                        <th>Realized P&L</th>
                        <th>Tags</th>
                        <th>Holding Days</th>
                    </tr>
            """
            for lot in data['closed_lots']:
                pnl = lot.get('realized_pnl', 0)
                html += f"""
                    <tr>
                        <td>{lot.get('symbol', '')}</td>
                        <td>{lot.get('closed_quantity', 0):.4f}</td>
                        <td>${lot.get('entry_price', 0):.2f}</td>
                        <td class="{'positive' if pnl >= 0 else 'negative'}">${pnl:.2f}</td>
                        <td>{" ".join([f'<span class="tag">{tag}</span>' for tag in lot.get('tags', [])])}</td>
                        <td>{lot.get('holding_days', 0)}</td>
                    </tr>
                """
            html += "</table>"
        
        # Performance Metrics
        if data.get('performance_metrics'):
            perf = data['performance_metrics']
            html += f"""
                <h2>Performance Metrics</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="label">Win Rate</div>
                        <div class="value">{perf.get('win_rate', 0):.1%}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Profit Factor</div>
                        <div class="value">{perf.get('profit_factor', 0):.2f}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Sharpe Ratio</div>
                        <div class="value">{perf.get('sharpe_ratio', 0):.2f}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Total Return</div>
                        <div class="value">{perf.get('total_return', 0):.1%}</div>
                    </div>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


# ============= Portfolio Manager =============

class PortfolioManagerV2:
    """
    Lot-based portfolio manager that preserves complete trade history.
    Thread-safe with comprehensive features.
    """
    
    def __init__(self, exchange_adapters: Dict[str, ExchangeInterface], config: Dict = None):
        self.config = config or {}
        self.exchanges = exchange_adapters
        self._ledger: List[Trade] = []
        self._positions: Dict[str, PositionV2] = {}
        self._balances: Dict[str, Balance] = {}
        self._orders: Dict[str, Order] = {}
        self._active_orders: Dict[str, Order] = {}
        self._last_prices: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._lot_strategy = self.config.get('lot_strategy', Config.LOT_ALLOCATION_STRATEGY)
        self._order_update_callbacks = []
        self._trade_callbacks = []
        self._lot_callbacks = []
        
        # Sub-systems
        self.risk_manager = RiskManagerV2(self, self.config)
        self.performance_analyzer = PerformanceAnalyzerV2(self)
        self.alert_system = AlertSystem(self.config)
        self.reporting = ReportingModuleV2(self, self.risk_manager, self.performance_analyzer, self.alert_system)
        
        logger.info(f"Lot-based PortfolioManager initialized with exchanges: {list(self.exchanges.keys())}")
    
    # ============= Commands =============
    
    def submit_order(self, exchange: str, symbol: str, side: OrderSide, order_type: OrderType,
                     quantity: float, price: Optional[float] = None, 
                     tags: List[str] = None, **kwargs) -> str:
        """Submit a new order."""
        with self._lock:
            if exchange not in self.exchanges:
                raise ValueError(f"Unknown exchange: {exchange}")
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            order = Order(exchange=exchange, symbol=symbol, side=side, order_type=order_type,
                         quantity=quantity, price=price, **kwargs)
            
            approved, reason = self.risk_manager.check_order(order)
            if not approved:
                order.status = OrderStatus.REJECTED
                order.reject_reason = reason
                self._orders[order.internal_id] = order
                self.alert_system.add_alert(AlertSeverity.WARNING, "Order", "Order Rejected", 
                                            f"Order {order.internal_id} rejected: {reason}")
                raise ValueError(f"Order rejected: {reason}")
            
            if not self._check_sufficient_balance(order):
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Insufficient balance"
                self._orders[order.internal_id] = order
                raise ValueError("Insufficient balance")
            
            self._lock_funds_for_order(order)
            self._orders[order.internal_id] = order
            self._active_orders[order.internal_id] = order
            
            try:
                success, exchange_order_id, response = self.exchanges[exchange].place_order(order)
                if success:
                    order.exchange_order_id = exchange_order_id
                    order.status = OrderStatus.SUBMITTED
                    order.submitted_at = datetime.now()
                    logger.info(f"Order {order.internal_id} submitted")
                    self.alert_system.add_alert(AlertSeverity.INFO, "Order", "Order Submitted",
                                               f"Order {order.internal_id} submitted")
                else:
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = str(response)
                    self._release_locked_funds(order)
                    del self._active_orders[order.internal_id]
                    raise RuntimeError(f"Exchange rejected: {response}")
            except Exception as e:
                order.status = OrderStatus.REJECTED
                order.reject_reason = str(e)
                self._release_locked_funds(order)
                if order.internal_id in self._active_orders:
                    del self._active_orders[order.internal_id]
                raise
            
            self._trigger_order_update(order)
            return order.internal_id
    
    def cancel_order(self, internal_order_id: str) -> bool:
        """Cancel an active order."""
        with self._lock:
            if internal_order_id not in self._active_orders:
                return False
            order = self._active_orders[internal_order_id]
            if not order.is_active:
                return False
            
            if order.exchange_order_id:
                try:
                    success = self.exchanges[order.exchange].cancel_order(
                        order.exchange_order_id, order.symbol)
                    if not success:
                        return False
                except Exception as e:
                    logger.exception(f"Error cancelling order {internal_order_id}")
                    return False
            
            self._release_locked_funds(order)
            order.status = OrderStatus.CANCELLED
            order.closed_at = datetime.now()
            del self._active_orders[internal_order_id]
            self._trigger_order_update(order)
            self.alert_system.add_alert(AlertSeverity.INFO, "Order", "Order Cancelled",
                                       f"Order {internal_order_id} cancelled")
            return True
    
    # ============= Lot Management =============
    
    def add_lot(self, exchange: str, symbol: str, quantity: float, entry_price: float,
                order_id: str = "", tags: List[str] = None) -> str:
        """Manually add a lot to the portfolio."""
        with self._lock:
            position_key = f"{exchange}:{symbol}"
            if position_key not in self._positions:
                self._positions[position_key] = PositionV2(symbol=symbol, exchange=exchange)
            
            lot = Lot(
                exchange=exchange,
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                entry_price=entry_price,
                remaining_quantity=quantity,
                order_id=order_id,
                tags=tags or []
            )
            
            self._positions[position_key].add_lot(lot)
            self._trigger_lot_created(lot)
            return lot.lot_id
    
    def close_lots(self, exchange: str, symbol: str, quantity: float, exit_price: float,
                   fee: float = 0.0, strategy: Optional[str] = None) -> List[Tuple[Lot, float]]:
        """Close lots using specified strategy."""
        with self._lock:
            position_key = f"{exchange}:{symbol}"
            if position_key not in self._positions:
                raise ValueError(f"No position found for {symbol} on {exchange}")
            
            position = self._positions[position_key]
            strategy = strategy or self._lot_strategy
            
            results = position.close_lots(quantity, exit_price, fee, strategy)
            
            for lot, pnl in results:
                trade = Trade(
                    trade_id=f"close_{lot.lot_id}_{datetime.now().timestamp()}",
                    exchange=exchange,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=lot.closed_quantity,
                    price=exit_price,
                    timestamp=datetime.now(),
                    order_id="",
                    fee=fee * (lot.closed_quantity / quantity)
                )
                self._ledger.append(trade)
                self._trigger_trade(trade)
                self._trigger_lot_closed(lot)
                
                # Update daily P&L
                self.risk_manager.daily_pnl += pnl
            
            self._update_balances_for_sell(exchange, symbol, quantity, exit_price, fee)
            
            return results
    
    def get_lot(self, lot_id: str) -> Optional[Lot]:
        """Get a specific lot by ID."""
        with self._lock:
            for position in self._positions.values():
                for lot in position.lots:
                    if lot.lot_id == lot_id:
                        return lot
            return None
    
    def get_open_lots(self, exchange: Optional[str] = None, 
                      symbol: Optional[str] = None) -> List[Lot]:
        """Get all open lots."""
        with self._lock:
            lots = []
            for position in self._positions.values():
                if exchange and position.exchange != exchange:
                    continue
                if symbol and position.symbol != symbol:
                    continue
                lots.extend([l for l in position.lots if not l.is_closed])
            return lots
    
    def get_lot_statistics(self) -> Dict[str, Any]:
        """Get statistics about lots."""
        with self._lock:
            open_lots = self.get_open_lots()
            closed_lots = []
            for pos in self._positions.values():
                closed_lots.extend([l for l in pos.lots if l.is_closed])
            
            total_pnl = sum(l.realized_pnl for l in closed_lots)
            winning_lots = [l for l in closed_lots if l.realized_pnl > 0]
            losing_lots = [l for l in closed_lots if l.realized_pnl < 0]
            
            return {
                'total_lots': len(open_lots) + len(closed_lots),
                'open_lots': len(open_lots),
                'closed_lots': len(closed_lots),
                'total_realized_pnl': total_pnl,
                'winning_lots': len(winning_lots),
                'losing_lots': len(losing_lots),
                'win_rate': len(winning_lots) / max(len(closed_lots), 1),
                'avg_pnl_per_lot': total_pnl / max(len(closed_lots), 1),
                'best_lot': max(closed_lots, key=lambda l: l.realized_pnl).realized_pnl if closed_lots else 0,
                'worst_lot': min(closed_lots, key=lambda l: l.realized_pnl).realized_pnl if closed_lots else 0,
            }
    
    # ============= Queries =============
    
    def get_position(self, exchange: str, symbol: str) -> Optional[PositionV2]:
        with self._lock:
            return self._positions.get(f"{exchange}:{symbol}")
    
    def get_all_positions(self) -> List[PositionV2]:
        with self._lock:
            return list(self._positions.values())
    
    def get_balance(self, exchange: str, currency: str) -> Optional[Balance]:
        with self._lock:
            return self._balances.get(f"{exchange}:{currency}")
    
    def get_all_balances(self) -> List[Balance]:
        with self._lock:
            return list(self._balances.values())
    
    def get_active_orders(self, exchange: Optional[str] = None, 
                          symbol: Optional[str] = None) -> List[Order]:
        with self._lock:
            orders = list(self._active_orders.values())
            if exchange:
                orders = [o for o in orders if o.exchange == exchange]
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            return orders
    
    def get_order(self, internal_order_id: str) -> Optional[Order]:
        with self._lock:
            return self._orders.get(internal_order_id)
    
    def get_trades(self, exchange: Optional[str] = None, 
                   symbol: Optional[str] = None, limit: int = 1000) -> List[Trade]:
        with self._lock:
            trades = self._ledger
            if exchange:
                trades = [t for t in trades if t.exchange == exchange]
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            trades.sort(key=lambda x: x.timestamp, reverse=True)
            return trades[:limit]
    
    def get_total_equity(self) -> float:
        with self._lock:
            total = sum(b.total for b in self._balances.values() if b.currency == Config.BASE_CURRENCY)
            for pos in self._positions.values():
                total += pos.market_value
            return total
    
    def calculate_realized_pnl(self) -> float:
        with self._lock:
            return sum(
                sum(lot.realized_pnl for lot in pos.lots if lot.is_closed)
                for pos in self._positions.values()
            )
    
    def calculate_unrealized_pnl(self) -> float:
        with self._lock:
            return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    def get_portfolio_summary(self) -> Dict:
        with self._lock:
            lot_stats = self.get_lot_statistics()
            return {
                'total_equity': self.get_total_equity(),
                'total_cash': sum(b.total for b in self._balances.values()),
                'total_positions_value': sum(pos.market_value for pos in self._positions.values()),
                'total_realized_pnl': self.calculate_realized_pnl(),
                'total_unrealized_pnl': self.calculate_unrealized_pnl(),
                'total_pnl': self.calculate_realized_pnl() + self.calculate_unrealized_pnl(),
                'lot_statistics': lot_stats,
                'active_orders_count': len(self._active_orders),
                'positions': [{
                    'symbol': pos.symbol,
                    'exchange': pos.exchange,
                    'total_quantity': pos.total_quantity,
                    'average_cost': pos.average_cost,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'lot_count': pos.lot_count,
                    'lots': [{
                        'lot_id': l.lot_id,
                        'quantity': l.remaining_quantity,
                        'entry_price': l.entry_price,
                        'entry_time': l.entry_time.isoformat(),
                        'pnl': l.realized_pnl if l.is_closed else None,
                        'status': l.status.value,
                        'tags': l.tags
                    } for l in pos.lots]
                } for pos in self._positions.values()],
                'balances': [{
                    'exchange': b.exchange,
                    'currency': b.currency,
                    'total': b.total,
                    'free': b.free,
                    'locked': b.locked
                } for b in self._balances.values()]
            }
    
    # ============= Internal Methods =============
    
    def _check_sufficient_balance(self, order: Order) -> bool:
        if order.side == OrderSide.BUY:
            quote = order.symbol.split('/')[-1] if '/' in order.symbol else Config.BASE_CURRENCY
            bal = self._balances.get(f"{order.exchange}:{quote}")
            if not bal:
                return False
            required = order.quantity * (order.price or self._last_prices.get(order.symbol, 0))
            return bal.free >= required * (1 + Config.SLIPPAGE_BUFFER)
        else:
            base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            bal = self._balances.get(f"{order.exchange}:{base}")
            return bal and bal.free >= order.quantity
    
    def _lock_funds_for_order(self, order: Order):
        if order.side == OrderSide.BUY:
            quote = order.symbol.split('/')[-1] if '/' in order.symbol else Config.BASE_CURRENCY
            bal = self._balances.get(f"{order.exchange}:{quote}")
            if bal:
                bal.locked += order.quantity * (order.price or self._last_prices.get(order.symbol, 0))
        else:
            base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            bal = self._balances.get(f"{order.exchange}:{base}")
            if bal:
                bal.locked += order.quantity
    
    def _release_locked_funds(self, order: Order):
        if order.side == OrderSide.BUY:
            quote = order.symbol.split('/')[-1] if '/' in order.symbol else Config.BASE_CURRENCY
            bal = self._balances.get(f"{order.exchange}:{quote}")
            if bal:
                bal.locked = max(0, bal.locked - order.quantity * (order.price or self._last_prices.get(order.symbol, 0)))
        else:
            base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            bal = self._balances.get(f"{order.exchange}:{base}")
            if bal:
                bal.locked = max(0, bal.locked - order.quantity)
    
    def _update_balances_for_sell(self, exchange: str, symbol: str, quantity: float, 
                                   exit_price: float, fee: float):
        base = symbol.split('/')[0] if '/' in symbol else symbol
        quote = symbol.split('/')[-1] if '/' in symbol else Config.BASE_CURRENCY
        
        base_bal = self._balances.get(f"{exchange}:{base}")
        if base_bal:
            base_bal.total -= quantity
        
        quote_bal = self._balances.get(f"{exchange}:{quote}")
        if quote_bal:
            quote_bal.total += quantity * exit_price - fee
    
    def _sync_exchange(self, exchange_name: str, exchange: ExchangeInterface):
        # Sync balances
        for currency, data in exchange.get_balances().items():
            key = f"{exchange_name}:{currency}"
            if key in self._balances:
                self._balances[key].update(data['total'], data.get('locked', 0))
            else:
                self._balances[key] = Balance(exchange_name, currency, data['total'], data.get('locked', 0))
        
        # Update prices
        for position in self._positions.values():
            if position.exchange == exchange_name:
                ticker = exchange.get_ticker(position.symbol)
                if ticker and 'last' in ticker:
                    position.last_price = float(ticker['last'])
                    self._last_prices[position.symbol] = float(ticker['last'])
    
    # ============= Callbacks =============
    
    def register_order_update_callback(self, callback: Callable):
        self._order_update_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable):
        self._trade_callbacks.append(callback)
    
    def register_lot_callback(self, callback: Callable):
        self._lot_callbacks.append(callback)
    
    def _trigger_order_update(self, order: Order):
        for cb in list(self._order_update_callbacks):
            try: cb(order)
            except Exception as e: logger.exception(f"Callback error: {e}")
    
    def _trigger_trade(self, trade: Trade):
        for cb in list(self._trade_callbacks):
            try: cb(trade)
            except Exception as e: logger.exception(f"Callback error: {e}")
    
    def _trigger_lot_created(self, lot: Lot):
        for cb in list(self._lot_callbacks):
            try: cb(lot)
            except Exception as e: logger.exception(f"Callback error: {e}")
    
    def _trigger_lot_closed(self, lot: Lot):
        for cb in list(self._lot_callbacks):
            try: cb(lot)
            except Exception as e: logger.exception(f"Callback error: {e}")
    
    # ============= Sync Thread =============
    
    def start_sync_thread(self):
        if self._sync_thread and self._sync_thread.is_alive():
            return
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Sync thread started")
    
    def stop_sync_thread(self):
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        logger.info("Sync thread stopped")
    
    def _sync_loop(self):
        while self._running:
            try:
                self.sync_with_exchanges()
            except Exception as e:
                logger.exception(f"Sync error: {e}")
            time.sleep(Config.SYNC_INTERVAL)
    
    def sync_with_exchanges(self):
        with self._lock:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    self._sync_exchange(exchange_name, exchange)
                except Exception as e:
                    logger.exception(f"Error syncing {exchange_name}: {e}")
            
            self.risk_manager.update_daily_pnl()
            self.risk_manager.update_drawdown()
            self.alert_system.check_portfolio_alerts(self, self.risk_manager)
    
    # ============= Persistence =============
    
    def save_state(self, filepath: str = None):
        """Save portfolio state to disk."""
        if filepath is None:
            filepath = Config.STATE_FILE
        with self._lock:
            state = {
                'ledger': [t.to_dict() for t in self._ledger],
                'positions': {k: v.to_dict() for k, v in self._positions.items()},
                'balances': {k: v.to_dict() for k, v in self._balances.items()},
                'orders': {k: v.to_dict() for k, v in self._orders.items()},
                'last_prices': self._last_prices,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str = None):
        """Load portfolio state from disk."""
        if filepath is None:
            filepath = Config.STATE_FILE
        if not Path(filepath).exists():
            logger.warning(f"State file {filepath} not found")
            return
        with self._lock:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self._ledger = [Trade.from_dict(t) for t in state['ledger']]
            self._positions = {k: PositionV2.from_dict(v) for k, v in state['positions'].items()}
            self._balances = {k: Balance.from_dict(v) for k, v in state['balances'].items()}
            self._orders = {k: Order.from_dict(v) for k, v in state['orders'].items()}
            self._last_prices = state['last_prices']
            if 'config' in state:
                self.config.update(state['config'])
            self._active_orders = {oid: o for oid, o in self._orders.items() if o.is_active}
        logger.info(f"State loaded from {filepath}")


# ============= Demo =============

def run_demo():
    """Run a comprehensive demo of the lot-based portfolio system."""
    print("\n" + "="*70)
    print("  LOT-BASED PORTFOLIO MANAGEMENT SYSTEM - COMPLETE DEMO")
    print("="*70)
    
    # Setup
    exchanges = {'binance': MockExchange('binance')}
    portfolio = PortfolioManagerV2(exchanges)
    
    # Initialize balances
    for exchange_name, exchange in exchanges.items():
        for currency, data in exchange.get_balances().items():
            portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                exchange_name, currency, data['total'], data['locked'])
        print(f"  Loaded {exchange_name} balances")
    
    portfolio.start_sync_thread()
    
    try:
        # 1. Add lots
        print("\n" + "-"*70)
        print("  STEP 1: ADDING LOTS (Different Entry Prices)")
        print("-"*70)
        
        lots = [
            (0.1, 48000, "Strategy A", ["entry1", "scalp"]),
            (0.15, 49000, "Strategy A", ["entry2"]),
            (0.2, 50000, "Strategy B", ["swing"]),
            (0.08, 51000, "Strategy C", ["long"]),
        ]
        
        for qty, price, tag, tags in lots:
            lot_id = portfolio.add_lot('binance', 'BTC/USD', qty, price, tags=tags)
            print(f"  Added: {qty:.2f} BTC @ ${price:.0f} (tags: {tags})")
        
        # Update current price
        portfolio._last_prices['BTC/USD'] = 52000
        for pos in portfolio._positions.values():
            pos.last_price = 52000
        
        # 2. Show position
        print("\n" + "-"*70)
        print("  STEP 2: CURRENT POSITION")
        print("-"*70)
        
        pos = portfolio.get_position('binance', 'BTC/USD')
        print(f"  Total BTC: {pos.total_quantity:.4f}")
        print(f"  Average Cost: ${pos.average_cost:.2f}")
        print(f"  Market Value: ${pos.market_value:.2f}")
        print(f"  Unrealized PnL: ${pos.unrealized_pnl:.2f}")
        print(f"  Number of Open Lots: {pos.lot_count}")
        
        print("\n  Open Lots:")
        for i, lot in enumerate(pos.get_open_lots_sorted("FIFO"), 1):
            pnl = lot.remaining_quantity * (52000 - lot.entry_price)
            print(f"    {i}. {lot.remaining_quantity:.4f} BTC @ ${lot.entry_price:.0f} "
                  f"(PnL: ${pnl:.2f}) tags: {lot.tags}")
        
        # 3. Close lots with FIFO
        print("\n" + "-"*70)
        print("  STEP 3: CLOSING 0.2 BTC USING FIFO")
        print("-"*70)
        
        results = portfolio.close_lots('binance', 'BTC/USD', 0.2, 52000, strategy="FIFO")
        for lot, pnl in results:
            print(f"  Closed from ${lot.entry_price:.0f}: PnL ${pnl:.2f}")
        
        # 4. Show remaining lots
        print("\n" + "-"*70)
        print("  STEP 4: REMAINING LOTS")
        print("-"*70)
        
        for lot in pos.get_open_lots_sorted("FIFO"):
            print(f"  {lot.remaining_quantity:.4f} BTC @ ${lot.entry_price:.0f} tags: {lot.tags}")
        
        # 5. Lot Statistics
        print("\n" + "-"*70)
        print("  STEP 5: LOT STATISTICS")
        print("-"*70)
        
        stats = portfolio.get_lot_statistics()
        print(f"  Closed Lots: {stats['closed_lots']}")
        print(f"  Open Lots: {stats['open_lots']}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Total Realized PnL: ${stats['total_realized_pnl']:.2f}")
        print(f"  Best Lot: ${stats['best_lot']:.2f}")
        print(f"  Worst Lot: ${stats['worst_lot']:.2f}")
        
        # 6. Risk Metrics
        print("\n" + "-"*70)
        print("  STEP 6: RISK METRICS")
        print("-"*70)
        
        risk_metrics = portfolio.risk_manager.get_risk_metrics()
        print(f"  VaR (95%): {risk_metrics.var_95:.2%}")
        print(f"  VaR (99%): {risk_metrics.var_99:.2%}")
        print(f"  CVaR (95%): {risk_metrics.cvar_95:.2%}")
        print(f"  Volatility: {risk_metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        print(f"  Exposure: {risk_metrics.exposure:.2%}")
        print(f"  Risk Level: {risk_metrics.risk_level.value}")
        print(f"  Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        
        # 7. Performance Metrics
        print("\n" + "-"*70)
        print("  STEP 7: PERFORMANCE METRICS")
        print("-"*70)
        
        perf = portfolio.performance_analyzer.get_performance_metrics()
        trade_stats = portfolio.performance_analyzer.calculate_trade_statistics()
        returns = portfolio.performance_analyzer.calculate_returns()
        
        print(f"  Total Return: {returns.get('total_return', 0):.2%}")
        print(f"  Annualized Return: {returns.get('annualized_return', 0):.2%}")
        print(f"  Win Rate: {perf.win_rate:.1%}")
        print(f"  Profit Factor: {perf.profit_factor:.2f}")
        print(f"  Sharpe Ratio: {perf.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {perf.sortino_ratio:.2f}")
        print(f"  Calmar Ratio: {perf.calmar_ratio:.2f}")
        print(f"  Omega Ratio: {perf.omega_ratio:.2f}")
        print(f"  Consecutive Wins: {perf.consecutive_wins}")
        print(f"  Consecutive Losses: {perf.consecutive_losses}")
        
        # 8. Reports
        print("\n" + "-"*70)
        print("  STEP 8: GENERATING REPORTS")
        print("-"*70)
        
        reports = [
            ("Lot Report", portfolio.reporting.generate_lot_report),
            ("PnL Report", portfolio.reporting.generate_pnl_report),
            ("Risk Report", portfolio.reporting.generate_risk_report),
            ("Performance Report", portfolio.reporting.generate_performance_report),
            ("Tax Report", lambda: portfolio.reporting.generate_tax_report(2024)),
        ]
        
        for name, func in reports:
            try:
                filepath = func()
                print(f"  {name}: {filepath}")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
        
        # 9. Summary
        print("\n" + "-"*70)
        print("  STEP 9: PORTFOLIO SUMMARY")
        print("-"*70)
        
        summary = portfolio.get_portfolio_summary()
        print(f"  Total Equity: ${summary['total_equity']:.2f}")
        print(f"  Total Cash: ${summary['total_cash']:.2f}")
        print(f"  Positions Value: ${summary['total_positions_value']:.2f}")
        print(f"  Realized PnL: ${summary['total_realized_pnl']:.2f}")
        print(f"  Unrealized PnL: ${summary['total_unrealized_pnl']:.2f}")
        print(f"  Total PnL: ${summary['total_pnl']:.2f}")
        print(f"  Active Orders: {summary['active_orders_count']}")
        
        print("\n" + "="*70)
        print("  ✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        logger.exception("Demo error")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        portfolio.stop_sync_thread()
        portfolio.save_state()
        print("\n  Portfolio state saved.")


# ============= Main =============

if __name__ == "__main__":
    run_demo()