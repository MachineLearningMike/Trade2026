"""
portfolio_core.py — domain model, lot strategies, exchange interface, manager.
"""
import math
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
import os

logger = logging.getLogger(__name__)

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
    
    # Paths (package-relative so demos/tests don't depend on CWD)
    REPORT_DIR = Path(__file__).resolve().parent / "reports"
    STATE_FILE = str(Path(__file__).resolve().parent / "portfolio_state.pkl")
    
    # Email
    EMAIL_ENABLED = os.environ.get('EMAIL_ENABLED', 'false').lower() == 'true'
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    EMAIL_USER = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
    EMAIL_RECIPIENTS = os.environ.get('EMAIL_RECIPIENTS', '').split(',')
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}


# ============================================================================
# ENUMS
# ============================================================================

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderTimeInForce(Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    DAY = "day"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LotStatus(Enum):
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    ARCHIVED = "archived"


class ReportFormat(Enum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"


# ============================================================================
# HELPERS
# ============================================================================

def to_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


def normalize_symbol(symbol: str) -> str:
    """
    Normalize exchange symbols to ``BASE/QUOTE`` form used by the portfolio.

    Examples:
      BTCUSDT → BTC/USD
      ETHUSD  → ETH/USD
      BTC/USD → BTC/USD (unchanged)
    """
    s = str(symbol).strip().upper().replace('-', '/')
    if '/' in s:
        base, quote = s.split('/', 1)
        if quote in ('USDT', 'BUSD', 'USDC'):
            quote = 'USD'
        return f"{base}/{quote}"
    for quote in ('USDT', 'BUSD', 'USDC', 'USD'):
        if s.endswith(quote) and len(s) > len(quote):
            base = s[: -len(quote)]
            q = 'USD' if quote in ('USDT', 'BUSD', 'USDC') else quote
            return f"{base}/{q}"
    return s


def split_symbol(symbol: str) -> Tuple[str, str]:
    """Return (base, quote) after normalize_symbol."""
    sym = normalize_symbol(symbol)
    if '/' in sym:
        base, quote = sym.split('/', 1)
        return base, quote
    return sym, Config.BASE_CURRENCY


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns."""
    if not returns or len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252
    if np.std(returns_array) > 0:
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns_array)
    return 0.0


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (uses downside deviation)."""
    if not returns or len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
    return 0.0


def calculate_calmar_ratio(returns: List[float], max_drawdown: float) -> float:
    """Calculate Calmar ratio."""
    if not returns or max_drawdown <= 0:
        return 0.0
    avg_return = np.mean(returns) * 252
    return avg_return / max_drawdown


def calculate_omega_ratio(returns: List[float], threshold: float = 0.0) -> float:
    """Calculate Omega ratio."""
    if not returns:
        return 0.0
    returns_array = np.array(returns)
    gains = returns_array[returns_array > threshold]
    losses = returns_array[returns_array < threshold]
    if len(losses) > 0 and np.sum(losses) != 0:
        return np.sum(gains) / abs(np.sum(losses))
    return float('inf') if len(gains) > 0 else 0.0


# ============================================================================
# EXCHANGE INTERFACE
# ============================================================================

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
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        pass


# ============================================================================
# DATA CLASSES
# ============================================================================

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
    lot_id: Optional[str] = None
    fee_currency: str = ""
    commission_asset: str = ""
    commission_paid: float = 0.0
    # Realized PnL for sells (entry cost + fees already accounted); None for buys / legacy
    realized_pnl: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
    
    @property
    def value(self) -> float:
        return self.quantity * self.price
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    def to_dict(self) -> Dict:
        return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trade':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['side'] = OrderSide(data['side'])
        data.setdefault('realized_pnl', None)
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
    def free(self) -> float:
        return self.total - self.locked
    
    def update(self, total: float, locked: float):
        self.total = total
        self.locked = locked
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict:
        return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Balance':
        data = data.copy()
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
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.time_in_force, str):
            self.time_in_force = OrderTimeInForce(self.time_in_force)
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status)
        if self.remaining_quantity == 0 and self.quantity > 0:
            self.remaining_quantity = self.quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        ]
    
    @property
    def is_done(self) -> bool:
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]
    
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
    
    def to_dict(self) -> Dict:
        return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        data = data.copy()
        
        # Convert datetime fields
        for field in ['created_at', 'submitted_at', 'updated_at', 'closed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert enum fields
        data['side'] = OrderSide(data['side'])
        data['order_type'] = OrderType(data['order_type'])
        data['time_in_force'] = OrderTimeInForce(data['time_in_force'])
        data['status'] = OrderStatus(data['status'])
        
        # Convert trades if present
        if 'trades' in data and data['trades']:
            data['trades'] = [Trade.from_dict(t) for t in data['trades']]
        else:
            data['trades'] = []
        
        return cls(**data)


@dataclass
class Lot:
    """
    A lot represents a batch of assets bought at a specific price and time.
    Strategy decides which lots to sell. Closed lots are archived.
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
    # Total entry fee paid for the full original quantity (allocated pro-rata on close)
    entry_fee: float = 0.0
    order_id: str = ""
    tags: List[str] = field(default_factory=list)
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.status, str):
            self.status = LotStatus(self.status)
        # Only default remaining to full size for brand-new open lots
        if (
            self.remaining_quantity == 0
            and self.quantity > 0
            and self.closed_quantity == 0
            and self.status == LotStatus.OPEN
        ):
            self.remaining_quantity = self.quantity
        self.exchange = self.exchange or ""
        if self.symbol:
            self.symbol = normalize_symbol(self.symbol)
        # Snap fully closed residual
        if self.remaining_quantity <= 1e-12 and self.closed_quantity > 0:
            self.remaining_quantity = 0.0
            if self.status == LotStatus.OPEN:
                self.status = LotStatus.CLOSED
                if self.closed_time is None:
                    self.closed_time = datetime.now()
    
    @property
    def is_closed(self) -> bool:
        return self.status in [LotStatus.CLOSED, LotStatus.ARCHIVED]
    
    @property
    def is_active(self) -> bool:
        return self.status in [LotStatus.OPEN, LotStatus.PARTIALLY_CLOSED]
    
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
    
    @property
    def fill_percentage(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (self.closed_quantity / self.quantity) * 100
    
    def close_partial(self, quantity: float, exit_price: float, fee: float = 0.0) -> float:
        """
        Close a portion of the lot and return realized PnL.

        PnL (currency) = exit_value - entry_cost - exit_fee - pro_rata_entry_fee
        so round-trip fees are counted once.
        """
        if quantity <= 0 or quantity > self.remaining_quantity:
            raise ValueError(f"Cannot close {quantity} from lot with {self.remaining_quantity} remaining")
        
        entry_cost = quantity * self.entry_price
        exit_value = quantity * exit_price
        entry_fee_alloc = 0.0
        if self.quantity > 0 and self.entry_fee:
            entry_fee_alloc = float(self.entry_fee) * (quantity / self.quantity)
        pnl = exit_value - entry_cost - fee - entry_fee_alloc
        
        self.remaining_quantity -= quantity
        self.closed_quantity += quantity
        self.realized_pnl += pnl
        
        if self.remaining_quantity <= 1e-12:
            self.remaining_quantity = 0.0
            self.status = LotStatus.CLOSED
            self.closed_time = datetime.now()
        else:
            self.status = LotStatus.PARTIALLY_CLOSED
        
        return pnl
    
    def archive(self):
        """Archive a closed lot (remove from active position)."""
        if self.status != LotStatus.CLOSED:
            raise ValueError(f"Cannot archive lot with status {self.status}")
        self.status = LotStatus.ARCHIVED
    
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
            'entry_fee': self.entry_fee,
            'order_id': self.order_id,
            'tags': self.tags,
            'strategy_metadata': self.strategy_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Lot':
        data = data.copy()
        data['side'] = OrderSide(data['side'])
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        data['status'] = LotStatus(data['status'])
        if data.get('closed_time'):
            data['closed_time'] = datetime.fromisoformat(data['closed_time'])
        data.setdefault('entry_fee', 0.0)
        return cls(**data)


@dataclass
class PositionV2:
    """
    Enhanced position that tracks individual lots.
    Closed lots are moved to archive to keep active position clean.
    """
    symbol: str
    exchange: str
    lots: List[Lot] = field(default_factory=list)
    archived_lots: List[Lot] = field(default_factory=list)
    last_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def total_quantity(self) -> float:
        return sum(lot.remaining_quantity for lot in self.lots if lot.is_active)
    
    @property
    def total_cost_basis(self) -> float:
        return sum(lot.remaining_quantity * lot.entry_price for lot in self.lots if lot.is_active)
    
    @property
    def average_cost(self) -> float:
        qty = self.total_quantity
        if qty == 0:
            return 0.0
        return self.total_cost_basis / qty
    
    @property
    def market_value(self) -> float:
        return self.total_quantity * self.last_price
    
    @property
    def unrealized_pnl(self) -> float:
        return sum(lot.remaining_quantity * (self.last_price - lot.entry_price) for lot in self.lots if lot.is_active)
    
    @property
    def realized_pnl(self) -> float:
        # Include PARTIALLY_CLOSED lots: they hold realized_pnl on closed_quantity
        archived_pnl = sum(lot.realized_pnl for lot in self.archived_lots)
        lot_pnl = sum(lot.realized_pnl for lot in self.lots if lot.closed_quantity > 0)
        return archived_pnl + lot_pnl
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def lot_count(self) -> int:
        return len([l for l in self.lots if l.is_active])
    
    @property
    def total_lots_ever(self) -> int:
        return len(self.lots) + len(self.archived_lots)
    
    def get_lots_by_status(self, status: LotStatus) -> List[Lot]:
        if status == LotStatus.ARCHIVED:
            return self.archived_lots
        return [l for l in self.lots if l.status == status]
    
    def get_lots_by_strategy(self, tag: str, include_archived: bool = False) -> List[Lot]:
        result = [l for l in self.lots if tag in l.tags]
        if include_archived:
            result.extend([l for l in self.archived_lots if tag in l.tags])
        return result
    
    def get_open_lots_sorted(self, strategy: str = "FIFO") -> List[Lot]:
        """Get active lots sorted by strategy.

        ``self.lots`` is append-order. When ``entry_time`` collides (common when
        several lots are added in the same microsecond), FIFO/LIFO use insertion
        index as a stable tie-breaker so LIFO still sells newest-added first.
        """
        active_lots = [(i, l) for i, l in enumerate(self.lots) if l.is_active]
        if strategy == "FIFO":
            active_lots.sort(key=lambda t: (t[1].entry_time, t[0]))
            return [l for _, l in active_lots]
        if strategy == "LIFO":
            active_lots.sort(key=lambda t: (t[1].entry_time, t[0]), reverse=True)
            return [l for _, l in active_lots]
        if strategy == "HIFO":
            return [l for _, l in sorted(active_lots, key=lambda t: t[1].entry_price, reverse=True)]
        if strategy == "LOFO":
            return [l for _, l in sorted(active_lots, key=lambda t: t[1].entry_price)]
        return [l for _, l in active_lots]
    
    def get_equity_curve(self) -> List[Dict]:
        """Build equity curve from all lots."""
        curve = []
        equity = Config.INITIAL_EQUITY
        
        all_events = []
        for lot in self.lots + self.archived_lots:
            all_events.append({
                'time': lot.entry_time,
                'type': 'buy',
                'lot': lot,
                'quantity': lot.quantity,
                'price': lot.entry_price
            })
            if lot.is_closed and lot.closed_time:
                exit_price = lot.entry_price + (lot.realized_pnl / lot.closed_quantity) if lot.closed_quantity > 0 else lot.entry_price
                all_events.append({
                    'time': lot.closed_time,
                    'type': 'sell',
                    'lot': lot,
                    'quantity': lot.closed_quantity,
                    'price': exit_price
                })
        
        all_events.sort(key=lambda x: x['time'])
        
        for event in all_events:
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
        """Add a new lot to the position."""
        self.lots.append(lot)
        self.last_update = datetime.now()
    
    def close_lots(self, quantity: float, exit_price: float, fee: float = 0.0, 
                   strategy: str = "FIFO") -> List[Tuple[Lot, float]]:
        """
        Close lots using specified strategy.
        Returns list of (lot, pnl) tuples.
        """
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
        
        # Archive fully closed lots
        self._archive_closed_lots()
        self.last_update = datetime.now()
        return results
    
    def _archive_closed_lots(self):
        """Move fully closed lots to archive."""
        to_archive = []
        remaining = []
        
        for lot in self.lots:
            if lot.status == LotStatus.CLOSED:
                to_archive.append(lot)
            else:
                remaining.append(lot)
        
        for lot in to_archive:
            lot.archive()
            self.archived_lots.append(lot)
        
        self.lots = remaining
    
    def clear_archived(self):
        """Clear archived lots (free memory)."""
        self.archived_lots = []
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'lots': [l.to_dict() for l in self.lots],
            'archived_lots': [l.to_dict() for l in self.archived_lots],
            'last_price': self.last_price,
            'last_update': self.last_update.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PositionV2':
        data = data.copy()
        data['lots'] = [Lot.from_dict(l) for l in data['lots']]
        data['archived_lots'] = [Lot.from_dict(l) for l in data.get('archived_lots', [])]
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        return cls(**data)


# ============================================================================
# STRATEGY INTERFACE
# ============================================================================

class LotSelectionStrategy(ABC):
    """Abstract strategy for selecting which lots to sell."""
    
    @abstractmethod
    def select_lots(self, position: PositionV2, quantity: float, 
                    current_price: float, context: Dict = None) -> List[str]:
        """Select lot IDs to sell."""
        pass
    
    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__.replace('Strategy', '')


class FIFOStrategy(LotSelectionStrategy):
    """First In, First Out - sell oldest lots first."""
    
    def select_lots(self, position: PositionV2, quantity: float, 
                    current_price: float, context: Dict = None) -> List[str]:
        lots = position.get_open_lots_sorted("FIFO")
        return self._select_from_lots(lots, quantity)
    
    def _select_from_lots(self, lots: List[Lot], quantity: float) -> List[str]:
        selected = []
        remaining = quantity
        for lot in lots:
            if remaining <= 0:
                break
            selected.append(lot.lot_id)
            remaining -= lot.remaining_quantity
        return selected


class LIFOStrategy(LotSelectionStrategy):
    """Last In, First Out - sell newest lots first."""
    
    def select_lots(self, position: PositionV2, quantity: float, 
                    current_price: float, context: Dict = None) -> List[str]:
        lots = position.get_open_lots_sorted("LIFO")
        selected = []
        remaining = quantity
        for lot in lots:
            if remaining <= 0:
                break
            selected.append(lot.lot_id)
            remaining -= lot.remaining_quantity
        return selected


class TaxOptimizedStrategy(LotSelectionStrategy):
    """Sell lots with highest cost basis first (HIFO) to minimize capital gains."""
    
    def select_lots(self, position: PositionV2, quantity: float, 
                    current_price: float, context: Dict = None) -> List[str]:
        lots = position.get_open_lots_sorted("HIFO")
        selected = []
        remaining = quantity
        for lot in lots:
            if remaining <= 0:
                break
            selected.append(lot.lot_id)
            remaining -= lot.remaining_quantity
        return selected


class LossHarvestingStrategy(LotSelectionStrategy):
    """Tax-loss harvesting - sell largest losses first, then FIFO for remainder."""
    
    def select_lots(self, position: PositionV2, quantity: float, 
                    current_price: float, context: Dict = None) -> List[str]:
        active_lots = [l for l in position.lots if l.is_active]
        # Largest losses first (most negative unrealized PnL per unit)
        losers = [l for l in active_lots if l.entry_price > current_price]
        losers.sort(key=lambda l: current_price - l.entry_price)  # ascending: worst first
        
        selected: List[str] = []
        selected_ids = set()
        remaining = quantity
        for lot in losers:
            if remaining <= 0:
                break
            selected.append(lot.lot_id)
            selected_ids.add(lot.lot_id)
            remaining -= lot.remaining_quantity
        
        if remaining > 0:
            for lot in position.get_open_lots_sorted("FIFO"):
                if remaining <= 0:
                    break
                if lot.lot_id in selected_ids:
                    continue
                selected.append(lot.lot_id)
                selected_ids.add(lot.lot_id)
                remaining -= lot.remaining_quantity
        
        return selected


class GainHarvestingStrategy(LotSelectionStrategy):
    """Gain harvesting - sell largest gains first, then FIFO for remainder."""
    
    def select_lots(self, position: PositionV2, quantity: float, 
                    current_price: float, context: Dict = None) -> List[str]:
        active_lots = [l for l in position.lots if l.is_active]
        winners = [l for l in active_lots if l.entry_price < current_price]
        winners.sort(key=lambda l: current_price - l.entry_price, reverse=True)
        
        selected: List[str] = []
        selected_ids = set()
        remaining = quantity
        for lot in winners:
            if remaining <= 0:
                break
            selected.append(lot.lot_id)
            selected_ids.add(lot.lot_id)
            remaining -= lot.remaining_quantity
        
        if remaining > 0:
            for lot in position.get_open_lots_sorted("FIFO"):
                if remaining <= 0:
                    break
                if lot.lot_id in selected_ids:
                    continue
                selected.append(lot.lot_id)
                selected_ids.add(lot.lot_id)
                remaining -= lot.remaining_quantity
        
        return selected


class SpecificLotStrategy(LotSelectionStrategy):
    """Sell specific lots by ID."""
    
    def __init__(self, lot_ids: List[str]):
        self.lot_ids = list(lot_ids)
    
    def select_lots(self, position: PositionV2, quantity: float, 
                    current_price: float, context: Dict = None) -> List[str]:
        by_id = {l.lot_id: l for l in position.lots if l.is_active}
        selected: List[str] = []
        remaining = quantity
        for lot_id in self.lot_ids:
            if remaining <= 1e-12:
                break
            lot = by_id.get(lot_id)
            if lot is None:
                continue
            selected.append(lot_id)
            remaining -= lot.remaining_quantity
        if remaining > 1e-12:
            raise ValueError(
                f"Specific lots cannot cover quantity {quantity}; shortfall {remaining:.8f}"
            )
        return selected

    def get_name(self) -> str:
        return "Specific"


class StrategyFactory:
    """Factory for creating strategies."""
    
    _strategies = {
        'FIFO': FIFOStrategy,
        'LIFO': LIFOStrategy,
        'HIFO': TaxOptimizedStrategy,
        'TaxLoss': LossHarvestingStrategy,
        'TaxGain': GainHarvestingStrategy,
        'Specific': SpecificLotStrategy,
    }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> LotSelectionStrategy:
        """Create a strategy by name. For Specific, pass lot_ids=[...]."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        if name == 'Specific':
            lot_ids = kwargs.get('lot_ids')
            if not lot_ids:
                raise ValueError("Specific strategy requires lot_ids=[...]")
            return SpecificLotStrategy(lot_ids)
        return cls._strategies[name]()


# ============================================================================
# MOCK EXCHANGE
# ============================================================================

class MockExchange(ExchangeInterface):
    """Mock exchange for testing."""
    
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
        self._historical_data = self._generate_historical_data()
    
    def _generate_historical_data(self):
        """Generate mock OHLCV data."""
        data = {}
        for symbol in ['BTC/USD', 'ETH/USD']:
            base_price = self.prices.get(symbol, 50000)
            symbol_data = []
            for i in range(100):
                day = datetime.now() - timedelta(days=i)
                noise = np.random.normal(0, 0.02) * base_price
                close = base_price * (1 + noise * 0.1)
                symbol_data.append({
                    'timestamp': day.isoformat(),
                    'open': close * (1 - 0.005),
                    'high': close * (1 + 0.01),
                    'low': close * (1 - 0.01),
                    'close': close,
                    'volume': 1000 + np.random.randint(0, 500)
                })
            data[symbol] = symbol_data
        return data
    
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
            return {
                'id': exchange_order_id,
                'status': o.status.value,
                'filled': o.filled_quantity,
                'remaining': o.remaining_quantity,
                'avg_price': o.average_fill_price
            }
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        return [{'id': oid, 'symbol': o.symbol, 'status': o.status.value} 
                for oid, o in self.orders.items() if o.is_active and (symbol is None or o.symbol == symbol)]
    
    def get_balances(self) -> Dict[str, Dict[str, float]]:
        return self.balances
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        return {'last': self.prices.get(symbol, 50000)}
    
    def get_historical_ohlcv(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        return self._historical_data.get(symbol, [])[:limit]
    
    def get_positions(self) -> List[Dict]:
        return [{'symbol': f"{c}/USD", 'quantity': b['total'], 'price': self.prices.get(f"{c}/USD", 0)}
                for c, b in self.balances.items() if c not in ['USD', 'USDT', 'USDC'] and b['total'] > 0]
    
    def update_price(self, symbol: str, price: float):
        self.prices[symbol] = price


# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManagerV2:
    """
    Lot-based portfolio manager with full features.
    Strategy decides which lots to sell. Closed lots are archived.
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
        
        from portfolio_risk import RiskManagerV2
        from portfolio_reporting import (
            PerformanceAnalyzerV2, AlertSystem, ReportingModuleV2,
        )
        self.risk_manager = RiskManagerV2(self, self.config)
        self.performance_analyzer = PerformanceAnalyzerV2(self)
        self.alert_system = AlertSystem(self.config)
        self.reporting = ReportingModuleV2(
            self, self.risk_manager, self.performance_analyzer, self.alert_system
        )
        
        logger.info("PortfolioManager initialized")
    
    # ============= Lot Management =============
    
    def add_lot(self, exchange: str, symbol: str, quantity: float, entry_price: float,
                order_id: str = "", tags: List[str] = None, strategy_metadata: Dict = None,
                fee: float = 0.0) -> str:
        """
        Add a new lot and debit quote cash (cost + entry fee).

        Args:
            fee: Entry fee in quote currency (stored on the lot; allocated on close).
        """
        with self._lock:
            symbol = normalize_symbol(symbol)
            if quantity <= 0 or entry_price <= 0:
                raise ValueError("quantity and entry_price must be positive")
            fee = float(fee or 0.0)
            cost = quantity * entry_price + fee
            base, quote = split_symbol(symbol)

            quote_key = f"{exchange}:{quote}"
            quote_bal = self._balances.get(quote_key)
            if quote_bal is None:
                # Auto-provision (tests / incomplete setup); prefer seeding balances
                logger.warning(
                    f"No {quote} balance on {exchange}; auto-funding {cost:.4f} for buy"
                )
                self._balances[quote_key] = Balance(exchange, quote, cost, 0.0)
                quote_bal = self._balances[quote_key]
            if quote_bal.free < cost:
                raise ValueError(
                    f"Insufficient {quote} on {exchange}: need {cost:.4f}, "
                    f"free {quote_bal.free:.4f}"
                )
            quote_bal.total -= cost

            base_key = f"{exchange}:{base}"
            base_bal = self._balances.get(base_key)
            if base_bal is None:
                self._balances[base_key] = Balance(exchange, base, 0.0, 0.0)
                base_bal = self._balances[base_key]
            base_bal.total += quantity

            position_key = f"{exchange}:{symbol}"
            if position_key not in self._positions:
                self._positions[position_key] = PositionV2(symbol=symbol, exchange=exchange)

            lot = Lot(
                exchange=exchange,
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                remaining_quantity=quantity,
                entry_fee=fee,
                order_id=order_id,
                tags=tags or [],
                strategy_metadata=strategy_metadata or {},
            )

            pos = self._positions[position_key]
            pos.add_lot(lot)
            pos.last_price = float(entry_price)
            pos.last_update = datetime.now()
            self._last_prices[symbol] = float(entry_price)

            trade = Trade(
                trade_id=f"buy_{lot.lot_id}_{datetime.now().timestamp()}",
                exchange=exchange,
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                price=entry_price,
                timestamp=datetime.now(),
                order_id=order_id,
                fee=fee,
                lot_id=lot.lot_id,
                fee_currency=quote,
            )
            self._ledger.append(trade)

            return lot.lot_id
    
    def close_lots(self, exchange: str, symbol: str, quantity: float, exit_price: float,
                   strategy: LotSelectionStrategy, fee: float = 0.0) -> List[Tuple[Lot, float]]:
        """Close lots using the provided strategy. ``fee`` is total exit fee for ``quantity``."""
        with self._lock:
            symbol = normalize_symbol(symbol)
            position_key = f"{exchange}:{symbol}"
            if position_key not in self._positions:
                raise ValueError(f"No position found for {symbol} on {exchange}")
            
            position = self._positions[position_key]
            
            if quantity > position.total_quantity + 1e-12:
                raise ValueError(f"Insufficient quantity: have {position.total_quantity}, want {quantity}")
            
            context = {'exit_price': exit_price, 'fee': fee}
            selected_lot_ids = strategy.select_lots(position, quantity, exit_price, context)
            
            if not selected_lot_ids:
                raise ValueError("Strategy selected no lots to sell")

            # Pre-validate coverage so we never partially mutate on under-selection
            active_by_id = {l.lot_id: l for l in position.lots if l.is_active}
            coverage = 0.0
            seen_ids = set()
            for lot_id in selected_lot_ids:
                if lot_id in seen_ids:
                    continue
                lot = active_by_id.get(lot_id)
                if lot is None:
                    continue
                coverage += lot.remaining_quantity
                seen_ids.add(lot_id)
            if coverage + 1e-12 < quantity:
                raise ValueError(
                    f"Strategy lots cover only {coverage:.8f} of requested {quantity:.8f}"
                )
            
            remaining = quantity
            results = []
            closed_total = 0.0
            
            for lot_id in selected_lot_ids:
                if remaining <= 1e-12:
                    break
                
                lot = None
                for l in position.lots:
                    if l.lot_id == lot_id and l.is_active:
                        lot = l
                        break
                
                if not lot:
                    logger.warning(f"Lot {lot_id} not found or not active, skipping")
                    continue
                
                close_qty = min(remaining, lot.remaining_quantity)
                exit_fee_part = fee * (close_qty / quantity) if quantity > 0 else 0.0
                # close_partial also allocates lot.entry_fee pro-rata
                pnl = lot.close_partial(close_qty, exit_price, exit_fee_part)
                results.append((lot, pnl))
                remaining -= close_qty
                closed_total += close_qty
                
                trade = Trade(
                    trade_id=f"sell_{lot.lot_id}_{datetime.now().timestamp()}",
                    exchange=exchange,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=close_qty,
                    price=exit_price,
                    timestamp=datetime.now(),
                    order_id="",
                    fee=exit_fee_part,
                    lot_id=lot.lot_id,
                    fee_currency=split_symbol(symbol)[1],
                    realized_pnl=pnl,
                )
                self._ledger.append(trade)
                self.risk_manager.daily_pnl += pnl
            
            if remaining > 1e-12:
                raise ValueError(
                    f"Could not fully close quantity {quantity}: shortfall {remaining:.8f} "
                    f"(strategy selected {len(selected_lot_ids)} lot id(s), closed {closed_total:.8f})"
                )
            
            position.last_price = float(exit_price)
            position.last_update = datetime.now()
            self._last_prices[symbol] = float(exit_price)
            position._archive_closed_lots()
            self._update_balances_for_sell(exchange, symbol, closed_total, exit_price, fee)
            
            return results
    
    def sell_with_strategy(self, exchange: str, symbol: str, quantity: float, 
                          exit_price: float, strategy_name: str = "FIFO", 
                          fee: float = 0.0) -> List[Tuple[Lot, float]]:
        """Sell using a named strategy."""
        strategy = StrategyFactory.create(strategy_name)
        return self.close_lots(exchange, symbol, quantity, exit_price, strategy, fee)
    
    # ============= Order Management =============
    
    def submit_order(self, exchange: str, symbol: str, side: OrderSide, order_type: OrderType,
                     quantity: float, price: Optional[float] = None, **kwargs) -> str:
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
                                            f"Order rejected: {reason}")
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
            return True
    
    def get_order(self, internal_order_id: str) -> Optional[Order]:
        with self._lock:
            return self._orders.get(internal_order_id)
    
    def get_active_orders(self, exchange: Optional[str] = None, 
                          symbol: Optional[str] = None) -> List[Order]:
        with self._lock:
            orders = list(self._active_orders.values())
            if exchange:
                orders = [o for o in orders if o.exchange == exchange]
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            return orders
    
    # ============= Queries =============
    
    def get_position(self, exchange: str, symbol: str) -> Optional[PositionV2]:
        with self._lock:
            symbol = normalize_symbol(symbol)
            return self._positions.get(f"{exchange}:{symbol}")
    
    def get_all_positions(self) -> List[PositionV2]:
        with self._lock:
            return list(self._positions.values())
    
    def get_open_lots(self, exchange: Optional[str] = None, 
                      symbol: Optional[str] = None) -> List[Lot]:
        with self._lock:
            lots = []
            for position in self._positions.values():
                if exchange and position.exchange != exchange:
                    continue
                if symbol and position.symbol != symbol:
                    continue
                lots.extend([l for l in position.lots if l.is_active])
            return lots
    
    def get_archived_lots(self, exchange: Optional[str] = None,
                          symbol: Optional[str] = None) -> List[Lot]:
        with self._lock:
            lots = []
            for position in self._positions.values():
                if exchange and position.exchange != exchange:
                    continue
                if symbol and position.symbol != symbol:
                    continue
                lots.extend(position.archived_lots)
            return lots
    
    def get_lot(self, lot_id: str) -> Optional[Lot]:
        with self._lock:
            for position in self._positions.values():
                for lot in position.lots + position.archived_lots:
                    if lot.lot_id == lot_id:
                        return lot
            return None
    
    def get_balance(self, exchange: str, currency: str) -> Optional[Balance]:
        with self._lock:
            return self._balances.get(f"{exchange}:{currency}")
    
    def get_all_balances(self) -> List[Balance]:
        with self._lock:
            return list(self._balances.values())
    
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
        """
        Equity = base-currency cash + lot market value + free base inventory
        not already represented by lots (e.g. mock exchange seed balances).
        """
        with self._lock:
            total = sum(
                b.total for b in self._balances.values()
                if b.currency == Config.BASE_CURRENCY
            )
            for pos in self._positions.values():
                total += pos.market_value

            # Free base assets beyond open lot qty, marked at last known price
            lot_qty_by_key: Dict[str, float] = defaultdict(float)
            for pos in self._positions.values():
                base, _ = split_symbol(pos.symbol)
                lot_qty_by_key[f"{pos.exchange}:{base}"] += pos.total_quantity

            for key, bal in self._balances.items():
                if bal.currency == Config.BASE_CURRENCY:
                    continue
                free_base = bal.total - lot_qty_by_key.get(key, 0.0)
                if free_base <= 1e-12:
                    continue
                mark = self._mark_price_for_currency(bal.exchange, bal.currency)
                if mark is not None and mark > 0:
                    total += free_base * mark
            return total

    def _mark_price_for_currency(self, exchange: str, currency: str) -> Optional[float]:
        """Best-effort mark for a base currency (prefer pair with BASE_CURRENCY)."""
        pair = f"{currency}/{Config.BASE_CURRENCY}"
        if pair in self._last_prices:
            return self._last_prices[pair]
        for sym, px in self._last_prices.items():
            if sym.startswith(f"{currency}/"):
                return px
        for pos in self._positions.values():
            if pos.exchange == exchange:
                base, _ = split_symbol(pos.symbol)
                if base == currency and pos.last_price > 0:
                    return pos.last_price
        return None

    def update_mark(self, symbol: str, price: float, exchange: Optional[str] = None) -> None:
        """Thread-safe mark-to-market update for a symbol."""
        with self._lock:
            symbol = normalize_symbol(symbol)
            price = float(price)
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid mark price for {symbol}: {price}")
            self._last_prices[symbol] = price
            for pos in self._positions.values():
                if pos.symbol != symbol:
                    continue
                if exchange is not None and pos.exchange != exchange:
                    continue
                pos.last_price = price
                pos.last_update = datetime.now()
    
    def calculate_realized_pnl(self) -> float:
        with self._lock:
            return sum(pos.realized_pnl for pos in self._positions.values())
    
    def calculate_unrealized_pnl(self) -> float:
        with self._lock:
            return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    def get_lot_statistics(self) -> Dict[str, Any]:
        with self._lock:
            open_lots = self.get_open_lots()
            archived_lots = self.get_archived_lots()
            # Realized PnL on still-active partially closed lots
            partial_lots = [l for l in open_lots if l.closed_quantity > 0]
            realized_sources = list(archived_lots) + partial_lots
            total_pnl = sum(l.realized_pnl for l in realized_sources)
            winning_lots = [l for l in realized_sources if l.realized_pnl > 0]
            losing_lots = [l for l in realized_sources if l.realized_pnl < 0]
            n_realized = max(len(realized_sources), 1)
            
            return {
                'total_lots': len(open_lots) + len(archived_lots),
                'open_lots': len(open_lots),
                'archived_lots': len(archived_lots),
                'total_realized_pnl': total_pnl,
                'winning_lots': len(winning_lots),
                'losing_lots': len(losing_lots),
                'win_rate': len(winning_lots) / n_realized if realized_sources else 0.0,
                'avg_pnl_per_lot': total_pnl / n_realized if realized_sources else 0.0,
                'best_lot': max((l.realized_pnl for l in realized_sources), default=0),
                'worst_lot': min((l.realized_pnl for l in realized_sources), default=0),
            }
    
    def get_portfolio_summary(self) -> Dict:
        with self._lock:
            return {
                'total_equity': self.get_total_equity(),
                # Quote/base cash only — do not sum BTC qty with USD dollars
                'total_cash': sum(
                    b.total for b in self._balances.values()
                    if b.currency == Config.BASE_CURRENCY
                ),
                'total_positions_value': sum(pos.market_value for pos in self._positions.values()),
                'total_realized_pnl': self.calculate_realized_pnl(),
                'total_unrealized_pnl': self.calculate_unrealized_pnl(),
                'total_pnl': self.calculate_realized_pnl() + self.calculate_unrealized_pnl(),
                'lot_statistics': self.get_lot_statistics(),
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
                    'archived_lot_count': len(pos.archived_lots),
                } for pos in self._positions.values()]
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
        base, quote = split_symbol(symbol)
        
        base_bal = self._balances.get(f"{exchange}:{base}")
        if base_bal:
            base_bal.total -= quantity
        
        quote_bal = self._balances.get(f"{exchange}:{quote}")
        if quote_bal:
            quote_bal.total += quantity * exit_price - fee
        else:
            self._balances[f"{exchange}:{quote}"] = Balance(
                exchange, quote, quantity * exit_price - fee, 0.0
            )
    
    def _sync_exchange(self, exchange_name: str, exchange: ExchangeInterface):
        for currency, data in exchange.get_balances().items():
            key = f"{exchange_name}:{currency}"
            if key in self._balances:
                self._balances[key].update(data['total'], data.get('locked', 0))
            else:
                self._balances[key] = Balance(exchange_name, currency, data['total'], data.get('locked', 0))
        
        for position in self._positions.values():
            if position.exchange == exchange_name:
                ticker = exchange.get_ticker(position.symbol)
                if ticker and 'last' in ticker:
                    position.last_price = float(ticker['last'])
                    self._last_prices[position.symbol] = float(ticker['last'])
    
    # ============= Sync Thread =============
    
    def start_sync_thread(self):
        if self._sync_thread and self._sync_thread.is_alive():
            return
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
    
    def stop_sync_thread(self):
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
    
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
        if filepath is None:
            filepath = Config.STATE_FILE
        with self._lock:
            state = {
                'ledger': [t.to_dict() for t in self._ledger],
                'positions': {k: v.to_dict() for k, v in self._positions.items()},
                'balances': {k: v.to_dict() for k, v in self._balances.items()},
                'orders': {k: v.to_dict() for k, v in self._orders.items()},
                'last_prices': self._last_prices,
                'timestamp': datetime.now().isoformat()
            }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str = None):
        if filepath is None:
            filepath = Config.STATE_FILE
        if not Path(filepath).exists():
            return
        with self._lock:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self._ledger = [Trade.from_dict(t) for t in state['ledger']]
            self._positions = {k: PositionV2.from_dict(v) for k, v in state['positions'].items()}
            self._balances = {k: Balance.from_dict(v) for k, v in state['balances'].items()}
            self._orders = {k: Order.from_dict(v) for k, v in state['orders'].items()}
            self._last_prices = state['last_prices']
            # Rebuild active-order index (not persisted separately)
            self._active_orders = {
                oid: o for oid, o in self._orders.items() if o.is_active
            }
        logger.info(f"State loaded from {filepath}")


# ============================================================================
# COMPLETE TEST SUITE
# ============================================================================

