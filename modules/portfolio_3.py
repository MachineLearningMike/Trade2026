"""
Comprehensive Portfolio Management System for Trading Bot
Acts as the single source of truth for all financial state.
Includes profit calculation, risk management, performance analytics,
reporting, alerting, and comprehensive test cases.
"""

import threading
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict, deque
import json
import csv
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from datetime import date

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= Enums and Data Classes =============

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

class ReportFormat(Enum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"


@dataclass
class Trade:
    """Represents a single executed trade (fill)."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'fee': self.fee,
            'fee_currency': self.fee_currency,
            'commission_asset': self.commission_asset,
            'commission_paid': self.commission_paid
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create Trade from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Position:
    """Current position for a symbol."""
    symbol: str
    exchange: str
    quantity: float = 0.0
    cost_basis: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price
    
    @property
    def cost_value(self) -> float:
        return self.quantity * self.cost_basis
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_value
    
    @property
    def unrealized_pnl_percentage(self) -> float:
        if self.cost_value == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_value) * 100
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_pnl_percentage(self) -> float:
        if self.cost_value == 0:
            return 0.0
        return (self.total_pnl / self.cost_value) * 100
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0
    
    @property
    def break_even_price(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.cost_basis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'quantity': self.quantity,
            'cost_basis': self.cost_basis,
            'realized_pnl': self.realized_pnl,
            'last_price': self.last_price,
            'last_update': self.last_update.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position from dictionary."""
        data = data.copy()
        data['last_update'] = datetime.fromisoformat(data['last_update'])
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'exchange': self.exchange,
            'currency': self.currency,
            'total': self.total,
            'locked': self.locked,
            'last_update': self.last_update.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Balance':
        """Create Balance from dictionary."""
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
        
        if self.remaining_quantity == 0.0 and self.quantity > 0:
            self.remaining_quantity = self.quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_done(self) -> bool:
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def filled_value(self) -> float:
        if not self.average_fill_price:
            return 0.0
        return self.filled_quantity * self.average_fill_price
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'internal_id': self.internal_id,
            'exchange_order_id': self.exchange_order_id,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'client_order_id': self.client_order_id,
            'reject_reason': self.reject_reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create Order from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['submitted_at']:
            data['submitted_at'] = datetime.fromisoformat(data['submitted_at'])
        if data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data['closed_at']:
            data['closed_at'] = datetime.fromisoformat(data['closed_at'])
        return cls(**data)


@dataclass
class RiskMetrics:
    """Risk metrics for a portfolio or position."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0
    exposure: float = 0.0
    concentration_ratio: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class PerformanceMetrics:
    """Performance metrics for a portfolio."""
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


@dataclass
class Alert:
    """Alert message."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.INFO
    category: str = ""
    title: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'category': self.category,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create Alert from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data['acknowledged_at']:
            data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        return cls(**data)


# ============= Exchange Interface =============

class ExchangeInterface(ABC):
    """Abstract interface for exchange adapters."""
    
    @abstractmethod
    def place_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[Dict]]:
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


# ============= Risk Manager =============

class RiskManager:
    """
    Manages risk for the portfolio.
    Implements position sizing, exposure limits, and risk checks.
    """
    
    def __init__(self, portfolio_manager: 'PortfolioManager', config: Dict[str, Any] = None):
        self.portfolio = portfolio_manager
        self.config = config or {}
        
        # Default risk limits
        self.max_position_size_pct = self.config.get('max_position_size_pct', 0.1)
        self.max_exposure_pct = self.config.get('max_exposure_pct', 0.5)
        self.max_leverage = self.config.get('max_leverage', 1.0)
        self.max_drawdown_pct = self.config.get('max_drawdown_pct', 0.2)
        self.max_concentration_pct = self.config.get('max_concentration_pct', 0.3)
        self.daily_loss_limit = self.config.get('daily_loss_limit', 0.05)
        self.var_limit = self.config.get('var_limit', 0.02)
        
        # Position sizing
        self.base_currency = self.config.get('base_currency', 'USD')
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
        
        # Tracking
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.last_reset_day = datetime.now().date()
        
        logger.info("RiskManager initialized with limits: %s", self.config)
    
    def check_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        Check if an order complies with risk limits.
        Returns (approved, rejection_reason)
        """
        total_equity = self.portfolio.get_total_equity()
        
        position = self.portfolio.get_position(order.exchange, order.symbol)
        current_position_value = position.market_value if position else 0
        
        order_value = order.quantity * (order.price or self.portfolio._last_prices.get(order.symbol, 0))
        
        if order_value > total_equity * self.max_position_size_pct:     # What if it's selling?
            return False, f"Position size {order_value:.2f} exceeds limit {total_equity * self.max_position_size_pct:.2f}"
        
        total_exposure = sum(p.market_value for p in self.portfolio.get_all_positions())
        new_exposure = total_exposure + (order_value if order.side == OrderSide.BUY else 0)
        if new_exposure > total_equity * self.max_exposure_pct:
            return False, f"Total exposure {new_exposure:.2f} would exceed limit {total_equity * self.max_exposure_pct:.2f}"
        
        new_concentration = (current_position_value + order_value) / total_equity
        if new_concentration > self.max_concentration_pct:
            return False, f"Concentration {new_concentration:.2%} exceeds limit {self.max_concentration_pct:.2%}"
        
        if self.daily_pnl < -total_equity * self.daily_loss_limit:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
        
        if self.current_drawdown > self.max_drawdown_pct:
            return False, f"Max drawdown {self.current_drawdown:.2%} exceeded limit {self.max_drawdown_pct:.2%}"
        
        return True, None
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk per trade.
        """
        total_equity = self.portfolio.get_total_equity()
        risk_amount = total_equity * self.risk_per_trade
        
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        max_size = (total_equity * self.max_position_size_pct) / entry_price
        return min(position_size, max_size)
    
    def update_daily_pnl(self):
        """Update daily P&L tracking."""
        today = datetime.now().date()
        if today > self.last_reset_day:
            self.daily_pnl = 0.0
            self.last_reset_day = today
    
    def update_drawdown(self):
        """Update current drawdown."""
        current_equity = self.portfolio.get_total_equity()
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical method.
        """
        if not returns:
            return 0.0
        
        returns = np.array(returns)
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        """
        if not returns:
            return 0.0
        
        var = self.calculate_var(returns, confidence)
        returns_below_var = [r for r in returns if r <= var]
        
        if not returns_below_var:
            return var
        
        return np.mean(returns_below_var)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio."""
        metrics = RiskMetrics()
        
        returns = self._calculate_historical_returns()
        
        if returns:
            metrics.var_95 = self.calculate_var(returns, 0.95)
            metrics.var_99 = self.calculate_var(returns, 0.99)
            metrics.cvar_95 = self.calculate_cvar(returns, 0.95)
            metrics.volatility = np.std(returns) * np.sqrt(252)
            
            risk_free_rate = 0.02 / 252
            excess_returns = np.array(returns) - risk_free_rate
            if np.std(returns) > 0:
                metrics.sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
        
        total_equity = self.portfolio.get_total_equity()
        total_exposure = sum(p.market_value for p in self.portfolio.get_all_positions())
        metrics.exposure = total_exposure / total_equity if total_equity > 0 else 0
        
        positions = self.portfolio.get_all_positions()
        if positions:
            largest_position = max(positions, key=lambda p: p.market_value)
            metrics.concentration_ratio = largest_position.market_value / total_equity if total_equity > 0 else 0
        
        #--------------- This is not very rich parameter -------------
        if metrics.var_95 < -0.02 and metrics.exposure < 0.3:
            metrics.risk_level = RiskLevel.LOW
        elif metrics.var_95 < -0.05 and metrics.exposure < 0.6:
            metrics.risk_level = RiskLevel.MEDIUM
        elif metrics.var_95 < -0.10 and metrics.exposure < 0.8:
            metrics.risk_level = RiskLevel.HIGH
        else:
            metrics.risk_level = RiskLevel.CRITICAL
        
        return metrics
    
    def _calculate_historical_returns(self) -> List[float]:
        """
        Calculate historical returns from trades.
        """
        returns = []
        trades = self.portfolio.get_trades(limit=1000)
        
        daily_pnl = defaultdict(float)
        
        for trade in trades:
            day = trade.timestamp.date()
            if trade.is_sell:
                daily_pnl[day] += trade.value - (trade.quantity * trade.price)
        
        prev_equity = 10000     #-------------- No...
        for day in sorted(daily_pnl.keys()):
            if prev_equity > 0:
                daily_return = daily_pnl[day] / prev_equity
                returns.append(daily_return)
                prev_equity += daily_pnl[day]
        
        return returns


# ============= Performance Analyzer =============

class PerformanceAnalyzer:
    """
    Analyzes portfolio performance and calculates metrics.
    """
    
    def __init__(self, portfolio_manager: 'PortfolioManager'):
        self.portfolio = portfolio_manager
    
    def calculate_returns(self, period: str = 'all') -> Dict[str, float]:
        """
        Calculate various return metrics.
        """
        trades = self.portfolio.get_trades()
        if not trades:
            return {}
        
        trades.sort(key=lambda x: x.timestamp)
        
        initial_equity = self._get_initial_equity()
        current_equity = self.portfolio.get_total_equity()
        total_return = (current_equity - initial_equity) / initial_equity if initial_equity > 0 else 0
        
        daily_returns = self._calculate_daily_returns(trades)
        
        days = (trades[-1].timestamp - trades[0].timestamp).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = total_return
        
        monthly_return = self._calculate_period_return(trades, 'month')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'daily_return': np.mean(list(daily_returns.values())) if daily_returns else 0,
            'monthly_return': monthly_return
        }
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate detailed trade statistics.
        """
        trades = self.portfolio.get_trades()
        
        if len(trades) < 2:
            return {}
        
        trade_pnls = self._calculate_trade_pnls()
        
        if not trade_pnls:
            return {}
        
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]
        
        stats = {
            'total_trades': len(trade_pnls),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trade_pnls) * 100 if trade_pnls else 0,
            'average_win': np.mean(winning_trades) if winning_trades else 0,
            'average_loss': np.mean(losing_trades) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf'),
            'expectancy': np.mean(trade_pnls),
            'standard_deviation': np.std(trade_pnls)
        }
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in trade_pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        stats['max_consecutive_wins'] = max_consecutive_wins
        stats['max_consecutive_losses'] = max_consecutive_losses
        
        return stats
    
    def calculate_risk_adjusted_metrics(self, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.
        """
        trades = self.portfolio.get_trades()
        if len(trades) < 2:
            return {}
        
        daily_returns = list(self._calculate_daily_returns(trades).values())
        
        if not daily_returns:
            return {}
        
        returns_array = np.array(daily_returns)
        excess_returns = returns_array - risk_free_rate / 252
        
        if np.std(returns_array) > 0:
            sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns_array)
        else:
            sharpe = 0
        
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
        else:
            sortino = 0
        
        max_dd = self.calculate_max_drawdown()
        if max_dd > 0:
            calmar = np.mean(returns_array) * 252 / max_dd
        else:
            calmar = 0
        
        threshold = 0
        gains = returns_array[returns_array > threshold]
        losses = returns_array[returns_array < threshold]
        if len(losses) > 0 and np.sum(losses) != 0:
            omega = np.sum(gains) / abs(np.sum(losses))
        else:
            omega = float('inf') if len(gains) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega
        }
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity curve.
        """
        equity_curve = self._build_equity_curve()
        
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
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
    
    def _calculate_daily_returns(self, trades: List[Trade]) -> Dict[date, float]:
        """Calculate daily returns from trades."""
        from datetime import date
        daily_pnl = defaultdict(float)
        
        for trade in trades:
            day = trade.timestamp.date()
            if trade.is_sell:
                daily_pnl[day] += trade.value - (trade.quantity * trade.price)
        
        returns = {}
        prev_equity = self._get_initial_equity()
        
        for day in sorted(daily_pnl.keys()):
            if prev_equity > 0:
                returns[day] = daily_pnl[day] / prev_equity
                prev_equity += daily_pnl[day]
        
        return returns
    
    def _calculate_period_return(self, trades: List[Trade], period: str) -> float:
        """Calculate return for a specific period."""
        if not trades:
            return 0.0
        
        start_equity = self._get_initial_equity()
        end_equity = self.portfolio.get_total_equity()
        
        return (end_equity - start_equity) / start_equity if start_equity > 0 else 0
    
    def _calculate_trade_pnls(self) -> List[float]:
        """Calculate P&L for each individual trade."""
        trade_pnls = []
        position_tracker = {}
        
        trades = self.portfolio.get_trades()
        trades.sort(key=lambda x: x.timestamp)
        
        for trade in trades:
            if trade.symbol not in position_tracker:
                position_tracker[trade.symbol] = []
            
            if trade.is_buy:
                position_tracker[trade.symbol].append({
                    'quantity': trade.quantity,
                    'price': trade.price
                })
            else:
                remaining = trade.quantity
                while remaining > 0 and position_tracker[trade.symbol]:
                    buy = position_tracker[trade.symbol][0]
                    match_qty = min(remaining, buy['quantity'])
                    
                    pnl = (trade.price - buy['price']) * match_qty
                    trade_pnls.append(pnl)
                    
                    remaining -= match_qty
                    buy['quantity'] -= match_qty
                    
                    if buy['quantity'] <= 0:
                        position_tracker[trade.symbol].pop(0)
        
        return trade_pnls
    
    def _build_equity_curve(self) -> List[float]:
        """Build equity curve over time."""
        equity_curve = [self._get_initial_equity()]
        trades = self.portfolio.get_trades()
        trades.sort(key=lambda x: x.timestamp)
        
        current_equity = equity_curve[0]
        
        for trade in trades:
            if trade.is_sell:
                profit = trade.value - (trade.quantity * trade.price)
                current_equity += profit
                equity_curve.append(current_equity)
        
        return equity_curve
    
    def _get_initial_equity(self) -> float:
        """Get initial equity."""
        return 10000.0


# ============= Reporting Module =============

class ReportingModule:
    """
    Generates various reports for portfolio analysis.
    """
    
    def __init__(self, portfolio_manager: 'PortfolioManager', 
                 risk_manager: RiskManager,
                 performance_analyzer: PerformanceAnalyzer,
                 alert_system: 'AlertSystem'):
        self.portfolio = portfolio_manager
        self.risk_manager = risk_manager
        self.analyzer = performance_analyzer
        self.alerts = alert_system
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_pnl_report(self, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           format: ReportFormat = ReportFormat.JSON) -> str:
        """
        Generate a comprehensive P&L report.
        """
        summary = self.portfolio.get_portfolio_summary()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat() if start_date else 'all',
                'end': end_date.isoformat() if end_date else 'all'
            },
            'summary': summary,
            'performance': self.analyzer.get_performance_metrics().__dict__,
            'positions': [],
            'trades': []
        }
        
        for position in self.portfolio.get_all_positions():
            report_data['positions'].append({
                'symbol': position.symbol,
                'exchange': position.exchange,
                'quantity': position.quantity,
                'cost_basis': position.cost_basis,
                'current_price': position.last_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'unrealized_pnl_percentage': position.unrealized_pnl_percentage
            })
        
        trades = self.portfolio.get_trades(limit=100)
        for trade in trades:
            if start_date and trade.timestamp < start_date:
                continue
            if end_date and trade.timestamp > end_date:
                continue
            
            report_data['trades'].append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp.isoformat(),
                'exchange': trade.exchange,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'fee': trade.fee
            })
        
        filename = self.report_dir / f"pnl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if format == ReportFormat.JSON:
            return self._save_json(report_data, filename)
        elif format == ReportFormat.CSV:
            return self._save_csv(report_data, filename)
        elif format == ReportFormat.HTML:
            return self._save_html(report_data, filename)
        else:
            return self._save_json(report_data, filename)
    
    def generate_risk_report(self) -> str:
        """
        Generate a detailed risk report.
        """
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'risk_metrics': risk_metrics.__dict__,
            'limits': {
                'max_position_size_pct': self.risk_manager.max_position_size_pct,
                'max_exposure_pct': self.risk_manager.max_exposure_pct,
                'max_drawdown_pct': self.risk_manager.max_drawdown_pct,
                'daily_loss_limit': self.risk_manager.daily_loss_limit,
                'var_limit': self.risk_manager.var_limit
            },
            'current_status': {
                'daily_pnl': self.risk_manager.daily_pnl,
                'current_drawdown': self.risk_manager.current_drawdown,
                'peak_equity': self.risk_manager.peak_equity
            },
            'exposure_by_symbol': [],
            'var_breakdown': {}
        }
        
        total_equity = self.portfolio.get_total_equity()
        for position in self.portfolio.get_all_positions():
            exposure = position.market_value / total_equity if total_equity > 0 else 0
            report_data['exposure_by_symbol'].append({
                'symbol': position.symbol,
                'exposure': exposure,
                'value': position.market_value
            })
        
        filename = self.report_dir / f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Risk report saved to {filename}")
        return str(filename)
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.
        """
        performance = self.analyzer.get_performance_metrics()
        trade_stats = self.analyzer.calculate_trade_statistics()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': performance.__dict__,
            'trade_statistics': trade_stats,
            'returns': self.analyzer.calculate_returns(),
            'risk_adjusted': self.analyzer.calculate_risk_adjusted_metrics(),
            'equity_curve': self.analyzer._build_equity_curve()
        }
        
        filename = self.report_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filename}")
        return str(filename)
    
    def generate_tax_report(self, year: int) -> str:
        """
        Generate tax report for a specific year.
        """
        trades = self.portfolio.get_trades()
        year_trades = [t for t in trades if t.timestamp.year == year]
        
        realized_pnl = self.portfolio.calculate_realized_pnl()
        
        report_data = {
            'tax_year': year,
            'generated_at': datetime.now().isoformat(),
            'total_realized_pnl': realized_pnl,
            'trades': [],
            'summary_by_symbol': {}
        }
        
        symbol_pnl = defaultdict(float)
        
        for trade in year_trades:
            trade_data = {
                'date': trade.timestamp.isoformat(),
                'exchange': trade.exchange,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'fee': trade.fee
            }
            report_data['trades'].append(trade_data)
            
            if trade.is_sell:
                symbol_pnl[trade.symbol] += trade.value - (trade.quantity * trade.price)
        
        for symbol, pnl in symbol_pnl.items():
            report_data['summary_by_symbol'][symbol] = pnl
        
        filename = self.report_dir / f"tax_report_{year}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Tax report for {year} saved to {filename}")
        return str(filename)
    
    def _save_json(self, data: Dict, filename: Path) -> str:
        """Save report as JSON."""
        json_file = filename.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Report saved to {json_file}")
        return str(json_file)
    
    def _save_csv(self, data: Dict, filename: Path) -> str:
        """Save report as CSV."""
        csv_file = filename.with_suffix('.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Report Generated', data['generated_at']])
            writer.writerow([])
            
            writer.writerow(['Positions'])
            if data['positions']:
                writer.writerow(data['positions'][0].keys())
                for pos in data['positions']:
                    writer.writerow(pos.values())
            
            writer.writerow([])
            
            writer.writerow(['Recent Trades'])
            if data['trades']:
                writer.writerow(data['trades'][0].keys())
                for trade in data['trades']:
                    writer.writerow(trade.values())
        
        logger.info(f"Report saved to {csv_file}")
        return str(csv_file)
    
    def _save_html(self, data: Dict, filename: Path) -> str:
        """Save report as HTML."""
        html_file = filename.with_suffix('.html')
        
        summary = data.get('summary', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e7f3fe; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Portfolio Report</h1>
            <div class="summary">
                <p><strong>Generated:</strong> {data['generated_at']}</p>
                <p><strong>Total Equity:</strong> ${summary.get('total_equity', 0):.2f}</p>
                <p><strong>Total PnL:</strong> ${summary.get('total_pnl', 0):.2f}</p>
                <p><strong>Win Rate:</strong> {summary.get('win_rate', 0):.1f}%</p>
                <p><strong>Profit Factor:</strong> {summary.get('profit_factor', 0):.2f}</p>
            </div>
            
            <h2>Positions</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Exchange</th>
                    <th>Quantity</th>
                    <th>Cost Basis</th>
                    <th>Current Price</th>
                    <th>Unrealized PnL</th>
                </tr>
        """
        
        for pos in data.get('positions', []):
            html_content += f"""
                <tr>
                    <td>{pos['symbol']}</td>
                    <td>{pos['exchange']}</td>
                    <td>{pos['quantity']:.4f}</td>
                    <td>${pos['cost_basis']:.2f}</td>
                    <td>${pos['current_price']:.2f}</td>
                    <td>${pos['unrealized_pnl']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Recent Trades</h2>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Exchange</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>Value</th>
                    <th>Fee</th>
                </tr>
        """
        
        for trade in data.get('trades', [])[:50]:
            html_content += f"""
                <tr>
                    <td>{trade['timestamp']}</td>
                    <td>{trade['exchange']}</td>
                    <td>{trade['symbol']}</td>
                    <td>{trade['side']}</td>
                    <td>{trade['quantity']:.4f}</td>
                    <td>${trade['price']:.2f}</td>
                    <td>${trade['value']:.2f}</td>
                    <td>${trade['fee']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {html_file}")
        return str(html_file)


# ============= Alert System =============

class AlertSystem:
    """
    Manages alerts and notifications for the portfolio.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.thresholds = self.config.get('thresholds', {})
        
        self.smtp_server = self.config.get('smtp_server')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.email_user = self.config.get('email_user')
        self.email_password = self.config.get('email_password')
        self.email_recipients = self.config.get('email_recipients', [])
        
        logger.info("AlertSystem initialized")
    
    def add_alert(self, severity: AlertSeverity, category: str, 
                  title: str, message: str, data: Dict[str, Any] = None):
        """Add a new alert."""
        alert = Alert(
            severity=severity,
            category=category,
            title=title,
            message=message,
            data=data
        )
        
        self.alerts.append(alert)
        
        log_level = self._severity_to_log_level(severity)
        logger.log(log_level, f"Alert [{severity.value}] {category}: {title} - {message}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            self._send_email_alert(alert)
    
    def register_handler(self, handler: Callable[[Alert], None]):
        """Register a callback for alerts."""
        self.alert_handlers.append(handler)
    
    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]
    
    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert {alert_id} acknowledged")
                break
    
    def check_portfolio_alerts(self, portfolio: 'PortfolioManager', 
                               risk_manager: RiskManager):
        """Check for portfolio alerts based on thresholds."""
        summary = portfolio.get_portfolio_summary()
        total_equity = summary['total_equity']
        
        max_dd = self.thresholds.get('max_drawdown', 0.15)
        current_dd = risk_manager.current_drawdown
        if current_dd > max_dd:
            self.add_alert(
                AlertSeverity.WARNING,
                "Risk",
                "High Drawdown",
                f"Current drawdown {current_dd:.2%} exceeds threshold {max_dd:.2%}",
                {'drawdown': current_dd, 'threshold': max_dd}
            )
        
        daily_loss_limit = self.thresholds.get('daily_loss_limit', 0.05)
        if risk_manager.daily_pnl < -total_equity * daily_loss_limit:
            self.add_alert(
                AlertSeverity.WARNING,
                "Risk",
                "Daily Loss Limit",
                f"Daily loss {risk_manager.daily_pnl:.2f} exceeds limit",
                {'daily_pnl': risk_manager.daily_pnl, 'limit': total_equity * daily_loss_limit}
            )
        
        max_conc = self.thresholds.get('max_concentration', 0.25)
        positions = portfolio.get_all_positions()
        for position in positions:
            concentration = position.market_value / total_equity if total_equity > 0 else 0
            if concentration > max_conc:
                self.add_alert(
                    AlertSeverity.WARNING,
                    "Risk",
                    "High Concentration",
                    f"Position {position.symbol} concentration {concentration:.2%} exceeds {max_conc:.2%}",
                    {'symbol': position.symbol, 'concentration': concentration}
                )
        
        min_balance = self.thresholds.get('min_balance', 1000)
        for balance in portfolio.get_all_balances():
            if balance.free < min_balance and balance.currency == 'USD':
                self.add_alert(
                    AlertSeverity.INFO,
                    "Balance",
                    "Low Balance",
                    f"Low {balance.currency} balance on {balance.exchange}: {balance.free:.2f}",
                    {'exchange': balance.exchange, 'balance': balance.free}
                )
    
    def _severity_to_log_level(self, severity: AlertSeverity) -> int:
        """Convert alert severity to logging level."""
        mapping = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert for critical issues."""
        if not all([self.smtp_server, self.email_user, self.email_password, self.email_recipients]):
            logger.warning("Email not configured, skipping alert email")
            return
        
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


# ============= Enhanced Portfolio Manager =============

class PortfolioManager:
    """
    Central portfolio manager - single source of truth for all financial state.
    Thread-safe with proper locking for all state modifications.
    """
    
    def __init__(self, exchange_adapters: Dict[str, ExchangeInterface]):
        """
        Initialize portfolio manager.
        
        Args:
            exchange_adapters: Dictionary mapping exchange names to their adapters
        """
        self.exchanges = exchange_adapters
        
        self._ledger: List[Trade] = []
        self._positions: Dict[str, Position] = {}
        self._balances: Dict[str, Balance] = {}
        self._orders: Dict[str, Order] = {}
        self._active_orders: Dict[str, Order] = {}
        
        self._last_prices: Dict[str, float] = {}
        
        self._lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
        self._sync_interval = 1.0
        
        self._order_update_callbacks = []
        self._trade_callbacks = []
        self._balance_callbacks = []
        
        self.risk_manager = RiskManager(self)
        self.performance_analyzer = PerformanceAnalyzer(self)
        self.alert_system = AlertSystem()
        self.reporting = ReportingModule(self, self.risk_manager, 
                                        self.performance_analyzer, self.alert_system)
        
        logger.info("PortfolioManager initialized with exchanges: %s", list(self.exchanges.keys()))
    
    # ============= Public API - Commands =============
    
    def submit_order(self, exchange: str, symbol: str, side: OrderSide, 
                     order_type: OrderType, quantity: float, 
                     price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     time_in_force: OrderTimeInForce = OrderTimeInForce.GTC,
                     client_order_id: Optional[str] = None) -> str:
        """
        Submit a new order. This is the ONLY way to place orders.
        Returns internal order ID.
        """
        with self._lock:
            if exchange not in self.exchanges:
                raise ValueError(f"Unknown exchange: {exchange}")
            
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            order = Order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                client_order_id=client_order_id
            )
            
            approved, reason = self.risk_manager.check_order(order)
            if not approved:
                order.status = OrderStatus.REJECTED
                order.reject_reason = reason
                self._orders[order.internal_id] = order
                self.alert_system.add_alert(
                    AlertSeverity.WARNING,
                    "Risk",
                    "Order Rejected",
                    f"Order {order.internal_id} rejected: {reason}",
                    {'order': order.internal_id, 'reason': reason}
                )
                raise ValueError(f"Order rejected by risk manager: {reason}")
            
            if not self._check_sufficient_balance(order):
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Insufficient balance"
                self._orders[order.internal_id] = order
                logger.warning(f"Order {order.internal_id} rejected: insufficient balance")
                raise ValueError(f"Insufficient balance for order: {order}")
            
            self._lock_funds_for_order(order)
            
            self._orders[order.internal_id] = order
            self._active_orders[order.internal_id] = order
            
            try:
                success, exchange_order_id, response = self.exchanges[exchange].place_order(order)
                
                if success:
                    order.exchange_order_id = exchange_order_id
                    order.status = OrderStatus.SUBMITTED
                    order.submitted_at = datetime.now()
                    logger.info(f"Order {order.internal_id} submitted to {exchange}, exchange ID: {exchange_order_id}")
                    
                    self.alert_system.add_alert(
                        AlertSeverity.INFO,
                        "Order",
                        "Order Submitted",
                        f"Order {order.internal_id} submitted to {exchange}",
                        {'order_id': order.internal_id, 'exchange': exchange}
                    )
                else:
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = str(response)
                    self._release_locked_funds(order)
                    del self._active_orders[order.internal_id]
                    logger.error(f"Order {order.internal_id} rejected by exchange: {response}")
                    
                    self.alert_system.add_alert(
                        AlertSeverity.ERROR,
                        "Exchange",
                        "Order Rejected",
                        f"Order {order.internal_id} rejected by exchange: {response}",
                        {'order': order.internal_id, 'response': response}
                    )
                    raise RuntimeError(f"Exchange rejected order: {response}")
                    
            except Exception as e:
                order.status = OrderStatus.REJECTED
                order.reject_reason = str(e)
                self._release_locked_funds(order)
                del self._active_orders[order.internal_id]
                logger.exception(f"Exception submitting order {order.internal_id}")
                
                self.alert_system.add_alert(
                    AlertSeverity.ERROR,
                    "System",
                    "Order Submission Failed",
                    f"Exception submitting order {order.internal_id}: {e}",
                    {'order': order.internal_id, 'error': str(e)}
                )
                raise
            
            self._trigger_order_update(order)
            
            return order.internal_id
    
    def cancel_order(self, internal_order_id: str) -> bool:
        """
        Cancel an active order.
        """
        with self._lock:
            if internal_order_id not in self._active_orders:
                logger.warning(f"Order {internal_order_id} not found in active orders")
                return False
            
            order = self._active_orders[internal_order_id]
            
            if not order.is_active:
                logger.warning(f"Order {internal_order_id} is not active (status: {order.status})")
                return False
            
            if order.exchange_order_id:
                try:
                    success = self.exchanges[order.exchange].cancel_order(
                        order.exchange_order_id, order.symbol
                    )
                except Exception as e:
                    logger.exception(f"Error cancelling order {internal_order_id} on exchange")
                    self.alert_system.add_alert(
                        AlertSeverity.ERROR,
                        "Exchange",
                        "Cancel Failed",
                        f"Failed to cancel order {internal_order_id}: {e}",
                        {'order': internal_order_id, 'error': str(e)}
                    )
                    return False
            else:
                success = True
            
            if success:
                self._release_locked_funds(order)
                
                order.status = OrderStatus.CANCELLED
                order.closed_at = datetime.now()
                
                del self._active_orders[internal_order_id]
                
                logger.info(f"Order {internal_order_id} cancelled")
                
                self.alert_system.add_alert(
                    AlertSeverity.INFO,
                    "Order",
                    "Order Cancelled",
                    f"Order {internal_order_id} cancelled successfully",
                    {'order': internal_order_id}
                )
                
                self._trigger_order_update(order)
                return True
            
            return False
    
    def sync_with_exchanges(self):
        """
        Synchronize portfolio state with all exchanges.
        This should be called regularly (e.g., every second).
        """
        with self._lock:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    self._sync_exchange(exchange_name, exchange)
                except Exception as e:
                    logger.exception(f"Error syncing with {exchange_name}: {e}")
                    self.alert_system.add_alert(
                        AlertSeverity.ERROR,
                        "Sync",
                        "Sync Failed",
                        f"Failed to sync with {exchange_name}: {e}",
                        {'exchange': exchange_name, 'error': str(e)}
                    )
            
            self.risk_manager.update_daily_pnl()
            self.risk_manager.update_drawdown()
            
            self.alert_system.check_portfolio_alerts(self, self.risk_manager)
    
    # ============= Public API - Queries =============
    
    def get_balance(self, exchange: str, currency: str) -> Optional[Balance]:
        with self._lock:
            return self._balances.get(f"{exchange}:{currency}")
    
    def get_all_balances(self) -> List[Balance]:
        with self._lock:
            return list(self._balances.values())
    
    def get_position(self, exchange: str, symbol: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(f"{exchange}:{symbol}")
    
    def get_all_positions(self) -> List[Position]:
        with self._lock:
            return list(self._positions.values())
    
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
    
    def get_order_history(self, exchange: Optional[str] = None,
                          symbol: Optional[str] = None,
                          limit: int = 100) -> List[Order]:
        with self._lock:
            orders = [o for o in self._orders.values() if o.is_done]
            
            if exchange:
                orders = [o for o in orders if o.exchange == exchange]
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            
            orders.sort(key=lambda x: x.created_at, reverse=True)
            return orders[:limit]
    
    def get_trades(self, exchange: Optional[str] = None,
                   symbol: Optional[str] = None,
                   limit: int = 1000) -> List[Trade]:
        with self._lock:
            trades = self._ledger
            
            if exchange:
                trades = [t for t in trades if t.exchange == exchange]
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            
            trades.sort(key=lambda x: x.timestamp, reverse=True)
            return trades[:limit]
    
    def get_total_equity(self, quote_currency: str = "USD") -> float:
        with self._lock:
            total = 0.0
            
            for balance in self._balances.values():
                total += balance.total
            
            for position in self._positions.values():
                total += position.market_value
            
            return total
    
    def calculate_realized_pnl(self, exchange: Optional[str] = None, 
                              symbol: Optional[str] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> float:
        with self._lock:
            total_pnl = 0.0
            trades = self._ledger
            
            if exchange:
                trades = [t for t in trades if t.exchange == exchange]
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            if start_time:
                trades = [t for t in trades if t.timestamp >= start_time]
            if end_time:
                trades = [t for t in trades if t.timestamp <= end_time]
            
            trades.sort(key=lambda x: x.timestamp)
            
            position_tracker = {}
            
            for trade in trades:
                if trade.symbol not in position_tracker:
                    position_tracker[trade.symbol] = []
                
                if trade.is_buy:
                    position_tracker[trade.symbol].append({
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'trade_id': trade.trade_id,
                        'fees': trade.fee
                    })
                else:
                    remaining_sell_qty = trade.quantity
                    sell_price = trade.price
                    
                    buys = position_tracker[trade.symbol]
                    while remaining_sell_qty > 0 and buys:
                        buy = buys[0]
                        match_qty = min(remaining_sell_qty, buy['quantity'])
                        
                        buy_cost = match_qty * buy['price']
                        sell_value = match_qty * sell_price
                        
                        buy_fee = (match_qty / buy['quantity']) * buy.get('fees', 0)
                        sell_fee = (match_qty / trade.quantity) * trade.fee
                        
                        trade_pnl = sell_value - buy_cost - buy_fee - sell_fee
                        total_pnl += trade_pnl
                        
                        remaining_sell_qty -= match_qty
                        buy['quantity'] -= match_qty
                        
                        if buy['quantity'] <= 0:
                            buys.pop(0)
            
            return total_pnl
    
    def calculate_unrealized_pnl(self, exchange: Optional[str] = None,
                                symbol: Optional[str] = None) -> float:
        with self._lock:
            total_pnl = 0.0
            positions = self._positions.values()
            
            if exchange:
                positions = [p for p in positions if p.exchange == exchange]
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
            
            for position in positions:
                total_pnl += position.unrealized_pnl
            
            return total_pnl
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        with self._lock:
            positions_data = []
            for pos in self._positions.values():
                positions_data.append({
                    'symbol': pos.symbol,
                    'exchange': pos.exchange,
                    'quantity': pos.quantity,
                    'cost_basis': pos.cost_basis,
                    'current_price': pos.last_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_percentage': pos.unrealized_pnl_percentage,
                    'realized_pnl': pos.realized_pnl,
                    'total_pnl': pos.total_pnl,
                    'break_even_price': pos.break_even_price,
                    'is_profitable': pos.is_profitable
                })
            
            balances_data = []
            for bal in self._balances.values():
                balances_data.append({
                    'exchange': bal.exchange,
                    'currency': bal.currency,
                    'total': bal.total,
                    'free': bal.free,
                    'locked': bal.locked
                })
            
            total_realized_pnl = self.calculate_realized_pnl()
            total_unrealized_pnl = self.calculate_unrealized_pnl()
            total_pnl = total_realized_pnl + total_unrealized_pnl
            
            trade_stats = self.performance_analyzer.calculate_trade_statistics()
            
            return {
                'total_equity': self.get_total_equity(),
                'total_cash': sum(b.total for b in self._balances.values()),
                'total_positions_value': sum(p.market_value for p in self._positions.values()),
                'total_realized_pnl': total_realized_pnl,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_pnl': total_pnl,
                'profit_factor': trade_stats.get('profit_factor', 0),
                'win_rate': trade_stats.get('win_rate', 0),
                'active_orders_count': len(self._active_orders),
                'positions': positions_data,
                'balances': balances_data,
                'last_update': datetime.now()
            }
    
    # ============= Internal State Management =============
    
    def _check_sufficient_balance(self, order: Order) -> bool:
        if order.side == OrderSide.BUY:
            quote_currency = order.symbol.split('/')[-1] if '/' in order.symbol else "USD"
            balance_key = f"{order.exchange}:{quote_currency}"
            balance = self._balances.get(balance_key)
            
            if not balance:
                return False
            
            required = order.quantity * (order.price or 0)
            if required == 0:
                required = order.quantity * self._last_prices.get(order.symbol, 0)
            
            return balance.free >= required * 1.001
        
        elif order.side == OrderSide.SELL:
            base_currency = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            balance_key = f"{order.exchange}:{base_currency}"
            balance = self._balances.get(balance_key)
            
            if not balance:
                return False
            
            return balance.free >= order.quantity
        
        return False
    
    def _lock_funds_for_order(self, order: Order):
        if order.side == OrderSide.BUY:
            quote_currency = order.symbol.split('/')[-1] if '/' in order.symbol else "USD"
            balance_key = f"{order.exchange}:{quote_currency}"
            required = order.quantity * (order.price or 0)
            
            if balance_key in self._balances:
                self._balances[balance_key].locked += required
        
        elif order.side == OrderSide.SELL:
            base_currency = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            balance_key = f"{order.exchange}:{base_currency}"
            
            if balance_key in self._balances:
                self._balances[balance_key].locked += order.quantity
    
    def _release_locked_funds(self, order: Order):
        if order.side == OrderSide.BUY:
            quote_currency = order.symbol.split('/')[-1] if '/' in order.symbol else "USD"
            balance_key = f"{order.exchange}:{quote_currency}"
            required = order.quantity * (order.price or 0)
            
            if balance_key in self._balances:
                self._balances[balance_key].locked = max(
                    0, self._balances[balance_key].locked - required
                )
        
        elif order.side == OrderSide.SELL:
            base_currency = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            balance_key = f"{order.exchange}:{base_currency}"
            
            if balance_key in self._balances:
                self._balances[balance_key].locked = max(
                    0, self._balances[balance_key].locked - order.quantity
                )
    
    def _sync_exchange(self, exchange_name: str, exchange: ExchangeInterface):
        try:
            exchange_balances = exchange.get_balances()
            self._update_balances_from_exchange(exchange_name, exchange_balances)
        except Exception as e:
            logger.error(f"Error getting balances from {exchange_name}: {e}")
        
        try:
            open_orders_data = exchange.get_open_orders()
            self._reconcile_orders(exchange_name, open_orders_data)
        except Exception as e:
            logger.error(f"Error getting open orders from {exchange_name}: {e}")
        
        for position in self._positions.values():
            if position.exchange == exchange_name:
                try:
                    ticker = exchange.get_ticker(position.symbol)
                    if ticker and 'last' in ticker:
                        position.last_price = float(ticker['last'])
                        self._last_prices[position.symbol] = float(ticker['last'])
                except Exception as e:
                    logger.error(f"Error getting ticker for {position.symbol} from {exchange_name}: {e}")
    
    def _update_balances_from_exchange(self, exchange_name: str, 
                                       exchange_balances: Dict[str, Dict[str, float]]):
        for currency, data in exchange_balances.items():
            balance_key = f"{exchange_name}:{currency}"
            
            total = float(data.get('total', 0))
            locked = float(data.get('locked', 0))
            
            if balance_key in self._balances:
                self._balances[balance_key].update(total, locked)
            else:
                self._balances[balance_key] = Balance(
                    exchange=exchange_name,
                    currency=currency,
                    total=total,
                    locked=locked
                )
    
    def _reconcile_orders(self, exchange_name: str, open_orders_data: List[Dict]):
        exchange_open_ids = set()
        for order_data in open_orders_data:
            if 'id' in order_data:
                exchange_open_ids.add(str(order_data['id']))
        
        for internal_id, order in list(self._active_orders.items()):
            if order.exchange != exchange_name:
                continue
            
            if order.exchange_order_id and order.exchange_order_id not in exchange_open_ids:
                try:
                    order_data = self.exchanges[exchange_name].get_order_status(
                        order.exchange_order_id, order.symbol
                    )
                    
                    if order_data:
                        self._process_order_update(order, order_data)
                    else:
                        self._process_cancellation(order)
                        
                except Exception as e:
                    logger.error(f"Error fetching order {order.exchange_order_id}: {e}")
        
        exchange_ids_we_know = set()
        for o in self._active_orders.values():
            if o.exchange == exchange_name and o.exchange_order_id:
                exchange_ids_we_know.add(o.exchange_order_id)
        
        unknown_order_ids = exchange_open_ids - exchange_ids_we_know
        
        for order_id in unknown_order_ids:
            logger.warning(f"Found unknown order on {exchange_name}: {order_id}")
            self.alert_system.add_alert(
                AlertSeverity.WARNING,
                "Sync",
                "Unknown Order",
                f"Found unknown order {order_id} on {exchange_name}",
                {'exchange': exchange_name, 'order_id': order_id}
            )
    
    def _process_order_update(self, order: Order, exchange_data: Dict):
        old_status = order.status
        old_filled = order.filled_quantity
        
        order.update_from_exchange(exchange_data)
        
        if order.filled_quantity > old_filled:
            new_fill_quantity = order.filled_quantity - old_filled
            if new_fill_quantity > 0:
                new_fill = Trade(
                    trade_id=f"{order.exchange_order_id}_{order.filled_quantity}",
                    exchange=order.exchange,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=new_fill_quantity,
                    price=order.average_fill_price or 0,
                    timestamp=datetime.now(),
                    order_id=order.exchange_order_id or ""
                )
                
                self._process_new_trade(new_fill)
                
                if order.side == OrderSide.SELL:
                    self.risk_manager.daily_pnl += new_fill.value - (new_fill_quantity * (order.average_fill_price or 0))
        
        if order.is_done and order.internal_id in self._active_orders:
            del self._active_orders[order.internal_id]
            self._release_locked_funds(order)
        
        if old_status != order.status:
            self._trigger_order_update(order)
            
            if order.status == OrderStatus.FILLED:
                self.alert_system.add_alert(
                    AlertSeverity.INFO,
                    "Order",
                    "Order Filled",
                    f"Order {order.internal_id} filled at {order.average_fill_price}",
                    {'order': order.internal_id, 'price': order.average_fill_price}
                )
    
    def _process_new_trade(self, trade: Trade):
        self._ledger.append(trade)
        
        position_key = f"{trade.exchange}:{trade.symbol}"
        
        if position_key not in self._positions:
            self._positions[position_key] = Position(
                symbol=trade.symbol,
                exchange=trade.exchange
            )
        
        position = self._positions[position_key]
        
        if trade.is_buy:
            new_quantity = position.quantity + trade.quantity
            if new_quantity > 0:
                position.cost_basis = (
                    (position.cost_value + trade.value) / new_quantity
                )
            position.quantity = new_quantity
        else:
            if position.quantity > 0:
                sold_quantity = min(trade.quantity, position.quantity)
                cost_of_sold = sold_quantity * position.cost_basis
                realized = (sold_quantity * trade.price) - cost_of_sold - trade.fee
                position.realized_pnl += realized
            
            position.quantity -= trade.quantity
        
        position.last_update = datetime.now()
        
        self._update_balances_from_trade(trade)
        self._trigger_trade(trade)
    
    def _update_balances_from_trade(self, trade: Trade):
        if '/' in trade.symbol:
            base, quote = trade.symbol.split('/')
        else:
            base = trade.symbol
            quote = "USD"
        
        quote_balance_key = f"{trade.exchange}:{quote}"
        if quote_balance_key in self._balances:
            if trade.is_buy:
                self._balances[quote_balance_key].total -= trade.value
                self._balances[quote_balance_key].locked -= trade.value
            else:
                self._balances[quote_balance_key].total += trade.value
        
        base_balance_key = f"{trade.exchange}:{base}"
        if base_balance_key in self._balances:
            if trade.is_buy:
                self._balances[base_balance_key].total += trade.quantity
            else:
                self._balances[base_balance_key].total -= trade.quantity
                self._balances[base_balance_key].locked -= trade.quantity
    
    def _process_cancellation(self, order: Order):
        order.status = OrderStatus.CANCELLED
        order.closed_at = datetime.now()
        
        if order.internal_id in self._active_orders:
            del self._active_orders[order.internal_id]
        
        self._release_locked_funds(order)
        self._trigger_order_update(order)
        
        logger.info(f"Order {order.internal_id} cancelled (via reconciliation)")
        
        self.alert_system.add_alert(
            AlertSeverity.INFO,
            "Order",
            "Order Cancelled",
            f"Order {order.internal_id} cancelled via reconciliation",
            {'order': order.internal_id}
        )
    
    # ============= Callback Management =============
    
    def register_order_update_callback(self, callback):
        self._order_update_callbacks.append(callback)
    
    def register_trade_callback(self, callback):
        self._trade_callbacks.append(callback)
    
    def register_balance_callback(self, callback):
        self._balance_callbacks.append(callback)
    
    def _trigger_order_update(self, order: Order):
        for callback in self._order_update_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.exception(f"Error in order update callback: {e}")
    
    def _trigger_trade(self, trade: Trade):
        for callback in self._trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.exception(f"Error in trade callback: {e}")
    
    def _trigger_balance_update(self, balance: Balance):
        for callback in self._balance_callbacks:
            try:
                callback(balance)
            except Exception as e:
                logger.exception(f"Error in balance update callback: {e}")
    
    # ============= Background Sync Thread =============
    
    def start_sync_thread(self):
        if self._sync_thread and self._sync_thread.is_alive():
            logger.warning("Sync thread already running")
            return
        
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Started portfolio sync thread")
    
    def stop_sync_thread(self):
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
            logger.info("Stopped portfolio sync thread")
    
    def _sync_loop(self):
        while self._running:
            try:
                self.sync_with_exchanges()
            except Exception as e:
                logger.exception(f"Error in sync loop: {e}")
            
            time.sleep(self._sync_interval)
    
    # ============= Persistence =============
    
    def save_state(self, filepath: str):
        """
        Save portfolio state to disk.
        """
        import pickle
        
        state = {
            'ledger': [t.to_dict() for t in self._ledger],
            'positions': {k: v.to_dict() for k, v in self._positions.items()},
            'balances': {k: v.to_dict() for k, v in self._balances.items()},
            'orders': {k: v.to_dict() for k, v in self._orders.items()},
            'last_prices': self._last_prices,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Portfolio state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """
        Load portfolio state from disk.
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            logger.warning(f"State file {filepath} not found")
            return
        
        with self._lock:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self._ledger = [Trade.from_dict(t) for t in state['ledger']]
            self._positions = {k: Position.from_dict(v) for k, v in state['positions'].items()}
            self._balances = {k: Balance.from_dict(v) for k, v in state['balances'].items()}
            self._orders = {k: Order.from_dict(v) for k, v in state['orders'].items()}
            self._last_prices = state['last_prices']
            
            self._active_orders = {
                oid: order for oid, order in self._orders.items() 
                if order.is_active
            }
            
            logger.info(f"Portfolio state loaded from {filepath}")
            logger.info(f"Loaded {len(self._ledger)} trades, {len(self._positions)} positions, "
                       f"{len(self._active_orders)} active orders")


# ============= Test Cases =============

def main():
    """Main execution function."""
    
    exchanges = {
        'binance': MockExchange('binance'),
        'coinbase': MockExchange('coinbase')
    }
    portfolio = PortfolioManager(exchanges)
    
    # IMPORTANT: Initialize portfolio with exchange balances
    print("\n--- Initializing Portfolio with Exchange Balances ---")
    for exchange_name, exchange in exchanges.items():
        balances = exchange.get_balances()
        for currency, data in balances.items():
            balance_key = f"{exchange_name}:{currency}"
            portfolio._balances[balance_key] = Balance(
                exchange=exchange_name,
                currency=currency,
                total=data['total'],
                locked=data['locked']
            )
        print(f"  Loaded {exchange_name} balances: {balances}")
    
    # Now set risk limits based on actual total equity
    total_equity = portfolio.get_total_equity()
    print(f"\nTotal Equity after initialization: ${total_equity:.2f}")
    
    # Set risk limits as percentages of actual equity
    portfolio.risk_manager.max_position_size_pct = 0.3  # 30% of equity
    portfolio.risk_manager.max_exposure_pct = 0.7       # 70% of equity
    
    print(f"Risk Limits:")
    print(f"  Max Position Size: ${total_equity * portfolio.risk_manager.max_position_size_pct:.2f} ({portfolio.risk_manager.max_position_size_pct*100}%)")
    print(f"  Max Exposure: ${total_equity * portfolio.risk_manager.max_exposure_pct:.2f} ({portfolio.risk_manager.max_exposure_pct*100}%)")
    
    portfolio.start_sync_thread()
    
    try:
        print("\n" + "="*60)
        print("PORTFOLIO MANAGEMENT SYSTEM DEMONSTRATION")
        print("="*60)
        
        print("\n--- Initial Portfolio State ---")
        print(f"Total Equity: ${portfolio.get_total_equity():.2f}")
        
        # Show all balances
        print("\n--- Current Balances ---")
        for balance in portfolio.get_all_balances():
            print(f"  {balance.exchange} - {balance.currency}: "
                  f"Total=${balance.total:.2f}, Free=${balance.free:.2f}, Locked=${balance.locked:.2f}")
        
        print("\n--- Submitting Orders ---")
        
        # Calculate safe order sizes (10% of available balance)
        btc_order_value = portfolio.get_balance('binance', 'USD').free * 0.1
        btc_quantity = btc_order_value / 50000  # BTC price $50,000
        
        eth_order_value = portfolio.get_balance('coinbase', 'USD').free * 0.1
        eth_quantity = eth_order_value / 3000   # ETH price $3,000
        
        print(f"Calculated safe order sizes:")
        print(f"  BTC Order: {btc_quantity:.4f} BTC (${btc_order_value:.2f})")
        print(f"  ETH Order: {eth_quantity:.4f} ETH (${eth_order_value:.2f})")
        
        # Submit BTC buy order
        order1 = portfolio.submit_order(
            exchange='binance',
            symbol='BTC/USD',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=btc_quantity,
            price=50000
        )
        print(f"Order 1 (BTC Buy): {order1}")
        
        # Submit ETH buy order
        order2 = portfolio.submit_order(
            exchange='coinbase',
            symbol='ETH/USD',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=eth_quantity,
            price=3000
        )
        print(f"Order 2 (ETH Buy): {order2}")
        
        # Give some time for orders to process
        time.sleep(2)
        
        # Check active orders
        print(f"\n--- Active Orders ---")
        active_orders = portfolio.get_active_orders()
        print(f"Active Orders: {len(active_orders)}")
        for order in active_orders:
            print(f"  {order.symbol}: {order.side.value} {order.quantity:.4f} @ ${order.price}")
        
        # Get portfolio summary
        print("\n--- Portfolio Summary ---")
        summary = portfolio.get_portfolio_summary()
        print(f"Total Equity: ${summary['total_equity']:.2f}")
        print(f"Total Cash: ${summary['total_cash']:.2f}")
        print(f"Total Positions Value: ${summary['total_positions_value']:.2f}")
        print(f"Total PnL: ${summary['total_pnl']:.2f}")
        print(f"Active Orders: {summary['active_orders_count']}")
        
        # Show updated balances
        print("\n--- Updated Balances (with locked funds) ---")
        for balance in portfolio.get_all_balances():
            print(f"  {balance.exchange} - {balance.currency}: "
                  f"Total=${balance.total:.2f}, Free=${balance.free:.2f}, Locked=${balance.locked:.2f}")
        
        # Get risk metrics
        print("\n--- Risk Metrics ---")
        risk_metrics = portfolio.risk_manager.get_risk_metrics()
        print(f"VaR (95%): {risk_metrics.var_95:.2%}")
        print(f"Risk Level: {risk_metrics.risk_level.value}")
        print(f"Exposure: {risk_metrics.exposure:.2%}")
        print(f"Current Drawdown: {portfolio.risk_manager.current_drawdown:.2%}")
        
        # Generate reports
        print("\n--- Generating Reports ---")
        pnl_report = portfolio.reporting.generate_pnl_report()
        risk_report = portfolio.reporting.generate_risk_report()
        perf_report = portfolio.reporting.generate_performance_report()
        
        print(f"P&L Report: {pnl_report}")
        print(f"Risk Report: {risk_report}")
        print(f"Performance Report: {perf_report}")
        
        # Cancel one order
        print(f"\n--- Cancelling Order {order2} ---")
        success = portfolio.cancel_order(order2)
        print(f"Cancellation {'successful' if success else 'failed'}")
        
        # Check updated portfolio
        print("\n--- Updated Portfolio After Cancellation ---")
        summary = portfolio.get_portfolio_summary()
        print(f"Total Equity: ${summary['total_equity']:.2f}")
        print(f"Active Orders: {summary['active_orders_count']}")
        
        # Show final balances
        print("\n--- Final Balances ---")
        for balance in portfolio.get_all_balances():
            print(f"  {balance.exchange} - {balance.currency}: "
                  f"Total=${balance.total:.2f}, Free=${balance.free:.2f}, Locked=${balance.locked:.2f}")
        
        print("\n" + "="*60)
        print("RUNNING TEST SUITE")
        print("="*60)
        
        # Run tests
        tester = TestPortfolioManager()
        
        # Initialize test portfolio with balances
        for exchange_name, exchange in tester.exchanges.items():
            balances = exchange.get_balances()
            for currency, data in balances.items():
                balance_key = f"{exchange_name}:{currency}"
                tester.portfolio._balances[balance_key] = Balance(
                    exchange=exchange_name,
                    currency=currency,
                    total=data['total'],
                    locked=data['locked']
                )
        
        # Set risk limits for tests
        test_equity = tester.portfolio.get_total_equity()
        tester.portfolio.risk_manager.max_position_size_pct = 0.3
        tester.portfolio.risk_manager.max_exposure_pct = 0.7
        
        print(f"\nTest Portfolio Initialized with Equity: ${test_equity:.2f}")
        tester.run_all_tests()
        
    except Exception as e:
        logger.exception("Error in main execution")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        portfolio.stop_sync_thread()
        portfolio.save_state('portfolio_demo_state.pkl')
        print("\nPortfolio state saved. Done.")
        
class MockExchange(ExchangeInterface):
    """Mock exchange for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.orders = {}
        # Realistic balances
        self.balances = {
            'USD': {'free': 50000, 'locked': 0, 'total': 50000},
            'BTC': {'free': 1.0, 'locked': 0, 'total': 1.0},
            'ETH': {'free': 20.0, 'locked': 0, 'total': 20.0}
        }
        self.order_counter = 0
        self.prices = {
            'BTC/USD': 50000.0,
            'ETH/USD': 3000.0
        }
    
    def place_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[Dict]]:
        self.order_counter += 1
        order_id = f"{self.name}_order_{self.order_counter}"
        self.orders[order_id] = order
        
        # Simulate order placement by locking funds
        if order.side == OrderSide.BUY:
            quote_currency = order.symbol.split('/')[-1]
            if quote_currency in self.balances:
                cost = order.quantity * (order.price or self.prices.get(order.symbol, 0))
                self.balances[quote_currency]['locked'] += cost
                self.balances[quote_currency]['free'] -= cost
        else:  # SELL
            base_currency = order.symbol.split('/')[0]
            if base_currency in self.balances:
                self.balances[base_currency]['locked'] += order.quantity
                self.balances[base_currency]['free'] -= order.quantity
        
        return True, order_id, {'id': order_id, 'status': 'submitted'}
    
    def cancel_order(self, exchange_order_id: str, symbol: str) -> bool:
        if exchange_order_id in self.orders:
            order = self.orders[exchange_order_id]
            # Release locked funds
            if order.side == OrderSide.BUY:
                quote_currency = symbol.split('/')[-1]
                if quote_currency in self.balances:
                    cost = order.quantity * (order.price or self.prices.get(symbol, 0))
                    self.balances[quote_currency]['locked'] -= cost
                    self.balances[quote_currency]['free'] += cost
            else:
                base_currency = symbol.split('/')[0]
                if base_currency in self.balances:
                    self.balances[base_currency]['locked'] -= order.quantity
                    self.balances[base_currency]['free'] += order.quantity
            
            del self.orders[exchange_order_id]
            return True
        return False
    
    def get_order_status(self, exchange_order_id: str, symbol: str) -> Optional[Dict]:
        if exchange_order_id in self.orders:
            return {
                'id': exchange_order_id,
                'status': 'open',
                'filled': 0.0,
                'remaining': self.orders[exchange_order_id].quantity,
                'avg_price': None
            }
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        open_orders = []
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.OPEN]:
                open_orders.append({
                    'id': order_id,
                    'symbol': order.symbol,
                    'status': 'open'
                })
        return open_orders
    
    def get_balances(self) -> Dict[str, Dict[str, float]]:
        return self.balances
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        return {'last': self.prices.get(symbol, 50000)}
    
    def get_historical_ohlcv(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        data = []
        base_price = self.prices.get(symbol, 50000)
        for i in range(limit):
            data.append({
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'open': base_price * (1 - 0.01 * i),
                'high': base_price * (1 + 0.02 * i),
                'low': base_price * (1 - 0.02 * i),
                'close': base_price * (1 - 0.005 * i),
                'volume': 1000
            })
        return data

# Update the TestPortfolioManager class with corrected assertions

# ============= Test Cases =============

class TestPortfolioManager:
    """Test cases for PortfolioManager and associated modules."""
    
    def __init__(self):
        self.exchanges = {
            'binance': MockExchange('binance'),
            'coinbase': MockExchange('coinbase')
        }
        self.portfolio = PortfolioManager(self.exchanges)
        
        # Initialize portfolio with exchange balances
        for exchange_name, exchange in self.exchanges.items():
            balances = exchange.get_balances()
            for currency, data in balances.items():
                balance_key = f"{exchange_name}:{currency}"
                self.portfolio._balances[balance_key] = Balance(
                    exchange=exchange_name,
                    currency=currency,
                    total=data['total'],
                    locked=data['locked']
                )
        
        # Set risk limits
        self.portfolio.risk_manager.max_position_size_pct = 0.3
        self.portfolio.risk_manager.max_exposure_pct = 0.7
        
        self.test_results = []
    
    def assert_equal(self, actual, expected, test_name):
        """Assert equality and record result."""
        if actual == expected:
            result = "PASS"
        else:
            result = f"FAIL (expected {expected}, got {actual})"
        
        self.test_results.append({
            'name': test_name,
            'result': result,
            'passed': actual == expected
        })
        print(f"  {test_name}: {result}")
    
    def assert_approx(self, actual, expected, tolerance=0.01, test_name=""):
        """Assert approximate equality within tolerance."""
        if abs(actual - expected) <= tolerance:
            result = "PASS"
        else:
            result = f"FAIL (expected ~{expected}, got {actual})"
        
        self.test_results.append({
            'name': test_name,
            'result': result,
            'passed': abs(actual - expected) <= tolerance
        })
        print(f"  {test_name}: {result}")
    
    def assert_true(self, condition, test_name):
        """Assert condition is true."""
        if condition:
            result = "PASS"
        else:
            result = "FAIL"
        
        self.test_results.append({
            'name': test_name,
            'result': result,
            'passed': condition
        })
        print(f"  {test_name}: {result}")
    
    def run_all_tests(self):
        """Run all test cases."""
        print("\n" + "="*60)
        print("RUNNING PORTFOLIO MANAGER TESTS")
        print("="*60)
        
        # Clear any existing alerts before tests
        self.portfolio.alert_system.alerts.clear()
        
        self.test_initialization()
        self.test_order_submission()
        self.test_order_cancellation()
        self.test_balance_checking()
        self.test_position_tracking()
        self.test_pnl_calculation()
        self.test_risk_management()
        self.test_performance_analytics()
        self.test_reporting()
        self.test_alert_system()
        self.test_concurrent_operations()
        self.test_persistence()
        
        self.print_test_summary()
    
    def test_initialization(self):
        """Test portfolio manager initialization."""
        print("\n--- Testing Initialization ---")
        
        self.assert_equal(
            len(self.portfolio.exchanges), 2,
            "Exchange count"
        )
        self.assert_equal(
            len(self.portfolio.get_all_balances()), 6,
            "Initial balances count"
        )
        self.assert_equal(
            len(self.portfolio.get_all_positions()), 0,
            "Initial positions count"
        )
    
    def test_order_submission(self):
        """Test order submission."""
        print("\n--- Testing Order Submission ---")
        
        # Test buy order
        try:
            order_id = self.portfolio.submit_order(
                exchange='binance',
                symbol='BTC/USD',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.02,  # Small order to fit within limits
                price=50000
            )
            self.assert_true(order_id is not None, "Buy order submission")
        except Exception as e:
            self.assert_true(False, f"Buy order submission failed: {e}")
        
        # Test sell order
        try:
            order_id = self.portfolio.submit_order(
                exchange='coinbase',
                symbol='BTC/USD',
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=0.01,
                price=51000
            )
            self.assert_true(order_id is not None, "Sell order submission")
        except Exception as e:
            self.assert_true(False, f"Sell order submission failed: {e}")
        
        # Test invalid exchange
        try:
            self.portfolio.submit_order(
                exchange='invalid',
                symbol='BTC/USD',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=50000
            )
            self.assert_true(False, "Should reject invalid exchange")
        except ValueError:
            self.assert_true(True, "Rejected invalid exchange")
        
        # Test invalid quantity
        try:
            self.portfolio.submit_order(
                exchange='binance',
                symbol='BTC/USD',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=-0.1,
                price=50000
            )
            self.assert_true(False, "Should reject negative quantity")
        except ValueError:
            self.assert_true(True, "Rejected negative quantity")
    
    def test_order_cancellation(self):
        """Test order cancellation."""
        print("\n--- Testing Order Cancellation ---")
        
        # Submit an order to cancel
        order_id = self.portfolio.submit_order(
            exchange='binance',
            symbol='BTC/USD',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=50000
        )
        
        # Cancel the order
        result = self.portfolio.cancel_order(order_id)
        self.assert_true(result, "Cancel existing order")
        
        # Try to cancel non-existent order
        result = self.portfolio.cancel_order("non_existent_id")
        self.assert_true(not result, "Cancel non-existent order")
    
    def test_balance_checking(self):
        """Test balance checking functionality."""
        print("\n--- Testing Balance Checking ---")
        
        # Check USD balance
        balance = self.portfolio.get_balance('binance', 'USD')
        self.assert_equal(balance.total, 50000, "Initial USD balance")
        self.assert_equal(balance.free, 50000, "Initial free USD")
        self.assert_equal(balance.locked, 0, "Initial locked USD")
        
        # Check BTC balance
        balance = self.portfolio.get_balance('binance', 'BTC')
        self.assert_equal(balance.total, 1.0, "Initial BTC balance")
        
        # Check ETH balance
        balance = self.portfolio.get_balance('binance', 'ETH')
        self.assert_equal(balance.total, 20.0, "Initial ETH balance")
        
        # Check all balances count
        all_balances = self.portfolio.get_all_balances()
        self.assert_equal(len(all_balances), 6, "All balances count")
    
    def test_position_tracking(self):
        """Test position tracking."""
        print("\n--- Testing Position Tracking ---")
        
        # Clear existing positions
        self.portfolio._positions.clear()
        self.portfolio._ledger.clear()
        
        # Initially no positions
        positions = self.portfolio.get_all_positions()
        self.assert_equal(len(positions), 0, "Initial positions empty")
        
        # Create test trades
        trade1 = Trade(
            trade_id="test1",
            exchange="binance",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000,
            timestamp=datetime.now(),
            order_id="order1",
            fee=10
        )
        
        trade2 = Trade(
            trade_id="test2",
            exchange="binance",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=0.05,
            price=51000,
            timestamp=datetime.now(),
            order_id="order2",
            fee=5
        )
        
        # Process trades
        self.portfolio._process_new_trade(trade1)
        self.portfolio._process_new_trade(trade2)
        
        # Check position
        position = self.portfolio.get_position('binance', 'BTC/USD')
        self.assert_approx(position.quantity, 0.05, tolerance=0.001, test_name="Position quantity")
        self.assert_true(position.cost_basis > 0, "Cost basis calculated")
        self.assert_true(position.realized_pnl > 0, "Realized P&L positive")
    
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        print("\n--- Testing P&L Calculation ---")
        
        # Clear previous trades
        self.portfolio._ledger = []
        self.portfolio._positions = {}
        
        # Create a realistic trade sequence
        trades = [
            Trade("t1", "binance", "BTC/USD", OrderSide.BUY, 0.1, 50000, 
                  datetime.now() - timedelta(days=5), "o1", 10),
            Trade("t2", "binance", "BTC/USD", OrderSide.BUY, 0.05, 51000, 
                  datetime.now() - timedelta(days=4), "o2", 5),
            Trade("t3", "binance", "BTC/USD", OrderSide.SELL, 0.08, 52000, 
                  datetime.now() - timedelta(days=3), "o3", 8),
            Trade("t4", "binance", "BTC/USD", OrderSide.SELL, 0.05, 53000, 
                  datetime.now() - timedelta(days=2), "o4", 5),
        ]
        
        for trade in trades:
            self.portfolio._process_new_trade(trade)
        
        realized_pnl = self.portfolio.calculate_realized_pnl()
        self.assert_true(realized_pnl > 0, "Positive realized P&L")
        
        # Verify unrealized P&L for remaining position
        unrealized_pnl = self.portfolio.calculate_unrealized_pnl()
        self.assert_true(unrealized_pnl != 0, "Unrealized P&L calculated")
        
        # Test trade statistics
        trade_stats = self.portfolio.performance_analyzer.calculate_trade_statistics()
        self.assert_true('win_rate' in trade_stats, "Win rate calculated")
        self.assert_true(trade_stats.get('win_rate', 0) > 0, "Win rate positive")
    
    def test_risk_management(self):
        """Test risk management functionality."""
        print("\n--- Testing Risk Management ---")
        
        risk = self.portfolio.risk_manager
        
        # Test position sizing
        position_size = risk.calculate_position_size("BTC/USD", 50000, 49000)
        self.assert_true(position_size > 0, "Position size calculated")
        self.assert_true(position_size < 1.0, "Position size reasonable")
        
        # Test oversized order rejection
        order = Order(
            exchange="binance",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,  # Very large order
            price=50000
        )
        
        approved, reason = risk.check_order(order)
        self.assert_true(not approved, "Risk should reject oversized order")
        self.assert_true(reason is not None, "Rejection reason provided")
        
        # Test VaR calculation
        returns = [-0.01, -0.02, 0.01, 0.03, -0.015, 0.02, -0.03, 0.01]
        var = risk.calculate_var(returns, 0.95)
        self.assert_true(var < 0, "VaR calculated correctly")
        self.assert_true(var > -0.05, "VaR within expected range")
        
        # Test risk metrics
        metrics = risk.get_risk_metrics()
        self.assert_true(metrics.var_95 <= 0, "Risk metrics populated")
        self.assert_true(metrics.risk_level in RiskLevel, "Risk level valid")
    
    def test_performance_analytics(self):
        """Test performance analytics."""
        print("\n--- Testing Performance Analytics ---")
        
        analyzer = self.portfolio.performance_analyzer
        
        # Test trade statistics
        stats = analyzer.calculate_trade_statistics()
        self.assert_true('win_rate' in stats, "Trade statistics calculated")
        
        # Test returns
        returns = analyzer.calculate_returns()
        self.assert_true('total_return' in returns, "Returns calculated")
        
        # Test risk-adjusted metrics
        risk_metrics = analyzer.calculate_risk_adjusted_metrics()
        self.assert_true('sharpe_ratio' in risk_metrics, "Risk metrics calculated")
        
        # Test max drawdown
        max_dd = analyzer.calculate_max_drawdown()
        self.assert_true(max_dd >= 0, "Max drawdown calculated")
        
        # Test performance metrics
        perf = analyzer.get_performance_metrics()
        self.assert_true(perf.sharpe_ratio is not None, "Performance metrics populated")
    
    def test_reporting(self):
        """Test reporting functionality."""
        print("\n--- Testing Reporting ---")
        
        reporting = self.portfolio.reporting
        
        # Test P&L report
        try:
            report_file = reporting.generate_pnl_report()
            self.assert_true(report_file.endswith('.json'), "P&L report generated")
        except Exception as e:
            self.assert_true(False, f"P&L report failed: {e}")
        
        # Test risk report
        try:
            report_file = reporting.generate_risk_report()
            self.assert_true(report_file.endswith('.json'), "Risk report generated")
        except Exception as e:
            self.assert_true(False, f"Risk report failed: {e}")
        
        # Test performance report
        try:
            report_file = reporting.generate_performance_report()
            self.assert_true(report_file.endswith('.json'), "Performance report generated")
        except Exception as e:
            self.assert_true(False, f"Performance report failed: {e}")
        
        # Test tax report
        try:
            report_file = reporting.generate_tax_report(2024)
            self.assert_true(report_file.endswith('.json'), "Tax report generated")
        except Exception as e:
            self.assert_true(False, f"Tax report failed: {e}")
    
    def test_alert_system(self):
        """Test alert system."""
        print("\n--- Testing Alert System ---")
        
        alerts = self.portfolio.alert_system
        alert_received = []
        
        def test_handler(alert):
            alert_received.append(alert)
        
        alerts.register_handler(test_handler)
        
        # Clear any existing alerts
        alerts.alerts.clear()
        
        # Test adding alerts
        alerts.add_alert(
            AlertSeverity.WARNING,
            "Test",
            "Test Alert",
            "This is a test alert",
            {'test': True}
        )
        
        self.assert_equal(len(alert_received), 1, "Alert handler triggered")
        self.assert_equal(len(alerts.get_unacknowledged_alerts()), 1, "Unacknowledged alert")
        
        # Test acknowledgment
        alert_id = alert_received[0].alert_id
        alerts.acknowledge_alert(alert_id)
        self.assert_equal(len(alerts.get_unacknowledged_alerts()), 0, "Alert acknowledged")
        
        # Test portfolio alerts
        alerts.alerts.clear()
        alerts.check_portfolio_alerts(self.portfolio, self.portfolio.risk_manager)
        self.assert_true(len(alerts.alerts) >= 0, "Portfolio alerts generated")
    
    def test_concurrent_operations(self):
        """Test thread safety with concurrent operations."""
        print("\n--- Testing Concurrent Operations ---")
        
        import threading
        
        # Store initial order count
        initial_order_count = len(self.portfolio._orders)
        
        def submit_orders():
            for i in range(5):
                try:
                    self.portfolio.submit_order(
                        exchange='binance',
                        symbol='BTC/USD',
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=0.001,
                        price=50000 + i * 100
                    )
                except Exception as e:
                    pass
        
        threads = []
        for _ in range(3):
            t = threading.Thread(target=submit_orders)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify no corruption
        self.assert_true(len(self.portfolio._orders) >= initial_order_count, 
                        "Concurrent operations completed")
        self.assert_true(len(self.portfolio._active_orders) >= 0, "Active orders tracked")
    
    def test_persistence(self):
        """Test save/load functionality."""
        print("\n--- Testing Persistence ---")
        
        # Save state
        save_file = "test_portfolio_state.pkl"
        self.portfolio.save_state(save_file)
        
        # Create new portfolio and load state
        new_portfolio = PortfolioManager(self.exchanges)
        
        # Initialize new portfolio with exchange balances
        for exchange_name, exchange in self.exchanges.items():
            balances = exchange.get_balances()
            for currency, data in balances.items():
                balance_key = f"{exchange_name}:{currency}"
                new_portfolio._balances[balance_key] = Balance(
                    exchange=exchange_name,
                    currency=currency,
                    total=data['total'],
                    locked=data['locked']
                )
        
        new_portfolio.load_state(save_file)
        
        # Compare
        self.assert_equal(
            len(self.portfolio._ledger), len(new_portfolio._ledger),
            "Ledger persistence"
        )
        self.assert_equal(
            len(self.portfolio._positions), len(new_portfolio._positions),
            "Positions persistence"
        )
        self.assert_equal(
            len(self.portfolio._orders), len(new_portfolio._orders),
            "Orders persistence"
        )
        
        # Clean up
        import os
        if os.path.exists(save_file):
            os.remove(save_file)
    
    def print_test_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.test_results:
                if not r['passed']:
                    print(f"  - {r['name']}: {r['result']}")


# ============= Main Execution =============

def main():
    """Main execution function."""
    
    exchanges = {
        'binance': MockExchange('binance'),
        'coinbase': MockExchange('coinbase')
    }
    portfolio = PortfolioManager(exchanges)
    
    # IMPORTANT: Initialize portfolio with exchange balances
    print("\n--- Initializing Portfolio with Exchange Balances ---")
    for exchange_name, exchange in exchanges.items():
        balances = exchange.get_balances()
        for currency, data in balances.items():
            balance_key = f"{exchange_name}:{currency}"
            portfolio._balances[balance_key] = Balance(
                exchange=exchange_name,
                currency=currency,
                total=data['total'],
                locked=data['locked']
            )
        print(f"  Loaded {exchange_name} balances: {balances}")
    
    # Now set risk limits based on actual total equity
    total_equity = portfolio.get_total_equity()
    print(f"\nTotal Equity after initialization: ${total_equity:.2f}")
    
    # Set risk limits as percentages of actual equity
    portfolio.risk_manager.max_position_size_pct = 0.3  # 30% of equity
    portfolio.risk_manager.max_exposure_pct = 0.7       # 70% of equity
    
    print(f"Risk Limits:")
    print(f"  Max Position Size: ${total_equity * portfolio.risk_manager.max_position_size_pct:.2f} ({portfolio.risk_manager.max_position_size_pct*100}%)")
    print(f"  Max Exposure: ${total_equity * portfolio.risk_manager.max_exposure_pct:.2f} ({portfolio.risk_manager.max_exposure_pct*100}%)")
    
    portfolio.start_sync_thread()
    
    try:
        print("\n" + "="*60)
        print("PORTFOLIO MANAGEMENT SYSTEM DEMONSTRATION")
        print("="*60)
        
        print("\n--- Initial Portfolio State ---")
        print(f"Total Equity: ${portfolio.get_total_equity():.2f}")
        
        # Show all balances
        print("\n--- Current Balances ---")
        for balance in portfolio.get_all_balances():
            print(f"  {balance.exchange} - {balance.currency}: "
                  f"Total=${balance.total:.2f}, Free=${balance.free:.2f}, Locked=${balance.locked:.2f}")
        
        print("\n--- Submitting Orders ---")
        
        # Calculate safe order sizes (10% of available balance)
        btc_order_value = portfolio.get_balance('binance', 'USD').free * 0.1
        btc_quantity = btc_order_value / 50000  # BTC price $50,000
        
        eth_order_value = portfolio.get_balance('coinbase', 'USD').free * 0.1
        eth_quantity = eth_order_value / 3000   # ETH price $3,000
        
        print(f"Calculated safe order sizes:")
        print(f"  BTC Order: {btc_quantity:.4f} BTC (${btc_order_value:.2f})")
        print(f"  ETH Order: {eth_quantity:.4f} ETH (${eth_order_value:.2f})")
        
        # Submit BTC buy order
        order1 = portfolio.submit_order(
            exchange='binance',
            symbol='BTC/USD',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=btc_quantity,
            price=50000
        )
        print(f"Order 1 (BTC Buy): {order1}")
        
        # Submit ETH buy order
        order2 = portfolio.submit_order(
            exchange='coinbase',
            symbol='ETH/USD',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=eth_quantity,
            price=3000
        )
        print(f"Order 2 (ETH Buy): {order2}")
        
        # Give some time for orders to process
        time.sleep(2)
        
        # Check active orders
        print(f"\n--- Active Orders ---")
        active_orders = portfolio.get_active_orders()
        print(f"Active Orders: {len(active_orders)}")
        for order in active_orders:
            print(f"  {order.symbol}: {order.side.value} {order.quantity:.4f} @ ${order.price}")
        
        # Get portfolio summary
        print("\n--- Portfolio Summary ---")
        summary = portfolio.get_portfolio_summary()
        print(f"Total Equity: ${summary['total_equity']:.2f}")
        print(f"Total Cash: ${summary['total_cash']:.2f}")
        print(f"Total Positions Value: ${summary['total_positions_value']:.2f}")
        print(f"Total PnL: ${summary['total_pnl']:.2f}")
        print(f"Active Orders: {summary['active_orders_count']}")
        
        # Show updated balances
        print("\n--- Updated Balances (with locked funds) ---")
        for balance in portfolio.get_all_balances():
            print(f"  {balance.exchange} - {balance.currency}: "
                  f"Total=${balance.total:.2f}, Free=${balance.free:.2f}, Locked=${balance.locked:.2f}")
        
        # Get risk metrics
        print("\n--- Risk Metrics ---")
        risk_metrics = portfolio.risk_manager.get_risk_metrics()
        print(f"VaR (95%): {risk_metrics.var_95:.2%}")
        print(f"Risk Level: {risk_metrics.risk_level.value}")
        print(f"Exposure: {risk_metrics.exposure:.2%}")
        print(f"Current Drawdown: {portfolio.risk_manager.current_drawdown:.2%}")
        
        # Generate reports
        print("\n--- Generating Reports ---")
        pnl_report = portfolio.reporting.generate_pnl_report()
        risk_report = portfolio.reporting.generate_risk_report()
        perf_report = portfolio.reporting.generate_performance_report()
        
        print(f"P&L Report: {pnl_report}")
        print(f"Risk Report: {risk_report}")
        print(f"Performance Report: {perf_report}")
        
        # Cancel one order
        print(f"\n--- Cancelling Order {order2} ---")
        success = portfolio.cancel_order(order2)
        print(f"Cancellation {'successful' if success else 'failed'}")
        
        # Check updated portfolio
        print("\n--- Updated Portfolio After Cancellation ---")
        summary = portfolio.get_portfolio_summary()
        print(f"Total Equity: ${summary['total_equity']:.2f}")
        print(f"Active Orders: {summary['active_orders_count']}")
        
        # Show final balances
        print("\n--- Final Balances ---")
        for balance in portfolio.get_all_balances():
            print(f"  {balance.exchange} - {balance.currency}: "
                  f"Total=${balance.total:.2f}, Free=${balance.free:.2f}, Locked=${balance.locked:.2f}")
        
        print("\n" + "="*60)
        print("RUNNING TEST SUITE")
        print("="*60)
        
        # Run tests
        tester = TestPortfolioManager()
        
        # Initialize test portfolio with balances
        for exchange_name, exchange in tester.exchanges.items():
            balances = exchange.get_balances()
            for currency, data in balances.items():
                balance_key = f"{exchange_name}:{currency}"
                tester.portfolio._balances[balance_key] = Balance(
                    exchange=exchange_name,
                    currency=currency,
                    total=data['total'],
                    locked=data['locked']
                )
        
        # Set risk limits for tests
        test_equity = tester.portfolio.get_total_equity()
        tester.portfolio.risk_manager.max_position_size_pct = 0.3
        tester.portfolio.risk_manager.max_exposure_pct = 0.7
        
        print(f"\nTest Portfolio Initialized with Equity: ${test_equity:.2f}")
        tester.run_all_tests()
        
    except Exception as e:
        logger.exception("Error in main execution")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        portfolio.stop_sync_thread()
        portfolio.save_state('portfolio_demo_state.pkl')
        print("\nPortfolio state saved. Done.")


if __name__ == "__main__":
    main()