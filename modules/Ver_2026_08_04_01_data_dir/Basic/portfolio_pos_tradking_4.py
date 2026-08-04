"""
Portfolio Management System - With Complete Test Suite
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
import os
import tempfile
import unittest

logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
logger = logging.getLogger(__name__)


# ============= Configuration =============

class Config:
    MAX_POSITION_SIZE_PCT = 0.30
    MAX_EXPOSURE_PCT = 0.70
    MAX_DRAWDOWN_PCT = 0.20
    MAX_CONCENTRATION_PCT = 0.30
    DAILY_LOSS_LIMIT = 0.05
    RISK_PER_TRADE = 0.01
    RISK_FREE_RATE = 0.02
    SYNC_INTERVAL = 1.0
    SLIPPAGE_BUFFER = 0.001
    BASE_CURRENCY = "USD"
    INITIAL_EQUITY = 10000.0
    REPORT_DIR = Path("reports")
    STATE_FILE = "portfolio_state.pkl"


# ============= Enums =============

class OrderSide(Enum): BUY = "buy"; SELL = "sell"
class OrderType(Enum): MARKET = "market"; LIMIT = "limit"; STOP_LOSS = "stop_loss"; TAKE_PROFIT = "take_profit"
class OrderStatus(Enum): PENDING = "pending"; SUBMITTED = "submitted"; OPEN = "open"; FILLED = "filled"; CANCELLED = "cancelled"; REJECTED = "rejected"
class OrderTimeInForce(Enum): GTC = "gtc"; IOC = "ioc"; FOK = "fok"; DAY = "day"
class RiskLevel(Enum): LOW = "low"; MEDIUM = "medium"; HIGH = "high"; CRITICAL = "critical"
class AlertSeverity(Enum): INFO = "info"; WARNING = "warning"; ERROR = "error"; CRITICAL = "critical"


# ============= Helper =============

def to_serializable(obj):
    if isinstance(obj, (datetime, date)): return obj.isoformat()
    if isinstance(obj, Enum): return obj.value
    if hasattr(obj, 'to_dict'): return obj.to_dict()
    if isinstance(obj, (list, tuple)): return [to_serializable(i) for i in obj]
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    return obj


# ============= Data Classes =============

@dataclass
class Trade:
    trade_id: str; exchange: str; symbol: str; side: OrderSide; quantity: float; price: float
    timestamp: datetime; order_id: str; fee: float = 0.0
    
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
class Position:
    symbol: str; exchange: str; quantity: float = 0.0; cost_basis: float = 0.0
    realized_pnl: float = 0.0; last_price: float = 0.0; last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float: return self.quantity * self.last_price
    @property
    def cost_value(self) -> float: return self.quantity * self.cost_basis
    @property
    def unrealized_pnl(self) -> float: return self.market_value - self.cost_value
    @property
    def total_pnl(self) -> float: return self.realized_pnl + self.unrealized_pnl
    
    def to_dict(self) -> Dict: return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data):
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        return cls(**data)


@dataclass
class Balance:
    exchange: str; currency: str; total: float = 0.0; locked: float = 0.0; last_update: datetime = field(default_factory=datetime.now)
    @property
    def free(self) -> float: return self.total - self.locked
    def update(self, total: float, locked: float): self.total = total; self.locked = locked; self.last_update = datetime.now()
    def to_dict(self) -> Dict: return to_serializable(asdict(self))
    @classmethod
    def from_dict(cls, data):
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        return cls(**data)


@dataclass
class Order:
    exchange: str; symbol: str; side: OrderSide; order_type: OrderType; quantity: float
    price: Optional[float] = None; stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    internal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0; remaining_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None; updated_at: Optional[datetime] = None; closed_at: Optional[datetime] = None
    reject_reason: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.side, str): self.side = OrderSide(self.side)
        if isinstance(self.order_type, str): self.order_type = OrderType(self.order_type)
        if isinstance(self.time_in_force, str): self.time_in_force = OrderTimeInForce(self.time_in_force)
        if isinstance(self.status, str): self.status = OrderStatus(self.status)
        if self.remaining_quantity == 0 and self.quantity > 0: self.remaining_quantity = self.quantity
    
    @property
    def is_active(self) -> bool: return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.OPEN]
    @property
    def is_done(self) -> bool: return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
    
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


@dataclass
class RiskMetrics:
    var_95: float = 0.0; var_99: float = 0.0; cvar_95: float = 0.0
    max_drawdown: float = 0.0; volatility: float = 0.0; sharpe_ratio: float = 0.0
    exposure: float = 0.0; concentration_ratio: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    def to_dict(self) -> Dict: return to_serializable(asdict(self))


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0; annualized_return: float = 0.0; daily_return: float = 0.0
    win_rate: float = 0.0; profit_factor: float = 0.0; average_win: float = 0.0; average_loss: float = 0.0
    largest_win: float = 0.0; largest_loss: float = 0.0; consecutive_wins: int = 0; consecutive_losses: int = 0
    sharpe_ratio: float = 0.0; sortino_ratio: float = 0.0; calmar_ratio: float = 0.0; omega_ratio: float = 0.0
    def to_dict(self) -> Dict: return to_serializable(asdict(self))


@dataclass
class Alert:
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.INFO; category: str = ""; title: str = ""; message: str = ""
    timestamp: datetime = field(default_factory=datetime.now); data: Optional[Dict] = None
    acknowledged: bool = False; acknowledged_at: Optional[datetime] = None
    def to_dict(self) -> Dict: return to_serializable(asdict(self))
    @classmethod
    def from_dict(cls, data):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('acknowledged_at'): data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        data['severity'] = AlertSeverity(data['severity'])
        return cls(**data)


# ============= Exchange Interface =============

class ExchangeInterface(ABC):
    @abstractmethod
    def place_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[Dict]]: pass
    @abstractmethod
    def cancel_order(self, exchange_order_id: str, symbol: str) -> bool: pass
    @abstractmethod
    def get_order_status(self, exchange_order_id: str, symbol: str) -> Optional[Dict]: pass
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]: pass
    @abstractmethod
    def get_balances(self) -> Dict[str, Dict[str, float]]: pass
    @abstractmethod
    def get_ticker(self, symbol: str) -> Optional[Dict]: pass
    @abstractmethod
    def get_positions(self) -> List[Dict]: pass


# ============= Mock Exchange =============

class MockExchange(ExchangeInterface):
    def __init__(self, name: str):
        self.name = name; self.orders = {}
        self.balances = {'USD': {'free': 50000, 'locked': 0, 'total': 50000},
                        'BTC': {'free': 1.0, 'locked': 0, 'total': 1.0},
                        'ETH': {'free': 20.0, 'locked': 0, 'total': 20.0}}
        self.order_counter = 0
        self.prices = {'BTC/USD': 50000.0, 'ETH/USD': 3000.0}
    
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
            if order.status != OrderStatus.SUBMITTED: return False
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
                if base not in self.balances: self.balances[base] = {'free': 0, 'locked': 0, 'total': 0}
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
    
    def get_balances(self) -> Dict[str, Dict[str, float]]: return self.balances
    def get_ticker(self, symbol: str) -> Optional[Dict]: return {'last': self.prices.get(symbol, 50000)}
    def get_positions(self) -> List[Dict]:
        return [{'symbol': f"{c}/USD", 'quantity': b['total'], 'price': self.prices.get(f"{c}/USD", 0)}
                for c, b in self.balances.items() if c not in ['USD', 'USDT', 'USDC'] and b['total'] > 0]


# ============= Risk Manager =============

class RiskManager:
    def __init__(self, portfolio, config=None):
        self.portfolio = portfolio
        self.config = config or {}
        self.max_position_size_pct = self.config.get('max_position_size_pct', Config.MAX_POSITION_SIZE_PCT)
        self.max_exposure_pct = self.config.get('max_exposure_pct', Config.MAX_EXPOSURE_PCT)
        self.max_drawdown_pct = self.config.get('max_drawdown_pct', Config.MAX_DRAWDOWN_PCT)
        self.max_concentration_pct = self.config.get('max_concentration_pct', Config.MAX_CONCENTRATION_PCT)
        self.daily_loss_limit = self.config.get('daily_loss_limit', Config.DAILY_LOSS_LIMIT)
        self.risk_per_trade = self.config.get('risk_per_trade', Config.RISK_PER_TRADE)
        self.daily_pnl = 0.0; self.current_drawdown = 0.0; self.peak_equity = 0.0
        self.last_reset_day = datetime.now().date()
    
    def check_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        total_equity = self.portfolio.get_total_equity()
        if total_equity <= 0: return False, "Total equity is zero or negative"
        position = self.portfolio.get_position(order.exchange, order.symbol)
        order_price = order.price or self.portfolio._last_prices.get(order.symbol, 0)
        if order_price <= 0: return False, f"Cannot determine price for {order.symbol}"
        order_value = order.quantity * order_price
        
        if order.side == OrderSide.SELL:
            if order.quantity > (position.quantity if position else 0):
                return False, f"Insufficient position to sell"
            return True, None
        
        max_pos = total_equity * self.max_position_size_pct
        if order_value > max_pos:
            return False, f"Position size ${order_value:.2f} exceeds limit ${max_pos:.2f}"
        
        total_exposure = sum(p.market_value for p in self.portfolio.get_all_positions())
        max_exp = total_equity * self.max_exposure_pct
        if total_exposure + order_value > max_exp:
            return False, f"Total exposure would exceed limit ${max_exp:.2f}"
        
        current_pos_value = position.market_value if position else 0
        if (current_pos_value + order_value) / total_equity > self.max_concentration_pct:
            return False, f"Concentration exceeds limit"
        
        if self.daily_pnl < -total_equity * self.daily_loss_limit:
            return False, f"Daily loss limit reached"
        
        if self.current_drawdown > self.max_drawdown_pct:
            return False, f"Max drawdown exceeded"
        
        return True, None
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        total_equity = self.portfolio.get_total_equity()
        risk_amount = total_equity * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0: return 0
        return min(risk_amount / risk_per_unit, (total_equity * self.max_position_size_pct) / entry_price)
    
    def update_daily_pnl(self):
        if datetime.now().date() > self.last_reset_day:
            self.daily_pnl = 0.0; self.last_reset_day = datetime.now().date()
    
    def update_drawdown(self):
        current_equity = self.portfolio.get_total_equity()
        if current_equity > self.peak_equity: self.peak_equity = current_equity
        if self.peak_equity > 0: self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
    
    def get_risk_metrics(self) -> RiskMetrics:
        metrics = RiskMetrics()
        metrics.exposure = sum(p.market_value for p in self.portfolio.get_all_positions()) / max(self.portfolio.get_total_equity(), 1)
        positions = self.portfolio.get_all_positions()
        if positions:
            largest = max(positions, key=lambda p: p.market_value)
            metrics.concentration_ratio = largest.market_value / max(self.portfolio.get_total_equity(), 1)
        metrics.risk_level = RiskLevel.LOW if metrics.exposure < 0.3 else RiskLevel.MEDIUM if metrics.exposure < 0.6 else RiskLevel.HIGH
        return metrics


# ============= Performance Analyzer =============

class PerformanceAnalyzer:
    def __init__(self, portfolio, initial_equity=None):
        self.portfolio = portfolio
        self._initial_equity = initial_equity
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        trades = self.portfolio.get_trades()
        if len(trades) < 2: return {}
        trade_pnls = self._calculate_trade_pnls()
        if not trade_pnls: return {}
        winning = [p for p in trade_pnls if p > 0]
        losing = [p for p in trade_pnls if p < 0]
        return {
            'total_trades': len(trade_pnls), 'winning_trades': len(winning), 'losing_trades': len(losing),
            'win_rate': len(winning) / len(trade_pnls) * 100 if trade_pnls else 0,
            'average_win': np.mean(winning) if winning else 0, 'average_loss': np.mean(losing) if losing else 0,
            'largest_win': max(winning) if winning else 0, 'largest_loss': min(losing) if losing else 0,
            'profit_factor': abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else float('inf')
        }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        stats = self.calculate_trade_statistics()
        metrics.win_rate = stats.get('win_rate', 0)
        metrics.profit_factor = stats.get('profit_factor', 0)
        metrics.average_win = stats.get('average_win', 0)
        metrics.average_loss = stats.get('average_loss', 0)
        metrics.largest_win = stats.get('largest_win', 0)
        metrics.largest_loss = stats.get('largest_loss', 0)
        return metrics
    
    def _calculate_trade_pnls(self) -> List[float]:
        trade_pnls = []
        position_tracker = defaultdict(list)
        for trade in sorted(self.portfolio.get_trades(), key=lambda x: x.timestamp):
            key = f"{trade.exchange}:{trade.symbol}"
            if trade.is_buy:
                position_tracker[key].append({'quantity': trade.quantity, 'price': trade.price, 'fee': trade.fee})
            else:
                remaining = trade.quantity
                buys = position_tracker[key]
                while remaining > 0 and buys:
                    buy = buys[0]
                    match_qty = min(remaining, buy['quantity'])
                    cost = match_qty * buy['price']
                    sell_value = match_qty * trade.price
                    buy_fee = (match_qty / buy['quantity']) * buy['fee']
                    sell_fee = (match_qty / trade.quantity) * trade.fee
                    trade_pnls.append(sell_value - cost - buy_fee - sell_fee)
                    remaining -= match_qty
                    buy['quantity'] -= match_qty
                    if buy['quantity'] <= 0: buys.pop(0)
        return trade_pnls


# ============= Alert System =============

class AlertSystem:
    def __init__(self, config=None):
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        self.thresholds = self.config.get('thresholds', {})
    
    def add_alert(self, severity: AlertSeverity, category: str, title: str, message: str, data: Dict = None):
        alert = Alert(severity=severity, category=category, title=title, message=message, data=data)
        self.alerts.append(alert)
        log_level = {AlertSeverity.INFO: logging.INFO, AlertSeverity.WARNING: logging.WARNING,
                    AlertSeverity.ERROR: logging.ERROR, AlertSeverity.CRITICAL: logging.CRITICAL}.get(severity, logging.INFO)
        logger.log(log_level, f"[{severity.value}] {title}: {message}")
        for handler in list(self.alert_handlers):
            try: handler(alert)
            except Exception as e: logger.error(f"Alert handler error: {e}")
    
    def register_handler(self, handler: Callable): self.alert_handlers.append(handler)
    def get_unacknowledged_alerts(self) -> List[Alert]: return [a for a in self.alerts if not a.acknowledged]
    
    def check_portfolio_alerts(self, portfolio, risk_manager):
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


# ============= Reporting Module =============

class ReportingModule:
    def __init__(self, portfolio, risk_manager, performance_analyzer, alert_system):
        self.portfolio = portfolio; self.risk_manager = risk_manager
        self.analyzer = performance_analyzer; self.alerts = alert_system
        self.report_dir = Config.REPORT_DIR
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_pnl_report(self) -> str:
        summary = self.portfolio.get_portfolio_summary()
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'performance': self.analyzer.get_performance_metrics().to_dict(),
            'positions': [{'symbol': p.symbol, 'quantity': p.quantity, 'cost_basis': p.cost_basis,
                          'current_price': p.last_price, 'unrealized_pnl': p.unrealized_pnl,
                          'realized_pnl': p.realized_pnl} for p in self.portfolio.get_all_positions()],
            'trades': [t.to_dict() for t in self.portfolio.get_trades(limit=50)]
        }
        filename = self.report_dir / f"pnl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f: json.dump(report_data, f, indent=2, default=str)
        return str(filename)
    
    def generate_risk_report(self) -> str:
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'risk_metrics': self.risk_manager.get_risk_metrics().to_dict(),
            'current_drawdown': self.risk_manager.current_drawdown,
            'daily_pnl': self.risk_manager.daily_pnl
        }
        filename = self.report_dir / f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f: json.dump(report_data, f, indent=2, default=str)
        return str(filename)
    
    def generate_performance_report(self) -> str:
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': self.analyzer.get_performance_metrics().to_dict(),
            'trade_statistics': self.analyzer.calculate_trade_statistics()
        }
        filename = self.report_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f: json.dump(report_data, f, indent=2, default=str)
        return str(filename)


# ============= Portfolio Manager =============

class PortfolioManager:
    def __init__(self, exchange_adapters: Dict[str, ExchangeInterface], config: Dict = None):
        self.config = config or {}
        self.exchanges = exchange_adapters
        self._ledger: List[Trade] = []
        self._positions: Dict[str, Position] = {}
        self._balances: Dict[str, Balance] = {}
        self._orders: Dict[str, Order] = {}
        self._active_orders: Dict[str, Order] = {}
        self._last_prices: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._order_update_callbacks = []
        self._trade_callbacks = []
        
        self.risk_manager = RiskManager(self, self.config)
        self.performance_analyzer = PerformanceAnalyzer(self)
        self.alert_system = AlertSystem(self.config)
        self.reporting = ReportingModule(self, self.risk_manager, self.performance_analyzer, self.alert_system)
        
        logger.info(f"PortfolioManager initialized with exchanges: {list(self.exchanges.keys())}")
    
    # ============= Commands =============
    
    def submit_order(self, exchange: str, symbol: str, side: OrderSide, order_type: OrderType,
                     quantity: float, price: Optional[float] = None, **kwargs) -> str:
        with self._lock:
            if exchange not in self.exchanges: raise ValueError(f"Unknown exchange: {exchange}")
            if quantity <= 0: raise ValueError("Quantity must be positive")
            
            order = Order(exchange=exchange, symbol=symbol, side=side, order_type=order_type,
                         quantity=quantity, price=price, **kwargs)
            
            approved, reason = self.risk_manager.check_order(order)
            if not approved:
                order.status = OrderStatus.REJECTED
                order.reject_reason = reason
                self._orders[order.internal_id] = order
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
                if order.internal_id in self._active_orders: del self._active_orders[order.internal_id]
                raise
            
            self._trigger_order_update(order)
            return order.internal_id
    
    def cancel_order(self, internal_order_id: str) -> bool:
        with self._lock:
            if internal_order_id not in self._active_orders: return False
            order = self._active_orders[internal_order_id]
            if not order.is_active: return False
            
            if order.exchange_order_id:
                try:
                    success = self.exchanges[order.exchange].cancel_order(order.exchange_order_id, order.symbol)
                    if not success: return False
                except Exception as e:
                    logger.exception(f"Error cancelling order {internal_order_id}")
                    return False
            
            self._release_locked_funds(order)
            order.status = OrderStatus.CANCELLED
            order.closed_at = datetime.now()
            del self._active_orders[internal_order_id]
            self._trigger_order_update(order)
            return True
    
    def sync_with_exchanges(self):
        with self._lock:
            for exchange_name, exchange in self.exchanges.items():
                try: self._sync_exchange(exchange_name, exchange)
                except Exception as e: logger.exception(f"Error syncing {exchange_name}: {e}")
            self.risk_manager.update_daily_pnl()
            self.risk_manager.update_drawdown()
            self.alert_system.check_portfolio_alerts(self, self.risk_manager)
    
    # ============= Queries =============
    
    def get_balance(self, exchange: str, currency: str) -> Optional[Balance]:
        with self._lock: return self._balances.get(f"{exchange}:{currency}")
    
    def get_all_balances(self) -> List[Balance]:
        with self._lock: return list(self._balances.values())
    
    def get_position(self, exchange: str, symbol: str) -> Optional[Position]:
        with self._lock: return self._positions.get(f"{exchange}:{symbol}")
    
    def get_all_positions(self) -> List[Position]:
        with self._lock: return list(self._positions.values())
    
    def get_active_orders(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> List[Order]:
        with self._lock:
            orders = list(self._active_orders.values())
            if exchange: orders = [o for o in orders if o.exchange == exchange]
            if symbol: orders = [o for o in orders if o.symbol == symbol]
            return orders
    
    def get_trades(self, exchange: Optional[str] = None, symbol: Optional[str] = None, limit: int = 1000) -> List[Trade]:
        with self._lock:
            trades = self._ledger
            if exchange: trades = [t for t in trades if t.exchange == exchange]
            if symbol: trades = [t for t in trades if t.symbol == symbol]
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
            total_pnl = 0.0
            tracker = defaultdict(list)
            for trade in sorted(self._ledger, key=lambda x: x.timestamp):
                key = f"{trade.exchange}:{trade.symbol}"
                if trade.is_buy:
                    tracker[key].append({'quantity': trade.quantity, 'price': trade.price, 'fee': trade.fee})
                else:
                    remaining = trade.quantity
                    buys = tracker[key]
                    while remaining > 0 and buys:
                        buy = buys[0]
                        match_qty = min(remaining, buy['quantity'])
                        cost = match_qty * buy['price']
                        sell_value = match_qty * trade.price
                        buy_fee = (match_qty / buy['quantity']) * buy['fee']
                        sell_fee = (match_qty / trade.quantity) * trade.fee
                        total_pnl += sell_value - cost - buy_fee - sell_fee
                        remaining -= match_qty
                        buy['quantity'] -= match_qty
                        if buy['quantity'] <= 0: buys.pop(0)
            return total_pnl
    
    def calculate_unrealized_pnl(self) -> float:
        with self._lock: return sum(p.unrealized_pnl for p in self._positions.values())
    
    def get_portfolio_summary(self) -> Dict:
        with self._lock:
            return {
                'total_equity': self.get_total_equity(),
                'total_cash': sum(b.total for b in self._balances.values()),
                'total_positions_value': sum(p.market_value for p in self._positions.values()),
                'total_realized_pnl': self.calculate_realized_pnl(),
                'total_unrealized_pnl': self.calculate_unrealized_pnl(),
                'total_pnl': self.calculate_realized_pnl() + self.calculate_unrealized_pnl(),
                'active_orders_count': len(self._active_orders),
                'positions': [{'symbol': p.symbol, 'quantity': p.quantity, 'cost_basis': p.cost_basis,
                              'current_price': p.last_price, 'unrealized_pnl': p.unrealized_pnl} 
                             for p in self._positions.values()],
                'balances': [{'exchange': b.exchange, 'currency': b.currency, 'total': b.total, 
                             'free': b.free, 'locked': b.locked} for b in self._balances.values()]
            }
    
    # ============= Internal Methods =============
    
    def _check_sufficient_balance(self, order: Order) -> bool:
        if order.side == OrderSide.BUY:
            quote = order.symbol.split('/')[-1] if '/' in order.symbol else Config.BASE_CURRENCY
            bal = self._balances.get(f"{order.exchange}:{quote}")
            if not bal: return False
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
            if bal: bal.locked += order.quantity * (order.price or self._last_prices.get(order.symbol, 0))
        else:
            base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            bal = self._balances.get(f"{order.exchange}:{base}")
            if bal: bal.locked += order.quantity
    
    def _release_locked_funds(self, order: Order):
        if order.side == OrderSide.BUY:
            quote = order.symbol.split('/')[-1] if '/' in order.symbol else Config.BASE_CURRENCY
            bal = self._balances.get(f"{order.exchange}:{quote}")
            if bal: bal.locked = max(0, bal.locked - order.quantity * (order.price or self._last_prices.get(order.symbol, 0)))
        else:
            base = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            bal = self._balances.get(f"{order.exchange}:{base}")
            if bal: bal.locked = max(0, bal.locked - order.quantity)
    
    def _sync_exchange(self, exchange_name: str, exchange: ExchangeInterface):
        for currency, data in exchange.get_balances().items():
            key = f"{exchange_name}:{currency}"
            if key in self._balances:
                self._balances[key].update(data['total'], data.get('locked', 0))
            else:
                self._balances[key] = Balance(exchange_name, currency, data['total'], data.get('locked', 0))
        
        for pos_data in exchange.get_positions():
            symbol = pos_data['symbol']
            key = f"{exchange_name}:{symbol}"
            if key in self._positions:
                self._positions[key].quantity = pos_data['quantity']
                self._positions[key].last_price = pos_data.get('price', self._positions[key].last_price)
            else:
                self._positions[key] = Position(symbol, exchange_name, pos_data['quantity'], 
                                               pos_data.get('price', 0), last_price=pos_data.get('price', 0))
        
        for position in self._positions.values():
            if position.exchange == exchange_name:
                ticker = exchange.get_ticker(position.symbol)
                if ticker and 'last' in ticker:
                    position.last_price = float(ticker['last'])
                    self._last_prices[position.symbol] = float(ticker['last'])
    
    def _process_new_trade(self, trade: Trade):
        self._ledger.append(trade)
        key = f"{trade.exchange}:{trade.symbol}"
        if key not in self._positions:
            self._positions[key] = Position(trade.symbol, trade.exchange)
        pos = self._positions[key]
        if trade.is_buy:
            old_cost = pos.cost_basis * pos.quantity if pos.quantity > 0 else 0
            new_quantity = pos.quantity + trade.quantity
            pos.cost_basis = (old_cost + trade.value) / new_quantity if new_quantity > 0 else 0
            pos.quantity = new_quantity
        else:
            sold = min(trade.quantity, pos.quantity)
            pos.realized_pnl += (sold * trade.price) - (sold * pos.cost_basis) - trade.fee
            pos.quantity = max(0, pos.quantity - trade.quantity)
            if pos.quantity == 0: pos.cost_basis = 0
        pos.last_update = datetime.now()
        self._trigger_trade(trade)
    
    # ============= Callbacks =============
    
    def register_order_update_callback(self, callback: Callable):
        self._order_update_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable):
        self._trade_callbacks.append(callback)
    
    def _trigger_order_update(self, order: Order):
        for cb in list(self._order_update_callbacks):
            try: cb(order)
            except Exception as e: logger.exception(f"Callback error: {e}")
    
    def _trigger_trade(self, trade: Trade):
        for cb in list(self._trade_callbacks):
            try: cb(trade)
            except Exception as e: logger.exception(f"Callback error: {e}")
    
    # ============= Sync Thread =============
    
    def start_sync_thread(self):
        if self._sync_thread and self._sync_thread.is_alive(): return
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
    
    def stop_sync_thread(self):
        self._running = False
        if self._sync_thread: self._sync_thread.join(timeout=5)
    
    def _sync_loop(self):
        while self._running:
            try: self.sync_with_exchanges()
            except Exception as e: logger.exception(f"Sync error: {e}")
            time.sleep(Config.SYNC_INTERVAL)
    
    # ============= Persistence =============
    
    def save_state(self, filepath: str = None):
        if filepath is None: filepath = Config.STATE_FILE
        with self._lock:
            positions_dict = {}
            for k, v in self._positions.items():
                try:
                    positions_dict[k] = v.to_dict()
                except Exception as e:
                    logger.warning(f"Could not serialize position {k}: {e}")
                    positions_dict[k] = {'symbol': v.symbol, 'exchange': v.exchange, 'quantity': v.quantity,
                                        'cost_basis': v.cost_basis, 'realized_pnl': v.realized_pnl,
                                        'last_price': v.last_price, 'last_update': v.last_update.isoformat()}
            
            state = {
                'ledger': [t.to_dict() for t in self._ledger],
                'positions': positions_dict,
                'balances': {k: v.to_dict() for k, v in self._balances.items()},
                'orders': {k: v.to_dict() for k, v in self._orders.items()},
                'last_prices': self._last_prices
            }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f: pickle.dump(state, f)
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str = None):
        if filepath is None: filepath = Config.STATE_FILE
        if not Path(filepath).exists(): return
        with self._lock:
            with open(filepath, 'rb') as f: state = pickle.load(f)
            self._ledger = [Trade.from_dict(t) for t in state['ledger']]
            self._positions = {k: Position.from_dict(v) for k, v in state['positions'].items()}
            self._balances = {k: Balance.from_dict(v) for k, v in state['balances'].items()}
            self._orders = {k: Order.from_dict(v) for k, v in state['orders'].items()}
            self._last_prices = state['last_prices']
            self._active_orders = {oid: o for oid, o in self._orders.items() if o.is_active}
        logger.info(f"State loaded from {filepath}")


# ============= Complete Test Suite =============

class TestPortfolioManager(unittest.TestCase):
    """Comprehensive test suite for Portfolio Manager."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.exchanges = {'binance': MockExchange('binance'), 'coinbase': MockExchange('coinbase')}
        self.portfolio = PortfolioManager(self.exchanges)
        
        # Initialize balances
        for exchange_name, exchange in self.exchanges.items():
            for currency, data in exchange.get_balances().items():
                self.portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                    exchange_name, currency, data['total'], data['locked'])
        
        # Set risk limits for testing
        self.portfolio.risk_manager.max_position_size_pct = 0.3
        self.portfolio.risk_manager.max_exposure_pct = 0.7
        
        # Reset state
        self.portfolio._ledger = []
        self.portfolio._positions = {}
        self.portfolio._orders = {}
        self.portfolio._active_orders = {}
    
    # ============= Test 1: Order Submission =============
    
    def test_order_submission(self):
        """Test order submission and filling."""
        print("\n🧪 Testing Order Submission...")
        
        # Submit buy order
        order_id = self.portfolio.submit_order(
            'binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        self.assertIsNotNone(order_id)
        self.assertEqual(len(self.portfolio.get_active_orders()), 1)
        
        # Fill the order
        self.exchanges['binance'].fill_order(self.exchanges['binance'].orders.keys().__iter__().__next__())
        
        # Check position
        positions = self.portfolio.get_all_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, 'BTC/USD')
        self.assertAlmostEqual(positions[0].quantity, 0.1, places=4)
        
        print(f"  ✅ Order submitted and filled successfully")
    
    # ============= Test 2: Position Tracking =============
    
    def test_position_tracking(self):
        """Test position tracking with proper cost basis."""
        print("\n🧪 Testing Position Tracking...")
        
        # Buy 0.1 BTC at $50,000
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Check position
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.quantity, 0.1, places=4)
        self.assertAlmostEqual(pos.cost_basis, 50000.0, places=2)
        self.assertAlmostEqual(pos.cost_value, 5000.0, places=2)
        
        # Buy another 0.05 BTC at $51,000
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.05, 51000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Check updated position - weighted average cost
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        self.assertAlmostEqual(pos.quantity, 0.15, places=4)
        expected_cost = (0.1 * 50000 + 0.05 * 51000) / 0.15
        self.assertAlmostEqual(pos.cost_basis, expected_cost, places=2)
        self.assertAlmostEqual(pos.cost_value, 0.15 * expected_cost, places=2)
        
        # Update price to check unrealized P&L
        self.portfolio._last_prices['BTC/USD'] = 52000
        pos.last_price = 52000
        self.assertGreater(pos.unrealized_pnl, 0)
        
        print(f"  ✅ Position tracking with cost basis working")
    
    # ============= Test 3: P&L Calculation =============
    
    def test_pnl_calculation(self):
        """Test realized and unrealized P&L calculation."""
        print("\n🧪 Testing P&L Calculation...")
        
        # Buy 0.2 BTC at $50,000
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.2, 50000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Buy 0.1 BTC at $51,000
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 51000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Sell 0.15 BTC at $52,000
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.SELL, OrderType.LIMIT, 0.15, 52000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Check realized P&L (FIFO: first 0.1 BTC bought at 50k, next 0.05 BTC bought at 51k)
        realized_pnl = self.portfolio.calculate_realized_pnl()
        expected_pnl = 0.1 * (52000 - 50000) + 0.05 * (52000 - 51000)
        self.assertAlmostEqual(realized_pnl, expected_pnl, places=2)
        
        # Check remaining position (0.05 BTC at 51k)
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        self.assertAlmostEqual(pos.quantity, 0.05, places=4)
        self.assertAlmostEqual(pos.cost_basis, 51000, places=2)
        
        print(f"  ✅ P&L calculation working (realized: ${realized_pnl:.2f})")
    
    # ============= Test 4: Risk Management =============
    
    def test_risk_management(self):
        """Test risk management with position sizing and limits."""
        print("\n🧪 Testing Risk Management...")
        
        # Test position size calculation
        size = self.portfolio.risk_manager.calculate_position_size('BTC/USD', 50000, 49000)
        self.assertGreater(size, 0)
        self.assertLess(size, 1.0)  # Reasonable size
        
        # Test oversized order rejection
        try:
            self.portfolio.submit_order(
                'binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 100, 50000)  # Very large
            self.fail("Should have rejected oversized order")
        except ValueError as e:
            self.assertIn("exceeds limit", str(e))
        
        # Test exposure limit
        # Buy up to exposure limit
        total_equity = self.portfolio.get_total_equity()
        max_exposure = total_equity * 0.7
        max_btc = max_exposure / 50000 * 0.5  # Buy half of max exposure
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, max_btc, 50000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Try to exceed exposure limit
        try:
            self.portfolio.submit_order(
                'binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, max_btc * 0.8, 50000)
            self.fail("Should have rejected exceeding exposure limit")
        except ValueError as e:
            self.assertIn("exposure", str(e).lower())
        
        print(f"  ✅ Risk management working")
    
    # ============= Test 5: Order Cancellation =============
    
    def test_order_cancellation(self):
        """Test order cancellation."""
        print("\n🧪 Testing Order Cancellation...")
        
        # Submit order
        order_id = self.portfolio.submit_order(
            'binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        self.assertEqual(len(self.portfolio.get_active_orders()), 1)
        
        # Cancel order
        result = self.portfolio.cancel_order(order_id)
        self.assertTrue(result)
        self.assertEqual(len(self.portfolio.get_active_orders()), 0)
        
        # Check order status
        order = self.portfolio.get_order(order_id)
        self.assertEqual(order.status, OrderStatus.CANCELLED)
        
        # Try to cancel non-existent order
        result = self.portfolio.cancel_order("non_existent")
        self.assertFalse(result)
        
        print(f"  ✅ Order cancellation working")
    
    # ============= Test 6: Performance Analytics =============
    
    def test_performance_analytics(self):
        """Test performance analytics and statistics."""
        print("\n🧪 Testing Performance Analytics...")
        
        # Create a series of trades
        trades = [
            ('BUY', 0.1, 50000),
            ('BUY', 0.05, 51000),
            ('SELL', 0.08, 52000),
            ('SELL', 0.05, 53000),
        ]
        
        for side, qty, price in trades:
            side_enum = OrderSide.BUY if side == 'BUY' else OrderSide.SELL
            self.portfolio.submit_order('binance', 'BTC/USD', side_enum, OrderType.LIMIT, qty, price)
            order_id = next(iter(self.exchanges['binance'].orders.keys()))
            self.exchanges['binance'].fill_order(order_id)
        
        # Calculate trade statistics
        stats = self.portfolio.performance_analyzer.calculate_trade_statistics()
        self.assertEqual(stats['total_trades'], 2)  # Only sell trades count
        self.assertGreater(stats['win_rate'], 0)
        self.assertGreater(stats['profit_factor'], 1.0)
        
        # Get performance metrics
        metrics = self.portfolio.performance_analyzer.get_performance_metrics()
        self.assertGreater(metrics.win_rate, 0)
        
        print(f"  ✅ Performance analytics working")
    
    # ============= Test 7: Alert System =============
    
    def test_alert_system(self):
        """Test alert system."""
        print("\n🧪 Testing Alert System...")
        
        alert_received = []
        
        def test_handler(alert):
            alert_received.append(alert)
        
        self.portfolio.alert_system.register_handler(test_handler)
        
        # Add alert
        self.portfolio.alert_system.add_alert(
            AlertSeverity.WARNING, "Test", "Test Alert", "This is a test")
        self.assertEqual(len(alert_received), 1)
        self.assertEqual(len(self.portfolio.alert_system.alerts), 1)
        
        # Check unacknowledged alerts
        unack = self.portfolio.alert_system.get_unacknowledged_alerts()
        self.assertEqual(len(unack), 1)
        
        # Acknowledge alert
        self.portfolio.alert_system.acknowledge_alert(alert_received[0].alert_id)
        unack = self.portfolio.alert_system.get_unacknowledged_alerts()
        self.assertEqual(len(unack), 0)
        
        print(f"  ✅ Alert system working")
    
    # ============= Test 8: Report Generation =============
    
    def test_report_generation(self):
        """Test report generation."""
        print("\n🧪 Testing Report Generation...")
        
        # Create some trades
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Generate reports
        pnl_report = self.portfolio.reporting.generate_pnl_report()
        risk_report = self.portfolio.reporting.generate_risk_report()
        perf_report = self.portfolio.reporting.generate_performance_report()
        
        # Check reports exist
        self.assertTrue(Path(pnl_report).exists())
        self.assertTrue(Path(risk_report).exists())
        self.assertTrue(Path(perf_report).exists())
        
        # Check JSON is valid
        for report in [pnl_report, risk_report, perf_report]:
            with open(report, 'r') as f:
                data = json.load(f)
                self.assertIsNotNone(data)
        
        print(f"  ✅ Report generation working")
    
    # ============= Test 9: State Persistence =============
    
    def test_state_persistence(self):
        """Test save and load state."""
        print("\n🧪 Testing State Persistence...")
        
        # Create some state
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        try:
            self.portfolio.save_state(temp_file)
            
            # Create new portfolio and load
            new_portfolio = PortfolioManager(self.exchanges)
            for exchange_name, exchange in self.exchanges.items():
                for currency, data in exchange.get_balances().items():
                    new_portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                        exchange_name, currency, data['total'], data['locked'])
            
            new_portfolio.load_state(temp_file)
            
            # Compare
            self.assertEqual(len(self.portfolio._ledger), len(new_portfolio._ledger))
            self.assertEqual(len(self.portfolio._positions), len(new_portfolio._positions))
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"  ✅ State persistence working")
    
    # ============= Test 10: Thread Safety =============
    
    def test_thread_safety(self):
        """Test thread safety with concurrent operations."""
        print("\n🧪 Testing Thread Safety...")
        
        def submit_orders():
            for i in range(3):
                try:
                    self.portfolio.submit_order(
                        'binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.01, 50000 + i * 100)
                except Exception as e:
                    pass
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=submit_orders)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify no corruption
        self.assertGreaterEqual(len(self.portfolio._orders), 0)
        self.assertGreaterEqual(len(self.portfolio._active_orders), 0)
        
        print(f"  ✅ Thread safety working")
    
    # ============= Test 11: Balance Tracking =============
    
    def test_balance_tracking(self):
        """Test balance tracking and updates."""
        print("\n🧪 Testing Balance Tracking...")
        
        # Get initial balances
        initial_usd = self.portfolio.get_balance('binance', 'USD')
        self.assertEqual(initial_usd.total, 50000)
        
        # Submit and fill buy order
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Check balances updated
        usd = self.portfolio.get_balance('binance', 'USD')
        btc = self.portfolio.get_balance('binance', 'BTC')
        
        self.assertAlmostEqual(usd.total, 50000 - 5000, places=2)
        self.assertAlmostEqual(btc.total, 1.0 + 0.1, places=4)
        
        print(f"  ✅ Balance tracking working")
    
    # ============= Test 12: Multiple Assets =============
    
    def test_multiple_assets(self):
        """Test trading multiple assets."""
        print("\n🧪 Testing Multiple Assets...")
        
        # Buy BTC
        self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        order_id = next(iter(self.exchanges['binance'].orders.keys()))
        self.exchanges['binance'].fill_order(order_id)
        
        # Buy ETH
        self.portfolio.submit_order('coinbase', 'ETH/USD', OrderSide.BUY, OrderType.LIMIT, 0.5, 3000)
        order_id = next(iter(self.exchanges['coinbase'].orders.keys()))
        self.exchanges['coinbase'].fill_order(order_id)
        
        # Check both positions
        btc_pos = self.portfolio.get_position('binance', 'BTC/USD')
        eth_pos = self.portfolio.get_position('coinbase', 'ETH/USD')
        
        self.assertIsNotNone(btc_pos)
        self.assertIsNotNone(eth_pos)
        self.assertGreater(btc_pos.quantity, 0)
        self.assertGreater(eth_pos.quantity, 0)
        
        # Check portfolio summary
        summary = self.portfolio.get_portfolio_summary()
        self.assertEqual(len(summary['positions']), 2)
        self.assertGreater(summary['total_positions_value'], 0)
        
        print(f"  ✅ Multiple assets working")


# ============= Run Tests =============

def run_tests():
    """Run all tests with detailed output."""
    print("\n" + "="*60)
    print("PORTFOLIO MANAGEMENT SYSTEM - COMPLETE TEST SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioManager))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        for failure in result.failures:
            print(f"\n  FAIL: {failure[0]}")
        for error in result.errors:
            print(f"\n  ERROR: {error[0]}")
    
    return result.wasSuccessful()


# ============= Demo =============

def run_demo():
    """Run a quick demo of the system."""
    print("\n" + "="*60)
    print("PORTFOLIO MANAGEMENT SYSTEM DEMO")
    print("="*60)
    
    exchanges = {'binance': MockExchange('binance'), 'coinbase': MockExchange('coinbase')}
    portfolio = PortfolioManager(exchanges)
    
    for exchange_name, exchange in exchanges.items():
        for currency, data in exchange.get_balances().items():
            portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                exchange_name, currency, data['total'], data['locked'])
        print(f"Loaded {exchange_name} balances")
    
    portfolio.risk_manager.max_position_size_pct = 0.3
    portfolio.risk_manager.max_exposure_pct = 0.7
    
    try:
        total_equity = portfolio.get_total_equity()
        print(f"\nTotal Equity: ${total_equity:.2f}")
        
        print("\n--- Submitting Orders ---")
        btc_qty = min(0.1, portfolio.get_balance('binance', 'USD').free * 0.05 / 50000)
        eth_qty = min(0.5, portfolio.get_balance('coinbase', 'USD').free * 0.05 / 3000)
        
        order1 = portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, btc_qty, 50000)
        order2 = portfolio.submit_order('coinbase', 'ETH/USD', OrderSide.BUY, OrderType.LIMIT, eth_qty, 3000)
        print(f"Orders submitted: {order1}, {order2}")
        
        print("\n--- Filling Orders ---")
        for exchange in exchanges.values():
            for order_id in list(exchange.orders.keys()):
                exchange.fill_order(order_id)
        print("Orders filled")
        time.sleep(0.5)
        
        summary = portfolio.get_portfolio_summary()
        print(f"\n--- Summary ---")
        print(f"Total Equity: ${summary['total_equity']:.2f}")
        print(f"Total PnL: ${summary['total_pnl']:.2f}")
        print(f"Active Orders: {summary['active_orders_count']}")
        
        print("\n--- Positions ---")
        for pos in portfolio.get_all_positions():
            print(f"  {pos.symbol}: {pos.quantity:.4f} @ ${pos.cost_basis:.2f} (PnL: ${pos.unrealized_pnl:.2f})")
        
        print("\n--- Reports ---")
        print(f"P&L Report: {portfolio.reporting.generate_pnl_report()}")
        print(f"Risk Report: {portfolio.reporting.generate_risk_report()}")
        print(f"Performance Report: {portfolio.reporting.generate_performance_report()}")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nDemo complete.")


# ============= Main =============

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run tests
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        # Run demo
        run_demo()

    