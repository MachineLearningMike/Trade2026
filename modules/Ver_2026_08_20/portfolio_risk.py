"""
portfolio_risk.py — risk metrics and RiskManagerV2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np

from portfolio_core import (
    Config,
    Order,
    OrderSide,
    OrderType,
    LotStatus,
    RiskLevel,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    normalize_symbol,
    split_symbol,
    to_serializable,
)

if TYPE_CHECKING:
    from portfolio_core import PortfolioManagerV2

logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> Dict:
        return to_serializable(asdict(self))


class RiskManagerV2:
    """Risk manager with lot-level awareness and VaR calculations."""
    
    def __init__(self, portfolio: 'PortfolioManagerV2', config: Dict = None):
        self.portfolio = portfolio
        self.config = config or {}
        
        self.max_position_size_pct = self.config.get('max_position_size_pct', Config.MAX_POSITION_SIZE_PCT)
        self.max_exposure_pct = self.config.get('max_exposure_pct', Config.MAX_EXPOSURE_PCT)
        self.max_lots_per_symbol = self.config.get('max_lots_per_symbol', Config.MAX_LOTS_PER_SYMBOL)
        self.daily_loss_limit = self.config.get('daily_loss_limit', Config.DAILY_LOSS_LIMIT)
        self.risk_free_rate = self.config.get('risk_free_rate', Config.RISK_FREE_RATE)
        
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.last_reset_day = datetime.now().date()
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
        
        if order.side == OrderSide.SELL:
            if position and order.quantity > position.total_quantity:
                return False, f"Insufficient position: have {position.total_quantity}, want {order.quantity}"
            return True, None
        
        if order_value > total_equity * self.max_position_size_pct:
            return False, f"Position size ${order_value:.2f} exceeds limit ${total_equity * self.max_position_size_pct:.2f}"
        
        total_exposure = sum(p.market_value for p in self.portfolio.get_all_positions())
        if total_exposure + order_value > total_equity * self.max_exposure_pct:
            return False, f"Exposure would exceed ${total_equity * self.max_exposure_pct:.2f}"
        
        active_lots = position.lot_count if position else 0
        if active_lots >= self.max_lots_per_symbol:
            return False, f"Max lots ({self.max_lots_per_symbol}) reached for {order.symbol}"
        
        if self.daily_pnl < -total_equity * self.daily_loss_limit:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        if self.current_drawdown > Config.MAX_DRAWDOWN_PCT:
            return False, f"Max drawdown {self.current_drawdown:.2%} exceeded"
        
        return True, None

    def check_buy(
        self,
        exchange: str,
        symbol: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Risk + cash check for a market buy (used by strategy hot path).
        """
        symbol = normalize_symbol(symbol)
        order = Order(
            exchange=exchange,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
        )
        ok, reason = self.check_order(order)
        if not ok:
            return ok, reason
        # Cash availability (quote + fee)
        base, quote = split_symbol(symbol)
        bal = self.portfolio._balances.get(f"{exchange}:{quote}")
        cost = quantity * price + fee
        if bal is None:
            return False, f"No {quote} balance on {exchange}"
        if bal.free < cost * (1 + Config.SLIPPAGE_BUFFER):
            return False, (
                f"Insufficient {quote}: need ${cost:.2f}, free ${bal.free:.2f}"
            )
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
        returns_array = np.array(returns)
        return np.percentile(returns_array, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk."""
        if not returns:
            return 0.0
        var = self.calculate_var(returns, confidence)
        returns_below_var = [r for r in returns if r <= var]
        if not returns_below_var:
            return var
        return np.mean(returns_below_var)
    
    def get_historical_returns(self, days: int = 30) -> List[float]:
        """Get historical returns from lot trades."""
        if self._returns_cache and self._cache_timestamp:
            if (datetime.now() - self._cache_timestamp).seconds < 60:
                return self._returns_cache
        
        trades = self.portfolio.get_trades(limit=1000)
        if not trades:
            return []
        
        daily_pnl = defaultdict(float)
        for trade in trades:
            if trade.is_sell:
                day = trade.timestamp.date()
                if trade.realized_pnl is not None:
                    daily_pnl[day] += trade.realized_pnl
                    continue
                # Legacy fallback: live average cost is wrong after partial closes
                position = self.portfolio.get_position(trade.exchange, trade.symbol)
                if position and position.average_cost > 0:
                    pnl = trade.quantity * (trade.price - position.average_cost) - trade.fee
                    daily_pnl[day] += pnl
        
        returns = []
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
        
        returns = self.get_historical_returns()
        if returns:
            metrics.var_95 = self.calculate_var(returns, 0.95)
            metrics.var_99 = self.calculate_var(returns, 0.99)
            metrics.cvar_95 = self.calculate_cvar(returns, 0.95)
            metrics.volatility = np.std(returns) * np.sqrt(252)
            metrics.sharpe_ratio = calculate_sharpe_ratio(returns, self.risk_free_rate)
            metrics.sortino_ratio = calculate_sortino_ratio(returns, self.risk_free_rate)
            metrics.calmar_ratio = calculate_calmar_ratio(returns, self.current_drawdown)
        
        total_equity = self.portfolio.get_total_equity()
        total_exposure = sum(p.market_value for p in self.portfolio.get_all_positions())
        metrics.exposure = total_exposure / max(total_equity, 1)
        
        positions = self.portfolio.get_all_positions()
        if positions:
            largest = max(positions, key=lambda p: p.market_value)
            metrics.concentration_ratio = largest.market_value / max(total_equity, 1)
        
        metrics.max_drawdown = self.current_drawdown
        metrics.max_drawdown_percentage = self.current_drawdown * 100
        
        # VaR is negative for losses; milder (closer to zero) + lower exposure = lower risk.
        # Escalate if either VaR is severe or exposure is high.
        if metrics.var_95 >= -0.02 and metrics.exposure < 0.3:
            metrics.risk_level = RiskLevel.LOW
        elif metrics.var_95 >= -0.05 and metrics.exposure < 0.6:
            metrics.risk_level = RiskLevel.MEDIUM
        elif metrics.var_95 >= -0.10 and metrics.exposure < 0.8:
            metrics.risk_level = RiskLevel.HIGH
        else:
            metrics.risk_level = RiskLevel.CRITICAL
        
        return metrics
