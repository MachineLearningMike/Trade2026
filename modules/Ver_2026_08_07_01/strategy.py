"""
EWMA Signal-Based Trading Strategy - Integrated with Lot-Based Portfolio Management
Adds:
- EWMA probability smoothing
- Dynamic lot quantity based on signal strength
- TP/SL exit logic per lot
- Market order execution

Run a self-contained demo (mock price + probability placeholders)::

    python strategy.py
"""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from portfolio import (
    AlertSeverity,
    Balance,
    MockExchange,
    OrderSide,
    OrderTimeInForce,
    OrderType,
    PortfolioManagerV2,
    SpecificLotStrategy,
    normalize_symbol,
)

logger = logging.getLogger(__name__)


# ============================================================================
# PLACEHOLDER PRICE / PROBABILITY FEEDS (demo / until real model is wired)
# ============================================================================

class PlaceholderPriceFeed:
    """
    Synthetic mid price with mild random walk.

    Replace with a real ticker (exchange / Binance) by passing
    ``get_price_func=...`` in the engine config.
    """

    def __init__(
        self,
        base_prices: Optional[Dict[str, float]] = None,
        rng: Optional[np.random.Generator] = None,
        vol: float = 0.0015,
    ):
        self.prices: Dict[str, float] = dict(
            base_prices
            or {
                'BTC/USD': 50_000.0,
                'ETH/USD': 3_000.0,
            }
        )
        self.rng = rng or np.random.default_rng(0)
        self.vol = float(vol)

    def __call__(self, symbol: str) -> float:
        px = float(self.prices.get(symbol, 50_000.0))
        # Log-normal-ish step so TP/SL can eventually fire in demos
        shock = float(self.rng.normal(0.0, self.vol))
        px = max(1e-8, px * (1.0 + shock))
        self.prices[symbol] = px
        return px

    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = float(price)


class PlaceholderProbabilityFeed:
    """
    Synthetic "P(favorable move)" in [0, 1].

    Mean is biased above the default EWMA threshold so the demo buys;
    noise keeps the path interesting. Replace with model inference via
    ``get_probability_func=...``.
    """

    def __init__(
        self,
        mean: float = 0.72,
        noise: float = 0.12,
        rng: Optional[np.random.Generator] = None,
    ):
        self.mean = float(mean)
        self.noise = float(noise)
        self.rng = rng or np.random.default_rng(1)

    def __call__(self, symbol: str) -> float:
        # symbol reserved for multi-asset model routing later
        p = self.mean + float(self.rng.normal(0.0, self.noise))
        return float(np.clip(p, 0.0, 1.0))


# ============================================================================
# CONFIGURATION EXTENSION
# ============================================================================

class TradingConfig:
    """Trading parameters for the EWMA signal strategy."""
    
    # EWMA parameters
    EWMA_ALPHA = 0.2           # Decay factor (0.2 = ~10-minute half-life)
    EWMA_THRESHOLD = 0.6       # Buy when smoothed prob exceeds this level
    
    # TP/SL parameters as *net* fractional return after round-trip fees
    TAKE_PROFIT = 0.005        # 0.5% net take-profit
    STOP_LOSS = -0.003         # -0.3% net stop-loss
    
    # Position sizing
    Q_MAX = 0.01               # Maximum quantity per lot (units)
    
    # Fees (constant)
    BUY_FEE = 0.001            # 0.1% buy fee
    SELL_FEE = 0.001           # 0.1% sell fee
    TOTAL_FEES = BUY_FEE + SELL_FEE  # 0.2% round-trip
    
    # Order execution
    USE_MARKET_ORDERS = True   # True = market orders, False = limit orders
    
    # Risk limits
    MAX_INVENTORY_LOTS = 10    # Maximum lots per symbol
    MAX_DAILY_TRADES = 50      # Maximum trades per day


# ============================================================================
# EWMA STRATEGY ENGINE
# ============================================================================

class EWMAStrategyEngine:
    """
    EWMA-based trading strategy engine.
    Calculates smoothed probability and determines buy/sell decisions.
    """
    
    def __init__(self, config: Dict = None, symbol: str = ''):
        self.config = config or {}
        self.symbol = symbol
        
        # EWMA state
        self.ewma = 0.0
        self._ewma_initialized = False
        self.alpha = float(self.config.get('ewma_alpha', TradingConfig.EWMA_ALPHA))
        self.threshold = float(self.config.get('ewma_threshold', TradingConfig.EWMA_THRESHOLD))
        if not (0.0 <= self.threshold < 1.0):
            raise ValueError(
                f"ewma_threshold must be in [0, 1); got {self.threshold}"
            )
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"ewma_alpha must be in (0, 1]; got {self.alpha}")
        
        # TP/SL parameters
        self.take_profit = self.config.get('take_profit', TradingConfig.TAKE_PROFIT)
        self.stop_loss = self.config.get('stop_loss', TradingConfig.STOP_LOSS)
        
        # Position sizing
        self.q_max = self.config.get('q_max', TradingConfig.Q_MAX)
        
        # Fees: entry + exit charged once each (not double-counted)
        self.buy_fee = float(self.config.get('buy_fee', TradingConfig.BUY_FEE))
        self.sell_fee = float(self.config.get('sell_fee', TradingConfig.SELL_FEE))
        self.total_fees = float(
            self.config.get('total_fees', self.buy_fee + self.sell_fee)
        )
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0
        
        # History for backtesting
        self.ewma_history: List[float] = []
        self.signal_history: List[Dict] = []
        
        logger.info(
            f"EWMAStrategyEngine[{symbol or '*'}] initialized: "
            f"alpha={self.alpha}, threshold={self.threshold}, "
            f"TP={self.take_profit:.2%}, SL={self.stop_loss:.2%}"
        )
    
    def update_ewma(self, probability: float) -> float:
        """Update EWMA; first observation seeds the filter (no cold-start bias)."""
        p = float(probability)
        if not self._ewma_initialized:
            self.ewma = p
            self._ewma_initialized = True
        else:
            self.ewma = self.alpha * p + (1.0 - self.alpha) * self.ewma
        self.ewma_history.append(self.ewma)
        return self.ewma
    
    def calculate_quantity(self) -> float:
        """
        Calculate buy quantity based on EWMA signal strength.
        Q = Q_MAX × (EWMA - threshold) / (1 - threshold)
        """
        if self.threshold >= 1.0:
            return 0.0
        if self.ewma <= self.threshold:
            return 0.0
        
        q = self.q_max * (self.ewma - self.threshold) / (1.0 - self.threshold)
        return min(q, self.q_max)
    
    def should_buy(self, current_inventory_count: int, max_lots: int = 10) -> Tuple[bool, float]:
        """
        Determine if we should buy.
        Returns: (should_buy, quantity)
        """
        if current_inventory_count >= max_lots:
            return False, 0.0
        
        if self.ewma <= self.threshold:
            return False, 0.0
        
        quantity = self.calculate_quantity()
        return quantity > 0, quantity
    
    def net_return(self, buy_price: float, sell_price: float) -> float:
        """
        Net fractional return after fees once each side:
          gross = (sell - buy) / buy
          net   = gross - buy_fee - sell_fee
        """
        if buy_price <= 0:
            return 0.0
        gross_pl = (sell_price - buy_price) / buy_price
        return gross_pl - self.buy_fee - self.sell_fee

    def should_sell_lot(self, lot, current_price: float) -> Tuple[bool, str]:
        """
        Determine if a specific lot should be sold using *net* return
        (gross price move minus buy_fee and sell_fee).

        Returns: (should_sell, reason: 'TP' or 'SL')
        """
        # Prefer lot.entry_fee rate if stored; else use engine fee rates
        entry_fee_rate = self.buy_fee
        if getattr(lot, 'quantity', 0) and getattr(lot, 'entry_fee', 0):
            notional = lot.quantity * lot.entry_price
            if notional > 0:
                entry_fee_rate = float(lot.entry_fee) / notional
        if lot.entry_price <= 0:
            return False, ''
        gross = (current_price - lot.entry_price) / lot.entry_price
        net_pl = gross - entry_fee_rate - self.sell_fee

        if net_pl >= self.take_profit:
            return True, 'TP'
        if net_pl <= self.stop_loss:
            return True, 'SL'
        return False, ''
    
    def calculate_net_pnl(self, buy_price: float, sell_price: float) -> float:
        """Alias for net_return (fractional)."""
        return self.net_return(buy_price, sell_price)

    def entry_fee(self, price: float, quantity: float) -> float:
        return float(price) * float(quantity) * self.buy_fee

    def exit_fee(self, price: float, quantity: float) -> float:
        return float(price) * float(quantity) * self.sell_fee
    
    def record_trade(self, net_pnl: float):
        """Record trade performance (pass net PnL only — fees already deducted)."""
        self.trade_count += 1
        self.total_pnl += net_pnl
        if net_pnl > 0:
            self.win_count += 1
    
    def get_statistics(self) -> Dict:
        """Get strategy statistics."""
        return {
            'symbol': self.symbol,
            'current_ewma': self.ewma,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'win_rate': self.win_count / max(self.trade_count, 1),
            'total_pnl': self.total_pnl,
            'avg_pnl': self.total_pnl / max(self.trade_count, 1),
            'ewma_history_length': len(self.ewma_history),
        }
    
    def reset(self):
        """Reset strategy state."""
        self.ewma = 0.0
        self._ewma_initialized = False
        self.ewma_history = []
        self.signal_history = []
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0


# ============================================================================
# INTEGRATED TRADING ENGINE
# ============================================================================

class EWMATradingEngine:
    """
    Full trading engine integrating EWMA strategy with portfolio management.

    One ``EWMAStrategyEngine`` per symbol (no cross-symbol EWMA bleed).
    """
    
    def __init__(self, portfolio: 'PortfolioManagerV2', config: Dict = None):
        """
        Args:
            portfolio: PortfolioManagerV2 instance
            config: Configuration dict with trading parameters.

                Required for non-demo use (fail-closed):
                  get_price_func(symbol) -> float
                  get_probability_func(symbol) -> float in [0, 1]

                Set allow_placeholders=True to use Placeholder* feeds (demo only).
        """
        self.portfolio = portfolio
        self.config = config or {}
        
        # Trading state
        self._running = False
        self._lock = threading.RLock()
        self._last_run_time: Optional[datetime] = None
        self._trade_log: List[Dict] = []
        
        # Extract symbols to trade (normalized BASE/QUOTE)
        raw_symbols = list(self.config.get('symbols', ['BTC/USD']))
        self.symbols = [normalize_symbol(s) for s in raw_symbols]
        self.exchange = self.config.get('exchange', 'binance')
        
        # Per-symbol strategy engines (never share EWMA across symbols)
        self.strategies: Dict[str, EWMAStrategyEngine] = {
            sym: EWMAStrategyEngine(self.config, symbol=sym) for sym in self.symbols
        }
        
        # Risk limits
        self.max_lots = self.config.get('max_lots', TradingConfig.MAX_INVENTORY_LOTS)
        self.max_daily_trades = self.config.get(
            'max_daily_trades', TradingConfig.MAX_DAILY_TRADES
        )
        # Per-symbol daily trade cap (default = global max)
        self.max_daily_trades_per_symbol = int(
            self.config.get('max_daily_trades_per_symbol', self.max_daily_trades)
        )
        self.use_market_orders = bool(
            self.config.get('use_market_orders', TradingConfig.USE_MARKET_ORDERS)
        )
        self.enforce_risk = bool(self.config.get('enforce_risk', True))
        self.daily_trade_count = 0
        self.daily_trade_count_by_symbol: Dict[str, int] = {
            s: 0 for s in self.symbols
        }
        self._last_reset_day = datetime.now().date()
        
        # Price / probability feeds — fail closed unless placeholders allowed
        allow_ph = bool(self.config.get('allow_placeholders', False))
        price_fn = self.config.get('get_price_func')
        prob_fn = self.config.get('get_probability_func')
        if price_fn is None:
            if not allow_ph:
                raise ValueError(
                    "get_price_func is required (or set allow_placeholders=True for demo)."
                )
            price_fn = PlaceholderPriceFeed()
            logger.warning("Using PlaceholderPriceFeed (demo only).")
        if prob_fn is None:
            if not allow_ph:
                raise ValueError(
                    "get_probability_func is required "
                    "(or set allow_placeholders=True for demo)."
                )
            prob_fn = PlaceholderProbabilityFeed()
            logger.warning("Using PlaceholderProbabilityFeed (demo only).")
        self.price_feed: Callable[[str], float] = price_fn
        self.probability_feed: Callable[[str], float] = prob_fn
        
        logger.info(f"EWMATradingEngine initialized for {self.symbols} on {self.exchange}")

    def _strategy_for(self, symbol: str) -> EWMAStrategyEngine:
        symbol = normalize_symbol(symbol)
        if symbol not in self.strategies:
            self.strategies[symbol] = EWMAStrategyEngine(self.config, symbol=symbol)
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            self.daily_trade_count_by_symbol.setdefault(symbol, 0)
        return self.strategies[symbol]

    @property
    def strategy(self) -> EWMAStrategyEngine:
        """
        Deprecated single-engine accessor.

        Raises if more than one symbol is configured — use
        ``strategies[symbol]`` or ``_strategy_for(symbol)``.
        """
        if len(self.strategies) != 1:
            raise AttributeError(
                "self.strategy is ambiguous with multiple symbols; "
                "use engine.strategies[symbol] or engine._strategy_for(symbol)"
            )
        return next(iter(self.strategies.values()))
    
    def _get_lot_count(self, symbol: str) -> int:
        """Get number of open lots for a symbol."""
        symbol = normalize_symbol(symbol)
        position = self.portfolio.get_position(self.exchange, symbol)
        if not position:
            return 0
        return len([l for l in position.lots if l.is_active])
    
    def _reset_daily_counter(self):
        """Reset daily trade counters if a new day has started."""
        today = datetime.now().date()
        if today > self._last_reset_day:
            self.daily_trade_count = 0
            for s in self.daily_trade_count_by_symbol:
                self.daily_trade_count_by_symbol[s] = 0
            self._last_reset_day = today

    def _can_trade_today(self, symbol: str) -> bool:
        symbol = normalize_symbol(symbol)
        if self.daily_trade_count >= self.max_daily_trades:
            return False
        if self.daily_trade_count_by_symbol.get(symbol, 0) >= self.max_daily_trades_per_symbol:
            return False
        return True

    def _note_trade(self, symbol: str) -> None:
        symbol = normalize_symbol(symbol)
        self.daily_trade_count += 1
        self.daily_trade_count_by_symbol[symbol] = (
            self.daily_trade_count_by_symbol.get(symbol, 0) + 1
        )
    
    def _log_trade(self, trade_type: str, lot, price: float, pnl: float,
                   reason: str = '', symbol: str = '', quantity: float = 0.0,
                   ewma: float = 0.0):
        """Log a trade for record keeping."""
        self._trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': trade_type,
            'symbol': getattr(lot, 'symbol', symbol) if lot is not None else symbol,
            'quantity': getattr(lot, 'quantity', quantity) if lot is not None else quantity,
            'entry_price': getattr(lot, 'entry_price', 0.0) if lot is not None else 0.0,
            'exit_price': price,
            'pnl': pnl,
            'reason': reason,
            'lot_id': getattr(lot, 'lot_id', None) if lot is not None else None,
            'ewma': ewma,
        })

    def _mark_price(self, symbol: str, price: float) -> None:
        """Keep portfolio mark-to-market in sync with the price feed (under portfolio lock)."""
        self.portfolio.update_mark(symbol, float(price), exchange=self.exchange)
        ex = self.portfolio.exchanges.get(self.exchange)
        if ex is not None and hasattr(ex, 'update_price'):
            ex.update_price(symbol, float(price))
    
    def step(self, symbol: str) -> Dict:
        """
        Execute one trading step (one minute) for a symbol.
        
        Returns:
            Dict with actions taken this step
        """
        with self._lock:
            self._reset_daily_counter()
            symbol = normalize_symbol(symbol)
            strat = self._strategy_for(symbol)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'actions': [],
                'ewma': strat.ewma,
                'probability': 0.0,
                'price': 0.0,
                'bought': False,
                'sold': 0,
            }
            
            # --- Get current price and probability ---
            current_price = float(self.price_feed(symbol))
            probability = float(self.probability_feed(symbol))
            result['probability'] = probability
            
            if not math.isfinite(current_price) or current_price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {current_price}")
            if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
                raise ValueError(
                    f"Probability for {symbol} must be in [0, 1]; got {probability}"
                )
            
            result['price'] = current_price
            self._mark_price(symbol, current_price)
            
            # --- Update per-symbol EWMA ---
            ewma = strat.update_ewma(probability)
            result['ewma'] = ewma
            
            # --- SELL PHASE: close the exact lots that hit TP/SL (net of fees) ---
            position = self.portfolio.get_position(self.exchange, symbol)
            lots_to_sell = []
            
            if position:
                for lot in position.lots:
                    if not lot.is_active:
                        continue
                    should_sell, reason = strat.should_sell_lot(lot, current_price)
                    if should_sell:
                        lots_to_sell.append((lot, reason))
            
            sold_count = 0
            for lot, reason in lots_to_sell:
                try:
                    qty = float(lot.remaining_quantity)
                    # Exit fee only; entry fee is on the lot and applied in close_partial
                    fee = strat.exit_fee(current_price, qty)
                    
                    if self.use_market_orders:
                        lot_strategy = SpecificLotStrategy([lot.lot_id])
                        results = self.portfolio.close_lots(
                            self.exchange, symbol,
                            qty, current_price,
                            lot_strategy, fee,
                        )
                        
                        for closed_lot, pnl in results:
                            strat.record_trade(pnl)
                            self._note_trade(symbol)
                            sold_count += 1
                            self._log_trade(
                                'SELL', closed_lot, current_price, pnl, reason,
                                ewma=ewma,
                            )
                            
                            if abs(pnl) > 10:
                                sev = (
                                    AlertSeverity.INFO if pnl > 0
                                    else AlertSeverity.WARNING
                                )
                                self.portfolio.alert_system.add_alert(
                                    sev,
                                    'Trade Execution',
                                    f'{reason} Executed',
                                    f'Sold {qty:.4f} {symbol} '
                                    f'at ${current_price:.2f}, PnL: ${pnl:.2f}',
                                )
                    else:
                        order_id = self.portfolio.submit_order(
                            self.exchange, symbol,
                            OrderSide.SELL, OrderType.LIMIT,
                            qty,
                            current_price * (
                                1 + strat.take_profit if reason == 'TP'
                                else 1 + strat.stop_loss
                            ),
                            time_in_force=OrderTimeInForce.GTC,
                        )
                        self._log_trade(
                            'SELL_LIMIT', lot, current_price, 0, reason, ewma=ewma,
                        )
                        result['actions'].append(
                            {'sell_limit': order_id, 'reason': reason, 'lot_id': lot.lot_id}
                        )
                        
                except Exception as e:
                    logger.error(f"Error selling lot {lot.lot_id}: {e}")
                    self.portfolio.alert_system.add_alert(
                        AlertSeverity.ERROR, 'Trade Execution', 
                        'Sell Failed', f'Error selling {symbol}: {e}'
                    )
            
            result['sold'] = sold_count
            
            # --- BUY PHASE (only if no sells this step) ---
            if sold_count == 0:
                lot_count = self._get_lot_count(symbol)
                should_buy, quantity = strat.should_buy(lot_count, self.max_lots)
                
                if should_buy and self._can_trade_today(symbol):
                    try:
                        entry_fee = strat.entry_fee(current_price, quantity)
                        if self.enforce_risk:
                            ok, reason = self.portfolio.risk_manager.check_buy(
                                self.exchange, symbol, quantity, current_price,
                                fee=entry_fee,
                            )
                            if not ok:
                                result['actions'].append(
                                    {'buy_blocked': reason}
                                )
                                logger.info(f"Buy blocked {symbol}: {reason}")
                                should_buy = False

                        if should_buy and self.use_market_orders:
                            lot_id = self.portfolio.add_lot(
                                self.exchange, symbol,
                                quantity, current_price,
                                tags=['ewma_signal'],
                                fee=entry_fee,
                                strategy_metadata={
                                    'ewma': ewma,
                                    'probability': probability,
                                    'entry_fee': entry_fee,
                                    'buy_fee_rate': strat.buy_fee,
                                    'timestamp': datetime.now().isoformat(),
                                }
                            )
                            self._mark_price(symbol, current_price)
                            self._log_trade(
                                'BUY',
                                self.portfolio.get_lot(lot_id),
                                current_price, 0.0, 'EWMA Signal',
                                symbol=symbol, quantity=quantity, ewma=ewma,
                            )
                            result['bought'] = True
                            result['buy_quantity'] = quantity
                            result['entry_fee'] = entry_fee
                            self._note_trade(symbol)
                            
                            logger.info(
                                f"Bought {quantity:.4f} {symbol} at "
                                f"${current_price:.2f} (EWMA={ewma:.3f}, "
                                f"entry_fee=${entry_fee:.4f})"
                            )
                            
                        elif should_buy and not self.use_market_orders:
                            ok, reason = self.portfolio.risk_manager.check_buy(
                                self.exchange, symbol, quantity, current_price,
                                fee=entry_fee,
                            ) if self.enforce_risk else (True, None)
                            if not ok:
                                result['actions'].append({'buy_blocked': reason})
                            else:
                                order_id = self.portfolio.submit_order(
                                    self.exchange, symbol,
                                    OrderSide.BUY, OrderType.LIMIT,
                                    quantity, current_price,
                                    time_in_force=OrderTimeInForce.GTC
                                )
                                result['bought'] = True
                                result['buy_order'] = order_id
                            
                    except Exception as e:
                        logger.error(f"Error buying {symbol}: {e}")
                        self.portfolio.alert_system.add_alert(
                            AlertSeverity.ERROR, 'Trade Execution',
                            'Buy Failed', f'Error buying {symbol}: {e}'
                        )
            
            # --- Record signal history (per-symbol engine) ---
            strat.signal_history.append({
                'timestamp': result['timestamp'],
                'symbol': symbol,
                'ewma': ewma,
                'probability': probability,
                'price': current_price,
                'bought': result['bought'],
                'sold': sold_count,
                'inventory_count': self._get_lot_count(symbol),
            })
            
            self._last_run_time = datetime.now()
            return result
    
    def run_cycle(self) -> Dict:
        """
        Run one full cycle (all symbols).
        
        Returns:
            Dict with results for all symbols
        """
        results = {}
        for symbol in self.symbols:
            try:
                results[symbol] = self.step(symbol)
            except Exception as e:
                logger.exception(f"Error processing {symbol}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def run_loop(self, interval_seconds: float = 60.0):
        """Run the trading loop continuously."""
        self._running = True
        logger.info(f"Starting trading loop (interval={interval_seconds}s)")
        
        while self._running:
            start_time = datetime.now()
            
            try:
                results = self.run_cycle()
                
                if results:
                    for symbol, result in results.items():
                        if result.get('bought') or result.get('sold', 0) > 0:
                            logger.info(
                                f"Cycle: {symbol} - Bought: {result.get('bought', False)}, "
                                f"Sold: {result.get('sold', 0)}, "
                                f"EWMA: {result.get('ewma', 0):.3f}"
                            )
                
                self.portfolio.sync_with_exchanges()
                
            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, interval_seconds - elapsed)
            if sleep_time > 0 and self._running:
                time.sleep(sleep_time)
        
        logger.info("Trading loop stopped")
    
    def stop(self):
        """Stop the trading loop."""
        self._running = False
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics (per-symbol strategy + portfolio)."""
        by_symbol = {
            sym: eng.get_statistics() for sym, eng in self.strategies.items()
        }
        # Aggregate closed-trade stats across symbols
        trade_count = sum(s['trade_count'] for s in by_symbol.values())
        win_count = sum(s['win_count'] for s in by_symbol.values())
        total_pnl = sum(s['total_pnl'] for s in by_symbol.values())
        stats = {
            'strategy': {
                'by_symbol': by_symbol,
                'trade_count': trade_count,
                'win_count': win_count,
                'win_rate': win_count / max(trade_count, 1),
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / max(trade_count, 1),
            },
            'trading': {
                'last_run': self._last_run_time.isoformat() if self._last_run_time else None,
                'daily_trades': self.daily_trade_count,
                'daily_trades_by_symbol': dict(self.daily_trade_count_by_symbol),
                'max_daily_trades': self.max_daily_trades,
                'max_daily_trades_per_symbol': self.max_daily_trades_per_symbol,
                'symbols': self.symbols,
                'is_running': self._running,
                'use_market_orders': self.use_market_orders,
                'enforce_risk': self.enforce_risk,
            },
            'portfolio': {
                'total_equity': self.portfolio.get_total_equity(),
                'total_pnl': (
                    self.portfolio.calculate_realized_pnl()
                    + self.portfolio.calculate_unrealized_pnl()
                ),
                'open_lots': len(self.portfolio.get_open_lots()),
                'archived_lots': len(self.portfolio.get_archived_lots()),
            }
        }
        return stats
    
    def get_recent_trades(self, count: int = 20) -> List[Dict]:
        """Get recent trade logs."""
        return self._trade_log[-count:]


# ============================================================================
# DEMO INTEGRATION
# ============================================================================

def build_demo_engine(
    symbols: Optional[List[str]] = None,
    n_steps_hint: int = 30,
    seed: int = 42,
) -> Tuple[EWMATradingEngine, PortfolioManagerV2, PlaceholderPriceFeed, PlaceholderProbabilityFeed]:
    """
    Wire MockExchange + portfolio + placeholder price/probability feeds.

    Returns (engine, portfolio, price_feed, probability_feed).
    """
    symbols = symbols or ['BTC/USD']
    rng_px = np.random.default_rng(seed)
    rng_p = np.random.default_rng(seed + 1)

    exchanges = {'binance': MockExchange('binance')}
    portfolio = PortfolioManagerV2(exchanges)

    # Seed portfolio cash / inventory from the mock exchange balances
    for exchange_name, exchange in exchanges.items():
        for currency, data in exchange.get_balances().items():
            portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                exchange_name, currency, data['total'], data['locked']
            )

    price_feed = PlaceholderPriceFeed(
        base_prices={s: exchanges['binance'].prices.get(s, 50_000.0) for s in symbols},
        rng=rng_px,
        vol=0.002,
    )
    # Mean above EWMA threshold so the demo actually enters
    probability_feed = PlaceholderProbabilityFeed(mean=0.75, noise=0.10, rng=rng_p)

    config = {
        'symbols': symbols,
        'exchange': 'binance',
        'max_lots': 10,
        'ewma_alpha': 0.2,
        'ewma_threshold': 0.6,
        'take_profit': 0.005,    # 0.5%
        'stop_loss': -0.003,     # -0.3%
        'q_max': 0.01,
        'buy_fee': TradingConfig.BUY_FEE,
        'sell_fee': TradingConfig.SELL_FEE,
        'use_market_orders': True,
        'max_daily_trades': 50,
        'max_daily_trades_per_symbol': 50,
        'enforce_risk': True,
        'allow_placeholders': True,  # demo only — real path must pass feeds
        'get_price_func': price_feed,
        'get_probability_func': probability_feed,
    }

    engine = EWMATradingEngine(portfolio, config)
    _ = n_steps_hint  # reserved for future scripted scenarios
    return engine, portfolio, price_feed, probability_feed


def run_integrated_demo(n_steps: int = 25, symbols: Optional[List[str]] = None):
    """
    Run a self-contained demo with placeholder price & probability feeds.

    Usage:
        python strategy.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    print("\n" + "=" * 70)
    print("  EWMA STRATEGY DEMO  (placeholder price + probability)")
    print("=" * 70)

    engine, portfolio, price_feed, prob_feed = build_demo_engine(symbols=symbols)
    print(f"  Symbols:      {engine.symbols}")
    print(f"  Price feed:   {type(price_feed).__name__}")
    print(f"  Prob feed:    {type(prob_feed).__name__}  "
          f"(mean={prob_feed.mean}, noise={prob_feed.noise})")
    s0 = next(iter(engine.strategies.values()))
    print(f"  EWMA alpha/thr: {s0.alpha} / {s0.threshold}")
    print(f"  Per-symbol engines: {list(engine.strategies.keys())}")
    print(f"  Risk enforce: {engine.enforce_risk}")
    print(f"  Steps:        {n_steps}")

    print("\n  Running trading steps...")
    print(
        f"  {'step':>4}  {'symbol':<10}  {'price':>10}  {'prob':>6}  "
        f"{'ewma':>6}  {'buy':>4}  {'sold':>4}  {'lots':>4}"
    )
    for i in range(n_steps):
        cycle = engine.run_cycle()
        for symbol, data in cycle.items():
            if 'error' in data:
                print(f"  {i+1:4d}  {symbol:<10}  ERROR: {data['error']}")
                continue
            lots = engine._get_lot_count(symbol)
            print(
                f"  {i+1:4d}  {symbol:<10}  "
                f"${data.get('price', 0):>9.2f}  "
                f"{data.get('probability', 0):>6.3f}  "
                f"{data.get('ewma', 0):>6.3f}  "
                f"{'Y' if data.get('bought') else '-':>4}  "
                f"{data.get('sold', 0):>4}  "
                f"{lots:>4}"
            )

    print("\n  Strategy statistics:")
    stats = engine.get_stats()
    print(f"    Trade count:  {stats['strategy']['trade_count']}")
    print(f"    Win rate:     {stats['strategy']['win_rate']:.1%}")
    print(f"    Total PnL:    ${stats['strategy']['total_pnl']:.2f}")
    for sym, st in stats['strategy']['by_symbol'].items():
        print(
            f"    [{sym}] ewma={st['current_ewma']:.4f}  "
            f"trades={st['trade_count']}  pnl=${st['total_pnl']:.2f}"
        )
    print(f"    Open lots:    {stats['portfolio']['open_lots']}")
    print(f"    Equity:       ${stats['portfolio']['total_equity']:.2f}")

    print("\n  Inventory:")
    positions = portfolio.get_all_positions()
    if not positions:
        print("    (no open positions)")
    for pos in positions:
        print(
            f"    {pos.symbol}: qty={pos.total_quantity:.4f}, "
            f"avg_cost=${pos.average_cost:.2f}, mark=${pos.last_price:.2f}, "
            f"uPnL=${pos.unrealized_pnl:.2f}"
        )

    print("\n  Recent trades:")
    for t in engine.get_recent_trades(10):
        print(
            f"    {t['type']:<10} {t.get('symbol')} qty={t.get('quantity', 0):.4f} "
            f"@ {t.get('exit_price', 0):.2f}  pnl={t.get('pnl', 0):.4f}  "
            f"({t.get('reason', '')})"
        )

    try:
        report_path = portfolio.reporting.generate_lot_report()
        print(f"\n  Lot report written: {report_path}")
    except Exception as e:
        print(f"\n  [warn] Lot report skipped: {e}")

    print("\n  [OK] Demo complete.")
    return engine, portfolio


def make_model_probability_feed(
    model_fn: Optional[Callable[[str], float]] = None,
) -> Callable[[str], float]:
    """
    Adapter for a trained model probability head.

    ``model_fn(symbol) -> float in [0, 1]`` should run your checkpoint
    (e.g. yield-mode chance after denormalize / sigmoid). If omitted, raises
    so the real path cannot silently fall back to random placeholders.
    """
    if model_fn is None:
        def _missing(symbol: str) -> float:
            raise RuntimeError(
                f"No model probability wired for {symbol}. "
                f"Pass model_fn=... from INFER / your predictor."
            )
        return _missing

    def _feed(symbol: str) -> float:
        p = float(model_fn(normalize_symbol(symbol)))
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"model_fn({symbol}) must return [0,1], got {p}")
        return p

    return _feed


def make_infer_probability_feed(
    predict_fn: Optional[Callable[[str], float]] = None,
    *,
    chance_to_prob: str = 'clip01',
) -> Callable[[str], float]:
    """
    Wire a trained yield/trend head into the strategy.

    ``predict_fn(symbol)`` should return a raw model score for that symbol
    (e.g. denormalized ``chance`` from the yield label head, or a sigmoid
    probability). Mapping:

      clip01  — clip raw score to [0, 1] (chance already a mass / score)
      sigmoid — 1 / (1 + exp(-x)) for unbounded logits

    Example (after you load a checkpoint in TRAIN/INFER style)::

        def predict(symbol: str) -> float:
            # run model on latest window for symbol; return denormalized chance
            return float(chance)

        feed = make_infer_probability_feed(predict, chance_to_prob='clip01')
        engine = EWMATradingEngine(portfolio, {
            'get_probability_func': feed,
            'get_price_func': live_ticker,
            ...
        })
    """
    if predict_fn is None:
        return make_model_probability_feed(None)

    def _map(x: float) -> float:
        if chance_to_prob == 'sigmoid':
            return float(1.0 / (1.0 + np.exp(-float(x))))
        # default: treat as already a probability-like score
        return float(np.clip(x, 0.0, 1.0))

    def _feed(symbol: str) -> float:
        raw = float(predict_fn(normalize_symbol(symbol)))
        p = _map(raw)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"mapped probability out of range: {p}")
        return p

    return _feed


def real_usage_example():
    """
    Sketch of a real deployment: same engine, live ticker + model probability.

    Fail-closed: omit allow_placeholders; both feeds are required.
    """
    def get_price(symbol: str) -> float:
        # TODO: exchange.get_ticker(normalize_symbol(symbol))['last']
        raise NotImplementedError("Wire a live ticker here")

    # TODO: load checkpoint once, then:
    # def predict(symbol): return denormalized_chance_from_infer(symbol)
    get_prob = make_infer_probability_feed(predict_fn=None)

    config = {
        'symbols': ['BTCUSDT', 'ETHUSDT'],  # normalized to BTC/USD, ETH/USD
        'exchange': 'binance',
        'max_lots': 10,
        'ewma_alpha': 0.2,
        'ewma_threshold': 0.6,
        'take_profit': 0.005,
        'stop_loss': -0.003,
        'q_max': 0.01,
        'buy_fee': TradingConfig.BUY_FEE,
        'sell_fee': TradingConfig.SELL_FEE,
        'get_price_func': get_price,
        'get_probability_func': get_prob,
        'use_market_orders': True,
        'max_daily_trades': 50,
        'max_daily_trades_per_symbol': 20,
        'enforce_risk': True,
        # allow_placeholders deliberately omitted → fail closed without feeds
    }
    _ = config
    # portfolio = PortfolioManagerV2({'binance': live_adapter})
    # engine = EWMATradingEngine(portfolio, config)
    # engine.run_loop(interval_seconds=60.0)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_integrated_demo()