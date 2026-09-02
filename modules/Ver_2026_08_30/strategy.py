"""
SMA Signal-Based Trading Strategy — Adaptive Percentile Threshold

Every minute (per symbol):
  1. Get model prediction p_t  (P(+0.5% within model horizon, e.g. 60 min))
  2. Maintain rolling SMA window (default 10 minutes of predictions)
  3. Append SMA to history; set threshold = percentile(history) (default 80th)
  4. BUY if SMA > threshold (quantity scales with signal strength)
  5. SELL each open lot on gross TP (+0.5%) or SL (-0.3%) — no time exit

Integrated with lot-based PortfolioManagerV2.

Run a self-contained demo::

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
        shock = float(self.rng.normal(0.0, self.vol))
        px = max(1e-8, px * (1.0 + shock))
        self.prices[symbol] = px
        return px

    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = float(price)


class PlaceholderProbabilityFeed:
    """
    Synthetic P(favorable move) in [0, 1].

    Mean is biased high so the adaptive-threshold demo eventually buys.
    Replace with model inference via ``get_probability_func=...``.
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
        p = self.mean + float(self.rng.normal(0.0, self.noise))
        return float(np.clip(p, 0.0, 1.0))


# ============================================================================
# CONFIGURATION
# ============================================================================

class TradingConfig:
    """Trading parameters for the SMA + adaptive-threshold strategy."""

    # Model / signal
    # Model window is conceptual (e.g. 60 min horizon) — provided by the model feed
    SMA_WINDOW = 10                 # minutes of predictions in rolling SMA
    PERCENTILE = 0.80               # adaptive threshold = this quantile of SMA history
    THRESHOLD_MIN_HISTORY = 10      # need at least this many SMAs before adapting
    THRESHOLD_UPDATE_FREQUENCY = 10  # recompute percentile every N steps
    HISTORY_LIMIT = 10_000          # cap on sma_history length
    INITIAL_THRESHOLD = 0.50        # fallback until history is ready

    # TP/SL as *gross* fractional price moves (fees applied on settle only)
    TAKE_PROFIT = 0.005             # +0.5%
    STOP_LOSS = -0.003              # -0.3%

    # Position sizing
    Q_MAX = 0.01                    # max quantity per lot

    # Fees
    BUY_FEE = 0.001                 # 0.1% entry
    SELL_FEE = 0.001                # 0.1% exit
    TOTAL_FEES = BUY_FEE + SELL_FEE

    # Execution
    USE_MARKET_ORDERS = True

    # Risk limits
    MAX_INVENTORY_LOTS = 10
    MAX_DAILY_TRADES = 50


# ============================================================================
# PER-SYMBOL SIGNAL ENGINE (SMA + adaptive threshold)
# ============================================================================

class SMAStrategyEngine:
    """
    Per-symbol signal state:

      predictions → rolling SMA → sma_history → percentile threshold
      buy size scales with (SMA - θ) / (1 - θ)
      sell on gross TP / SL only
    """

    def __init__(self, config: Dict = None, symbol: str = ''):
        self.config = config or {}
        self.symbol = symbol

        self.sma_window = int(self.config.get('sma_window', TradingConfig.SMA_WINDOW))
        self.percentile = float(self.config.get('percentile', TradingConfig.PERCENTILE))
        self.threshold_min_history = int(
            self.config.get('threshold_min_history', TradingConfig.THRESHOLD_MIN_HISTORY)
        )
        self.threshold_update_frequency = int(
            self.config.get(
                'threshold_update_frequency',
                TradingConfig.THRESHOLD_UPDATE_FREQUENCY,
            )
        )
        self.history_limit = int(
            self.config.get('history_limit', TradingConfig.HISTORY_LIMIT)
        )
        self.threshold = float(
            self.config.get('initial_threshold', TradingConfig.INITIAL_THRESHOLD)
        )

        if self.sma_window < 1:
            raise ValueError(f"sma_window must be >= 1; got {self.sma_window}")
        if not (0.0 < self.percentile < 1.0):
            raise ValueError(
                f"percentile must be in (0, 1); got {self.percentile}"
            )
        if not (0.0 <= self.threshold < 1.0):
            raise ValueError(
                f"initial_threshold must be in [0, 1); got {self.threshold}"
            )

        self.take_profit = float(
            self.config.get('take_profit', TradingConfig.TAKE_PROFIT)
        )
        self.stop_loss = float(self.config.get('stop_loss', TradingConfig.STOP_LOSS))
        self.q_max = float(self.config.get('q_max', TradingConfig.Q_MAX))

        self.buy_fee = float(self.config.get('buy_fee', TradingConfig.BUY_FEE))
        self.sell_fee = float(self.config.get('sell_fee', TradingConfig.SELL_FEE))
        self.total_fees = float(
            self.config.get('total_fees', self.buy_fee + self.sell_fee)
        )

        # Rolling window of model predictions and SMA history
        self.predictions: List[float] = []
        self.sma_history: List[float] = []
        self.sma: float = 0.0
        self.step_count: int = 0

        # Performance
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0
        self.signal_history: List[Dict] = []

        logger.info(
            f"SMAStrategyEngine[{symbol or '*'}] initialized: "
            f"sma_window={self.sma_window}, percentile={self.percentile:.0%}, "
            f"thr0={self.threshold:.3f}, TP={self.take_profit:.2%}, SL={self.stop_loss:.2%}"
        )

    # --- signal path ---

    def update_signal(self, probability: float) -> float:
        """
        Append p_t, recompute SMA, extend history, maybe refresh threshold.

        Returns current SMA.
        """
        p = float(probability)
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"probability must be in [0, 1]; got {p}")

        self.step_count += 1
        self.predictions.append(p)
        if len(self.predictions) > self.sma_window:
            self.predictions.pop(0)

        self.sma = self.calculate_sma()
        self.sma_history.append(self.sma)
        if len(self.sma_history) > self.history_limit:
            self.sma_history = self.sma_history[-self.history_limit :]

        if (
            self.threshold_update_frequency > 0
            and self.step_count % self.threshold_update_frequency == 0
        ):
            self.update_threshold()

        return self.sma

    def calculate_sma(self) -> float:
        if not self.predictions:
            return 0.0
        return float(sum(self.predictions) / len(self.predictions))

    def update_threshold(self) -> float:
        """Set threshold to configured percentile of SMA history."""
        if len(self.sma_history) < self.threshold_min_history:
            return self.threshold
        sorted_hist = sorted(self.sma_history)
        # nearest-rank style index in [0, n-1]
        idx = int(self.percentile * len(sorted_hist))
        idx = min(max(idx, 0), len(sorted_hist) - 1)
        self.threshold = float(sorted_hist[idx])
        # keep quantity formula safe
        if self.threshold >= 1.0:
            self.threshold = 0.999
        return self.threshold

    def calculate_quantity(self, sma: Optional[float] = None) -> float:
        """
        Q = Q_max × (SMA − θ) / (1 − θ), clipped to Q_max.
        """
        s = self.sma if sma is None else float(sma)
        if self.threshold >= 1.0:
            return 0.0
        if s <= self.threshold:
            return 0.0
        q = self.q_max * (s - self.threshold) / (1.0 - self.threshold)
        return float(min(q, self.q_max))

    def should_buy(
        self, current_inventory_count: int, max_lots: int = 10
    ) -> Tuple[bool, float]:
        if current_inventory_count >= max_lots:
            return False, 0.0
        if self.sma <= self.threshold:
            return False, 0.0
        quantity = self.calculate_quantity(self.sma)
        return quantity > 0, quantity

    def should_sell_lot(self, lot, current_price: float) -> Tuple[bool, str]:
        """
        Gross TP/SL on price only (fees settle on close, not in the trigger).

        Returns: (should_sell, reason 'TP'|'SL'|'')
        """
        if lot.entry_price <= 0:
            return False, ''
        gross_pl = (float(current_price) - lot.entry_price) / lot.entry_price
        if gross_pl >= self.take_profit:
            return True, 'TP'
        if gross_pl <= self.stop_loss:
            return True, 'SL'
        return False, ''

    def tp_price(self, buy_price: float) -> float:
        return float(buy_price) * (1.0 + self.take_profit)

    def sl_price(self, buy_price: float) -> float:
        return float(buy_price) * (1.0 + self.stop_loss)

    def net_return(self, buy_price: float, sell_price: float) -> float:
        """Net fractional return after round-trip fees (analytics only)."""
        if buy_price <= 0:
            return 0.0
        gross = (sell_price - buy_price) / buy_price
        return gross - self.buy_fee - self.sell_fee

    def calculate_net_pnl(self, buy_price: float, sell_price: float) -> float:
        return self.net_return(buy_price, sell_price)

    def entry_fee(self, price: float, quantity: float) -> float:
        return float(price) * float(quantity) * self.buy_fee

    def exit_fee(self, price: float, quantity: float) -> float:
        return float(price) * float(quantity) * self.sell_fee

    def record_trade(self, net_pnl: float):
        self.trade_count += 1
        self.total_pnl += net_pnl
        if net_pnl > 0:
            self.win_count += 1

    def get_statistics(self) -> Dict:
        return {
            'symbol': self.symbol,
            'current_sma': self.sma,
            'current_threshold': self.threshold,
            'sma_history_count': len(self.sma_history),
            'prediction_window': len(self.predictions),
            'step_count': self.step_count,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'win_rate': self.win_count / max(self.trade_count, 1),
            'total_pnl': self.total_pnl,
            'avg_pnl': self.total_pnl / max(self.trade_count, 1),
            # legacy aliases
            'current_ewma': self.sma,
            'ewma_history_length': len(self.sma_history),
        }

    def reset(self):
        self.predictions = []
        self.sma_history = []
        self.sma = 0.0
        self.step_count = 0
        self.threshold = float(
            self.config.get('initial_threshold', TradingConfig.INITIAL_THRESHOLD)
        )
        self.signal_history = []
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0

    # --- legacy EWMA-shaped aliases (tests / old call sites) ---

    @property
    def ewma(self) -> float:
        return self.sma

    def update_ewma(self, probability: float) -> float:
        return self.update_signal(probability)


# Back-compat name
EWMAStrategyEngine = SMAStrategyEngine


# ============================================================================
# INTEGRATED TRADING ENGINE
# ============================================================================

class SMATradingEngine:
    """
    SMA + adaptive-threshold engine wired to PortfolioManagerV2.

    One ``SMAStrategyEngine`` per symbol (no cross-symbol signal bleed).
    """

    def __init__(self, portfolio: 'PortfolioManagerV2', config: Dict = None):
        """
        Args:
            portfolio: PortfolioManagerV2 instance
            config: Trading parameters.

                Required for non-demo use (fail-closed):
                  get_price_func(symbol) -> float
                  get_probability_func(symbol) -> float in [0, 1]

                Set allow_placeholders=True for demo feeds only.
        """
        self.portfolio = portfolio
        self.config = config or {}

        self._running = False
        self._lock = threading.RLock()
        self._last_run_time: Optional[datetime] = None
        self._trade_log: List[Dict] = []

        raw_symbols = list(self.config.get('symbols', ['BTC/USD']))
        self.symbols = [normalize_symbol(s) for s in raw_symbols]
        self.exchange = self.config.get('exchange', 'binance')

        self.strategies: Dict[str, SMAStrategyEngine] = {
            sym: SMAStrategyEngine(self.config, symbol=sym) for sym in self.symbols
        }

        self.max_lots = int(
            self.config.get('max_lots', TradingConfig.MAX_INVENTORY_LOTS)
        )
        self.max_daily_trades = int(
            self.config.get('max_daily_trades', TradingConfig.MAX_DAILY_TRADES)
        )
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

        logger.info(
            f"SMATradingEngine initialized for {self.symbols} on {self.exchange}"
        )

    def _strategy_for(self, symbol: str) -> SMAStrategyEngine:
        symbol = normalize_symbol(symbol)
        if symbol not in self.strategies:
            self.strategies[symbol] = SMAStrategyEngine(self.config, symbol=symbol)
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            self.daily_trade_count_by_symbol.setdefault(symbol, 0)
        return self.strategies[symbol]

    @property
    def strategy(self) -> SMAStrategyEngine:
        if len(self.strategies) != 1:
            raise AttributeError(
                "self.strategy is ambiguous with multiple symbols; "
                "use engine.strategies[symbol] or engine._strategy_for(symbol)"
            )
        return next(iter(self.strategies.values()))

    def _get_lot_count(self, symbol: str) -> int:
        symbol = normalize_symbol(symbol)
        position = self.portfolio.get_position(self.exchange, symbol)
        if not position:
            return 0
        return len([l for l in position.lots if l.is_active])

    def _reset_daily_counter(self):
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
        if (
            self.daily_trade_count_by_symbol.get(symbol, 0)
            >= self.max_daily_trades_per_symbol
        ):
            return False
        return True

    def _note_trade(self, symbol: str) -> None:
        symbol = normalize_symbol(symbol)
        self.daily_trade_count += 1
        self.daily_trade_count_by_symbol[symbol] = (
            self.daily_trade_count_by_symbol.get(symbol, 0) + 1
        )

    def _log_trade(
        self,
        trade_type: str,
        lot,
        price: float,
        pnl: float,
        reason: str = '',
        symbol: str = '',
        quantity: float = 0.0,
        sma: float = 0.0,
        threshold: float = 0.0,
    ):
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
            'sma': sma,
            'threshold': threshold,
            'ewma': sma,  # legacy field name
        })

    def _mark_price(self, symbol: str, price: float) -> None:
        self.portfolio.update_mark(symbol, float(price), exchange=self.exchange)
        ex = self.portfolio.exchanges.get(self.exchange)
        if ex is not None and hasattr(ex, 'update_price'):
            ex.update_price(symbol, float(price))

    def step(self, symbol: str) -> Dict:
        """
        One trading step (one minute) for a symbol.

        1. p_t from model
        2. rolling SMA + adaptive θ
        3. sell lots on gross TP/SL
        4. buy if SMA > θ and no sells this minute
        """
        with self._lock:
            self._reset_daily_counter()
            symbol = normalize_symbol(symbol)
            strat = self._strategy_for(symbol)

            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'actions': [],
                'sma': strat.sma,
                'threshold': strat.threshold,
                'probability': 0.0,
                'price': 0.0,
                'bought': False,
                'buy_quantity': 0.0,
                'sold': 0,
                'lot_count': 0,
                'threshold_updated': False,
                # legacy keys
                'ewma': strat.sma,
            }

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

            # --- Signal: rolling SMA + adaptive threshold ---
            prev_step = strat.step_count
            sma = strat.update_signal(probability)
            result['sma'] = sma
            result['ewma'] = sma
            result['threshold'] = strat.threshold
            result['threshold_updated'] = (
                strat.threshold_update_frequency > 0
                and strat.step_count % strat.threshold_update_frequency == 0
                and strat.step_count > prev_step
            )
            result['sma_history_count'] = len(strat.sma_history)

            # --- SELL PHASE (gross TP/SL per lot) ---
            position = self.portfolio.get_position(self.exchange, symbol)
            lots_to_sell: List[Tuple[Any, str]] = []

            if position:
                for lot in list(position.lots):
                    if not lot.is_active:
                        continue
                    should_sell, reason = strat.should_sell_lot(lot, current_price)
                    if should_sell:
                        lots_to_sell.append((lot, reason))

            sold_count = 0
            for lot, reason in lots_to_sell:
                try:
                    qty = float(lot.remaining_quantity)
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
                                sma=sma, threshold=strat.threshold,
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
                            'SELL_LIMIT', lot, current_price, 0, reason,
                            sma=sma, threshold=strat.threshold,
                        )
                        result['actions'].append(
                            {
                                'sell_limit': order_id,
                                'reason': reason,
                                'lot_id': lot.lot_id,
                            }
                        )
                except Exception as e:
                    logger.error(f"Error selling lot {lot.lot_id}: {e}")
                    self.portfolio.alert_system.add_alert(
                        AlertSeverity.ERROR, 'Trade Execution',
                        'Sell Failed', f'Error selling {symbol}: {e}',
                    )

            result['sold'] = sold_count

            # --- BUY PHASE (skip if any sell this minute) ---
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
                                result['actions'].append({'buy_blocked': reason})
                                logger.info(f"Buy blocked {symbol}: {reason}")
                                should_buy = False

                        if should_buy and self.use_market_orders:
                            lot_id = self.portfolio.add_lot(
                                self.exchange, symbol,
                                quantity, current_price,
                                tags=['sma_signal'],
                                fee=entry_fee,
                                strategy_metadata={
                                    'sma': sma,
                                    'threshold': strat.threshold,
                                    'probability': probability,
                                    'tp_price': strat.tp_price(current_price),
                                    'sl_price': strat.sl_price(current_price),
                                    'entry_fee': entry_fee,
                                    'buy_fee_rate': strat.buy_fee,
                                    'timestamp': datetime.now().isoformat(),
                                },
                            )
                            self._mark_price(symbol, current_price)
                            self._log_trade(
                                'BUY',
                                self.portfolio.get_lot(lot_id),
                                current_price, 0.0, 'SMA Signal',
                                symbol=symbol, quantity=quantity,
                                sma=sma, threshold=strat.threshold,
                            )
                            result['bought'] = True
                            result['buy_quantity'] = quantity
                            result['entry_fee'] = entry_fee
                            self._note_trade(symbol)
                            logger.info(
                                f"Bought {quantity:.4f} {symbol} at "
                                f"${current_price:.2f} (SMA={sma:.3f}, "
                                f"thr={strat.threshold:.3f}, "
                                f"entry_fee=${entry_fee:.4f})"
                            )
                        elif should_buy and not self.use_market_orders:
                            ok, reason = (
                                self.portfolio.risk_manager.check_buy(
                                    self.exchange, symbol, quantity, current_price,
                                    fee=entry_fee,
                                )
                                if self.enforce_risk
                                else (True, None)
                            )
                            if not ok:
                                result['actions'].append({'buy_blocked': reason})
                            else:
                                order_id = self.portfolio.submit_order(
                                    self.exchange, symbol,
                                    OrderSide.BUY, OrderType.LIMIT,
                                    quantity, current_price,
                                    time_in_force=OrderTimeInForce.GTC,
                                )
                                result['bought'] = True
                                result['buy_order'] = order_id
                                result['buy_quantity'] = quantity
                    except Exception as e:
                        logger.error(f"Error buying {symbol}: {e}")
                        self.portfolio.alert_system.add_alert(
                            AlertSeverity.ERROR, 'Trade Execution',
                            'Buy Failed', f'Error buying {symbol}: {e}',
                        )

            result['lot_count'] = self._get_lot_count(symbol)

            strat.signal_history.append({
                'timestamp': result['timestamp'],
                'symbol': symbol,
                'sma': sma,
                'threshold': strat.threshold,
                'probability': probability,
                'price': current_price,
                'bought': result['bought'],
                'sold': sold_count,
                'inventory_count': result['lot_count'],
            })

            self._last_run_time = datetime.now()
            return result

    def run_cycle(self) -> Dict:
        results = {}
        for symbol in self.symbols:
            try:
                results[symbol] = self.step(symbol)
            except Exception as e:
                logger.exception(f"Error processing {symbol}")
                results[symbol] = {'error': str(e)}
        return results

    def run_loop(self, interval_seconds: float = 60.0):
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
                                f"SMA: {result.get('sma', 0):.3f}, "
                                f"thr: {result.get('threshold', 0):.3f}"
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
        self._running = False

    def get_stats(self) -> Dict:
        by_symbol = {
            sym: eng.get_statistics() for sym, eng in self.strategies.items()
        }
        trade_count = sum(s['trade_count'] for s in by_symbol.values())
        win_count = sum(s['win_count'] for s in by_symbol.values())
        total_pnl = sum(s['total_pnl'] for s in by_symbol.values())
        return {
            'strategy': {
                'by_symbol': by_symbol,
                'trade_count': trade_count,
                'win_count': win_count,
                'win_rate': win_count / max(trade_count, 1),
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / max(trade_count, 1),
            },
            'trading': {
                'last_run': (
                    self._last_run_time.isoformat() if self._last_run_time else None
                ),
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
            },
        }

    def get_recent_trades(self, count: int = 20) -> List[Dict]:
        return self._trade_log[-count:]


# Back-compat name
EWMATradingEngine = SMATradingEngine


# ============================================================================
# DEMO INTEGRATION
# ============================================================================

def build_demo_engine(
    symbols: Optional[List[str]] = None,
    n_steps_hint: int = 30,
    seed: int = 42,
) -> Tuple[SMATradingEngine, PortfolioManagerV2, PlaceholderPriceFeed, PlaceholderProbabilityFeed]:
    """
    Wire MockExchange + portfolio + placeholder price/probability feeds.

    Returns (engine, portfolio, price_feed, probability_feed).
    """
    symbols = symbols or ['BTC/USD']
    rng_px = np.random.default_rng(seed)
    rng_p = np.random.default_rng(seed + 1)

    exchanges = {'binance': MockExchange('binance')}
    portfolio = PortfolioManagerV2(exchanges)

    for exchange_name, exchange in exchanges.items():
        for currency, data in exchange.get_balances().items():
            portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                exchange_name, currency, data['total'], data['locked']
            )

    price_feed = PlaceholderPriceFeed(
        base_prices={
            s: exchanges['binance'].prices.get(s, 50_000.0) for s in symbols
        },
        rng=rng_px,
        vol=0.002,
    )
    # High mean so SMA eventually sits above adaptive threshold
    probability_feed = PlaceholderProbabilityFeed(mean=0.78, noise=0.08, rng=rng_p)

    config = {
        'symbols': symbols,
        'exchange': 'binance',
        'max_lots': 10,
        'sma_window': TradingConfig.SMA_WINDOW,
        'percentile': TradingConfig.PERCENTILE,
        'threshold_min_history': TradingConfig.THRESHOLD_MIN_HISTORY,
        'threshold_update_frequency': TradingConfig.THRESHOLD_UPDATE_FREQUENCY,
        'initial_threshold': TradingConfig.INITIAL_THRESHOLD,
        'take_profit': TradingConfig.TAKE_PROFIT,
        'stop_loss': TradingConfig.STOP_LOSS,
        'q_max': TradingConfig.Q_MAX,
        'buy_fee': TradingConfig.BUY_FEE,
        'sell_fee': TradingConfig.SELL_FEE,
        'use_market_orders': True,
        'max_daily_trades': 50,
        'max_daily_trades_per_symbol': 50,
        'enforce_risk': True,
        'allow_placeholders': True,
        'get_price_func': price_feed,
        'get_probability_func': probability_feed,
    }

    engine = SMATradingEngine(portfolio, config)
    _ = n_steps_hint
    return engine, portfolio, price_feed, probability_feed


def run_integrated_demo(n_steps: int = 40, symbols: Optional[List[str]] = None):
    """
    Self-contained demo with placeholder price & probability feeds.

        python strategy.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    print("\n" + "=" * 70)
    print("  SMA + ADAPTIVE THRESHOLD DEMO  (placeholder price + probability)")
    print("=" * 70)

    engine, portfolio, price_feed, prob_feed = build_demo_engine(symbols=symbols)
    print(f"  Symbols:      {engine.symbols}")
    print(f"  Price feed:   {type(price_feed).__name__}")
    print(
        f"  Prob feed:    {type(prob_feed).__name__}  "
        f"(mean={prob_feed.mean}, noise={prob_feed.noise})"
    )
    s0 = next(iter(engine.strategies.values()))
    print(
        f"  SMA window/pct: {s0.sma_window} / {s0.percentile:.0%}  "
        f"(thr0={s0.threshold:.3f})"
    )
    print(f"  TP/SL (gross): {s0.take_profit:.2%} / {s0.stop_loss:.2%}")
    print(f"  Per-symbol engines: {list(engine.strategies.keys())}")
    print(f"  Risk enforce: {engine.enforce_risk}")
    print(f"  Steps:        {n_steps}")

    print("\n  Running trading steps...")
    print(
        f"  {'step':>4}  {'symbol':<10}  {'price':>10}  {'prob':>6}  "
        f"{'sma':>6}  {'thr':>6}  {'buy':>4}  {'sold':>4}  {'lots':>4}"
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
                f"{data.get('sma', 0):>6.3f}  "
                f"{data.get('threshold', 0):>6.3f}  "
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
            f"    [{sym}] sma={st['current_sma']:.4f}  thr={st['current_threshold']:.4f}  "
            f"hist={st['sma_history_count']}  trades={st['trade_count']}  "
            f"pnl=${st['total_pnl']:.2f}"
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

    ``model_fn(symbol) -> float in [0, 1]`` (e.g. yield-mode chance after denorm).
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

    ``predict_fn(symbol)`` returns a raw model score; mapped via clip01 or sigmoid.

    Example::

        feed = make_infer_probability_feed(predict, chance_to_prob='clip01')
        engine = SMATradingEngine(portfolio, {
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
        return float(np.clip(x, 0.0, 1.0))

    def _feed(symbol: str) -> float:
        raw = float(predict_fn(normalize_symbol(symbol)))
        p = _map(raw)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"mapped probability out of range: {p}")
        return p

    return _feed


def real_usage_example():
    """Sketch: live ticker + model probability, fail-closed feeds."""
    def get_price(symbol: str) -> float:
        raise NotImplementedError("Wire a live ticker here")

    get_prob = make_infer_probability_feed(predict_fn=None)

    config = {
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'exchange': 'binance',
        'max_lots': 10,
        'sma_window': 10,
        'percentile': 0.80,
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
    }
    _ = config
    # portfolio = PortfolioManagerV2({'binance': live_adapter})
    # engine = SMATradingEngine(portfolio, config)
    # engine.run_loop(interval_seconds=60.0)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_integrated_demo()
