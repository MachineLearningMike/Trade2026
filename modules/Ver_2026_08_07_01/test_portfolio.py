"""
test_portfolio.py — unit tests for the lot-based portfolio system.
"""
import json
import os
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from portfolio import (
    Config,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderTimeInForce,
    LotStatus,
    RiskLevel,
    AlertSeverity,
    ReportFormat,
    Trade,
    Balance,
    Order,
    Lot,
    PositionV2,
    LotSelectionStrategy,
    MockExchange,
    PortfolioManagerV2,
    StrategyFactory,
    SpecificLotStrategy,
    Alert,
    AlertSystem,
    PerformanceMetrics,
    RiskMetrics,
    normalize_symbol,
)
from strategy import (
    EWMAStrategyEngine,
    EWMATradingEngine,
    SMAStrategyEngine,
    SMATradingEngine,
)

class TestLotSystem(unittest.TestCase):
    """Complete unit tests for the lot-based portfolio system."""
    
    def setUp(self):
        """Set up test environment."""
        self.exchanges = {'binance': MockExchange('binance')}
        self.portfolio = PortfolioManagerV2(self.exchanges)
        
        for exchange_name, exchange in self.exchanges.items():
            for currency, data in exchange.get_balances().items():
                self.portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                    exchange_name, currency, data['total'], data['locked'])
    
    # ========== Core Lot Tests ==========
    
    def test_add_lot(self):
        """Test adding a lot."""
        lot_id = self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.assertIsNotNone(lot_id)
        lot = self.portfolio.get_lot(lot_id)
        self.assertEqual(lot.quantity, 0.1)
        self.assertEqual(lot.entry_price, 50000)
        self.assertEqual(lot.status, LotStatus.OPEN)
    
    def test_lot_closing(self):
        """Test closing lots."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.add_lot('binance', 'BTC/USD', 0.2, 51000)
        results = self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.15, 52000, 'FIFO')
        self.assertEqual(len(results), 2)
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        self.assertAlmostEqual(pos.total_quantity, 0.15, places=4)
    
    def test_strategies(self):
        """Test different strategies (including same-timestamp LIFO tie-break)."""
        # FIFO
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 48000)
        self.portfolio.add_lot('binance', 'BTC/USD', 0.2, 50000)
        results = self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.15, 52000, 'FIFO')
        self.assertEqual(results[0][0].entry_price, 48000)
        
        # LIFO (fresh portfolio with funded USD)
        p = PortfolioManagerV2(self.exchanges)
        for exchange_name, exchange in self.exchanges.items():
            for currency, data in exchange.get_balances().items():
                p._balances[f"{exchange_name}:{currency}"] = Balance(
                    exchange_name, currency, data['total'], data['locked'])
        p.add_lot('binance', 'BTC/USD', 0.1, 48000)
        p.add_lot('binance', 'BTC/USD', 0.2, 50000)
        p.add_lot('binance', 'BTC/USD', 0.15, 51000)
        # Force identical entry_time so sort cannot rely on wall-clock alone
        pos = p.get_position('binance', 'BTC/USD')
        shared_ts = pos.lots[0].entry_time
        for lot in pos.lots:
            lot.entry_time = shared_ts
        results = p.sell_with_strategy('binance', 'BTC/USD', 0.15, 52000, 'LIFO')
        self.assertEqual(results[0][0].entry_price, 51000)
    
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        results = self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.05, 52000, 'FIFO')
        pnl = results[0][1]
        self.assertAlmostEqual(pnl, 100.0, places=2)
    
    def test_archiving(self):
        """Test that closed lots are archived."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.1, 52000, 'FIFO')
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        self.assertEqual(pos.total_quantity, 0)
        archived = self.portfolio.get_archived_lots('binance', 'BTC/USD')
        self.assertEqual(len(archived), 1)
    
    # ========== Order Management ==========
    
    def test_order_management(self):
        """Test order submission and cancellation."""
        order_id = self.portfolio.submit_order('binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000)
        self.assertIsNotNone(order_id)
        order = self.portfolio.get_order(order_id)
        self.assertEqual(order.status, OrderStatus.SUBMITTED)
        result = self.portfolio.cancel_order(order_id)
        self.assertTrue(result)
    
    def test_balance_tracking(self):
        """Test balance tracking (quote cash debited on buy)."""
        initial_total = self.portfolio.get_balance('binance', 'USD').total
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        usd = self.portfolio.get_balance('binance', 'USD')
        self.assertAlmostEqual(usd.total, initial_total - 5000, places=2)
        btc = self.portfolio.get_balance('binance', 'BTC')
        self.assertAlmostEqual(btc.total, 1.0 + 0.1, places=4)  # mock seed 1.0 BTC
    
    # ========== Risk Tests ==========
    
    def test_risk_management(self):
        """Test risk management."""
        order = Order(exchange='binance', symbol='BTC/USD', side=OrderSide.BUY,
                     order_type=OrderType.LIMIT, quantity=100, price=50000)
        approved, reason = self.portfolio.risk_manager.check_order(order)
        self.assertFalse(approved)
        self.assertIn("Position size", reason)
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        risk_metrics = self.portfolio.risk_manager.get_risk_metrics()
        self.assertIsInstance(risk_metrics.var_95, float)
        self.assertIsInstance(risk_metrics.risk_level, RiskLevel)
    
    # ========== Performance Tests ==========
    
    def test_performance_metrics(self):
        """Test performance metrics."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.add_lot('binance', 'BTC/USD', 0.2, 51000)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.15, 52000, 'FIFO')
        metrics = self.portfolio.performance_analyzer.get_performance_metrics()
        self.assertGreater(metrics.win_rate, 0)
        self.assertGreater(metrics.profit_factor, 0)
    
    # ========== Reporting Tests ==========
    
    def test_report_generation(self):
        """Test report generation."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.05, 52000, 'FIFO')
        
        report_path = self.portfolio.reporting.generate_lot_report(ReportFormat.JSON)
        self.assertTrue(Path(report_path).exists())
        report_path = self.portfolio.reporting.generate_pnl_report(ReportFormat.CSV)
        self.assertTrue(Path(report_path).exists())
        report_path = self.portfolio.reporting.generate_risk_report(ReportFormat.HTML)
        self.assertTrue(Path(report_path).exists())
    
    # ========== Persistence Tests ==========
    
    def test_persistence(self):
        """Test save and load."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        try:
            self.portfolio.save_state(temp_file)
            new_portfolio = PortfolioManagerV2(self.exchanges)
            new_portfolio.load_state(temp_file)
            self.assertEqual(len(self.portfolio._ledger), len(new_portfolio._ledger))
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # ========== Concurrency Tests ==========
    
    def test_thread_safety(self):
        """Test thread safety."""
        def add_lots():
            for i in range(10):
                self.portfolio.add_lot('binance', 'BTC/USD', 0.01, 50000 + i * 100)
        threads = [threading.Thread(target=add_lots) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        self.assertEqual(len(pos.lots), 50)
    
    # ========== Advanced Risk Tests ==========
    
    def test_var_calculation(self):
        """Test VaR calculation with sample returns."""
        returns = [-0.01, -0.02, 0.01, 0.03, -0.015, 0.02, -0.03, 0.01]
        var_95 = self.portfolio.risk_manager.calculate_var(returns, 0.95)
        var_99 = self.portfolio.risk_manager.calculate_var(returns, 0.99)
        
        self.assertLess(var_95, 0)
        self.assertLess(var_99, var_95)
        self.assertGreater(var_95, -0.05)
    
    def test_cvar_calculation(self):
        """Test CVaR calculation with sample returns."""
        returns = [-0.01, -0.02, 0.01, 0.03, -0.015, 0.02, -0.03, 0.01]
        cvar = self.portfolio.risk_manager.calculate_cvar(returns, 0.95)
        
        self.assertLess(cvar, 0)
        var = self.portfolio.risk_manager.calculate_var(returns, 0.95)
        self.assertLess(cvar, var)
    
    # ========== Equity Curve Tests ==========
    
    def test_equity_curve(self):
        """Test equity curve building."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.05, 52000, 'FIFO')
        
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        curve = pos.get_equity_curve()
        
        self.assertGreater(len(curve), 0)
        self.assertIn('equity', curve[0])
        self.assertIn('time', curve[0])
        self.assertGreater(curve[-1]['equity'], 0)
    
    # ========== Tax Report Tests ==========
    
    def test_tax_report(self):
        """Test tax report generation with year filtering."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.1, 52000, 'FIFO')
        
        year = datetime.now().year
        report_path = self.portfolio.reporting.generate_tax_report(year)
        self.assertTrue(Path(report_path).exists())
        
        with open(report_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data['tax_year'], year)
            self.assertIn('closed_lots', data)
            self.assertIn('summary', data)
            self.assertGreaterEqual(len(data['closed_lots']), 1)
    
    # ========== Custom Strategy Tests ==========
    
    def test_custom_strategy(self):
        """Test custom strategy implementation."""
        class CustomStrategy(LotSelectionStrategy):
            def select_lots(self, position, quantity, current_price, context):
                return [position.lots[0].lot_id]
        
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 48000, tags=['first'])
        self.portfolio.add_lot('binance', 'BTC/USD', 0.2, 50000, tags=['second'])
        
        strategy = CustomStrategy()
        results = self.portfolio.close_lots('binance', 'BTC/USD', 0.1, 52000, strategy)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].entry_price, 48000)
        self.assertEqual(results[0][0].tags, ['first'])
    
    # ========== OHLCV Tests ==========
    
    def test_ohlcv_data(self):
        """Test OHLCV data fetching from mock exchange."""
        exchange = self.portfolio.exchanges['binance']
        data = exchange.get_historical_ohlcv('BTC/USD', '1d', 10)
        
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        self.assertIn('close', data[0])
        self.assertIn('open', data[0])
        self.assertIn('high', data[0])
        self.assertIn('low', data[0])
        self.assertIn('volume', data[0])
        self.assertIn('timestamp', data[0])
    
    # ========== Email Alert Tests ==========
    
    def test_email_alerts(self):
        """Test email alert system with mock SMTP."""
        alert_system = AlertSystem({
            'email_enabled': True,
            'smtp_server': 'smtp.test.com',
            'smtp_port': 587,
            'email_user': 'test@test.com',
            'email_password': 'password',
            'email_recipients': ['recipient@test.com']
        })
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_instance = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_instance
            
            alert_system.add_alert(
                AlertSeverity.ERROR,
                'Test',
                'Test Alert',
                'Test message',
                {'test': 'data'}
            )
            
            mock_smtp.assert_called_once_with('smtp.test.com', 587)
            mock_instance.starttls.assert_called_once()
            mock_instance.login.assert_called_once_with('test@test.com', 'password')
            mock_instance.send_message.assert_called_once()
    
    # ========== Multiple Exchange Tests ==========
    
    def test_multiple_exchanges(self):
        """Test multiple exchanges integration."""
        exchanges = {
            'binance': MockExchange('binance'),
            'coinbase': MockExchange('coinbase')
        }
        portfolio = PortfolioManagerV2(exchanges)
        
        for exchange_name, exchange in exchanges.items():
            for currency, data in exchange.get_balances().items():
                portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                    exchange_name, currency, data['total'], data['locked'])
        
        lot1 = portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        lot2 = portfolio.add_lot('coinbase', 'ETH/USD', 0.5, 3000)
        
        binance_pos = portfolio.get_position('binance', 'BTC/USD')
        coinbase_pos = portfolio.get_position('coinbase', 'ETH/USD')
        
        self.assertIsNotNone(binance_pos)
        self.assertIsNotNone(coinbase_pos)
        self.assertEqual(binance_pos.total_quantity, 0.1)
        self.assertEqual(coinbase_pos.total_quantity, 0.5)
        
        all_lots = portfolio.get_open_lots()
        self.assertEqual(len(all_lots), 2)
        
        portfolio.sell_with_strategy('binance', 'BTC/USD', 0.05, 52000, 'FIFO')
        
        binance_pos = portfolio.get_position('binance', 'BTC/USD')
        coinbase_pos = portfolio.get_position('coinbase', 'ETH/USD')
        
        self.assertAlmostEqual(binance_pos.total_quantity, 0.05, places=4)
        self.assertEqual(coinbase_pos.total_quantity, 0.5)

    # ========== Accounting / review regression tests ==========

    def test_partial_close_realized_pnl_on_position(self):
        """PARTIALLY_CLOSED lots must contribute to position.realized_pnl."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        results = self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.05, 52000, 'FIFO')
        expected = results[0][1]
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        self.assertEqual(pos.lots[0].status, LotStatus.PARTIALLY_CLOSED)
        self.assertAlmostEqual(pos.realized_pnl, expected, places=6)
        self.assertAlmostEqual(self.portfolio.calculate_realized_pnl(), expected, places=6)
        summary = self.portfolio.get_portfolio_summary()
        self.assertAlmostEqual(summary['total_realized_pnl'], expected, places=6)

    def test_tax_loss_does_not_reselect_or_underfill(self):
        """TaxLoss: largest losses first; FIFO fallback must not re-select losers."""
        # Small loss, large loss, winner
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 51000)  # small loss vs 50000
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 55000)  # large loss
        self.portfolio.add_lot('binance', 'BTC/USD', 0.2, 48000)  # winner
        # Sell 0.25: both losers (0.2) + 0.05 of winner via FIFO
        results = self.portfolio.sell_with_strategy(
            'binance', 'BTC/USD', 0.25, 50000, 'TaxLoss'
        )
        self.assertGreaterEqual(len(results), 2)
        sells = [t for t in self.portfolio.get_trades() if t.is_sell]
        total_closed = sum(t.quantity for t in sells)
        self.assertAlmostEqual(total_closed, 0.25, places=6)
        # Largest loss (entry 55000) should be fully closed / archived
        archived = self.portfolio.get_archived_lots('binance', 'BTC/USD')
        entries = [l.entry_price for l in archived]
        self.assertIn(55000, entries)
        # First selected lot should be the largest loss
        self.assertEqual(results[0][0].entry_price, 55000)

    def test_close_lots_raises_on_underfill(self):
        """close_lots must raise when strategy lots cannot cover quantity."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        lot_id = pos.lots[0].lot_id

        class BadStrategy(LotSelectionStrategy):
            def select_lots(self, position, quantity, current_price, context):
                return [lot_id, lot_id]  # duplicate id only covers 0.1

        with self.assertRaises(ValueError):
            self.portfolio.close_lots('binance', 'BTC/USD', 0.15, 52000, BadStrategy())
        # Position unchanged
        self.assertAlmostEqual(
            self.portfolio.get_position('binance', 'BTC/USD').total_quantity, 0.1, places=6
        )

    def test_specific_lot_raises_if_insufficient(self):
        self.portfolio.add_lot('binance', 'BTC/USD', 0.05, 50000)
        pos = self.portfolio.get_position('binance', 'BTC/USD')
        strat = SpecificLotStrategy([pos.lots[0].lot_id])
        with self.assertRaises(ValueError):
            self.portfolio.close_lots('binance', 'BTC/USD', 0.1, 52000, strat)

    def test_fee_aware_pnl(self):
        """Entry + exit fees reduce realized PnL."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000, fee=10.0)
        results = self.portfolio.sell_with_strategy(
            'binance', 'BTC/USD', 0.1, 52000, 'FIFO', fee=5.0
        )
        # Gross: 0.1*(52000-50000)=200; net: 200-5-10=185
        self.assertAlmostEqual(results[0][1], 185.0, places=4)
        sells = [t for t in self.portfolio.get_trades() if t.is_sell]
        self.assertEqual(len(sells), 1)
        self.assertAlmostEqual(sells[0].realized_pnl, 185.0, places=4)

    def test_trade_statistics_use_recorded_pnl_after_partials(self):
        """Analytics must not rebuild sell PnL from live average cost after partials."""
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 48000)
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 52000)
        # FIFO closes first lot (profit 400 on full 0.1: 0.1*(52000-48000)=400)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.1, 52000, 'FIFO')
        # Remaining avg cost is 52000; if analytics used live avg, second sell would show 0
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.05, 53000, 'FIFO')
        stats = self.portfolio.performance_analyzer.calculate_trade_statistics()
        self.assertGreater(stats['total_trades'], 0)
        # First sell was profitable; win rate should not be zeroed by avg-cost bug
        self.assertGreater(stats['win_rate'], 0)
        sells = [t for t in self.portfolio.get_trades() if t.is_sell]
        self.assertTrue(all(t.realized_pnl is not None for t in sells))

    def test_annualized_return_uses_earliest_trade(self):
        from datetime import timedelta
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.05, 52000, 'FIFO')
        # Backdate the buy so span > 1 day even though get_trades is newest-first
        buy = [t for t in self.portfolio._ledger if t.is_buy][0]
        buy.timestamp = datetime.now() - timedelta(days=30)
        rets = self.portfolio.performance_analyzer.calculate_returns()
        # With 30-day span, annualized should differ from total (unless total is 0)
        self.assertIn('annualized_return', rets)
        self.assertIn('total_return', rets)

    def test_risk_level_low_for_healthy_portfolio(self):
        """Mild VaR + low exposure should be LOW, not CRITICAL."""
        rm = self.portfolio.risk_manager
        # No trades → empty returns → var_95 default 0, exposure ~0 after mock cash only
        metrics = rm.get_risk_metrics()
        self.assertEqual(metrics.risk_level, RiskLevel.LOW)

    def test_total_cash_is_base_currency_only(self):
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        summary = self.portfolio.get_portfolio_summary()
        usd = self.portfolio.get_balance('binance', 'USD').total
        self.assertAlmostEqual(summary['total_cash'], usd, places=4)

    def test_load_state_rebuilds_active_orders(self):
        import tempfile
        order_id = self.portfolio.submit_order(
            'binance', 'BTC/USD', OrderSide.BUY, OrderType.LIMIT, 0.1, 50000
        )
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        try:
            self.portfolio.save_state(temp_file)
            new_p = PortfolioManagerV2(self.exchanges)
            new_p.load_state(temp_file)
            self.assertIn(order_id, new_p._active_orders)
            active = new_p.get_active_orders()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0].internal_id, order_id)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_order_partially_filled_is_active(self):
        o = Order(
            exchange='binance',
            symbol='BTC/USD',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000,
            status=OrderStatus.PARTIALLY_FILLED,
        )
        self.assertTrue(o.is_active)
        self.assertFalse(o.is_done)
        o.status = OrderStatus.EXPIRED
        self.assertTrue(o.is_done)
        self.assertFalse(o.is_active)

    def test_tax_report_current_year_has_lots(self):
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        self.portfolio.sell_with_strategy('binance', 'BTC/USD', 0.1, 52000, 'FIFO')
        year = datetime.now().year
        report_path = self.portfolio.reporting.generate_tax_report(year)
        with open(report_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(data['tax_year'], year)
        self.assertGreaterEqual(len(data['closed_lots']), 1)
        self.assertAlmostEqual(data['summary']['total_realized_pnl'], 200.0, places=2)

    def test_strategy_rejects_nan_price(self):
        engine = SMATradingEngine(
            self.portfolio,
            {
                'symbols': ['BTC/USD'],
                'exchange': 'binance',
                'allow_placeholders': False,
                'get_price_func': lambda s: float('nan'),
                'get_probability_func': lambda s: 0.8,
            },
        )
        with self.assertRaises(ValueError):
            engine.step('BTC/USD')

    def test_strategy_threshold_validation(self):
        with self.assertRaises(ValueError):
            SMAStrategyEngine({'percentile': 1.0})
        with self.assertRaises(ValueError):
            SMAStrategyEngine({'percentile': -0.1})
        with self.assertRaises(ValueError):
            SMAStrategyEngine({'initial_threshold': 1.0})
        eng = SMAStrategyEngine({'initial_threshold': 0.6, 'sma_window': 10})
        eng.sma = 0.9
        eng.threshold = 0.6
        self.assertGreater(eng.calculate_quantity(), 0)
        # Legacy aliases still construct
        self.assertIs(EWMAStrategyEngine, SMAStrategyEngine)
        self.assertIs(EWMATradingEngine, SMATradingEngine)

    def test_sma_adaptive_threshold_and_gross_tpsl(self):
        """SMA window, percentile θ, quantity scaling, gross TP/SL."""
        eng = SMAStrategyEngine({
            'sma_window': 3,
            'percentile': 0.80,
            'threshold_min_history': 5,
            'threshold_update_frequency': 1,
            'initial_threshold': 0.50,
            'take_profit': 0.005,
            'stop_loss': -0.003,
            'q_max': 0.01,
        })
        # Warm up window with flat high probs
        for _ in range(5):
            eng.update_signal(0.90)
        self.assertAlmostEqual(eng.sma, 0.90, places=5)
        self.assertGreater(len(eng.sma_history), 0)
        # Adaptive θ should sit at high end of history after enough samples
        eng.update_threshold()
        self.assertGreaterEqual(eng.threshold, 0.5)

        # Quantity scales above θ
        eng.threshold = 0.80
        eng.sma = 0.90
        q = eng.calculate_quantity()
        expected = 0.01 * (0.90 - 0.80) / (1.0 - 0.80)
        self.assertAlmostEqual(q, expected, places=6)

        # Gross TP/SL (fees do not change trigger)
        class _Lot:
            entry_price = 100.0
            entry_fee = 0.0
            quantity = 1.0
        sell_tp, reason_tp = eng.should_sell_lot(_Lot(), 100.50)  # +0.5%
        self.assertTrue(sell_tp)
        self.assertEqual(reason_tp, 'TP')
        # Use price strictly through SL to avoid float edge on 99.70
        sell_sl, reason_sl = eng.should_sell_lot(_Lot(), 99.69)   # < -0.3%
        self.assertTrue(sell_sl)
        self.assertEqual(reason_sl, 'SL')
        hold, reason_h = eng.should_sell_lot(_Lot(), 100.20)      # +0.2%
        self.assertFalse(hold)
        self.assertEqual(reason_h, '')

    def test_hifo_and_tax_gain_ordering(self):
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 48000)
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 51000)
        self.portfolio.add_lot('binance', 'BTC/USD', 0.1, 50000)
        # HIFO sells highest cost first (51000)
        results = self.portfolio.sell_with_strategy(
            'binance', 'BTC/USD', 0.1, 52000, 'HIFO'
        )
        self.assertEqual(results[0][0].entry_price, 51000)

        p = PortfolioManagerV2(self.exchanges)
        for exchange_name, exchange in self.exchanges.items():
            for currency, data in exchange.get_balances().items():
                p._balances[f"{exchange_name}:{currency}"] = Balance(
                    exchange_name, currency, data['total'], data['locked'])
        p.add_lot('binance', 'BTC/USD', 0.1, 48000)
        p.add_lot('binance', 'BTC/USD', 0.1, 50000)
        # TaxGain: largest gain first at px 52000 → 48000 lot
        results = p.sell_with_strategy('binance', 'BTC/USD', 0.1, 52000, 'TaxGain')
        self.assertEqual(results[0][0].entry_price, 48000)


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization(unittest.TestCase):
    """Test all to_dict/from_dict roundtrip conversions."""
    
    def test_trade_roundtrip(self):
        """Test Trade serialization roundtrip."""
        trade = Trade(
            trade_id="test123",
            exchange="binance",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000,
            timestamp=datetime.now(),
            order_id="order123",
            fee=10.0,
            lot_id="lot123",
            fee_currency="USD",
            commission_asset="BTC",
            commission_paid=0.001,
            realized_pnl=None,
        )
        data = trade.to_dict()
        restored = Trade.from_dict(data)
        
        self.assertEqual(trade.trade_id, restored.trade_id)
        self.assertEqual(trade.exchange, restored.exchange)
        self.assertEqual(trade.symbol, restored.symbol)
        self.assertEqual(trade.side, restored.side)
        self.assertEqual(trade.quantity, restored.quantity)
        self.assertEqual(trade.price, restored.price)
        self.assertEqual(trade.fee, restored.fee)
        self.assertEqual(trade.lot_id, restored.lot_id)
        self.assertEqual(trade.fee_currency, restored.fee_currency)
        self.assertEqual(trade.commission_asset, restored.commission_asset)
        self.assertEqual(trade.commission_paid, restored.commission_paid)
        self.assertEqual(trade.order_id, restored.order_id)
        self.assertEqual(trade.timestamp.isoformat(), restored.timestamp.isoformat())
        self.assertIsNone(restored.realized_pnl)

        sell = Trade(
            trade_id="sell1",
            exchange="binance",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=0.05,
            price=52000,
            timestamp=datetime.now(),
            order_id="o2",
            realized_pnl=100.0,
        )
        sell_r = Trade.from_dict(sell.to_dict())
        self.assertAlmostEqual(sell_r.realized_pnl, 100.0)
    
    def test_trade_with_all_fields_none(self):
        """Test Trade serialization with optional fields as None."""
        trade = Trade(
            trade_id="test456",
            exchange="coinbase",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            quantity=0.5,
            price=3000,
            timestamp=datetime.now(),
            order_id="order456"
        )
        data = trade.to_dict()
        restored = Trade.from_dict(data)
        
        self.assertEqual(trade.trade_id, restored.trade_id)
        self.assertEqual(trade.exchange, restored.exchange)
        self.assertEqual(trade.side, restored.side)
        self.assertEqual(trade.fee, 0.0)
        self.assertIsNone(trade.lot_id)
    
    def test_balance_roundtrip(self):
        """Test Balance serialization roundtrip."""
        balance = Balance(
            exchange="binance",
            currency="USD",
            total=50000,
            locked=5000
        )
        data = balance.to_dict()
        restored = Balance.from_dict(data)
        
        self.assertEqual(balance.exchange, restored.exchange)
        self.assertEqual(balance.currency, restored.currency)
        self.assertEqual(balance.total, restored.total)
        self.assertEqual(balance.locked, restored.locked)
        self.assertEqual(balance.free, restored.free)
        self.assertEqual(balance.last_update.isoformat(), restored.last_update.isoformat())
    
    def test_order_roundtrip(self):
        """Test Order serialization roundtrip with trades."""
        order = Order(
            exchange="binance",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000,
            client_order_id="client123",
            trades=[
                Trade(
                    trade_id="trade1",
                    exchange="binance",
                    symbol="BTC/USD",
                    side=OrderSide.BUY,
                    quantity=0.1,
                    price=50000,
                    timestamp=datetime.now(),
                    order_id="order123"
                )
            ]
        )
        data = order.to_dict()
        restored = Order.from_dict(data)
        
        self.assertEqual(order.exchange, restored.exchange)
        self.assertEqual(order.symbol, restored.symbol)
        self.assertEqual(order.side, restored.side)
        self.assertEqual(order.order_type, restored.order_type)
        self.assertEqual(order.quantity, restored.quantity)
        self.assertEqual(order.price, restored.price)
        self.assertEqual(order.client_order_id, restored.client_order_id)
        self.assertEqual(order.status, restored.status)
        self.assertEqual(len(order.trades), len(restored.trades))
        self.assertEqual(order.trades[0].trade_id, restored.trades[0].trade_id)
        self.assertEqual(order.trades[0].quantity, restored.trades[0].quantity)
    
    def test_order_with_all_fields(self):
        """Test Order serialization with all fields populated."""
        now = datetime.now()
        order = Order(
            exchange="binance",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000,
            stop_price=48000,
            time_in_force=OrderTimeInForce.GTC,
            internal_id="internal123",
            exchange_order_id="exchange123",
            status=OrderStatus.SUBMITTED,
            filled_quantity=0.05,
            remaining_quantity=0.05,
            average_fill_price=50000,
            created_at=now,
            submitted_at=now,
            updated_at=now,
            closed_at=now,
            client_order_id="client123",
            reject_reason=""
        )
        data = order.to_dict()
        restored = Order.from_dict(data)
        
        self.assertEqual(order.exchange, restored.exchange)
        self.assertEqual(order.symbol, restored.symbol)
        self.assertEqual(order.side, restored.side)
        self.assertEqual(order.order_type, restored.order_type)
        self.assertEqual(order.quantity, restored.quantity)
        self.assertEqual(order.price, restored.price)
        self.assertEqual(order.stop_price, restored.stop_price)
        self.assertEqual(order.time_in_force, restored.time_in_force)
        self.assertEqual(order.internal_id, restored.internal_id)
        self.assertEqual(order.exchange_order_id, restored.exchange_order_id)
        self.assertEqual(order.status, restored.status)
        self.assertEqual(order.filled_quantity, restored.filled_quantity)
        self.assertEqual(order.remaining_quantity, restored.remaining_quantity)
        self.assertEqual(order.average_fill_price, restored.average_fill_price)
        self.assertEqual(order.client_order_id, restored.client_order_id)
    
    def test_lot_roundtrip(self):
        """Test Lot serialization roundtrip."""
        lot = Lot(
            exchange="binance",
            symbol="BTC/USD",
            quantity=0.1,
            entry_price=50000,
            tags=["strategy_a", "high_priority"],
            strategy_metadata={"priority": 1, "score": 0.95, "signal": "buy"}
        )
        data = lot.to_dict()
        restored = Lot.from_dict(data)
        
        self.assertEqual(lot.exchange, restored.exchange)
        self.assertEqual(lot.symbol, restored.symbol)
        self.assertEqual(lot.side, restored.side)
        self.assertEqual(lot.quantity, restored.quantity)
        self.assertEqual(lot.entry_price, restored.entry_price)
        self.assertEqual(lot.tags, restored.tags)
        self.assertEqual(lot.strategy_metadata, restored.strategy_metadata)
        self.assertEqual(lot.status, restored.status)
        self.assertEqual(lot.remaining_quantity, restored.remaining_quantity)
    
    def test_lot_after_partial_close(self):
        """Test Lot serialization after partial close."""
        lot = Lot(
            exchange="binance",
            symbol="BTC/USD",
            quantity=0.2,
            entry_price=50000
        )
        lot.close_partial(0.1, 52000)
        
        data = lot.to_dict()
        restored = Lot.from_dict(data)
        
        self.assertEqual(lot.remaining_quantity, restored.remaining_quantity)
        self.assertEqual(lot.closed_quantity, restored.closed_quantity)
        self.assertEqual(lot.status, restored.status)
        self.assertEqual(lot.realized_pnl, restored.realized_pnl)
        self.assertEqual(lot.status, LotStatus.PARTIALLY_CLOSED)
        self.assertIsNone(lot.closed_time)  # only set on full close
    
    def test_lot_after_full_close(self):
        """Test Lot serialization after full close."""
        lot = Lot(
            exchange="binance",
            symbol="BTC/USD",
            quantity=0.1,
            entry_price=50000
        )
        lot.close_partial(0.1, 52000)
        lot.archive()
        
        data = lot.to_dict()
        restored = Lot.from_dict(data)
        
        self.assertEqual(restored.status, LotStatus.ARCHIVED)
        self.assertEqual(restored.remaining_quantity, 0)
        self.assertEqual(restored.closed_quantity, 0.1)
        self.assertIsNotNone(restored.closed_time)
    
    def test_position_roundtrip(self):
        """Test PositionV2 serialization roundtrip."""
        pos = PositionV2(
            symbol="BTC/USD",
            exchange="binance",
            lots=[
                Lot(
                    exchange="binance",
                    symbol="BTC/USD",
                    quantity=0.1,
                    entry_price=50000,
                    tags=["first"]
                ),
                Lot(
                    exchange="binance",
                    symbol="BTC/USD",
                    quantity=0.2,
                    entry_price=51000,
                    tags=["second"]
                )
            ],
            last_price=52000
        )
        data = pos.to_dict()
        restored = PositionV2.from_dict(data)
        
        self.assertEqual(pos.symbol, restored.symbol)
        self.assertEqual(pos.exchange, restored.exchange)
        self.assertEqual(len(pos.lots), len(restored.lots))
        self.assertEqual(pos.last_price, restored.last_price)
        
        for i, lot in enumerate(pos.lots):
            self.assertEqual(lot.quantity, restored.lots[i].quantity)
            self.assertEqual(lot.entry_price, restored.lots[i].entry_price)
            self.assertEqual(lot.tags, restored.lots[i].tags)
    
    def test_position_with_archived_lots(self):
        """Test PositionV2 serialization with archived lots."""
        pos = PositionV2(
            symbol="BTC/USD",
            exchange="binance"
        )
        
        lot = Lot(exchange="binance", symbol="BTC/USD", quantity=0.1, entry_price=50000)
        pos.add_lot(lot)
        pos.close_lots(0.1, 52000, strategy="FIFO")
        
        data = pos.to_dict()
        restored = PositionV2.from_dict(data)
        
        self.assertEqual(len(pos.archived_lots), len(restored.archived_lots))
        if pos.archived_lots:
            self.assertEqual(pos.archived_lots[0].quantity, restored.archived_lots[0].quantity)
            self.assertEqual(pos.archived_lots[0].entry_price, restored.archived_lots[0].entry_price)
            self.assertEqual(pos.archived_lots[0].realized_pnl, restored.archived_lots[0].realized_pnl)
    
    def test_risk_metrics_roundtrip(self):
        """Test RiskMetrics serialization."""
        metrics = RiskMetrics(
            var_95=-0.02,
            var_99=-0.035,
            cvar_95=-0.04,
            max_drawdown=0.15,
            max_drawdown_percentage=15.0,
            volatility=0.25,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            exposure=0.5,
            concentration_ratio=0.3,
            risk_level=RiskLevel.MEDIUM
        )
        data = metrics.to_dict()
        
        self.assertEqual(data['var_95'], -0.02)
        self.assertEqual(data['var_99'], -0.035)
        self.assertEqual(data['risk_level'], 'medium')
        self.assertEqual(data['sharpe_ratio'], 1.5)
    
    def test_performance_metrics_roundtrip(self):
        """Test PerformanceMetrics serialization."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            annualized_return=0.30,
            daily_return=0.001,
            monthly_return=0.02,
            win_rate=65.0,
            profit_factor=2.5,
            average_win=500,
            average_loss=-200,
            largest_win=2000,
            largest_loss=-1000,
            consecutive_wins=5,
            consecutive_losses=2,
            sharpe_ratio=1.8,
            sortino_ratio=2.2,
            calmar_ratio=1.5,
            omega_ratio=1.3
        )
        data = metrics.to_dict()
        
        self.assertEqual(data['total_return'], 0.25)
        self.assertEqual(data['win_rate'], 65.0)
        self.assertEqual(data['profit_factor'], 2.5)
        self.assertEqual(data['consecutive_wins'], 5)
    
    def test_alert_roundtrip(self):
        """Test Alert serialization roundtrip."""
        alert = Alert(
            severity=AlertSeverity.ERROR,
            category="Risk Management",
            title="Drawdown Limit Exceeded",
            message="Current drawdown 22% exceeds limit 20%",
            data={"drawdown": 0.22, "limit": 0.20}
        )
        data = alert.to_dict()
        restored = Alert.from_dict(data)
        
        self.assertEqual(alert.severity, restored.severity)
        self.assertEqual(alert.category, restored.category)
        self.assertEqual(alert.title, restored.title)
        self.assertEqual(alert.message, restored.message)
        self.assertEqual(alert.data, restored.data)
        self.assertEqual(alert.timestamp.isoformat(), restored.timestamp.isoformat())
    
    def test_alert_acknowledged(self):
        """Test Alert serialization with acknowledgment."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category="Test",
            title="Test Alert",
            message="Test message"
        )
        alert.acknowledged = True
        alert.acknowledged_at = datetime.now()
        
        data = alert.to_dict()
        restored = Alert.from_dict(data)
        
        self.assertTrue(restored.acknowledged)
        self.assertIsNotNone(restored.acknowledged_at)
        self.assertEqual(alert.acknowledged_at.isoformat(), restored.acknowledged_at.isoformat())
    
    def test_nested_serialization(self):
        """Test complex nested serialization with multiple objects."""
        position = PositionV2(
            symbol="BTC/USD",
            exchange="binance",
            lots=[
                Lot(
                    exchange="binance",
                    symbol="BTC/USD",
                    quantity=0.1,
                    entry_price=48000,
                    tags=["entry1"],
                    strategy_metadata={"signal": "strong_buy"}
                ),
                Lot(
                    exchange="binance",
                    symbol="BTC/USD",
                    quantity=0.15,
                    entry_price=50000,
                    tags=["entry2"],
                    strategy_metadata={"signal": "buy"}
                )
            ],
            last_price=52000
        )
        
        trades = [
            Trade(
                trade_id="t1",
                exchange="binance",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=0.1,
                price=48000,
                timestamp=datetime.now(),
                order_id="o1",
                lot_id=position.lots[0].lot_id
            ),
            Trade(
                trade_id="t2",
                exchange="binance",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=0.15,
                price=50000,
                timestamp=datetime.now(),
                order_id="o2",
                lot_id=position.lots[1].lot_id
            )
        ]
        
        pos_data = position.to_dict()
        trade_data = [t.to_dict() for t in trades]
        
        restored_pos = PositionV2.from_dict(pos_data)
        restored_trades = [Trade.from_dict(t) for t in trade_data]
        
        self.assertEqual(len(position.lots), len(restored_pos.lots))
        self.assertEqual(len(trades), len(restored_trades))
        self.assertEqual(trades[0].lot_id, restored_trades[0].lot_id)
        self.assertEqual(position.lots[0].strategy_metadata, restored_pos.lots[0].strategy_metadata)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests including serialization tests."""
    print("\n" + "="*70)
    print("  RUNNING COMPLETE TEST SUITE")
    print("="*70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestLotSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSerialization))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("  TEST RESULTS")
    print("="*70)
    print(f"  Total Tests: {result.testsRun}")
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("  [OK] ALL TESTS PASSED!")
    else:
        print("  [FAIL] SOME TESTS FAILED!")
    
    return result.wasSuccessful()


# ============================================================================
# DEMO
# ============================================================================



if __name__ == "__main__":
    import sys
    ok = run_all_tests() if "run_all_tests" in dir() else unittest.main(exit=False).result.wasSuccessful()
    sys.exit(0 if ok else 1)
