"""
portfolio.py — public facade for the lot-based portfolio system.

Modules:
  portfolio_core.py       domain model, strategies, exchange, manager
  portfolio_risk.py       risk manager
  portfolio_reporting.py  analytics, alerts, reporting
  test_portfolio.py       unit tests (python test_portfolio.py / portfolio.py --test)
"""
from portfolio_core import (  # noqa: F401
    Config,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderTimeInForce,
    RiskLevel,
    AlertSeverity,
    LotStatus,
    ReportFormat,
    to_serializable,
    normalize_symbol,
    split_symbol,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    ExchangeInterface,
    Trade,
    Balance,
    Order,
    Lot,
    PositionV2,
    LotSelectionStrategy,
    FIFOStrategy,
    LIFOStrategy,
    TaxOptimizedStrategy,
    LossHarvestingStrategy,
    GainHarvestingStrategy,
    SpecificLotStrategy,
    StrategyFactory,
    MockExchange,
    PortfolioManagerV2,
)
from portfolio_risk import RiskMetrics, RiskManagerV2  # noqa: F401
from portfolio_reporting import (  # noqa: F401
    PerformanceMetrics,
    PerformanceAnalyzerV2,
    Alert,
    AlertSystem,
    ReportingModuleV2,
)


def run_all_tests():
    """Lazy-import tests so production `import portfolio` does not load unittest."""
    import unittest
    from test_portfolio import TestLotSystem, TestSerialization

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLotSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSerialization))
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def run_demo():
    """Run a comprehensive demo."""
    print("\n" + "="*70)
    print("  LOT-BASED PORTFOLIO SYSTEM - COMPLETE DEMO")
    print("="*70)
    
    exchanges = {'binance': MockExchange('binance')}
    portfolio = PortfolioManagerV2(exchanges)
    
    for exchange_name, exchange in exchanges.items():
        for currency, data in exchange.get_balances().items():
            portfolio._balances[f"{exchange_name}:{currency}"] = Balance(
                exchange_name, currency, data['total'], data['locked'])
    
    print("\n  Adding lots...")
    lots = [(0.1, 48000, "Strategy A"), (0.15, 49000, "Strategy A"),
            (0.2, 50000, "Strategy B"), (0.08, 51000, "Strategy C")]
    
    for qty, price, tag in lots:
        portfolio.add_lot('binance', 'BTC/USD', qty, price, tags=[tag])
        print(f"    Added: {qty:.2f} BTC @ ${price:.0f} (tag: {tag})")
    
    portfolio.update_mark('BTC/USD', 52000, exchange='binance')
    
    print("\n  Current Position:")
    pos = portfolio.get_position('binance', 'BTC/USD')
    print(f"    Total BTC: {pos.total_quantity:.4f}")
    print(f"    Unrealized PnL: ${pos.unrealized_pnl:.2f}")
    
    print("\n  Testing strategies:")
    for strategy in ["FIFO", "LIFO", "HIFO", "TaxLoss", "TaxGain"]:
        try:
            p = PortfolioManagerV2(exchanges)
            for qty, price, tag in lots:
                p.add_lot('binance', 'BTC/USD', qty, price, tags=[tag])
            p.update_mark('BTC/USD', 52000, exchange='binance')
            results = p.sell_with_strategy('binance', 'BTC/USD', 0.25, 52000, strategy)
            print(f"    {strategy}: {len(results)} lots sold")
        except Exception as e:
            print(f"    {strategy}: Error - {e}")
    
    print("\n  Generating reports...")
    tax_year = __import__('datetime').datetime.now().year
    reports = [
        ("Lot Report", portfolio.reporting.generate_lot_report),
        ("PnL Report", portfolio.reporting.generate_pnl_report),
        ("Risk Report", portfolio.reporting.generate_risk_report),
        ("Performance Report", portfolio.reporting.generate_performance_report),
        ("Tax Report", lambda: portfolio.reporting.generate_tax_report(tax_year)),
    ]
    
    for name, func in reports:
        try:
            filepath = func()
            print(f"    {name}: {filepath}")
        except Exception as e:
            print(f"    {name}: ERROR - {e}")
    
    print("\n  Portfolio Summary:")
    summary = portfolio.get_portfolio_summary()
    print(f"    Total Equity: ${summary['total_equity']:.2f}")
    print(f"    Total PnL: ${summary['total_pnl']:.2f}")
    print(f"    Open Lots: {summary['lot_statistics']['open_lots']}")
    print(f"    Archived Lots: {summary['lot_statistics']['archived_lots']}")
    print(f"    Win Rate: {summary['lot_statistics']['win_rate']:.1%}")
    
    print("\n  [OK] Demo complete!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        run_demo()
