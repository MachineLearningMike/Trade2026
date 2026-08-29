"""
portfolio_reporting.py — performance analytics, alerts, reporting.
"""
from __future__ import annotations

import logging
import os
import csv
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from pathlib import Path
from collections import defaultdict
import uuid
import numpy as np

from portfolio_core import (
    Config,
    ReportFormat,
    AlertSeverity,
    OrderSide,
    LotStatus,
    RiskLevel,
    to_serializable,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    normalize_symbol,
)

if TYPE_CHECKING:
    from portfolio_core import PortfolioManagerV2
    from portfolio_risk import RiskManagerV2

logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> Dict:
        return to_serializable(asdict(self))


class PerformanceAnalyzerV2:
    """Performance analyzer with lot-level metrics."""
    
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
        }
    
    def _sell_trade_pnl(self, trade) -> Optional[float]:
        """Prefer ledger-recorded realized PnL; avoid live average-cost rebuild."""
        if not trade.is_sell:
            return None
        if trade.realized_pnl is not None:
            return float(trade.realized_pnl)
        position = self.portfolio.get_position(trade.exchange, trade.symbol)
        if position and position.average_cost > 0:
            return trade.quantity * (trade.price - position.average_cost) - trade.fee
        return None

    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate trade statistics from ledger."""
        trades = self.portfolio.get_trades()
        if len(trades) < 2:
            return {}
        
        trade_pnls = []
        for trade in trades:
            pnl = self._sell_trade_pnl(trade)
            if pnl is not None:
                trade_pnls.append(pnl)
        
        if not trade_pnls:
            return {}
        
        winning = [p for p in trade_pnls if p > 0]
        losing = [p for p in trade_pnls if p < 0]
        
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
        
        # get_trades() is newest-first; span from earliest trade
        first_ts = min(t.timestamp for t in trades)
        days = max(1, (datetime.now() - first_ts).days)
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else total_return
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'daily_return': total_return / max(days, 1),
            'monthly_return': total_return / max(days / 30, 1)
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
        
        daily_returns = self._calculate_daily_returns()
        if daily_returns:
            metrics.sharpe_ratio = calculate_sharpe_ratio(daily_returns)
            metrics.sortino_ratio = calculate_sortino_ratio(daily_returns)
            metrics.omega_ratio = calculate_omega_ratio(daily_returns)
            max_dd = self.calculate_max_drawdown()
            metrics.calmar_ratio = calculate_calmar_ratio(daily_returns, max_dd)
        
        return metrics
    
    def _get_initial_equity(self) -> float:
        """Get initial equity."""
        if self._initial_equity is not None:
            return self._initial_equity
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
            pnl = self._sell_trade_pnl(trade)
            if pnl is not None:
                equity += pnl
                curve.append(equity)
        
        self._equity_curve_cache = curve
        self._cache_timestamp = datetime.now()
        return curve


# ============================================================================
# ALERT SYSTEM
# ============================================================================

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
    
    def to_dict(self) -> Dict:
        return to_serializable(asdict(self))
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('acknowledged_at'):
            data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        data['severity'] = AlertSeverity(data['severity'])
        return cls(**data)


class AlertSystem:
    """Alert system with email notifications."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        self.thresholds = self.config.get('thresholds', {})
        
        self.email_enabled = self.config.get('email_enabled', Config.EMAIL_ENABLED)
        self.smtp_server = self.config.get('smtp_server', Config.SMTP_SERVER)
        self.smtp_port = self.config.get('smtp_port', Config.SMTP_PORT)
        self.email_user = self.config.get('email_user', Config.EMAIL_USER)
        self.email_password = self.config.get('email_password', Config.EMAIL_PASSWORD)
        self.email_recipients = self.config.get('email_recipients', Config.EMAIL_RECIPIENTS)
        
        if self.email_enabled and not all([self.smtp_server, self.email_user, self.email_password, self.email_recipients]):
            logger.warning("Email alerts enabled but missing configuration, disabling")
            self.email_enabled = False
    
    def add_alert(self, severity: AlertSeverity, category: str, title: str, 
                  message: str, data: Dict = None):
        """Add a new alert."""
        alert = Alert(severity=severity, category=category, title=title, message=message, data=data)
        self.alerts.append(alert)
        
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)
        logger.log(log_level, f"[{severity.value}] {title}: {message}")
        
        for handler in list(self.alert_handlers):
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
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
        """Send email alert."""
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
        """Check for portfolio alerts."""
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
        
        if risk_manager.daily_pnl < -total_equity * self.thresholds.get('daily_loss_limit', 0.05):
            self.add_alert(AlertSeverity.ERROR, "Risk", "Daily Loss Limit", 
                          f"Daily loss ${risk_manager.daily_pnl:.2f} exceeds limit")


# ============================================================================
# REPORTING MODULE
# ============================================================================

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
            for lot in pos.lots + pos.archived_lots:
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
            'equity_curve': self.analyzer._get_equity_curve(),
            'positions': [{
                'symbol': p.symbol,
                'exchange': p.exchange,
                'total_quantity': p.total_quantity,
                'average_cost': p.average_cost,
                'market_value': p.market_value,
                'unrealized_pnl': p.unrealized_pnl,
                'realized_pnl': p.realized_pnl,
                'lot_count': p.lot_count,
                'archived_lot_count': len(p.archived_lots)
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
            for lot in pos.archived_lots:
                if lot.closed_time and lot.closed_time.year == year:
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
        filepath = filename.with_suffix('.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON report saved to {filepath}")
        return str(filepath)
    
    def _save_csv(self, data: Dict, filename: Path) -> str:
        filepath = filename.with_suffix('.csv')
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Report Generated', data.get('generated_at', '')])
            writer.writerow([])
            
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
        filepath = filename.with_suffix('.html')
        html = self._generate_html_report(data)
        with open(filepath, 'w', encoding='utf-8') as f:
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
        
        if data.get('open_lots'):
            html += """
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
            for lot in data['open_lots'][:50]:
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
        
        if data.get('closed_lots'):
            html += """
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
            for lot in data['closed_lots'][:50]:
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
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


# ============================================================================
# MOCK EXCHANGE
# ============================================================================

