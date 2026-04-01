"""
Email alert system for trading bot
Sends email notifications for trades and daily summaries via Gmail SMTP (FREE)
"""

import os
from dotenv import load_dotenv
load_dotenv()

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config: dict, params: dict):
        """
        Initialize email alert system
        """
        self.config = config.get('alerts', {})
        self.params = params
        self.enabled = self.config.get('enabled', False)
        self.to_email = self.config.get('email', '')
        
        # Gmail SMTP settings
        self.smtp_host = 'smtp.gmail.com'
        self.smtp_port = 587
        
        # Get credentials from environment variables
        self.from_email = os.getenv('GMAIL_USER')
        self.password = os.getenv('GMAIL_PASSWORD')
        
        # Validate setup
        if self.enabled:
            if not self.from_email or not self.password:
                logger.error("Gmail credentials not found in environment variables")
                logger.error("Add GMAIL_USER and GMAIL_PASSWORD to your .env file")
                self.enabled = False
            elif not self.to_email:
                logger.error("No recipient email configured in config")
                self.enabled = False
            else:
                logger.info(f"Email alerts initialized ({self.from_email} → {self.to_email})")
        else:
            logger.info("Email alerts disabled in config")
    
    def send_email(self, subject: str, body: str) -> bool:
        """
        Send email via Gmail SMTP
        
        Args:
            subject: Email subject line
            body: Email body (plain text or HTML)
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Alerts disabled - would have sent: {subject}")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            
            # Auto-detect if body is HTML
            if body.strip().startswith('<'):
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Send via Gmail SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)
            
            logger.info(f"Email sent: {subject}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            logger.error("Gmail authentication failed - check your app password")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    # ==================== Trading Alerts ====================
    
    def alert_trade_entry(self, signal: Dict) -> None:
        """
        Alert when entering a trade
        
        Args:
            signal: Trade signal dictionary with all trade details
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        trade_type = signal.get('type', 'UNKNOWN')
        entry = signal.get('entry', 0)
        stop = signal.get('stop', 0)
        target = signal.get('target', 0)
        shares = signal.get('shares', 0)
        risk = signal.get('risk', 0)
        reward = signal.get('reward', 0)
        
        subject = f"🟢 Trade Entered: {trade_type} {symbol}"
        
        body = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333;">
    <h2 style="color: #2E7D32;">🤖 Trade Entry Notification</h2>
    
    <h3>Trade Details</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Symbol:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{symbol}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Direction:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{trade_type}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Entry Price:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${entry:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Stop Loss:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${stop:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Take Profit:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${target:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Shares:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{shares}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Risk:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${risk:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Potential Reward:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${reward:.2f}</td>
        </tr>
    </table>
    
    <h3>Pattern Details</h3>
    <ul>
        <li><strong>Breakout Date:</strong> {signal.get('breakout_date', 'N/A')}</li>
        <li><strong>Retest Date:</strong> {signal.get('retest_date', 'N/A')}</li>
        <li><strong>Breakout Volume:</strong> {signal.get('breakout_volume_ratio', 0):.2f}x average</li>
        <li><strong>Retest Volume:</strong> {signal.get('retest_volume_ratio', 0):.2f}x average</li>
        <li><strong>MA Slope:</strong> {signal.get('ma_slope_pct', 0):.2f}%</li>
    </ul>
    
    <p style="color: #666; font-size: 12px; margin-top: 30px;">
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        Sent by Algo Trading Bot
    </p>
</body>
</html>
"""
        
        self.send_email(subject, body)
    
    def alert_trade_exit(self, symbol: str, exit_type: str, pnl: float, pnl_pct: float, 
                        entry_price: Optional[float] = None, exit_price: Optional[float] = None,
                        shares: Optional[int] = None) -> None:
        """
        Alert when a trade exits
        
        Args:
            symbol: Stock symbol
            exit_type: How the trade exited (Stop Loss, Take Profit, Manual)
            pnl: Profit/Loss in dollars
            pnl_pct: Profit/Loss percentage
            entry_price: Original entry price (optional)
            exit_price: Exit price (optional)
            shares: Number of shares (optional)
        """
        emoji = "✅" if pnl > 0 else "❌"
        color = "#2E7D32" if pnl > 0 else "#C62828"
        
        subject = f"{emoji} Trade Closed: {symbol} ({exit_type})"
        
        additional_info = ""
        if entry_price and exit_price and shares:
            additional_info = f"""
    <h3>Trade Details</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Entry Price:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${entry_price:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Exit Price:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${exit_price:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Shares:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{shares}</td>
        </tr>
    </table>
"""
        
        body = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333;">
    <h2 style="color: {color};">{emoji} Trade Exit Notification</h2>
    
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Symbol:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{symbol}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Exit Type:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{exit_type}</td>
        </tr>
        <tr style="background-color: {'#E8F5E9' if pnl > 0 else '#FFEBEE'};">
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Profit/Loss:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {color}; font-weight: bold;">
                ${pnl:+,.2f} ({pnl_pct:+.2f}%)
            </td>
        </tr>
    </table>
    
    {additional_info}
    
    <p style="color: #666; font-size: 12px; margin-top: 30px;">
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        Sent by Algo Trading Bot
    </p>
</body>
</html>
"""
        
        self.send_email(subject, body)
    
    def alert_daily_summary(self, summary: Dict) -> None:
        """
        Send daily P&L summary
        
        Args:
            summary: Dictionary with daily statistics
        """
        trades_today = summary.get('trades_today', 0)
        pnl_today = summary.get('pnl_today', 0)
        win_rate_today = summary.get('win_rate_today', 0)
        active_positions = summary.get('active_positions', 0)
        account_value = summary.get('account_value', 0)
        
        emoji = "📈" if pnl_today > 0 else "📉" if pnl_today < 0 else "➡️"
        color = "#2E7D32" if pnl_today > 0 else "#C62828" if pnl_today < 0 else "#666"
        
        subject = f"{emoji} Daily Summary - {datetime.now().strftime('%Y-%m-%d')} - ${pnl_today:+,.2f}"
        
        # Build trade details table
        trade_rows = ""
        if 'trades' in summary and summary['trades']:
            for trade in summary['trades']:
                trade_color = "#2E7D32" if trade.get('pnl', 0) > 0 else "#C62828"
                trade_rows += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{trade.get('symbol', 'N/A')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{trade.get('type', 'N/A')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {trade_color}; font-weight: bold;">
                ${trade.get('pnl', 0):+,.2f}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {trade_color};">
                {trade.get('pnl_pct', 0):+.2f}%
            </td>
        </tr>
"""
            trades_table = f"""
    <h3>Today's Trades ({trades_today})</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
        <thead>
            <tr style="background-color: #f5f5f5;">
                <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: left;">Symbol</th>
                <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: left;">Type</th>
                <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: left;">P&L</th>
                <th style="padding: 8px; border-bottom: 2px solid #ddd; text-align: left;">%</th>
            </tr>
        </thead>
        <tbody>
{trade_rows}
        </tbody>
    </table>
"""
        else:
            trades_table = f"<p>No trades executed today.</p>"
        
        body = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333;">
    <h2 style="color: {color};">{emoji} Daily Trading Summary</h2>
    <p style="color: #666;">Date: {datetime.now().strftime('%A, %B %d, %Y')}</p>
    
    <h3>Performance</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <tr style="background-color: {'#E8F5E9' if pnl_today > 0 else '#FFEBEE' if pnl_today < 0 else '#f5f5f5'};">
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Daily P&L:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {color}; font-weight: bold; font-size: 18px;">
                ${pnl_today:+,.2f}
            </td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Trades Today:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{trades_today}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Win Rate (Today):</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{win_rate_today:.1f}%</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Account Value:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${account_value:,.2f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Active Positions:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{active_positions}</td>
        </tr>
    </table>
    
    {trades_table}
    
    <h3>Overall Statistics</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Total Trades:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{summary.get('total_trades', 0)}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Overall Win Rate:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{summary.get('overall_win_rate', 0):.1f}%</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Total P&L (All Time):</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${summary.get('total_pnl', 0):+,.2f}</td>
        </tr>
    </table>
    
    <p style="color: #666; font-size: 12px; margin-top: 30px;">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        Sent by Algo Trading Bot
    </p>
</body>
</html>
"""
        
        self.send_email(subject, body)
    
    def alert_error(self, error_type: str, message: str) -> None:
        """
        Alert for critical errors
        
        Args:
            error_type: Type of error
            message: Error message
        """
        subject = f"⚠️ Trading Bot Error: {error_type}"
        
        body = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333;">
    <h2 style="color: #C62828;">⚠️ Error Alert</h2>
    
    <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Error Type:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{error_type}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Time:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
        </tr>
    </table>
    
    <h3>Error Message</h3>
    <pre style="background-color: #f5f5f5; padding: 15px; border-left: 4px solid #C62828; overflow-x: auto;">
{message}
    </pre>
    
    <p style="color: #666; margin-top: 20px;">
        <strong>Action Required:</strong> Please check the logs and take appropriate action.
    </p>
    
    <p style="color: #666; font-size: 12px; margin-top: 30px;">
        Sent by Algo Trading Bot
    </p>
</body>
</html>
"""
        
        self.send_email(subject, body)
    
    def alert_bot_started(self) -> None:
        """Alert when bot starts"""
        subject = "🚀 Trading Bot Started"
        
        body = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333;">
    <h2 style="color: #1976D2;">🚀 Trading Bot Started</h2>
    
    <p>Your algorithmic trading bot has started successfully.</p>
    
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Start Time:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Status:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">✅ Active</td>
        </tr>
    </table>
    
    <p style="color: #666; font-size: 12px; margin-top: 30px;">
        Sent by Algo Trading Bot
    </p>
</body>
</html>
"""
        
        self.send_email(subject, body)
    
    def alert_bot_stopped(self, reason: str = "User initiated") -> None:
        """Alert when bot stops"""
        subject = "🛑 Trading Bot Stopped"
        
        body = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333;">
    <h2 style="color: #F57C00;">🛑 Trading Bot Stopped</h2>
    
    <p>Your trading bot has stopped.</p>
    
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Stop Time:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Reason:</strong></td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{reason}</td>
        </tr>
    </table>
    
    <p style="color: #666; font-size: 12px; margin-top: 30px;">
        Sent by Algo Trading Bot
    </p>
</body>
</html>
"""
        
        self.send_email(subject, body)