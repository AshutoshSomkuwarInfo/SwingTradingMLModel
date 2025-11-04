"""
Risk Management Module for Live Trading
Implements position sizing, stop-loss, and risk limits
"""

import pandas as pd
import numpy as np


class RiskManager:
    """
    Manages risk for live trading with position sizing, stop-loss, and limits
    """
    
    def __init__(
        self,
        initial_capital=100000,
        max_position_size_pct=0.20,  # 20% per position
        max_daily_loss_pct=0.05,  # 5% max loss per day
        max_drawdown_pct=0.15,  # 15% max drawdown
        stop_loss_pct=0.05,  # 5% stop loss
        risk_per_trade_pct=0.02  # Risk 2% of capital per trade
    ):
        """
        Initialize Risk Manager
        
        Args:
            initial_capital: Starting capital
            max_position_size_pct: Maximum percentage of capital per position
            max_daily_loss_pct: Maximum daily loss percentage
            max_drawdown_pct: Maximum portfolio drawdown
            stop_loss_pct: Stop loss percentage from entry
            risk_per_trade_pct: Risk per trade (capital at risk)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        
        # Tracking
        self.peak_capital = initial_capital
        self.daily_pnl = 0
        self.total_pnl = 0
        self.positions = []
        self.trades = []
        
    def calculate_position_size(self, entry_price, signal_type="BUY"):
        """
        Calculate position size based on risk parameters
        
        Args:
            entry_price: Entry price for the trade
            signal_type: BUY or SELL
            
        Returns:
            dict with position details including quantity, stop_loss, max_loss
        """
        # Check if we've hit max drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.max_drawdown_pct:
            return None  # Stop trading if drawdown exceeded
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss_pct * self.initial_capital:
            return None  # Stop trading for the day
        
        # Calculate capital at risk
        capital_at_risk = self.current_capital * self.risk_per_trade_pct
        
        # Calculate stop loss distance in currency
        stop_loss_distance = entry_price * self.stop_loss_pct
        
        # Position sizing based on risk
        # Quantity = (Capital at Risk) / (Stop Loss Distance)
        quantity = int(capital_at_risk / stop_loss_distance)
        
        # Apply maximum position size limit
        max_quantity_by_capital = int((self.current_capital * self.max_position_size_pct) / entry_price)
        quantity = min(quantity, max_quantity_by_capital)
        
        if quantity < 1:
            return None  # Position too small
        
        # Calculate position value and potential loss
        position_value = quantity * entry_price
        
        # Calculate stop loss price
        if signal_type == "BUY":
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        else:  # SELL
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)
        
        max_loss = abs(position_value - (quantity * stop_loss_price))
        
        return {
            'quantity': quantity,
            'entry_price': entry_price,
            'position_value': position_value,
            'stop_loss_price': stop_loss_price,
            'max_loss': max_loss,
            'capital_at_risk_pct': (max_loss / self.current_capital) * 100
        }
    
    def check_trade_allowed(self):
        """
        Check if new trades are allowed based on risk limits
        
        Returns:
            tuple: (allowed: bool, reason: str)
        """
        # Check max drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.max_drawdown_pct:
            return False, f"Max drawdown exceeded: {current_drawdown*100:.2f}%"
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss_pct * self.initial_capital:
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f}"
        
        return True, "OK"
    
    def update_position(self, trade_result):
        """
        Update tracking after a trade
        
        Args:
            trade_result: dict with trade details including pnl, type, etc.
        """
        self.total_pnl += trade_result.get('pnl', 0)
        self.daily_pnl += trade_result.get('pnl', 0)
        
        # Update current capital
        if 'entry_value' in trade_result and 'exit_value' in trade_result:
            capital_change = trade_result['exit_value'] - trade_result['entry_value']
            self.current_capital += capital_change
        
        # Update peak capital for drawdown tracking
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Record trade
        self.trades.append(trade_result)
    
    def reset_daily_tracking(self):
        """Reset daily tracking (call at start of each trading day)"""
        self.daily_pnl = 0
    
    def get_portfolio_status(self):
        """
        Get current portfolio status
        
        Returns:
            dict with portfolio metrics
        """
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        total_return_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'total_pnl': self.total_pnl,
            'total_return_pct': total_return_pct,
            'current_drawdown_pct': current_drawdown * 100,
            'daily_pnl': self.daily_pnl,
            'total_trades': len(self.trades),
            'max_drawdown_exceeded': current_drawdown >= self.max_drawdown_pct,
            'daily_loss_exceeded': self.daily_pnl < -self.max_daily_loss_pct * self.initial_capital
        }
    
    def get_recent_trades(self, n=10):
        """Get n most recent trades"""
        return self.trades[-n:] if self.trades else []


def kelly_criterion_position_size(win_rate, avg_win, avg_loss, capital):
    """
    Calculate position size using Kelly Criterion
    
    Args:
        win_rate: Win rate (0 to 1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount
        capital: Current capital
        
    Returns:
        Fraction of capital to risk (0 to 1)
    """
    if avg_loss == 0 or avg_win == 0:
        return 0.02  # Default 2% if can't calculate
    
    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1-p
    b = abs(avg_win / avg_loss)
    p = win_rate
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    
    # Use fractional Kelly (half) for safety
    safe_kelly = kelly_fraction * 0.5
    
    # Constrain between 0 and 0.05 (max 5% of capital)
    return max(0, min(0.05, safe_kelly))


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    """
    Calculate Sharpe ratio for returns
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (default 0 for daily)
        periods: Trading periods per year (252 for daily)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0
    
    excess_returns = returns - (risk_free_rate / periods)
    return np.sqrt(periods) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from equity curve
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        dict with max_drawdown_pct, max_drawdown_duration, etc.
    """
    if len(equity_curve) == 0:
        return {'max_drawdown_pct': 0, 'max_drawdown_duration': 0}
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown_pct = abs(drawdown.min()) * 100
    
    # Calculate duration of max drawdown
    # This is simplified - in practice you'd track each drawdown period
    max_dd_index = drawdown.idxmin()
    
    return {
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_date': max_dd_index
    }




