"""
Paper Trading System for Live Testing
Simulates live trading with virtual money using real market data
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from .risk import RiskManager


class PaperTradingSystem:
    """
    Paper trading system that simulates live trading with virtual capital
    """
    
    def __init__(
        self,
        initial_capital=100000,
        tickers=None,
        risk_config=None
    ):
        """
        Initialize Paper Trading System
        
        Args:
            initial_capital: Starting virtual capital
            tickers: List of tickers to trade
            risk_config: Risk management configuration
        """
        self.initial_capital = initial_capital
        self.tickers = tickers or []
        
        # Initialize risk manager
        risk_defaults = {
            'initial_capital': initial_capital,
            'max_position_size_pct': 0.20,
            'max_daily_loss_pct': 0.05,
            'max_drawdown_pct': 0.15,
            'stop_loss_pct': 0.05,
            'risk_per_trade_pct': 0.02
        }
        if risk_config:
            risk_defaults.update(risk_config)
        
        self.risk_manager = RiskManager(**risk_defaults)
        
        # Active positions tracking
        self.active_positions = {}  # {ticker: position_info}
        
        # Trading history
        self.trade_history = []
        
        # Daily tracking
        self.daily_data = []
        
    def get_current_price(self, ticker):
        """
        Get current price for a ticker (in production, use live feed)
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price or None
        """
        try:
            stock = yf.Ticker(ticker)
            current_data = stock.history(period="1d")
            if not current_data.empty:
                return float(current_data['Close'].iloc[-1])
        except Exception as e:
            print(f"Error getting price for {ticker}: {e}")
        return None
    
    def execute_trade(self, ticker, signal, confidence=None):
        """
        Execute a paper trade (virtual execution)
        
        Args:
            ticker: Stock ticker
            signal: BUY, SELL, or HOLD
            confidence: Signal confidence (optional)
            
        Returns:
            dict with trade execution details
        """
        # Check if trading is allowed
        allowed, reason = self.risk_manager.check_trade_allowed()
        if not allowed:
            return {'status': 'rejected', 'reason': reason}
        
        # Get current price
        current_price = self.get_current_price(ticker)
        if not current_price:
            return {'status': 'error', 'reason': 'Unable to get current price'}
        
        # Only trade BUY or SELL signals
        if signal == "HOLD":
            return {'status': 'no_action', 'reason': 'HOLD signal'}
        
        timestamp = datetime.now()
        
        # Handle BUY signal
        if signal == "BUY":
            # Check if we already have a position
            if ticker in self.active_positions:
                return {'status': 'rejected', 'reason': 'Position already exists'}
            
            # Calculate position size
            position_details = self.risk_manager.calculate_position_size(
                entry_price=current_price,
                signal_type="BUY"
            )
            
            if not position_details:
                return {'status': 'rejected', 'reason': 'Position size calculation failed'}
            
            # Create position
            position = {
                'ticker': ticker,
                'entry_price': current_price,
                'entry_time': timestamp,
                'quantity': position_details['quantity'],
                'stop_loss': position_details['stop_loss_price'],
                'signal_confidence': confidence,
                'position_value': position_details['position_value']
            }
            
            self.active_positions[ticker] = position
            
            # Record trade
            trade_record = {
                'timestamp': timestamp,
                'ticker': ticker,
                'action': 'OPEN_BUY',
                'price': current_price,
                'quantity': position_details['quantity'],
                'value': position_details['position_value'],
                'confidence': confidence
            }
            self.trade_history.append(trade_record)
            
            return {
                'status': 'executed',
                'action': 'OPEN_BUY',
                'ticker': ticker,
                'price': current_price,
                'quantity': position_details['quantity'],
                'value': position_details['position_value']
            }
        
        # Handle SELL signal
        elif signal == "SELL":
            # Check if we have a position to close
            if ticker not in self.active_positions:
                # Could open a short position if allowed
                return {'status': 'rejected', 'reason': 'No position to close'}
            
            position = self.active_positions.pop(ticker)
            
            # Calculate P&L
            entry_value = position['quantity'] * position['entry_price']
            exit_value = position['quantity'] * current_price
            
            if position['entry_time']:  # Long position
                pnl = exit_value - entry_value
            else:  # Short position (not implemented yet)
                pnl = entry_value - exit_value
            
            # Update risk manager
            trade_result = {
                'pnl': pnl,
                'entry_value': entry_value,
                'exit_value': exit_value,
                'type': 'LONG'
            }
            self.risk_manager.update_position(trade_result)
            
            # Record trade
            trade_record = {
                'timestamp': timestamp,
                'ticker': ticker,
                'action': 'CLOSE_LONG',
                'price': current_price,
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'pnl': pnl,
                'pnl_pct': (pnl / entry_value) * 100
            }
            self.trade_history.append(trade_record)
            
            return {
                'status': 'executed',
                'action': 'CLOSE_LONG',
                'ticker': ticker,
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_value) * 100
            }
        
        return {'status': 'unknown', 'reason': 'Invalid signal'}
    
    def check_stop_losses(self):
        """
        Check all active positions for stop loss triggers
        
        Returns:
            list of executed stop loss trades
        """
        executed_stops = []
        
        for ticker, position in list(self.active_positions.items()):
            current_price = self.get_current_price(ticker)
            if not current_price:
                continue
            
            # Check for stop loss trigger
            stop_loss = position.get('stop_loss')
            
            if current_price <= stop_loss:
                # Trigger stop loss - close position
                entry_value = position['quantity'] * position['entry_price']
                exit_value = position['quantity'] * current_price
                pnl = exit_value - entry_value
                
                # Remove position
                self.active_positions.pop(ticker)
                
                # Update risk manager
                trade_result = {
                    'pnl': pnl,
                    'entry_value': entry_value,
                    'exit_value': exit_value,
                    'type': 'LONG'
                }
                self.risk_manager.update_position(trade_result)
                
                # Record stop loss trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'STOP_LOSS',
                    'price': current_price,
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'stop_loss': stop_loss,
                    'pnl': pnl
                }
                self.trade_history.append(trade_record)
                executed_stops.append(trade_record)
        
        return executed_stops
    
    def update_positions_pnl(self):
        """
        Update P&L for all active positions (mark-to-market)
        
        Returns:
            dict with current unrealized P&L
        """
        total_unrealized_pnl = 0
        positions_pnl = {}
        
        for ticker, position in self.active_positions.items():
            current_price = self.get_current_price(ticker)
            if not current_price:
                continue
            
            entry_value = position['quantity'] * position['entry_price']
            current_value = position['quantity'] * current_price
            unrealized_pnl = current_value - entry_value
            
            positions_pnl[ticker] = {
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'quantity': position['quantity'],
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / entry_value) * 100
            }
            
            total_unrealized_pnl += unrealized_pnl
        
        return {
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions': positions_pnl
        }
    
    def get_portfolio_status(self):
        """
        Get complete portfolio status
        
        Returns:
            dict with comprehensive portfolio metrics
        """
        # Get unrealized P&L
        unrealized_data = self.update_positions_pnl()
        
        # Get risk manager status
        risk_status = self.risk_manager.get_portfolio_status()
        
        # Calculate realized P&L from closed trades
        closed_trades = [t for t in self.trade_history if 'pnl' in t]
        realized_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        
        return {
            **risk_status,
            'active_positions': len(self.active_positions),
            'total_trades': len(self.trade_history),
            'closed_trades': len(closed_trades),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_data['total_unrealized_pnl'],
            'total_pnl': realized_pnl + unrealized_data['total_unrealized_pnl'],
            'positions_detail': unrealized_data['positions']
        }
    
    def get_trade_history_df(self):
        """Get trade history as pandas DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)
    
    def reset_daily_stats(self):
        """Reset daily tracking (call at start of each day)"""
        self.risk_manager.reset_daily_tracking()




