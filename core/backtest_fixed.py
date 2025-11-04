"""
Fixed Backtesting Module
Addresses critical issues with proper position tracking, stop losses, and no look-ahead bias
"""

import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from .data import get_stock_data
from .model import BASE_FEATURES, EXTENDED_FEATURES


class Position:
    """Represents a trading position with trailing stop and take profit"""
    def __init__(self, ticker, entry_date, entry_price, quantity, stop_loss_pct=0.07, 
                 trail_stop=True, take_profit_pct=0.10):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.initial_stop_loss = entry_price * (1 - stop_loss_pct)
        self.stop_loss = self.initial_stop_loss
        self.take_profit = entry_price * (1 + take_profit_pct)
        self.trail_stop = trail_stop
        self.peak_price = entry_price
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0
        self.pnl_pct = 0
        self.exit_reason = None
    
    def update_trailing_stop(self, current_price, trail_pct=0.04):
        """Update stop loss as price moves up (trailing stop)"""
        if current_price > self.peak_price:
            self.peak_price = current_price
            if self.trail_stop:
                # Trail stop to trail_pct below peak
                new_stop = self.peak_price * (1 - trail_pct)
                self.stop_loss = max(self.stop_loss, new_stop)


def exit_position(position, exit_date, exit_price, capital, cost, reason="TIME_BASED"):
    """Helper function to exit a position and record trade"""
    position.exit_date = exit_date
    position.exit_price = exit_price
    position.pnl = (exit_price - position.entry_price) * position.quantity
    position.pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
    position.exit_reason = reason
    
    # Update capital (subtract transaction cost)
    capital += position.pnl - (position.entry_price * position.quantity * cost) - (exit_price * position.quantity * cost)
    
    return position.pnl, capital


def backtest_realistic_fixed(stock_list, initial_capital=100000, position_size=0.2, 
                             stop_loss_pct=0.07, cost=0.002, period="5y"):
    """
    Fixed backtest with proper position tracking and no look-ahead bias.
    
    Returns:
        portfolio_df (pd.DataFrame), trades (list), diagnostics (dict)
    """
    capital = initial_capital
    portfolio_history = []
    all_trades = []
    open_positions = {}  # {ticker: Position}
    overall_diagnostics = {
        "test_slice_length": 0, 
        "predicted_signal_counts": {"BUY": 0, "HOLD": 0, "SELL": 0}
    }

    for stock in stock_list:
        data = get_stock_data(stock, period=period)
        if data is None or data.empty:
            continue

        # Split into train/test to avoid lookahead
        if len(data) < 40:
            continue
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:].copy().reset_index()

        # Use BASE features only - simpler is better for accuracy
        features = BASE_FEATURES
        features = [f for f in features if f in data.columns]
        
        label_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}

        # Train model with original simpler hyperparameters
        X_train = train[features].astype(float)
        y_train = train["Signal"].map(label_mapping)
        
        # Remove NaN rows
        valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        # Original simpler hyperparameters
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0, use_label_encoder=False, eval_metric="mlogloss", verbosity=0
        )
        
        model.fit(X_train, y_train)

        # Get predictions
        preds = model.predict(test[features])
        test["Pred_Signal"] = preds

        overall_diagnostics["test_slice_length"] += len(test)
        overall_diagnostics["predicted_signal_counts"]["BUY"] += int((preds == 2).sum())
        overall_diagnostics["predicted_signal_counts"]["HOLD"] += int((preds == 1).sum())
        overall_diagnostics["predicted_signal_counts"]["SELL"] += int((preds == 0).sum())

        # Process each day in test period
        for day_idx in range(len(test)):
            current_date = test.iloc[day_idx]["Date"]
            current_price = test.iloc[day_idx]["Close"]
            signal = test.iloc[day_idx]["Pred_Signal"]
            
            # Check existing positions for exit conditions
            for ticker, position in list(open_positions.items()):
                # Update trailing stop
                position.update_trailing_stop(current_price)
                
                # Check take profit first (highest priority)
                if current_price >= position.take_profit:
                    # Exit with profit
                    _, capital = exit_position(position, current_date, current_price, capital, cost, "TAKE_PROFIT")
                    
                    # Record trade
                    all_trades.append({
                        "Stock": ticker,
                        "Date": position.entry_date,
                        "Signal": "BUY",
                        "Entry": round(position.entry_price, 2),
                        "Exit": round(position.exit_price, 2),
                        "Return%": round(position.pnl_pct, 2),
                    })
                    del open_positions[ticker]
                    continue
                
                # Check stop loss
                if current_price <= position.stop_loss:
                    # Execute stop loss
                    _, capital = exit_position(position, current_date, current_price, capital, cost, "STOP_LOSS")
                    
                    # Record trade
                    all_trades.append({
                        "Stock": ticker,
                        "Date": position.entry_date,
                        "Signal": "BUY",
                        "Entry": round(position.entry_price, 2),
                        "Exit": round(position.exit_price, 2),
                        "Return%": round(position.pnl_pct, 2),
                    })
                    del open_positions[ticker]
                    continue
                
                # Check dynamic exit window (10-20 days)
                days_held = (current_date - position.entry_date).days
                profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                
                # Exit if held long enough
                if days_held >= 10:  # Minimum hold period
                    # Exit early if profitable after 10 days
                    if profit_pct > 2 and days_held >= 10:
                        _, capital = exit_position(position, current_date, current_price, capital, cost, "EARLY_PROFIT")
                        all_trades.append({
                            "Stock": ticker,
                            "Date": position.entry_date,
                            "Signal": "BUY",
                            "Entry": round(position.entry_price, 2),
                            "Exit": round(position.exit_price, 2),
                            "Return%": round(position.pnl_pct, 2),
                        })
                        del open_positions[ticker]
                        continue
                    # Or exit if hit max hold time (20 days)
                    elif days_held >= 20:
                        _, capital = exit_position(position, current_date, current_price, capital, cost, "TIME_BASED")
                        all_trades.append({
                            "Stock": ticker,
                            "Date": position.entry_date,
                            "Signal": "BUY",
                            "Entry": round(position.entry_price, 2),
                            "Exit": round(position.exit_price, 2),
                            "Return%": round(position.pnl_pct, 2),
                        })
                        del open_positions[ticker]
                        continue
            
            # Handle new signals (only if no existing position in this stock)
            if signal == 2 and stock not in open_positions:  # BUY signal
                # Apply entry filters to improve signal quality
                entry_allowed = True
                
                # Check if indicators are available
                if "RSI" in test.columns:
                    current_rsi = test.iloc[day_idx]["RSI"]
                    # Only enter if not overbought
                    if current_rsi >= 70:
                        entry_allowed = False
                
                if "EMA_10" in test.columns:
                    price = test.iloc[day_idx]["Close"]
                    ema_10 = test.iloc[day_idx]["EMA_10"]
                    # Only enter if in uptrend
                    if price <= ema_10:
                        entry_allowed = False
                
                # Only enter if filters pass
                if entry_allowed:
                    available_capital = capital * position_size
                    
                    if available_capital > 0:
                        # Calculate position
                        quantity = int(available_capital / current_price)
                        
                        if quantity > 0:
                            # Create new position
                            open_positions[stock] = Position(
                                ticker=stock,
                                entry_date=current_date,
                                entry_price=current_price,
                                quantity=quantity,
                                stop_loss_pct=stop_loss_pct
                            )
                            
                            # Deduct entry cost
                            capital -= quantity * current_price * (1 + cost)
            
            # Calculate current portfolio value (mark-to-market)
            portfolio_value = capital
            for ticker, pos in open_positions.items():
                # Use current price for open positions
                portfolio_value += pos.quantity * current_price
            
            # Record daily portfolio value
            portfolio_history.append({
                "Date": current_date,
                "Capital": capital,
                "Portfolio_Value": portfolio_value
            })

    portfolio_df = pd.DataFrame(portfolio_history)
    if not portfolio_df.empty:
        portfolio_df = portfolio_df.set_index("Date")
    
    return portfolio_df, all_trades, overall_diagnostics


def backtest_simple_fixed(stock_list, period="5y"):
    """
    Fixed simple backtest with no look-ahead bias.
    
    Returns:
        growth_series (pd.Series), diagnostics (dict)
    """
    portfolio_returns = pd.Series(dtype=float)
    overall_diagnostics = {"test_slice_length": 0, "predicted_signal_counts": {"BUY": 0, "HOLD": 0, "SELL": 0}}

    for stock in stock_list:
        data = get_stock_data(stock, period=period)
        if data is None or data.empty:
            continue

        # Split into train/test
        if len(data) < 40:
            continue
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:].copy().reset_index()

        # Use BASE features only - simpler is better for accuracy
        features = BASE_FEATURES
        features = [f for f in features if f in data.columns]
        
        label_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}

        X_train = train[features].astype(float)
        y_train = train["Signal"].map(label_mapping)
        
        # Calculate class weights to handle imbalance
        class_counts = pd.Series(y_train).value_counts().sort_index()
        total_samples = len(y_train)
        if len(class_counts) >= 3:
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = {i: total_samples / (len(class_counts) * count) 
                           for i, count in class_counts.items()}
            sample_weights = compute_sample_weight(class_weight=class_weights, y=y_train)
        else:
            sample_weights = None
        
        # Improved hyperparameters
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="mlogloss", verbosity=0
        )
        
        # Fit with sample weights if available
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        preds = model.predict(test[features])
        test["Pred_Signal"] = preds

        overall_diagnostics["test_slice_length"] += len(test)
        overall_diagnostics["predicted_signal_counts"]["BUY"] += int((preds == 2).sum())
        overall_diagnostics["predicted_signal_counts"]["HOLD"] += int((preds == 1).sum())
        overall_diagnostics["predicted_signal_counts"]["SELL"] += int((preds == 0).sum())

        # Track positions properly
        open_positions = {}  # {index: entry_price}
        strategy_returns = []

        for i in range(len(test)):
            current_price = test.iloc[i]["Close"]
            signal = test.iloc[i]["Pred_Signal"]
            
            # Check existing positions for exit (15-day limit)
            for entry_idx in list(open_positions.keys()):
                if i - entry_idx >= 15:  # 15 days elapsed
                    entry_price = open_positions[entry_idx]
                    # Calculate return
                    ret = (current_price - entry_price) / entry_price
                    strategy_returns.append({"date": test.iloc[i]["Date"], "return": ret})
                    del open_positions[entry_idx]
            
            # Handle new signals (only if no overlapping position)
            if signal == 2:  # BUY signal
                if not open_positions:  # Only if no current position
                    open_positions[i] = current_price
            
            # Add zero return for non-trade days
            if i not in open_positions.values():  # Only if not holding a position
                if len(strategy_returns) == 0 or strategy_returns[-1]["date"] != test.iloc[i]["Date"]:
                    strategy_returns.append({"date": test.iloc[i]["Date"], "return": 0.0})
        
        # Close any remaining positions at end
        for entry_idx, entry_price in open_positions.items():
            exit_price = test.iloc[-1]["Close"]
            ret = (exit_price - entry_price) / entry_price
            strategy_returns.append({"date": test.iloc[-1]["Date"], "return": ret})
        
        # Create returns series
        if strategy_returns:
            ret_df = pd.DataFrame(strategy_returns)
            ret_df = ret_df.set_index("date")
            portfolio_returns = portfolio_returns.add(ret_df["return"].reindex(portfolio_returns.index).fillna(0), fill_value=0)

    # Convert to cumulative returns
    if len(portfolio_returns) > 0:
        growth = (1 + portfolio_returns.fillna(0)).cumprod() * 100
    else:
        growth = pd.Series(dtype=float)
    
    return growth, overall_diagnostics


def get_nifty50_benchmark(period="5y"):
    """Download NIFTY50 benchmark for comparison"""
    nifty = yf.download("^NSEI", period=period, progress=False)
    if nifty.empty:
        return None
    nifty_returns = nifty["Close"].pct_change().fillna(0)
    return (1 + nifty_returns).cumprod() * 100
