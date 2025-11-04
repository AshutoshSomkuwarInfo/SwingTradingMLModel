"""
Simple Signal-Based Backtest
Follows model predictions (BUY/HOLD/SELL) and measures accuracy
"""

import pandas as pd
from .data import get_stock_data
from .model import BASE_FEATURES, EXTENDED_FEATURES
from xgboost import XGBClassifier


def simple_signal_backtest(stock_list, period="5y", initial_capital=100000):
    """
    Simple backtest: Follow the signals exactly (BUY, HOLD, SELL)
    Track what happened vs what was predicted
    
    Returns:
        results (dict): Summary of performance
        trade_log (list): Detailed log of each trade
    """
    
    results = {
        'stock': None,
        'total_signals': 0,
        'buy_signals': 0,
        'hold_signals': 0,
        'sell_signals': 0,
        'correct_predictions': 0,
        'wrong_predictions': 0,
        'accuracy': 0,
        'total_profit': 0,
        'total_loss': 0,
        'net_pnl': 0,
        'total_return_pct': 0,
        'nifty_return_pct': 0,
        'vs_nifty': 0,
        'trades': [],
        'trade_details': []
    }
    
    for stock in stock_list:
        data = get_stock_data(stock, period=period)
        if data is None or data.empty or len(data) < 30:
            continue
            
        results['stock'] = stock
        
        # Split into train/test
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:].copy().reset_index()
        
        # Use BASE features only - simpler is better for this problem
        features = BASE_FEATURES
        features = [f for f in features if f in data.columns]
        
        label_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}
        signal_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        
        # Train model - use simpler, more reliable approach
        X_train = train[features].astype(float)
        y_train = train["Signal"].map(label_mapping)
        
        # Remove NaN rows
        valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        # Use original hyperparameters that tend to work better
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0, use_label_encoder=False, eval_metric="mlogloss", verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Get predictions
        X_test = test[features].astype(float)
        preds = model.predict(X_test)
        test["Pred_Signal"] = preds
        test["Pred_Signal_Str"] = test["Pred_Signal"].map(signal_names)
        test["Actual_Signal"] = test["Signal"]
        
        # Calculate forward returns (what actually happened)
        test["Future_Return"] = test["Close"].pct_change(15).shift(-15)
        
        # Analyze each prediction
        for i in range(len(test) - 15):  # Need 15 days forward data
            pred_signal = preds[i]
            actual_return = test.iloc[i]["Future_Return"]
            current_price = test.iloc[i]["Close"]
            
            if pd.isna(actual_return):
                continue
                
            results['total_signals'] += 1
            
            # Count signal types
            if pred_signal == 2:  # BUY
                results['buy_signals'] += 1
                
                # Check if prediction was correct
                if actual_return > 0.02:  # Actual gain > 2%
                    results['correct_predictions'] += 1
                    results['total_profit'] += actual_return * initial_capital * 0.2  # 20% position size
                    results['trades'].append({
                        'date': test.iloc[i]["Date"],
                        'signal': 'BUY',
                        'price': current_price,
                        'actual_return': actual_return * 100,
                        'result': 'CORRECT',
                        'pnl': actual_return * initial_capital * 0.2
                    })
                else:
                    results['wrong_predictions'] += 1
                    if actual_return < 0:
                        results['total_loss'] += abs(actual_return) * initial_capital * 0.2
                    results['trades'].append({
                        'date': test.iloc[i]["Date"],
                        'signal': 'BUY',
                        'price': current_price,
                        'actual_return': actual_return * 100,
                        'result': 'WRONG',
                        'pnl': actual_return * initial_capital * 0.2
                    })
                    
            elif pred_signal == 1:  # HOLD
                results['hold_signals'] += 1
                # HOLD means don't trade, so no profit/loss
                
            elif pred_signal == 0:  # SELL
                results['sell_signals'] += 1
                
                # For SELL, we're predicting decline
                if actual_return < -0.02:  # Actual drop > 2%
                    results['correct_predictions'] += 1
                    results['total_profit'] += abs(actual_return) * initial_capital * 0.2
                    results['trades'].append({
                        'date': test.iloc[i]["Date"],
                        'signal': 'SELL',
                        'price': current_price,
                        'actual_return': actual_return * 100,
                        'result': 'CORRECT',
                        'pnl': abs(actual_return) * initial_capital * 0.2
                    })
                else:
                    results['wrong_predictions'] += 1
                    if actual_return > 0:
                        results['total_loss'] += abs(actual_return) * initial_capital * 0.2
                    results['trades'].append({
                        'date': test.iloc[i]["Date"],
                        'signal': 'SELL',
                        'price': current_price,
                        'actual_return': actual_return * 100,
                        'result': 'WRONG',
                        'pnl': -abs(actual_return) * initial_capital * 0.2
                    })
        
        # Calculate accuracy
        total_evaluated = results['correct_predictions'] + results['wrong_predictions']
        if total_evaluated > 0:
            results['accuracy'] = (results['correct_predictions'] / total_evaluated) * 100
        
        # Calculate net P&L
        results['net_pnl'] = results['total_profit'] - results['total_loss']
        results['total_return_pct'] = (results['net_pnl'] / initial_capital) * 100
        
        # Get NIFTY benchmark
        try:
            nifty_data = get_stock_data("^NSEI", period=period)
            if nifty_data is not None and len(nifty_data) > 0:
                nifty_start = nifty_data.iloc[split]["Close"]
                nifty_end = nifty_data.iloc[-1]["Close"]
                results['nifty_return_pct'] = ((nifty_end - nifty_start) / nifty_start) * 100
                results['vs_nifty'] = results['total_return_pct'] - results['nifty_return_pct']
        except:
            pass
        
        # Create detailed trade log
        for trade in results['trades']:
            results['trade_details'].append({
                'Date': trade['date'],
                'Predicted': trade['signal'],
                'Price': f"₹{trade['price']:,.2f}",
                'Actual Return %': f"{trade['actual_return']:.2f}%",
                'Result': trade['result'],
                'P&L': f"₹{trade['pnl']:,.2f}"
            })
        
        break  # Process only first stock for now
    
    return results


def get_signal_summary(results):
    """Generate a summary of signal performance"""
    summary = {
        'signal_breakdown': {
            'BUY': results['buy_signals'],
            'HOLD': results['hold_signals'],
            'SELL': results['sell_signals']
        },
        'accuracy': f"{results['accuracy']:.2f}%",
        'correct': results['correct_predictions'],
        'wrong': results['wrong_predictions'],
        'total_profit': f"₹{results['total_profit']:,.2f}",
        'total_loss': f"₹{results['total_loss']:,.2f}",
        'net_pnl': f"₹{results['net_pnl']:,.2f}",
        'total_return': f"{results['total_return_pct']:.2f}%",
        'nifty_return': f"{results['nifty_return_pct']:.2f}%",
        'vs_nifty': f"{results['vs_nifty']:.2f}%"
    }
    
    return summary


