# 📊 Swing Trading Dashboard (15–20 Days)

A Streamlit-based machine learning dashboard for swing trading analysis, using advanced technical indicators, optimized XGBoost models, and comprehensive backtesting.

## ✨ Features

- **ML-Based Signal Prediction** - XGBoost classifier with enhanced feature engineering
- **Advanced Technical Indicators** - RSI, MACD, EMAs, Bollinger Bands, Stochastic, ATR
- **Comprehensive Backtesting** - Simple and realistic strategies with transaction costs
- **Performance Metrics** - Sharpe ratio, CAGR, maximum drawdown, trade analysis
- **Real-time Charts** - Interactive candlestick charts with technical indicators
- **NIFTY50 Benchmark** - Compare strategy performance against market index

---

## 🚀 Run Locally

1. Clone this repository or download it.
2. Open it in **VS Code**.
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate       # (Windows)
   # or
   source venv/bin/activate    # (Mac/Linux)
   
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
5. Run the dashboard:
   ```bash
   streamlit run app.py
   
## 📊 Model Improvements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed analysis of recent enhancements:
- Enhanced feature engineering with 9+ technical indicators
- Optimized XGBoost hyperparameters for better generalization
- Feature importance analysis and confidence scoring
- Flexible feature selection system
- Improved reproducibility and robustness
- **Enhanced 5-panel charts** - See [CHART_FEATURES.md](CHART_FEATURES.md) for details

## 🎯 How It Works

1. **Data Collection**: Downloads historical stock data using yfinance
2. **Feature Engineering**: Calculates multiple technical indicators
3. **Model Training**: Trains XGBoost classifier on historical data
4. **Signal Generation**: Predicts BUY/SELL/HOLD signals for 15-day swings
5. **Backtesting**: Validates strategy performance on historical data
6. **Metrics**: Calculates risk-adjusted returns and trade statistics

---

## 👨‍🎓 Beginner's Guide - Step by Step

> **📚 New! Paper Trading is now available in the Streamlit UI!** See [PAPER_TRADING_GUIDE.md](PAPER_TRADING_GUIDE.md) for the complete guide.

### **Week 1: Understand the Tool**

#### Day 1-2: Explore the Dashboard
1. Run the application: `streamlit run app.py`
2. Open your browser to `http://localhost:8501`
3. Select different stocks from the sidebar
4. Look at the charts and predictions
5. Click "Run Backtests" to see historical performance

**What you're learning**: How the ML model makes predictions

#### Day 3-4: Understand the Signals
- **BUY Signal**: Model predicts price will rise >5% in 15 days
- **SELL Signal**: Model predicts price will fall >5% in 15 days  
- **HOLD Signal**: Model predicts price will stay within ±5%

**Key insight**: These are predictions, not guarantees!

#### Day 5-7: Study the Charts
- Look at the 5-panel charts with technical indicators
- Understand RSI (overbought/oversold)
- See how Bollinger Bands show volatility
- Watch MACD for trend changes

**Learning goal**: Understanding technical analysis basics

---

### **Week 2-3: Paper Trading Setup**

#### Prerequisites Check
✅ You should have:
- Basic Python knowledge (variables, functions, loops)
- Understanding of risk (never risk more than you can afford)
- Time to check the system daily
- Realistic expectations (this is NOT get-rich-quick)

#### Setup Paper Trading

1. **Create a new file** `paper_trading_runner.py`:

```python
from core.paper_trading import PaperTradingSystem
from core.model import predict_latest_signal
from core.data import get_stock_data
import time
from datetime import datetime

# Initialize paper trader with $100,000 virtual money
trader = PaperTradingSystem(
    initial_capital=100000,
    risk_config={
        'max_position_size_pct': 0.20,  # Max 20% per stock
        'stop_loss_pct': 0.05,  # 5% stop loss
        'max_daily_loss_pct': 0.05,  # Stop if lose 5% in a day
        'risk_per_trade_pct': 0.02  # Risk only 2% per trade
    }
)

# Your watchlist (start with 3-5 stocks)
watchlist = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

def check_signals_daily():
    """Run this once per day to check for signals"""
    print(f"\n{'='*50}")
    print(f"Checking signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    for stock in watchlist:
        print(f"\n📊 Checking {stock}...")
        
        # Get data
        data = get_stock_data(stock, period="1y")
        if data is None or data.empty:
            print(f"  ❌ No data available")
            continue
        
        # Get prediction
        signal = predict_latest_signal(data)
        print(f"  📡 Signal: {signal}")
        
        # Execute trade
        result = trader.execute_trade(stock, signal)
        
        if result['status'] == 'executed':
            print(f"  ✅ {result['action']} at ₹{result.get('price', 'N/A')}")
            if 'quantity' in result:
                print(f"  📦 Quantity: {result['quantity']}")
        elif result['status'] == 'rejected':
            print(f"  ❌ Rejected: {result.get('reason', 'Unknown')}")
        else:
            print(f"  ⏸️  No action: {result.get('reason', 'HOLD signal')}")
        
        # Check for stop losses
        stops = trader.check_stop_losses()
        for stop in stops:
            print(f"  🛑 Stop loss triggered for {stop['ticker']}")

def print_portfolio_status():
    """Print current portfolio status"""
    status = trader.get_portfolio_status()
    
    print(f"\n{'='*50}")
    print("📊 PORTFOLIO STATUS")
    print(f"{'='*50}")
    print(f"💰 Capital: ₹{status['current_capital']:,.2f}")
    print(f"📈 Return: {status['total_return_pct']:.2f}%")
    print(f"📊 Total P&L: ₹{status['total_pnl']:,.2f}")
    print(f"🎯 Active Positions: {status['active_positions']}")
    print(f"📦 Total Trades: {status['total_trades']}")
    print(f"📉 Drawdown: {status['current_drawdown_pct']:.2f}%")
    
    if status['max_drawdown_exceeded']:
        print(f"  ⚠️  MAX DRAWDOWN EXCEEDED - Trading stopped!")
    
    if status['daily_loss_exceeded']:
        print(f"  ⚠️  DAILY LOSS LIMIT EXCEEDED!")

if __name__ == "__main__":
    print("🚀 Starting Paper Trading System")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Check signals (run this daily at market close ~3:30 PM)
            # For testing, run immediately
            check_signals_daily()
            print_portfolio_status()
            
            print("\n⏰ Waiting 24 hours before next check...")
            print("(In production, this would run once per day)")
            
            # Wait 24 hours (86400 seconds)
            # For testing, use 60 seconds = 1 minute
            time.sleep(60)  # Change to 86400 for daily checks
            
    except KeyboardInterrupt:
        print("\n\n👋 Stopping paper trader...")
        print_portfolio_status()
```

2. **Install schedule library** (optional for automatic daily runs):
```bash
pip install schedule
```

3. **Run paper trading**:
```bash
python paper_trading_runner.py
```

---

### **Week 4-12: Practice and Learn**

#### Daily Routine (15 minutes/day)

**Morning (9:00 AM):**
- Check portfolio status
- Review any overnight news
- Check if markets are open

**Evening (3:45 PM - after market close):**
- Run the signal checker
- Review any trades executed
- Note what happened in your trading journal

**Trading Journal Template**:
```
Date: 2024-XX-XX
Watchlist: RELIANCE.NS, TCS.NS, INFY.NS

Signals:
- RELIANCE: BUY at ₹2500
- TCS: HOLD
- INFY: SELL at ₹1500

Observations:
- Market was volatile today
- RELIANCE signal executed, quantity 20
- Learning: X indicator was accurate

Portfolio: ₹102,000 (up 2%)
```

#### Weekly Review (1 hour/week)

Every Sunday, review:
1. **What worked** - Which signals were profitable?
2. **What didn't** - Which signals lost money?
3. **Market conditions** - Bull market? Bear market? Trending? Sideways?
4. **Model accuracy** - How accurate were predictions?
5. **Risk management** - Did stop losses work as expected?

#### Monthly Analysis

Calculate these metrics:
- **Win Rate**: (Winning trades / Total trades) × 100
- **Average Win**: Average profit of winning trades
- **Average Loss**: Average loss of losing trades
- **Profit Factor**: Total wins / Total losses
- **Total Return**: (Current capital - Starting capital) / Starting capital

---

### **Month 4-6: Evaluation**

#### Decision Point:
After 3+ months of paper trading, evaluate:

**✅ Ready to go live IF:**
- Win rate > 50%
- Profit factor > 1.5
- Total return > 10%
- Max drawdown < 20%
- You understand why the model makes each decision
- You can explain the strategy to someone else

**❌ NOT ready IF:**
- Losing money consistently
- Don't understand how it works
- Don't have risk management discipline
- Can't stick to the plan

---

### **Month 7+: Going Live (If Ready)**

#### Start SMALL
1. **Initial Capital**: Start with ₹10,000-50,000 only
2. **Test for 1 month** with real money
3. **Compare** paper vs live results
4. **Scale up** slowly if successful (10% increase per month)

#### NEVER:
- ❌ Trade with rent money
- ❌ Trade with emergency funds  
- ❌ Risk more than you can afford to lose
- ❌ Skip paper trading
- ❌ Go "all in" on first trade

---

## 📚 Learning Resources

### Essential Reading:
1. **"Technical Analysis of Financial Markets"** by John Murphy
2. **"Algorithmic Trading"** by Ernest Chan
3. **"Risk Management"** by John Hull

### Online Courses:
- Coursera: Financial Markets (Yale University)
- edX: Trading and Investment courses
- Udemy: Technical Analysis courses

### Practice Sites:
- TradingView (free charts and indicators)
- Zerodha Varsity (free trading education)

---

## ⚠️ Important Disclaimers

### ✅ Backtest Status
The backtesting module has been **updated with fixes** for look-ahead bias, position tracking, and stop losses. See [BACKTEST_FIXES.md](BACKTEST_FIXES.md) for details. The fixed version is now integrated in the main application.

### This is NOT:
- ❌ A get-rich-quick scheme
- ❌ Guaranteed to make money
- ❌ Financial advice
- ❌ Suitable for everyone

### Trading Risks:
- 📉 You can lose money (potentially all of it)
- 📊 Past performance ≠ future results
- 🎲 ML models are not 100% accurate
- ⏰ Markets change constantly
- 🚨 System failures can happen

### Start ONLY IF:
- ✅ You understand the risks
- ✅ You can afford to lose what you invest
- ✅ You've paper traded successfully for 3+ months
- ✅ You have time to monitor the system
- ✅ You can follow the rules (especially stop losses)

---

## 🆘 Getting Help

### Common Issues:

**Issue**: "No signals being generated"
- **Solution**: Check if you have enough historical data (at least 1 year)

**Issue**: "Trades not executing"
- **Solution**: Check risk limits - you may have hit daily loss limit or max drawdown

**Issue**: "Unexpected losses"
- **Solution**: Review your stop losses and position sizing. Market conditions may have changed.

**Issue**: "Don't understand the indicators"
- **Solution**: Read [CHART_FEATURES.md](CHART_FEATURES.md) and study each indicator separately

---

## 📊 Success Metrics

Track these to measure progress:

### Technical Skills:
- [ ] Can identify technical patterns on charts
- [ ] Understand what RSI, MACD, Bollinger Bands mean
- [ ] Know when to trust vs ignore signals
- [ ] Can read and interpret backtest results

### Trading Discipline:
- [ ] Always follow stop losses
- [ ] Never risk more than planned
- [ ] Keep a trading journal
- [ ] Review performance regularly
- [ ] Stick to the system (don't override signals)

### Risk Management:
- [ ] Always use stop losses
- [ ] Limit position sizes
- [ ] Have max drawdown limits
- [ ] Never trade with money you need

---

**Remember**: This is a learning journey. Most successful traders take years to become profitable. Be patient, stay disciplined, and keep learning! 🌟
