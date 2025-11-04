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
   

## 🎯 How It Works

1. **Data Collection**: Downloads historical stock data using yfinance
2. **Feature Engineering**: Calculates multiple technical indicators
3. **Model Training**: Trains XGBoost classifier on historical data
4. **Signal Generation**: Predicts BUY/SELL/HOLD signals for 15-day swings
5. **Backtesting**: Validates strategy performance on historical data
6. **Metrics**: Calculates risk-adjusted returns and trade statistics

---

## 👨‍🎓 Beginner's Guide - Step by Step

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
