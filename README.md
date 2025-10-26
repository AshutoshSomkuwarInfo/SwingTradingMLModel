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
