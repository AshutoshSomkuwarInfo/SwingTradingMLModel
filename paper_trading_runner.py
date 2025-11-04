"""
Paper Trading Runner - Beginner Friendly
Start with this to practice trading without real money!

Usage:
    python paper_trading_runner.py

This will:
1. Check for trading signals on your watchlist
2. Execute virtual trades based on ML predictions
3. Track performance and risk
4. Help you learn before risking real money
"""

from core.paper_trading import PaperTradingSystem
from core.model import predict_latest_signal
from core.data import get_stock_data
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - Modify these settings
# ═══════════════════════════════════════════════════════════════

# Starting virtual capital (INR)
INITIAL_CAPITAL = 100000  # Start with ₹1 lakh

# Your watchlist - stocks to monitor
# Add Indian stocks with .NS suffix (e.g., "RELIANCE.NS")
WATCHLIST = [
    "RELIANCE.NS",  # Reliance Industries
    "TCS.NS",       # Tata Consultancy Services
    "INFY.NS",      # Infosys
]

# Risk Management Settings
RISK_CONFIG = {
    'max_position_size_pct': 0.20,  # Max 20% of capital per stock
    'stop_loss_pct': 0.05,          # 5% stop loss (auto-sell if down 5%)
    'max_daily_loss_pct': 0.05,     # Stop trading if lose 5% in a day
    'max_drawdown_pct': 0.15,       # Stop trading if portfolio down 15%
    'risk_per_trade_pct': 0.02      # Risk only 2% of capital per trade
}

# ═══════════════════════════════════════════════════════════════
# END OF CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Initialize paper trader
print("🚀 Initializing Paper Trading System...")
print(f"💰 Starting Capital: ₹{INITIAL_CAPITAL:,}")
print(f"📊 Watchlist: {len(WATCHLIST)} stocks\n")

trader = PaperTradingSystem(
    initial_capital=INITIAL_CAPITAL,
    tickers=WATCHLIST,
    risk_config=RISK_CONFIG
)


def check_signals_daily():
    """Check for trading signals and execute trades"""
    print(f"\n{'='*60}")
    print(f"📡 Checking signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    for stock in WATCHLIST:
        try:
            # Display ticker without .NS for readability
            display_name = stock.replace('.NS', '')
            print(f"\n📊 Analyzing {display_name}...")
            
            # Get historical data
            data = get_stock_data(stock, period="1y")
            if data is None or data.empty:
                print(f"   ❌ No data available")
                continue
            
            # Get ML prediction
            signal = predict_latest_signal(data)
            print(f"   🤖 ML Signal: {signal}")
            
            # Execute trade based on signal
            result = trader.execute_trade(stock, signal)
            
            # Display results
            if result['status'] == 'executed':
                print(f"   ✅ {result['action']} executed!")
                print(f"   💵 Price: ₹{result.get('price', 'N/A'):,.2f}")
                if 'quantity' in result:
                    print(f"   📦 Quantity: {result['quantity']} shares")
                    print(f"   💰 Value: ₹{result.get('value', 0):,.2f}")
                    
            elif result['status'] == 'rejected':
                print(f"   ❌ Trade rejected: {result.get('reason', 'Unknown')}")
                
            elif result['status'] == 'no_action':
                print(f"   ⏸️  {result.get('reason', 'No action needed')}")
                
            else:
                print(f"   ⚠️  Status: {result.get('status', 'Unknown')}")
            
            # Check for stop losses
            stops = trader.check_stop_losses()
            for stop in stops:
                print(f"\n   🛑 STOP LOSS TRIGGERED!")
                print(f"   📉 {stop['ticker'].replace('.NS', '')} sold at ₹{stop['price']:,.2f}")
                print(f"   💸 Loss: ₹{stop.get('pnl', 0):,.2f}")
            
        except Exception as e:
            print(f"   ❌ Error processing {stock}: {str(e)}")


def print_portfolio_status():
    """Display current portfolio status"""
    status = trader.get_portfolio_status()
    
    print(f"\n{'='*60}")
    print("📊 PORTFOLIO STATUS")
    print(f"{'='*60}")
    
    # Capital information
    print(f"\n💰 Capital:")
    print(f"   Starting:  ₹{status['initial_capital']:,.2f}")
    print(f"   Current:   ₹{status['current_capital']:,.2f}")
    print(f"   Change:    ₹{status['total_pnl']:,.2f} ({status['total_return_pct']:+.2f}%)")
    
    # Performance metrics
    print(f"\n📈 Performance:")
    print(f"   Total Trades:    {status['total_trades']}")
    print(f"   Active Positions: {status['active_positions']}")
    print(f"   Drawdown:        {status['current_drawdown_pct']:.2f}%")
    
    # Risk warnings
    print(f"\n⚠️  Risk Status:")
    if status['max_drawdown_exceeded']:
        print(f"   🚨 MAX DRAWDOWN EXCEEDED - Trading HALTED!")
    else:
        print(f"   ✅ Drawdown within limits")
    
    if status['daily_loss_exceeded']:
        print(f"   🚨 DAILY LOSS LIMIT EXCEEDED - Trading stopped for today!")
    else:
        print(f"   ✅ Daily loss limit OK")
    
    print(f"{'='*60}\n")


def run_interactive_mode():
    """Interactive mode for manual signal checking"""
    print("\n🎮 Interactive Mode")
    print("Press Enter to check signals now, or 'q' to quit")
    
    while True:
        user_input = input("\n> ").strip().lower()
        
        if user_input == 'q':
            print("\n👋 Exiting...")
            print_portfolio_status()
            break
        elif user_input == '':
            check_signals_daily()
            print_portfolio_status()
        elif user_input == 'status':
            print_portfolio_status()
        else:
            print("❓ Invalid input. Press Enter to check signals, 'status' for portfolio, or 'q' to quit")


def run_automatic_mode():
    """Automatic mode - runs on schedule"""
    print("⏰ Automatic Mode")
    print("Checking signals every minute (for testing)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            check_signals_daily()
            print_portfolio_status()
            
            print("⏰ Next check in 60 seconds...")
            print("(Press Ctrl+C to stop)\n")
            time.sleep(60)  # Check every minute (for testing)
            
    except KeyboardInterrupt:
        print("\n\n👋 Stopping paper trader...")
        print_portfolio_status()


def main():
    """Main entry point"""
    print("╔" + "═"*58 + "╗")
    print("║" + " "*10 + "PAPER TRADING SYSTEM - BEGINNER MODE" + " "*11 + "║")
    print("╚" + "═"*58 + "╝")
    print("\n📚 This is a learning tool to practice trading without real money.")
    print("💰 You start with virtual capital and make virtual trades.")
    print("🎯 Goal: Learn how the system works before risking real money.\n")
    
    print("Choose mode:")
    print("  1. Interactive (check signals when you press Enter)")
    print("  2. Automatic (check signals every minute)")
    print("  3. Exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        run_interactive_mode()
    elif choice == '2':
        run_automatic_mode()
    elif choice == '3':
        print("\n👋 Goodbye! Come back when ready to practice.")
    else:
        print("\n❌ Invalid choice. Exiting.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check the error message above and try again.")




