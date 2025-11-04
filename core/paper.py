import pandas as pd


def simulate_trades(prices: pd.Series, trades: list, initial_capital=100000, position_size=0.2):
    """Simple paper trading simulator.

    prices: pd.Series indexed by timestamps (close prices)
    trades: list of trade dicts with keys: Date (pd.Timestamp), Entry, Exit, Signal

    Returns:
        capital_history (pd.DataFrame) with Date index and Capital column
        executed_trades (list) enriched with pnl and capital after trade
    """
    capital = initial_capital
    portfolio = []
    executed = []

    # ensure trades are sorted by date
    trades_sorted = sorted(trades, key=lambda t: t.get("Date"))
    for t in trades_sorted:
        trade_capital = capital * position_size
        entry = t.get("Entry")
        exitp = t.get("Exit")
        if entry is None or exitp is None:
            continue
        gross = (exitp - entry) / entry if t.get("Signal") == "BUY" else (entry - exitp) / entry
        # simple; fees not modeled here (backtests already include costs)
        pnl = trade_capital * gross
        capital += pnl
        executed.append({**t, "PnL": pnl, "Capital": capital})
        portfolio.append({"Date": t.get("Date"), "Capital": capital})

    cap_df = pd.DataFrame(portfolio)
    if not cap_df.empty:
        cap_df = cap_df.set_index("Date")
    return cap_df, executed
