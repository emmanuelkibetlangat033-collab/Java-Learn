"""
Test the moving average crossover on many tickers at once (educational example).

Setup:  pip install yfinance pandas
Run:    python multi_backtest.py
"""
import yfinance as yf
import pandas as pd

# ---- Settings you can change ----
TICKERS = ["AAPL", "MSFT", "GOOG", "XOM", "SPY", "BTC-USD"]
START = "2015-01-01"
FAST, SLOW = 20, 50
FEE = 0.001  # 0.1% cost per trade


def backtest(ticker):
    """Run the crossover strategy on one ticker and return a dict of results."""
    prices = yf.download(ticker, start=START, auto_adjust=True, progress=False)["Close"].squeeze()
    if prices.empty or len(prices) < SLOW + 10:
        return None  # no data or too little history

    fast = prices.rolling(FAST).mean()
    slow = prices.rolling(SLOW).mean()
    position = (fast > slow).astype(int).shift(1).fillna(0)  # shift avoids look-ahead

    bh_ret = prices.pct_change().fillna(0)
    trades = position.diff().abs().fillna(0)
    strat_ret = position * bh_ret - trades * FEE

    strat_eq = (1 + strat_ret).cumprod()
    bh_eq = (1 + bh_ret).cumprod()

    def max_dd(eq):
        return (eq / eq.cummax() - 1).min()

    return {
        "Strategy return %": (strat_eq.iloc[-1] - 1) * 100,
        "Buy&hold return %": (bh_eq.iloc[-1] - 1) * 100,
        "Strategy max DD %": max_dd(strat_eq) * 100,
        "Buy&hold max DD %": max_dd(bh_eq) * 100,
        "Trades": int(trades.sum()),
    }


results = {}
for t in TICKERS:
    try:
        r = backtest(t)
    except Exception as e:
        print(f"{t}: failed ({e})")
        continue
    if r is None:
        print(f"{t}: no usable data, skipped")
    else:
        results[t] = r

table = pd.DataFrame(results).T.round(1)
table["Beat B&H?"] = table["Strategy return %"] > table["Buy&hold return %"]
table["Smaller DD?"] = table["Strategy max DD %"] > table["Buy&hold max DD %"]  # less negative = smaller

print(f"\nMA({FAST}/{SLOW}) crossover, fee {FEE:.2%}, from {START}\n")
print(table.to_string())
print(f"\nBeat buy-and-hold on return: {table['Beat B&H?'].sum()} of {len(table)}")
print(f"Had a smaller max drawdown:  {table['Smaller DD?'].sum()} of {len(table)}")
