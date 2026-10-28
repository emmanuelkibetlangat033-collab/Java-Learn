"""
Moving average crossover backtest (educational example, not a recommendation).

Setup:  pip install yfinance pandas matplotlib
Run:    python ma_backtest.py
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ---- Settings you can change ----
TICKER = "AAPL"
START = "2015-01-01"
FAST, SLOW = 20, 50
FEE = 0.001  # 0.1% cost per trade (commission + slippage estimate)

# ---- Data ----
df = yf.download(TICKER, start=START, auto_adjust=True)["Close"].squeeze().to_frame("price")

# ---- Signal: hold when fast average is above slow average ----
df["fast"] = df.price.rolling(FAST).mean()
df["slow"] = df.price.rolling(SLOW).mean()
# shift(1): act on yesterday's signal, which avoids look-ahead bias
df["position"] = (df.fast > df.slow).astype(int).shift(1).fillna(0)

# ---- Returns, with trading costs charged whenever the position changes ----
df["bh_ret"] = df.price.pct_change().fillna(0)
df["trade"] = df.position.diff().abs().fillna(0)
df["strat_ret"] = df.position * df.bh_ret - df.trade * FEE

df["strat_equity"] = (1 + df.strat_ret).cumprod()
df["bh_equity"] = (1 + df.bh_ret).cumprod()

# ---- Metrics ----
years = (df.index[-1] - df.index[0]).days / 365.25

def total_return(eq):
    return eq.iloc[-1] - 1

def cagr(eq):
    return eq.iloc[-1] ** (1 / years) - 1

def drawdown(eq):
    return eq / eq.cummax() - 1

summary = pd.DataFrame({
    "Total return": [total_return(df.strat_equity), total_return(df.bh_equity)],
    "Annual return (CAGR)": [cagr(df.strat_equity), cagr(df.bh_equity)],
    "Max drawdown": [drawdown(df.strat_equity).min(), drawdown(df.bh_equity).min()],
}, index=["Strategy", "Buy & hold"])

print(f"\n{TICKER}, {df.index[0].date()} to {df.index[-1].date()}")
print(f"Number of trades: {int(df.trade.sum())}")
print((summary * 100).round(1).astype(str) + "%")

# ---- Chart ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
df[["strat_equity", "bh_equity"]].plot(ax=ax1)
ax1.set_title(f"{TICKER}: growth of 1 unit of money")
ax1.legend(["Strategy", "Buy & hold"])

drawdown(df.strat_equity).plot(ax=ax2, label="Strategy")
drawdown(df.bh_equity).plot(ax=ax2, label="Buy & hold")
ax2.set_title("Drawdown (drop from previous peak)")
ax2.legend()

plt.tight_layout()
plt.savefig("backtest.png")
plt.show()
