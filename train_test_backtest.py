"""
Train/test split backtest (educational example).

Idea: pick the best moving-average settings using ONLY the earlier "train"
period, then check how those same settings did on the later "test" period
that the choice never saw. This is a basic guard against overfitting.

Setup:  pip install yfinance pandas
Run:    python train_test_backtest.py
"""
import itertools
import yfinance as yf
import pandas as pd

# ---- Settings you can change ----
TICKERS = ["AAPL", "MSFT", "GOOG", "XOM", "SPY", "BTC-USD"]
START = "2015-01-01"
SPLIT = "2021-01-01"          # train: START to SPLIT, test: SPLIT onward
FASTS = [10, 20, 30]
SLOWS = [50, 100, 200]
FEE = 0.001                   # 0.1% cost per trade


def strategy_returns(prices, fast, slow):
    """Daily strategy returns after fees. Averages use past data only."""
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    position = (ma_fast > ma_slow).astype(int).shift(1).fillna(0)
    trades = position.diff().abs().fillna(0)
    return position * prices.pct_change().fillna(0) - trades * FEE


def total_return(daily_ret):
    return (1 + daily_ret).prod() - 1


def max_drawdown(daily_ret):
    eq = (1 + daily_ret).cumprod()
    return (eq / eq.cummax() - 1).min()


rows = {}
for t in TICKERS:
    try:
        prices = yf.download(t, start=START, auto_adjust=True, progress=False)["Close"].squeeze()
    except Exception as e:
        print(f"{t}: failed ({e})")
        continue
    train_mask = prices.index < SPLIT
    test_mask = ~train_mask
    if train_mask.sum() < max(SLOWS) + 50 or test_mask.sum() == 0:
        print(f"{t}: not enough history for this split, skipped")
        continue

    # 1) Choose settings using the TRAIN period only
    best, best_ret = None, -float("inf")
    for fast, slow in itertools.product(FASTS, SLOWS):
        if fast >= slow:
            continue
        r = total_return(strategy_returns(prices, fast, slow)[train_mask])
        if r > best_ret:
            best, best_ret = (fast, slow), r

    # 2) Evaluate those settings on the unseen TEST period
    strat = strategy_returns(prices, *best)[test_mask]
    hold = prices.pct_change().fillna(0)[test_mask]
    rows[t] = {
        "Chosen (fast/slow)": f"{best[0]}/{best[1]}",
        "Train return %": best_ret * 100,
        "Test strat %": total_return(strat) * 100,
        "Test B&H %": total_return(hold) * 100,
        "Test strat DD %": max_drawdown(strat) * 100,
        "Test B&H DD %": max_drawdown(hold) * 100,
    }

if not rows:
    raise SystemExit("No tickers produced results.")

table = pd.DataFrame(rows).T
num_cols = [c for c in table.columns if c != "Chosen (fast/slow)"]
table[num_cols] = table[num_cols].astype(float).round(1)

print(f"\nTrain: {START} to {SPLIT} | Test: {SPLIT} onward | fee {FEE:.2%}\n")
print(table.to_string())

beat = (table["Test strat %"] > table["Test B&H %"]).sum()
smaller = (table["Test strat DD %"] > table["Test B&H DD %"]).sum()
print("\nOn the unseen test period, the strategy:")
print(f"  beat buy-and-hold on return in {beat} of {len(table)} tickers")
print(f"  had a smaller max drawdown in {smaller} of {len(table)} tickers")
print("\nNote: a big gap between 'Train return' and 'Test strat' is a sign of overfitting.")
