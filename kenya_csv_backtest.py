"""
Train/test backtest on stocks from CSV files (e.g. Nairobi Securities Exchange
data exported from your broker). Educational example, not a recommendation.

Setup:   pip install pandas
Usage:   1. Put one CSV per stock in a folder, e.g. ./nse_data/SCOM.csv
         2. Set DATA_DIR below
         3. python kenya_csv_backtest.py

Each CSV needs a date column and a closing-price column. Common header names
are detected automatically (Date, Close, Price, Closing Price, Last ...).
"""
import glob
import itertools
import os

import pandas as pd

# ---- Settings you can change ----
DATA_DIR = "./nse_data"
SPLIT = "2021-01-01"          # train: before this date, test: from this date
FASTS = [10, 20, 30]
SLOWS = [50, 100, 200]
FEE = 0.02                    # 2% per trade. NSE costs are much higher than US
                              # brokers; check your broker's real rate and edit.

DATE_NAMES = ["date", "day", "trade date", "trading date"]
PRICE_NAMES = ["close", "closing price", "close price", "price", "last", "adj close"]


def load_prices(path):
    """Read a broker CSV into a clean, oldest-first price Series."""
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = next((cols[n] for n in DATE_NAMES if n in cols), None)
    price_col = next((cols[n] for n in PRICE_NAMES if n in cols), None)
    if date_col is None or price_col is None:
        raise ValueError(f"need a date and a close/price column, found {list(df.columns)}")

    raw_dates = df[date_col].astype(str).str.strip()
    if raw_dates.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() > 0.9:
        dates = pd.to_datetime(raw_dates, errors="coerce")               # 2021-03-15
    else:
        dates = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")  # 15/03/2021 (Kenyan style)
    # Remove thousands separators ("1,250.50") and stray text before converting
    price = pd.to_numeric(
        df[price_col].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    s = pd.Series(price.values, index=dates).dropna()
    s = s[(s.index.notna()) & (s > 0)]
    s = s[~s.index.duplicated(keep="last")].sort_index()  # broker files are often newest-first
    return s


def strategy_returns(prices, fast, slow):
    """Daily strategy returns after fees. Uses yesterday's signal (no look-ahead)."""
    position = (prices.rolling(fast).mean() > prices.rolling(slow).mean()).astype(int).shift(1).fillna(0)
    trades = position.diff().abs().fillna(0)
    return position * prices.pct_change().fillna(0) - trades * FEE


def total_return(r):
    return (1 + r).prod() - 1


def max_drawdown(r):
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()


files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not files:
    raise SystemExit(f"No CSV files found in {DATA_DIR}")

rows = {}
for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        prices = load_prices(path)
    except Exception as e:
        print(f"{name}: could not read ({e})")
        continue

    train_mask = prices.index < pd.Timestamp(SPLIT)
    test_mask = ~train_mask
    if train_mask.sum() < max(SLOWS) + 50 or test_mask.sum() < 30:
        print(f"{name}: not enough history ({len(prices)} rows) for this split, skipped")
        continue

    # Share of days where the price didn't move: a sign of thin trading
    flat_pct = (prices.pct_change().fillna(0) == 0).mean() * 100

    best, best_ret = None, -float("inf")
    for fast, slow in itertools.product(FASTS, SLOWS):
        if fast >= slow:
            continue
        r = total_return(strategy_returns(prices, fast, slow)[train_mask])
        if r > best_ret:
            best, best_ret = (fast, slow), r

    strat = strategy_returns(prices, *best)[test_mask]
    hold = prices.pct_change().fillna(0)[test_mask]
    rows[name] = {
        "Rows": len(prices),
        "Flat days %": flat_pct,
        "Chosen": f"{best[0]}/{best[1]}",
        "Train %": best_ret * 100,
        "Test strat %": total_return(strat) * 100,
        "Test B&H %": total_return(hold) * 100,
        "Strat DD %": max_drawdown(strat) * 100,
        "B&H DD %": max_drawdown(hold) * 100,
    }

if not rows:
    raise SystemExit("No stocks produced results.")

table = pd.DataFrame(rows).T
num = [c for c in table.columns if c != "Chosen"]
table[num] = table[num].astype(float).round(1)

print(f"\nTrain: before {SPLIT} | Test: from {SPLIT} | fee {FEE:.1%} per trade\n")
print(table.to_string())
print(f"\nBeat buy-and-hold on return: {(table['Test strat %'] > table['Test B&H %']).sum()} of {len(table)}")
print(f"Smaller max drawdown:        {(table['Strat DD %'] > table['B&H DD %']).sum()} of {len(table)}")
print("\nCaveats: prices are not adjusted for dividends, so buy-and-hold is understated for")
print("high-dividend stocks. High 'Flat days %' means thin trading; real trades may not fill at these prices.")
