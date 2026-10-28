# Trading AI tests

Educational moving-average crossover backtests. Not financial advice. Uses
historical data only, and past results do not predict future results.

## Files
- `ma_backtest.py`: one stock, with fees, drawdown and a chart
- `multi_backtest.py`: several stocks in one table
- `train_test_backtest.py`: picks settings on early data, tests on later data
- `kenya_csv_backtest.py`: same test on CSV files from your broker (put them in `nse_data/`)
- `spreadsheets/`: paper-trading tracker and Treasury bill vs money market fund comparison

## Run on GitHub
1. Open the **Actions** tab, choose **Run backtests**, then **Run workflow**.
2. When it finishes, open the run and download **backtest-results** at the bottom.

## Run on your own computer
    pip install -r requirements.txt
    python multi_backtest.py

Change the settings (tickers, dates, fee) at the top of each script.
