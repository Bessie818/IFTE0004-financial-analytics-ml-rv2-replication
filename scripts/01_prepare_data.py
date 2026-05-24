from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TICKER = "^GSPC"
START_DATE = "1950-02-01"
END_DATE = "2018-01-01"


def main():
    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No data downloaded. Please check your internet connection.")

    if isinstance(df.columns, pd.MultiIndex):
        close_price = df["Close"][TICKER]
    else:
        close_price = df["Close"]

    daily_data = pd.DataFrame({"close_price": close_price}).dropna()

    daily_data["log_return"] = np.log(
        daily_data["close_price"] / daily_data["close_price"].shift(1)
    )
    daily_data = daily_data.dropna()

    monthly_rv = np.log(
        np.sqrt(daily_data["log_return"].pow(2).resample("ME").sum())
    )
    monthly_rv = monthly_rv.dropna()
    monthly_rv.name = "log_realized_volatility"

    daily_data.to_csv(DATA_DIR / "sp500_daily_data.csv")
    monthly_rv.to_csv(DATA_DIR / "sp500_monthly_realized_volatility.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(monthly_rv.index, monthly_rv.values)
    plt.title("S&P 500 Monthly Log Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Log Realized Volatility")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_realized_volatility.png", dpi=300)
    plt.close()

    print("Data preparation complete.")
    print(f"Daily observations: {len(daily_data)}")
    print(f"Monthly observations: {len(monthly_rv)}")
    print(f"Monthly sample: {monthly_rv.index.min().date()} to {monthly_rv.index.max().date()}")


if __name__ == "__main__":
    main()