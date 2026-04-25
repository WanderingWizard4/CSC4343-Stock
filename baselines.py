"""
baselines.py
============
Simple baseline models for volatility forecasting.
These give us a benchmark to compare the LSTM against in the final paper.
If the LSTM can't beat these, it's not useful.

Two baselines:
  1. Naive Persistence  — predict tomorrow's vol = today's vol
  2. Historical Mean    — predict tomorrow's vol = rolling 21-day average vol

USAGE
-----
Run on a single ticker to see baseline metrics:
    python baselines.py

Metrics reported:
  MSE  — Mean Squared Error
  MAE  — Mean Absolute Error
  RMSE — Root Mean Squared Error
  DA   — Directional Accuracy (did we predict vol going up vs down correctly?)
"""

import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pathlib import Path

from stock_data_loader import StockDataLoader

# =============================================================
# CONFIGURATION — match these to volatility_lstm.py
# =============================================================
DATA_PATH         = "../OHLC 1 minute data/extracted_files"
START_DATE        = "2000-01-01"
END_DATE          = "2026-02-28"
VOLATILITY_WINDOW = 21   # must match volatility_lstm.py
TRAIN_SPLIT       = 0.8  # same split so we evaluate on the same test window
ROLLING_MEAN_WINDOW = 21 # window for the historical mean baseline


# =============================================================
# REUSE: build daily vol DataFrame (mirrors volatility_lstm.py)
# =============================================================
def load_fred_data(start: str, end: str) -> pd.DataFrame:
    series_map = {
        "inflation_expectation": "T5YIE",
        "two_year_treasury":     "DGS2",
        "ten_year_treasury":     "DGS10",
        "economic_uncertainty":  "USEPUINDXD",
    }
    frames = {}
    for col_name, fred_code in series_map.items():
        raw = pdr.get_data_fred(fred_code, start=start, end=end)
        raw.index = pd.to_datetime(raw.index)
        frames[col_name] = raw[fred_code].shift(1)
    return pd.DataFrame(frames)


def build_daily_vol(raw_df: pd.DataFrame, vol_window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    """Resample to daily and compute realized vol + future_volatility target."""
    if raw_df.empty:
        return pd.DataFrame()

    daily = raw_df.resample("1D").agg({
        "open": "first", "high": "max",
        "low": "min",    "close": "last", "volume": "sum"
    }).dropna()

    if daily.empty:
        return pd.DataFrame()

    daily.index = daily.index.tz_localize(None)
    daily["return"]   = np.log(daily["close"] / daily["close"].shift(1))
    daily["volatility"] = daily["return"].rolling(window=vol_window).std()
    daily["future_volatility"] = daily["volatility"].shift(-vol_window)
    daily = daily.dropna()
    return daily[["volatility", "future_volatility"]]


# =============================================================
# METRICS
# =============================================================
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MSE, MAE, RMSE, and Directional Accuracy."""
    mse  = np.mean((actual - predicted) ** 2)
    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(mse)

    # Directional accuracy: did we correctly predict whether vol went up or down?
    actual_dir    = np.diff(actual) > 0
    predicted_dir = np.diff(predicted) > 0
    da = np.mean(actual_dir == predicted_dir) * 100  # as percentage

    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "DA (%)": da}


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n  {name}")
    print(f"    MSE  : {metrics['MSE']:.8f}")
    print(f"    MAE  : {metrics['MAE']:.8f}")
    print(f"    RMSE : {metrics['RMSE']:.8f}")
    print(f"    DA   : {metrics['DA (%)']:.2f}%")


# =============================================================
# BASELINE 1: Naive Persistence
# Predict: future_vol(t) = volatility(t)  (today's vol = tomorrow's vol)
# =============================================================
def naive_persistence(df: pd.DataFrame) -> np.ndarray:
    """Predict next period's volatility = current period's volatility."""
    return df["volatility"].values


# =============================================================
# BASELINE 2: Historical Rolling Mean
# Predict: future_vol(t) = mean of last N days of volatility
# =============================================================
def historical_mean(df: pd.DataFrame, window: int = ROLLING_MEAN_WINDOW) -> np.ndarray:
    """Predict next period's volatility = rolling mean of past N days."""
    return df["volatility"].rolling(window=window).mean().values


# =============================================================
# RUN BASELINES ON ONE TICKER
# =============================================================
def evaluate_baselines(ticker: str) -> None:
    print(f"\nLoading data for {ticker}...")
    loader = StockDataLoader(base_path=DATA_PATH)
    raw_df = loader.load1min(ticker=ticker, start=START_DATE, end=END_DATE)

    if raw_df.empty:
        print(f"No data found for {ticker}. Check your data path.")
        return

    df = build_daily_vol(raw_df)
    if df.empty or len(df) < 100:
        print(f"Not enough data for {ticker} after processing.")
        return

    # Use the same test split as volatility_lstm.py
    split   = int(len(df) * TRAIN_SPLIT)
    test_df = df.iloc[split:].copy()

    print(f"  Total bars : {len(df)}")
    print(f"  Test bars  : {len(test_df)}")
    print(f"  Test period: {test_df.index[0].date()} → {test_df.index[-1].date()}")

    actual = test_df["future_volatility"].values

    # --- Baseline 1: Naive Persistence ---
    persistence_preds = naive_persistence(test_df)
    valid = ~np.isnan(persistence_preds) & ~np.isnan(actual)
    p_metrics = compute_metrics(actual[valid], persistence_preds[valid])
    print_metrics("Baseline 1: Naive Persistence", p_metrics)

    # --- Baseline 2: Historical Rolling Mean ---
    mean_preds = historical_mean(test_df)
    valid = ~np.isnan(mean_preds) & ~np.isnan(actual)
    m_metrics = compute_metrics(actual[valid], mean_preds[valid])
    print_metrics(f"Baseline 2: Historical {ROLLING_MEAN_WINDOW}-Day Rolling Mean", m_metrics)

    print("\n  These are the numbers your LSTM needs to beat.")
    print("  Save them — they go in the Results section of the final paper.\n")

    # Save results to CSV for easy copy-paste into the paper
    results = pd.DataFrame({
        "Model": ["Naive Persistence", f"Historical {ROLLING_MEAN_WINDOW}-Day Mean"],
        "MSE":   [p_metrics["MSE"],  m_metrics["MSE"]],
        "MAE":   [p_metrics["MAE"],  m_metrics["MAE"]],
        "RMSE":  [p_metrics["RMSE"], m_metrics["RMSE"]],
        "DA (%)": [p_metrics["DA (%)"], m_metrics["DA (%)"]],
    })
    out_path = f"baseline_results_{ticker}.csv"
    results.to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    # Test on AAPL first — change to any ticker you have data for
    evaluate_baselines("AAPL")
