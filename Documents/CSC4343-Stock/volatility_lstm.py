"""
volatility_lstm.py
==================
LSTM pipeline for forecasting realized volatility.

Integrates:
- David's StockDataLoader (loading 1-min OHLCV data)
- David's MultiTimeFrameFeatures (resampling utility)
- Landin's DataPreprocessApril19 logic (returns, lags, FRED macros, vol target)
- Nathan's LSTM architecture (from lstm_ohlcv.py)

Target: future_volatility — the 21-day forward rolling std of daily log returns.
        This is what we actually want to forecast per the project proposal.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# David's modules
from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures

# =============================================================
# CONFIGURATION
# =============================================================
DATA_PATH      = "../OHLC 1 minute data/extracted_files"
TICKERS_FILE   = "s&p500tickers.txt"
START_DATE     = "2000-01-01"   # FRED series start reliably around 2000
END_DATE       = "2026-02-28"

VOLATILITY_WINDOW = 21   # rolling window for realized vol (trading days ~1 month)
RETURN_LAGS       = 60   # number of past daily returns fed as features (~3 months)

SEQ_LEN     = 60    # LSTM look-back window in daily bars
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 0.001
HIDDEN      = 64
NUM_LAYERS  = 2
TRAIN_SPLIT = 0.8
RESULTS_DIR = "results_vol"
SEED        = 42

# Minimum bars needed so both train and val sets have at least SEQ_LEN rows
MIN_BARS = int(SEQ_LEN / (1 - TRAIN_SPLIT)) + SEQ_LEN  # = 360 at defaults


# =============================================================
# REPRODUCIBILITY
# =============================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================
# 1. FRED MACRO DATA  (loaded once, shared across all tickers)
# =============================================================
def load_fred_data(start: str, end: str) -> pd.DataFrame:
    """
    Pull four macro series from FRED and return a daily DataFrame.
    Each series is shifted by 1 day to prevent look-ahead bias —
    today's model only sees yesterday's published macro figures.

    Series used (from Landin's DataPreprocessApril19.py):
      T5YIE      — 5-year breakeven inflation expectation
      DGS2       — 2-year US Treasury yield
      DGS10      — 10-year US Treasury yield
      USEPUINDXD — Daily Economic Policy Uncertainty Index
    """
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
        frames[col_name] = raw[fred_code].shift(1)  # 1-day lag for no look-ahead

    return pd.DataFrame(frames)


# =============================================================
# 2. VOLATILITY FEATURE BUILDER  (one ticker at a time)
# =============================================================
def build_volatility_df(
    raw_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    volatility_window: int = VOLATILITY_WINDOW,
    return_lags: int = RETURN_LAGS,
) -> pd.DataFrame:
    """
    Convert raw 1-minute OHLCV data into a daily feature DataFrame
    with future_volatility as the prediction target.

    Pipeline
    --------
    1. Resample 1-min bars → daily OHLCV
    2. Compute daily log returns
    3. Add `return_lags` lagged return columns as features
    4. Merge FRED macro series (forward-filled across non-trading days)
    5. Compute rolling realized volatility
    6. Shift volatility forward by `volatility_window` days → future_volatility
    7. Drop rows with any NaN values

    Parameters
    ----------
    raw_df           : 1-min OHLCV DataFrame from StockDataLoader.load1min()
    fred_df          : macro DataFrame from load_fred_data()
    volatility_window: rolling std window AND forward-shift amount (trading days)
    return_lags      : number of lagged daily return columns to create

    Returns
    -------
    Daily DataFrame ready for train/val split and scaling.
    Columns: volume, return, return_1…return_N, macro features,
             volatility, future_volatility (target — last column)
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Step 1 — resample 1-min bars to daily
    daily = raw_df.resample("1D").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    if daily.empty:
        return pd.DataFrame()

    # Step 2 — log return (more stationary and standard for financial data)
    daily["return"] = np.log(daily["close"] / daily["close"].shift(1))

    # Drop raw price levels — we keep volume as a feature
    daily = daily.drop(columns=["open", "high", "low", "close"])

    # Step 3 — lagged return features  (return_1 = yesterday, return_2 = two days ago…)
    for i in range(1, return_lags + 1):
        daily[f"return_{i}"] = daily["return"].shift(i)

    # Step 4 — merge FRED macro data
    # Strip timezone from index so it aligns cleanly with FRED's tz-naive dates
    daily.index = daily.index.tz_localize(None)
    daily = daily.join(fred_df, how="left")
    # Forward-fill gaps (weekends, holidays) then back-fill any leading NaN
    daily[fred_df.columns] = daily[fred_df.columns].ffill().bfill()

    # Step 5 — rolling realized volatility (std of log returns over window)
    daily["volatility"] = daily["return"].rolling(window=volatility_window).std()

    # Step 6 — forward-shift to create the TARGET: what vol will be over NEXT window
    # Shifting by -volatility_window means at row t, future_volatility =
    # the realized vol computed over bars [t+1 … t+volatility_window]
    daily["future_volatility"] = daily["volatility"].shift(-volatility_window)

    # Step 7 — drop NaN rows (from lags, rolling window, and the forward shift)
    daily = daily.dropna()

    return daily


# =============================================================
# 3. TICKER LIST
# =============================================================
def load_tickers(filepath: str) -> list:
    with open(filepath, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        raise ValueError(f"No tickers found in '{filepath}'")
    return tickers


# =============================================================
# 4. SCALING
# =============================================================
def scale_data(train_df: pd.DataFrame, *other_dfs: pd.DataFrame) -> tuple:
    """
    Fit MinMaxScaler on train_df ONLY, then transform all splits.
    Fitting only on train data prevents look-ahead / data leakage.
    """
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df.values.astype(np.float32))
    scaled_rest  = [scaler.transform(df.values.astype(np.float32)) for df in other_dfs]
    return scaler, scaled_train, *scaled_rest


# =============================================================
# 5. WINDOWED DATASET  — target is future_volatility
# =============================================================
class VolatilityDataset(Dataset):
    """
    Sliding-window PyTorch Dataset.

    Each sample:
      X : (seq_len, num_features) — the look-back window of daily features
      y : scalar                  — the scaled future_volatility at bar seq_len+1

    The target column is identified by name so any upstream column-order
    change raises an error immediately instead of silently training on the
    wrong variable.
    """
    TARGET_COL = "future_volatility"

    def __init__(self, data: np.ndarray, columns: list, seq_len: int = SEQ_LEN):
        if self.TARGET_COL not in columns:
            raise ValueError(
                f"'{self.TARGET_COL}' not found in columns. Got: {columns}"
            )
        self.data       = data
        self.seq_len    = seq_len
        self.target_idx = list(columns).index(self.TARGET_COL)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        X = self.data[idx : idx + self.seq_len]              # (seq_len, features)
        y = self.data[idx + self.seq_len, self.target_idx]   # future vol scalar
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# =============================================================
# 6. LSTM MODEL  (same architecture as lstm_ohlcv.py)
# =============================================================
class LSTMModel(nn.Module):
    """
    Two-layer LSTM with a fully-connected head predicting a single scalar
    (the next period's realized volatility).
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = HIDDEN,
        num_layers:  int   = NUM_LAYERS,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])   # last time-step → (batch, 1)
        return out.squeeze(-1)         # (batch,)


# =============================================================
# 7. TRAINING & EVALUATION LOOPS
# =============================================================
def train_one_epoch(
    model, loader, criterion, optimizer, device, max_grad_norm: float = 1.0
) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total += criterion(model(X), y).item()
    return total / len(loader)


def run_training(
    model, train_loader, val_loader, device,
    epochs: int = EPOCHS, lr: float = LR, save_path: str = ""
) -> float:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val  = float("inf")

    for epoch in range(1, epochs + 1):
        tr  = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)
        saved = ""
        if val < best_val:
            best_val = val
            if save_path:
                torch.save(model.state_dict(), save_path)
            saved = " ✓ saved"
        print(
            f"  Epoch {epoch:>2}/{epochs} | "
            f"Train: {tr:.6f} | Val: {val:.6f}{saved}"
        )
    return best_val


# =============================================================
# 8. SINGLE-TICKER END-TO-END PIPELINE
# =============================================================
def run_ticker(ticker: str, device, mtf: MultiTimeFrameFeatures, fred_df: pd.DataFrame) -> dict:
    result = dict(
        ticker=ticker, status="ok", best_val_loss=None,
        bars=0, num_features=0, skipped_reason=""
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # 1 — Load raw 1-min data via David's loader
        raw_df = mtf.loader.load1min(ticker=ticker, start=START_DATE, end=END_DATE)
        if raw_df.empty:
            result.update(status="skipped", skipped_reason="no raw data found")
            return result

        # 2 — Build volatility feature DataFrame (Landin's preprocessing)
        df = build_volatility_df(raw_df, fred_df)
        if df.empty:
            result.update(status="skipped", skipped_reason="empty after preprocessing")
            return result
        if len(df) < MIN_BARS:
            result.update(
                status="skipped",
                skipped_reason=f"only {len(df)} bars after processing (need ≥ {MIN_BARS})"
            )
            return result

        result["bars"] = len(df)

        # 3 — Chronological train / val split (no shuffling — time series!)
        split    = int(len(df) * TRAIN_SPLIT)
        train_df = df.iloc[:split]
        val_df   = df.iloc[split:]

        if len(train_df) <= SEQ_LEN or len(val_df) <= SEQ_LEN:
            result.update(status="skipped", skipped_reason="insufficient bars after split")
            return result

        # 4 — Scale (fit on train only to prevent data leakage)
        columns = list(df.columns)
        scaler, train_data, val_data = scale_data(train_df, val_df)

        # Drop any zero-variance columns (would produce NaN after MinMaxScaler)
        nan_mask = np.isnan(train_data).any(axis=0)
        if nan_mask.any():
            bad = [c for c, m in zip(columns, nan_mask) if m]
            print(f"  Dropping {len(bad)} zero-variance column(s): {bad}")
            train_data = train_data[:, ~nan_mask]
            val_data   = val_data[:,   ~nan_mask]
            columns    = [c for c, m in zip(columns, nan_mask) if not m]

        result["num_features"] = train_data.shape[1]

        # 5 — PyTorch Datasets and DataLoaders
        train_ds = VolatilityDataset(train_data, columns, SEQ_LEN)
        val_ds   = VolatilityDataset(val_data,   columns, SEQ_LEN)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=(device.type == "cuda")
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda")
        )

        # 6 — Build and train LSTM
        model     = LSTMModel(input_size=train_data.shape[1]).to(device)
        save_path = os.path.join(RESULTS_DIR, f"{ticker}.pt")
        best_val  = run_training(model, train_loader, val_loader, device, save_path=save_path)
        result["best_val_loss"] = best_val

        # 7 — Persist scaler for later inference / inverse-transform
        joblib.dump(scaler, os.path.join(RESULTS_DIR, f"{ticker}_scaler.pkl"))

    except Exception as e:
        result.update(status="skipped", skipped_reason=f"unexpected error: {e}")

    return result


# =============================================================
# 9. MAIN — MULTI-TICKER LOOP
# =============================================================
if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device            : {device}")
    print(f"Target            : future_volatility ({VOLATILITY_WINDOW}-day forward realized vol)")
    print(f"Seq length        : {SEQ_LEN} daily bars")
    print(f"Return lag features: {RETURN_LAGS}")
    print(f"Results directory : {RESULTS_DIR}\n")

    # Load FRED macro data ONCE — reused for every ticker
    print("Fetching FRED macro data (this takes a few seconds)...")
    fred_df = load_fred_data(START_DATE, END_DATE)
    print(f"FRED loaded: {fred_df.shape[0]} rows | columns: {list(fred_df.columns)}\n")

    loader  = StockDataLoader(base_path=DATA_PATH)
    mtf     = MultiTimeFrameFeatures(loader=loader)
    tickers = load_tickers(TICKERS_FILE)
    print(f"Tickers loaded: {len(tickers)}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>3}/{len(tickers)}] {ticker}")
        res = run_ticker(ticker, device, mtf, fred_df)
        if res["status"] == "skipped":
            print(f"  SKIPPED — {res['skipped_reason']}\n")
        else:
            print(
                f"  bars={res['bars']:,} | "
                f"features={res['num_features']} | "
                f"best_val_loss={res['best_val_loss']:.6f}\n"
            )
        summary.append(res)

    # Save run summary
    summary_df  = pd.DataFrame(summary)
    report_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary_df.to_csv(report_path, index=False)

    ok      = summary_df[summary_df.status == "ok"]
    skipped = summary_df[summary_df.status == "skipped"]

    print(f"\n{'='*55}")
    print(f"  Completed  : {len(ok)} tickers")
    print(f"  Skipped    : {len(skipped)} tickers")
    if not ok.empty:
        print(f"  Avg val loss : {ok.best_val_loss.mean():.6f}")
    print(f"  Models  → {RESULTS_DIR}/<ticker>.pt")
    print(f"  Scalers → {RESULTS_DIR}/<ticker>_scaler.pkl")
    print(f"  Report  → {report_path}")
    print(f"{'='*55}")
