import os
import gc
import joblib
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from pathlib import Path

# David's modules
from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures


# =============================================================
# CONFIGURATION
# =============================================================
DATA_PATH      = "OHLC 1 minute data/extracted_files"
TICKERS_FILE   = "s&p500tickers.txt"
START_DATE     = "2000-01-01"   # FRED series start reliably around 2000
END_DATE       = "2026-02-28"

VOLATILITY_WINDOW = 21   # rolling window for realized vol (trading days ~1 month)
RETURN_LAGS       = 60   # number of past daily returns fed as features (~3 months)

SEQ_LEN     = 60    # LSTM look-back window in daily bars
BATCH_SIZE  = 256
EPOCHS      = 25
LR          = 0.001
HIDDEN      = 64
NUM_LAYERS  = 2
TRAIN_SPLIT = 0.8
RESULTS_DIR = "results_vol"
SEED        = 4343

# Minimum bars needed so both train and val sets have at least SEQ_LEN rows
MIN_BARS = int(SEQ_LEN / (1 - TRAIN_SPLIT)) + SEQ_LEN  # = 360 at defaults


# For reproducibility
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Federal Reserve Economic Data loader
def load_fred_data(start: str, end: str) -> pd.DataFrame:
    """
    Pull four macro series from FRED and return a daily DataFrame.
    Each series (usually published with a lag) is shifted by 1 day to prevent look-ahead bias 

    Series used:
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


# Build dataset of daily features and returns, including past and future realized volatility
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


# Load list of tickers from a text file containing one ticker per line
def load_tickers(filepath: str) -> list:
    with open(filepath, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        raise ValueError(f"No tickers found in '{filepath}'")
    return tickers


# Fit a MinMaxScaler on the train split of one ticker, then transform all splits of that ticker
def scale_data(train_df: pd.DataFrame, *other_dfs: pd.DataFrame) -> tuple:
    """
    Fit MinMaxScaler on train_df ONLY, then transform all splits.
    Fitting only on train data prevents look-ahead / data leakage.
    """
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df.values.astype(np.float32))
    scaled_rest  = [scaler.transform(df.values.astype(np.float32)) for df in other_dfs]
    return scaler, scaled_train, *scaled_rest


# Create windowed PyTorch Dataset for one ticker
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


# LSTM model architecture
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


# Train LSTM on one epoch of data
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

# Train LSTM over all epochs, optionally saving the model that performs best on validation set
def run_training(
    model, train_loader, val_loader, device,
    epochs: int = EPOCHS, lr: float = LR, save_path: str = ""
) -> float:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val  = float("inf")

    for epoch in range(1, epochs + 1):

        print(f"Epoch {epoch:>2}/{epochs} — Training...", end="")
        tr  = train_one_epoch(model, train_loader, criterion, optimizer, device)

        print(f"Evaluating {epoch:>2}/{epochs}...", end="")
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


# Calculate r^2 and optionally save predicted/actual pairs to csv
@torch.no_grad()
def calculate_r2(
    model, loader, scaler, columns: list, ticker: str, device, results_dir: str = RESULTS_DIR, save_to_csv=False
) -> float:
    """
    Run inference on a DataLoader, inverse-transform predictions and actuals
    back to the original volatility scale, save them to a CSV, and return R².
    """
    model.eval()
    target_idx = list(columns).index(VolatilityDataset.TARGET_COL)
    n_features  = scaler.n_features_in_

    all_preds  = []
    all_actual = []

    for X, y in loader:
        X = X.to(device)
        preds = model(X).cpu().numpy()   # (batch,) — scaled
        all_preds.append(preds)
        all_actual.append(y.numpy())

    all_preds  = np.concatenate(all_preds)
    all_actual = np.concatenate(all_actual)

    def inverse_target(scaled_values: np.ndarray) -> np.ndarray:
        """Place values in target column of a zero dummy, inverse-transform, extract."""
        dummy = np.zeros((len(scaled_values), n_features), dtype=np.float32)
        dummy[:, target_idx] = scaled_values
        return scaler.inverse_transform(dummy)[:, target_idx]

    preds_real  = inverse_target(all_preds)
    actual_real = inverse_target(all_actual)

    r2 = r2_score(actual_real, preds_real)

    if save_to_csv:
        out = pd.DataFrame({
            "actual_volatility":    actual_real,
            "predicted_volatility": preds_real,
        })
        out_path = os.path.join(results_dir, f"{ticker}_predictions.csv")
        out.to_csv(out_path, index=False)
        print(f"  Predictions saved → {out_path}  |  R²: {r2:.4f}")

    return r2


# Helps prepare one ticker's data for use
def prepare_ticker_data(ticker: str, mtf: MultiTimeFrameFeatures, fred_df: pd.DataFrame):
    """Loads, processes, and chronologically splits data for a single ticker."""
    try:
        raw_df = mtf.loader.load1min(ticker=ticker, start=START_DATE, end=END_DATE)
        if raw_df.empty: 
            return None, None
        
        df = build_volatility_df(raw_df, fred_df)
        if df.empty or len(df) < MIN_BARS: 
            return None, None

        split = int(len(df) * TRAIN_SPLIT)
        return df.iloc[:split], df.iloc[split:]
    except Exception as e:
        print(f"  Skipping {ticker} due to error: {e}")
        return None, None


# Global multi-ticker loop
if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Target : future_volatility ({VOLATILITY_WINDOW}-day forward realized vol)")
    
    fred_df = load_fred_data(START_DATE, END_DATE)
    loader  = StockDataLoader(base_path=DATA_PATH)
    mtf     = MultiTimeFrameFeatures(loader=loader)
    tickers = load_tickers(TICKERS_FILE)
    
    PROCESSED_DIR = os.path.join(RESULTS_DIR, "processed_daily_temp")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Process incrementally, partial_fit scaler, free memory
    print("\nStep 1: Processing Raw Data & Fitting Global Scaler")
    scaler = MinMaxScaler()
    valid_tickers = []
    columns = None

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Extracting & processing {ticker}...")

        if not Path(os.path.join(PROCESSED_DIR, f"{ticker}_train.pkl")).is_file(): #skip tickers which were already processed 
            tr_df, vl_df = prepare_ticker_data(ticker, mtf, fred_df)

            if tr_df is not None:
                # Save the tiny daily dataframes to disk temporarily
                tr_df.to_pickle(os.path.join(PROCESSED_DIR, f"{ticker}_train.pkl"))
                vl_df.to_pickle(os.path.join(PROCESSED_DIR, f"{ticker}_val.pkl"))

                # Incrementally fit scaler ON TRAIN DATA ONLY to avoid leakage
                scaler.partial_fit(tr_df.values.astype(np.float32))
            
                if columns is None:
                    columns = list(tr_df.columns)
                valid_tickers.append(ticker)

            # Force Python to drop the raw data overhead for memory efficiency
            del tr_df, vl_df
            if hasattr(loader, 'cache'): 
                loader.cache.clear() 
            gc.collect() 

        else: # for tickers with available data
            valid_tickers.append(ticker)
            # Incrementally fit scaler on existing train split if it wasn't done in a previous run
            if columns is None:
                columns = list(pd.read_pickle(os.path.join(PROCESSED_DIR, f"{ticker}_train.pkl")).columns)
            scaler.partial_fit(pd.read_pickle(os.path.join(PROCESSED_DIR, f"{ticker}_train.pkl")).values.astype(np.float32))

    if not valid_tickers:
        raise RuntimeError("No valid data found for any ticker.")

    # Step 2: Load all the transformed data into RAM for training
    print("\nStep 2: Building Datasets & Training")
    train_datasets, val_datasets = [], []

    for ticker in valid_tickers:
        # Load the small daily files
        tr_df = pd.read_pickle(os.path.join(PROCESSED_DIR, f"{ticker}_train.pkl"))
        vl_df = pd.read_pickle(os.path.join(PROCESSED_DIR, f"{ticker}_val.pkl"))

        # Scale
        t_scaled = scaler.transform(tr_df.values.astype(np.float32))
        v_scaled = scaler.transform(vl_df.values.astype(np.float32))

        train_datasets.append(VolatilityDataset(t_scaled, columns, SEQ_LEN))
        val_datasets.append(VolatilityDataset(v_scaled, columns, SEQ_LEN))

    # Concatenate all the data into one location
    global_train_ds = ConcatDataset(train_datasets) 
    global_val_ds   = ConcatDataset(val_datasets)

    train_loader = DataLoader(global_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True)
    val_loader   = DataLoader(global_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True)

    print(f"Training global model: {len(global_train_ds)} Train sequences, {len(global_val_ds)} Val sequences")
    model = LSTMModel(input_size=len(columns)).to(device)
    save_path = os.path.join(RESULTS_DIR, "global_volatility_model.pt")

    best_val = run_training(model, train_loader, val_loader, device, save_path=save_path)

    # Step 3: Evaluation
    print("\n--- PHASE 3: Evaluation ---")
    model.load_state_dict(torch.load(save_path, map_location=device))
    r2 = calculate_r2(model, val_loader, scaler, columns, "global_all_tickers", device, save_to_csv=True)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "global_scaler.pkl"))

    print(f"\n{'='*55}")
    print(f"Completed : {len(valid_tickers)} tickers merged")
    print(f"Best Val Loss : {best_val:.6f}")
    print(f"Global R² : {r2:.4f}")
    print(f"Model -> {save_path}")
    print(f"Scaler -> {RESULTS_DIR}/global_scaler.pkl")
    print(f"Preds -> {RESULTS_DIR}/global_all_tickers_predictions.csv")
    print(f"{'='*55}")