import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# David' modules
from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures
from rolling_features import RollingFeatures


# =============================================================
# CONFIGURATION
# =============================================================

DATA_PATH    = "../OHLC 1 minute data/extracted_files"  # root folder for local CSV data
TICKERS_FILE = "s&p500tickers.txt"                   # one ticker per line
START_DATE   = "1992-01-01"
END_DATE     = "2026-02-28"

# Which timeframe to feed into the LSTM.
# Must match a key produced by MultiTimeFrameFeatures:
#   "5min", "15min", "30min", "1hour", "1week", "1month"
# MultiTimeFrameFeatures always computes all timeframes internally.
# Only PRIMARY_TF is passed to RollingFeatures and the LSTM.
PRIMARY_TF   = "1hour"

SEQ_LEN      = 60      # look-back window in bars  (60 × 1h = 60 hours of context)
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
HIDDEN       = 64
NUM_LAYERS   = 2
TRAIN_SPLIT  = 0.8
RESULTS_DIR  = "results"
SEED         = 42

# Minimum bars required AFTER feature engineering and the train/val split.
# Both train AND val must each have at least SEQ_LEN bars to form one window.
# Formula:  total × (1 − TRAIN_SPLIT) > SEQ_LEN
#       =>  total > SEQ_LEN / (1 − TRAIN_SPLIT)
# We add one extra SEQ_LEN buffer for indicator warm-up rows that get dropped.
MIN_BARS = int(SEQ_LEN / (1 - TRAIN_SPLIT)) + SEQ_LEN   # = 360 at defaults



# REPRODUCIBILITY

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False



# 1. TICKER LIST

def load_tickers(filepath: str) -> list:
    """
    Read ticker symbols from a plain-text file (one ticker per line).
    Skips blank lines and strips whitespace.
    """
    with open(filepath, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        raise ValueError(f"No tickers found in '{filepath}'")
    return tickers



# 2. DATA PIPELINE
#    MultiTimeFrameFeatures  →  RollingFeatures  →  PRIMARY_TF

def build_feature_df(
    ticker:     str,
    mtf:        MultiTimeFrameFeatures,
    roller:     RollingFeatures,
    primary_tf: str = PRIMARY_TF,
    start:      str = START_DATE,
    end:        str = END_DATE,
) -> pd.DataFrame:
    """
    Full data pipeline for one ticker:
      1. Load raw 1-min OHLCV + resample to all timeframes (MultiTimeFrameFeatures)
      2. Add rolling technical indicators to PRIMARY_TF only (RollingFeatures)
      3. Drop NaN rows left over from indicator warm-up periods
      4. Return a clean DataFrame ready for scaling

    Raises ValueError if the ticker has no data or if PRIMARY_TF is unavailable.
    """
    # Step 1 — load and resample
    multi_tf = mtf.create_features(ticker=ticker, start=start, end=end)

    if not multi_tf:
        raise ValueError(f"No data returned for '{ticker}'")

    if primary_tf not in multi_tf:
        raise ValueError(
            f"PRIMARY_TF '{primary_tf}' not in available timeframes: {list(multi_tf.keys())}"
        )

    # Step 2 — add rolling indicators to the chosen timeframe only
    processed = roller.process({primary_tf: multi_tf[primary_tf]})
    df = processed[primary_tf]

    if df.empty:
        raise ValueError(f"Feature DataFrame is empty after rolling features for '{ticker}'")

    # Step 3 — drop NaN rows from indicator warm-up (a single pass is sufficient)
    df = df.dropna()
    return df



# 3. SCALING

def scale_data(
    train_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
) -> tuple:
    """
    Fit a MinMaxScaler on train_df only, then transform all DataFrames.

    Fitting on train data only prevents data leakage into validation/test sets.

    Returns
    (scaler, scaled_train, scaled_other_1, scaled_other_2, ...)
    Keep the scaler — it is needed to inverse-transform predictions later.
    """
    scaler       = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df.values.astype(np.float32))
    scaled_rest  = [scaler.transform(df.values.astype(np.float32)) for df in other_dfs]
    return scaler, scaled_train, *scaled_rest


def save_scaler(scaler: MinMaxScaler, ticker: str, results_dir: str) -> str:
    """Persist the fitted scaler to disk alongside the model weights."""
    path = os.path.join(results_dir, f"{ticker}_scaler.pkl")
    joblib.dump(scaler, path)
    return path



# 4. WINDOWED DATASET

class OHLCVDataset(Dataset):
    """
    Sliding-window dataset over a scaled feature array.

    Each sample:
        X : (seq_len, num_features)  — the look-back window
        y : scalar                   — next bar's *scaled* close price

    Column layout after RollingFeatures (which preserves the original order):
        index 0=open  1=high  2=low  3=close  4=volume  5+=indicators
    CLOSE_IDX is verified at construction time against the actual column list
    so that any upstream column-order change raises an error immediately.
    """

    CLOSE_COL = "close"  # verified by name

    def __init__(
        self,
        data:     np.ndarray,
        columns:  list,
        seq_len:  int = SEQ_LEN,
    ):
        if self.CLOSE_COL not in columns:
            raise ValueError(
                f"Expected a '{self.CLOSE_COL}' column; got: {columns}"
            )
        self.data      = data
        self.seq_len   = seq_len
        self.close_idx = list(columns).index(self.CLOSE_COL)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        X = self.data[idx : idx + self.seq_len]               # (seq_len, features)
        y = self.data[idx + self.seq_len, self.close_idx]     # next bar close
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )



# 5. LSTM MODEL

class LSTMModel(nn.Module):
    """
    Two-layer LSTM → fully-connected head → scalar close-price prediction.

    Parameters
    ----------
    input_size  : number of input features per time step
    hidden_size : LSTM hidden dimension          (default: HIDDEN)
    num_layers  : number of stacked LSTM layers  (default: NUM_LAYERS)
    dropout     : dropout between LSTM layers    (default: 0.2)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = HIDDEN,
        num_layers:  int   = NUM_LAYERS,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        # PyTorch initialises h0 and c0 to zeros when not provided
        out, _ = self.lstm(x)       # (batch, seq_len, hidden_size)
        out     = out[:, -1, :]     # last time step → (batch, hidden_size)
        out     = self.fc(out)      # (batch, 1)
        return out.squeeze(-1)      # (batch,)



# 6. TRAINING & VALIDATION

def train_one_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    criterion:     nn.Module,
    optimizer:     torch.optim.Optimizer,
    device:        torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """One full pass over the training set. Returns mean batch loss."""
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        # Clip gradients to prevent the exploding-gradient problem common in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """Evaluate the model on a data loader. Returns mean batch loss."""
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total_loss += criterion(model(X), y).item()
    return total_loss / len(loader)


def run_training(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    epochs:       int   = EPOCHS,
    lr:           float = LR,
    save_path:    str   = "",
) -> float:
    """
    Train for `epochs` epochs with Adam optimiser.
    Saves model weights whenever validation loss improves.
    Returns the best validation loss achieved.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val  = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
            saved = "saved"

        print(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}{saved}"
        )

    return best_val



# 7. SINGLE-TICKER PIPELINE

def run_ticker(
    ticker: str,
    device: torch.device,
    mtf:    MultiTimeFrameFeatures,
    roller: RollingFeatures,
) -> dict:
    """
    End-to-end pipeline for one ticker:
      build_feature_df → split → scale → dataset → model → train

    Returns
    -------
    dict with keys:
        ticker, status, best_val_loss, bars, num_features, skipped_reason
    """
    result = {
        "ticker":         ticker,
        "status":         "ok",
        "best_val_loss":  None,
        "bars":           0,
        "num_features":   0,
        "skipped_reason": "",
    }

    # Ensure output directory exists whether called from __main__ or externally
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # 1. Build feature DataFrame
        df = build_feature_df(ticker, mtf, roller)
        result["bars"] = len(df)

        # 2. Guard: enough bars for both train AND val windows
        if len(df) < MIN_BARS:
            result.update(
                status="skipped",
                skipped_reason=(
                    f"only {len(df)} bars after processing "
                    f"(need ≥ {MIN_BARS} to guarantee non-empty train and val sets)"
                ),
            )
            return result

        # 3. Split BEFORE scaling to prevent data leakage
        split    = int(len(df) * TRAIN_SPLIT)
        train_df = df.iloc[:split]
        val_df   = df.iloc[split:]

        # Sanity-check: both partitions must have enough rows for at least one window.
        # Should never trigger given MIN_BARS above, but guards against edge cases.
        if len(train_df) <= SEQ_LEN or len(val_df) <= SEQ_LEN:
            result.update(
                status="skipped",
                skipped_reason=(
                    f"after split: train={len(train_df)} bars, val={len(val_df)} bars — "
                    f"both must exceed SEQ_LEN={SEQ_LEN}"
                ),
            )
            return result

        # 4. Scale
        columns = list(df.columns)
        scaler, train_data, val_data = scale_data(train_df, val_df)

        # Guard: zero-variance columns produce NaN after MinMaxScaler
        # (division by zero when max == min). Drop them before training.
        nan_mask = np.isnan(train_data).any(axis=0)
        if nan_mask.any():
            bad_cols = [c for c, bad in zip(columns, nan_mask) if bad]
            train_data = train_data[:, ~nan_mask]
            val_data   = val_data[:,   ~nan_mask]
            columns    = [c for c, bad in zip(columns, nan_mask) if not bad]
            print(f" Dropped {len(bad_cols)} zero-variance column(s): {bad_cols}")

        # 5. Datasets & DataLoaders
        train_dataset = OHLCVDataset(train_data, columns, SEQ_LEN)
        val_dataset   = OHLCVDataset(val_data,   columns, SEQ_LEN)

        train_loader = DataLoader(
            train_dataset,
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = 0,   # set > 0 for faster loading if OS supports it
            pin_memory  = device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = device.type == "cuda",
        )

        # 6. Build model
        num_features           = train_data.shape[1]
        result["num_features"] = num_features
        model                  = LSTMModel(input_size=num_features).to(device)
        model_path             = os.path.join(RESULTS_DIR, f"{ticker}.pt")

        # 7. Train
        best_val = run_training(
            model, train_loader, val_loader, device, save_path=model_path
        )
        result["best_val_loss"] = best_val

        # 8. Save scaler (needed to inverse-transform predictions later)
        save_scaler(scaler, ticker, RESULTS_DIR)

    except Exception as e:
        # Catch any unexpected error (GPU OOM, corrupt data, NaN loss, etc.)
        # so the multi-ticker loop continues rather than crashing mid-run.
        result.update(status="skipped", skipped_reason=f"unexpected error: {e}")

    return result



# 8. MAIN — MULTI-TICKER LOOP

if __name__ == "__main__":
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Primary TF : {PRIMARY_TF}  |  SEQ_LEN={SEQ_LEN} bars  |  MIN_BARS={MIN_BARS}\n")

    # Initialise shared objects once — reused across all 503 tickers
    loader = StockDataLoader(base_path=DATA_PATH)
    mtf    = MultiTimeFrameFeatures(loader=loader)
    roller = RollingFeatures()

    tickers = load_tickers(TICKERS_FILE)
    print(f"Loaded {len(tickers)} tickers from '{TICKERS_FILE}'\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Per-ticker pipeline
    summary = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>3}/{len(tickers)}] {ticker}")
        res = run_ticker(ticker, device, mtf, roller)

        if res["status"] == "skipped":
            print(f"  SKIPPED — {res['skipped_reason']}\n")
        else:
            print(
                f"  bars={res['bars']:,}  "
                f"features={res['num_features']}  "
                f"best_val_loss={res['best_val_loss']:.6f}\n"
            )
        summary.append(res)

    # Summary report
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
