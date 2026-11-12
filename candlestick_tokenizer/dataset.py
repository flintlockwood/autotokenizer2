"""
Dataset Management, Loading, and Chronological Splitting for Candlestick Autoencoders.

Supports:
- CSV file loading with robust column detection
- Synthetic Geometric Brownian Motion + jump diffusion OHLC generation
- Strict chronological train/validation/test splits (zero forward-leakage)
- PyTorch Dataset and DataLoader construction
"""

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .features import extract_features, CandlestickFeatureScaler, validate_ohlc_data


def generate_synthetic_ohlc(
    n_bars: int = 5000,
    s0: float = 100.0,
    mu: float = 0.0002,
    sigma: float = 0.015,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic synthetic OHLC candlestick series using intraday Geometric Brownian Motion
    with random jump shocks and occasional doji formations.
    """
    rng = np.random.default_rng(seed)
    opens = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    closes = np.empty(n_bars, dtype=np.float64)

    price = s0
    n_substeps = 24  # simulate hourly substeps for each daily candle

    for i in range(n_bars):
        o = price
        shocks = rng.normal(mu / n_substeps, sigma / np.sqrt(n_substeps), size=n_substeps)
        path = o * np.cumprod(1.0 + shocks)
        h = max(o, float(np.max(path)))
        l = min(o, float(np.min(path)))
        c = float(path[-1])

        # ~4% probability of doji (C == O)
        if rng.random() < 0.04:
            c = o

        opens[i] = o
        highs[i] = h
        lows[i] = l
        closes[i] = c

        # Overnight gap shock
        overnight_gap = rng.normal(0.0, sigma * 0.15)
        price = max(c * (1.0 + overnight_gap), 1.0)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes
    })
    return df


def load_ohlc_csv(filepath: str) -> pd.DataFrame:
    """
    Loads OHLC data from a CSV file, handling date indices and case-insensitive column names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath)
    # Check for index columns or multi-index headers
    col_map = {col.strip().lower(): col for col in df.columns}
    req = ["open", "high", "low", "close"]
    for r in req:
        if r not in col_map:
            raise ValueError(f"CSV missing column '{r}'. Available columns: {list(df.columns)}")

    cols = [col_map[r] for r in req]
    df_ohlc = df[cols].copy()
    df_ohlc.columns = ["Open", "High", "Low", "Close"]
    df_ohlc = df_ohlc.dropna().astype(np.float64)
    return df_ohlc


class CandlestickDataset(Dataset):
    """
    PyTorch Dataset providing:
    - norm_features: (5,) normalized stationary features
    - o_prev: scalar previous Open reference price
    - curr_ohlc: (4,) actual OHLC prices of the candlestick
    """

    def __init__(
        self,
        features_norm: np.ndarray,
        o_prev: np.ndarray,
        curr_ohlc: np.ndarray
    ):
        self.features_norm = torch.tensor(features_norm, dtype=torch.float32)
        self.o_prev = torch.tensor(o_prev, dtype=torch.float32)
        self.curr_ohlc = torch.tensor(curr_ohlc, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features_norm)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features_norm[idx], self.o_prev[idx], self.curr_ohlc[idx]


def create_dataloaders(
    ohlc_df_or_arr: Union[pd.DataFrame, np.ndarray],
    batch_size: int = 128,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    scaler: Optional[CandlestickFeatureScaler] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, CandlestickFeatureScaler, Dict[str, np.ndarray]]:
    """
    Extracts features, performs strict chronological train/val/test splitting,
    fits the feature scaler exclusively on the training set, and creates DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, fitted_scaler, split_arrays_dict
    """
    # 1. Extract stationary 5D features
    features, o_prev, curr_ohlc = extract_features(ohlc_df_or_arr)
    N = len(features)

    # 2. Strict chronological splitting
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val

    if n_train < 10 or n_val < 5 or n_test < 5:
        raise ValueError(f"Dataset too small ({N} samples) for chronological splitting.")

    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, N)

    feat_train = features[train_idx]
    feat_val = features[val_idx]
    feat_test = features[test_idx]

    # 3. Fit scaler ONLY on training data to avoid forward lookahead leakage
    if scaler is None:
        scaler = CandlestickFeatureScaler()
    scaler.fit(feat_train)

    feat_train_norm = scaler.transform(feat_train)
    feat_val_norm = scaler.transform(feat_val)
    feat_test_norm = scaler.transform(feat_test)

    # 4. Construct Datasets
    ds_train = CandlestickDataset(feat_train_norm, o_prev[train_idx], curr_ohlc[train_idx])
    ds_val = CandlestickDataset(feat_val_norm, o_prev[val_idx], curr_ohlc[val_idx])
    ds_test = CandlestickDataset(feat_test_norm, o_prev[test_idx], curr_ohlc[test_idx])

    # 5. Construct DataLoaders
    # Training loader may shuffle within training split; val/test remain strictly ordered
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    splits_data = {
        "train_features_norm": feat_train_norm,
        "val_features_norm": feat_val_norm,
        "test_features_norm": feat_test_norm,
        "train_o_prev": o_prev[train_idx],
        "val_o_prev": o_prev[val_idx],
        "test_o_prev": o_prev[test_idx],
        "train_ohlc": curr_ohlc[train_idx],
        "val_ohlc": curr_ohlc[val_idx],
        "test_ohlc": curr_ohlc[test_idx],
    }

    return train_loader, val_loader, test_loader, scaler, splits_data
