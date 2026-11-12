"""
Stationary Candlestick Feature Engineering and Exact Invertible Reconstruction.

Calculates the 5 specified features from OHLC market data:
1. f1: High Price Movement (%) relative to previous Open
2. f2: Low Price Movement (%) relative to previous Open
3. f3: Upper Wick Ratio relative to total range
4. f4: Lower Wick Ratio relative to total range
5. f5: Signed Body Ratio relative to total range (preserving bullish/bearish direction)

Also provides:
- Invertible scaling (CandlestickFeatureScaler)
- Exact mathematical reconstruction of OHLC prices given reconstructed features and previous Open
- OHLC physical validity checking and enforcement
"""

import json
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch


def validate_ohlc_data(df_or_array: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Validates that OHLC data contains no NaNs, Infs, or non-positive prices,
    and satisfies H >= max(O, C) and L <= min(O, C).
    
    Returns:
        numpy.ndarray of shape (N, 4) with columns [Open, High, Low, Close]
    """
    if isinstance(df_or_array, pd.DataFrame):
        req_cols = ["Open", "High", "Low", "Close"]
        # Case insensitive column matching
        col_map = {col.lower(): col for col in df_or_array.columns}
        for req in req_cols:
            if req.lower() not in col_map:
                raise ValueError(f"Missing required OHLC column: '{req}'")
        arr = df_or_array[[col_map[c.lower()] for c in req_cols]].to_numpy(dtype=np.float64)
    elif isinstance(df_or_array, np.ndarray):
        arr = df_or_array.astype(np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"Expected array of shape (N, 4), got {arr.shape}")
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray")

    # Check for NaN / Inf
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(~np.isfinite(arr))
        raise ValueError(f"Found {nan_count} non-finite (NaN or Inf) values in OHLC data")

    # Check strictly positive prices
    if np.any(arr <= 0):
        invalid_count = np.sum(arr <= 0)
        raise ValueError(f"Found {invalid_count} non-positive price values in OHLC data")

    # Check High >= Low
    o, h, l, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if np.any(h < l):
        inversions = np.sum(h < l)
        raise ValueError(f"Found {inversions} candles where High < Low")

    return arr


def extract_features(
    ohlc: Union[pd.DataFrame, np.ndarray],
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the 5 stationary features from OHLC data using the previous candle's Open as reference.
    
    For candlestick at index t (t >= 1):
    - f1 = (H_t - O_{t-1}) / O_{t-1} * 100
    - f2 = (L_t - O_{t-1}) / O_{t-1} * 100
    - f3 = (H_t - max(O_t, C_t)) / (H_t - L_t)
    - f4 = (min(O_t, C_t) - L_t) / (H_t - L_t)
    - f5 = (C_t - O_t) / (H_t - L_t) [SIGNED body ratio]
    
    Args:
        ohlc: Input OHLC dataframe or array of shape (T, 4)
        eps: Small threshold for zero-range candlestick handling
        
    Returns:
        features: (T-1, 5) numpy array of stationary features
        O_prev: (T-1,) numpy array of reference previous Open prices
        current_ohlc: (T-1, 4) numpy array of current OHLC candles (t >= 1)
    """
    arr = validate_ohlc_data(ohlc)
    T = len(arr)
    if T < 2:
        raise ValueError(f"Need at least 2 candlesticks to compute features, got {T}")

    # Reference prices: previous candlestick's Open
    o_prev = arr[:-1, 0]  # O_{t-1} for t = 1 ... T-1
    curr_ohlc = arr[1:, :]  # OHLC_t for t = 1 ... T-1

    o = curr_ohlc[:, 0]
    h = curr_ohlc[:, 1]
    l = curr_ohlc[:, 2]
    c = curr_ohlc[:, 3]

    # Feature 1: High Price Movement (%)
    f1 = (h - o_prev) / o_prev * 100.0

    # Feature 2: Low Price Movement (%)
    f2 = (l - o_prev) / o_prev * 100.0

    # Candlestick Range
    candle_range = h - l
    zero_range_mask = candle_range < eps

    # Safe denominator avoiding division by zero
    safe_range = np.where(zero_range_mask, 1.0, candle_range)

    # Feature 3: Upper Wick Ratio
    max_oc = np.maximum(o, c)
    upper_wick = h - max_oc
    f3 = np.where(zero_range_mask, 0.0, upper_wick / safe_range)
    # Clip numerical precision slightly to [0, 1]
    f3 = np.clip(f3, 0.0, 1.0)

    # Feature 4: Lower Wick Ratio
    min_oc = np.minimum(o, c)
    lower_wick = min_oc - l
    f4 = np.where(zero_range_mask, 0.0, lower_wick / safe_range)
    f4 = np.clip(f4, 0.0, 1.0)

    # Feature 5: Signed Body Ratio (positive for bullish, negative for bearish, zero for doji)
    body = c - o
    f5 = np.where(zero_range_mask, 0.0, body / safe_range)
    f5 = np.clip(f5, -1.0, 1.0)

    features = np.column_stack([f1, f2, f3, f4, f5])
    return features, o_prev, curr_ohlc


def reconstruct_ohlc(
    features: Union[np.ndarray, torch.Tensor],
    o_prev: Union[np.ndarray, torch.Tensor],
    enforce_constraints: bool = True
) -> Union[Tuple[np.ndarray, Dict[str, float]], Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Reconstructs OHLC prices from 5D feature vector and previous Open price.
    
    Formulas:
    H_hat = O_{t-1} * (1 + f1_hat / 100)
    L_hat = O_{t-1} * (1 + f2_hat / 100)
    R_hat = H_hat - L_hat
    B_hat = f5_hat * R_hat
    UW_hat = f3_hat * R_hat
    LW_hat = f4_hat * R_hat
    O_hat = L_hat + LW_hat + max(0, -B_hat)
    C_hat = O_hat + B_hat
    
    Constraints enforced (if enforce_constraints=True):
    L_hat <= min(O_hat, C_hat)
    H_hat >= max(O_hat, C_hat)
    
    Returns:
        reconstructed_ohlc: (N, 4) [Open, High, Low, Close]
        diagnostics: dictionary with constraint violation metrics
    """
    is_torch = isinstance(features, torch.Tensor)
    if is_torch:
        if not isinstance(o_prev, torch.Tensor):
            o_prev = torch.tensor(o_prev, dtype=features.dtype, device=features.device)
        o_prev = o_prev.view(-1)
        
        f1 = features[:, 0]
        f2 = features[:, 1]
        f3 = features[:, 2]
        f4 = features[:, 3]
        f5 = features[:, 4]

        h_hat = o_prev * (1.0 + f1 / 100.0)
        l_hat = o_prev * (1.0 + f2 / 100.0)
        r_hat = h_hat - l_hat
        b_hat = f5 * r_hat
        lw_hat = f4 * r_hat

        # max(0, -B_hat) = relu(-B_hat)
        neg_body_part = torch.clamp(-b_hat, min=0.0)
        o_hat = l_hat + lw_hat + neg_body_part
        c_hat = o_hat + b_hat

        # Diagnostics for constraint violations before enforcement
        min_oc = torch.minimum(o_hat, c_hat)
        max_oc = torch.maximum(o_hat, c_hat)
        low_violation = torch.clamp(l_hat - min_oc, min=0.0)
        high_violation = torch.clamp(max_oc - h_hat, min=0.0)
        range_inversion = torch.clamp(l_hat - h_hat, min=0.0)

        diag = {
            "low_violation_rate": float((low_violation > 1e-6).float().mean().item()),
            "max_low_violation": float(low_violation.max().item()),
            "high_violation_rate": float((high_violation > 1e-6).float().mean().item()),
            "max_high_violation": float(high_violation.max().item()),
            "range_inversion_rate": float((range_inversion > 1e-6).float().mean().item()),
        }

        if enforce_constraints:
            # Enforce L <= min(O, C) and H >= max(O, C)
            l_hat_final = torch.minimum(l_hat, min_oc)
            h_hat_final = torch.maximum(h_hat, max_oc)
            # Ensure H >= L strictly
            h_hat_final = torch.maximum(h_hat_final, l_hat_final)
        else:
            l_hat_final = l_hat
            h_hat_final = h_hat

        recon = torch.stack([o_hat, h_hat_final, l_hat_final, c_hat], dim=1)
        return recon, diag

    else:
        # NumPy version
        features = np.asarray(features, dtype=np.float64)
        o_prev = np.asarray(o_prev, dtype=np.float64).reshape(-1)

        f1 = features[:, 0]
        f2 = features[:, 1]
        f3 = features[:, 2]
        f4 = features[:, 3]
        f5 = features[:, 4]

        h_hat = o_prev * (1.0 + f1 / 100.0)
        l_hat = o_prev * (1.0 + f2 / 100.0)
        r_hat = h_hat - l_hat
        b_hat = f5 * r_hat
        lw_hat = f4 * r_hat

        neg_body_part = np.maximum(0.0, -b_hat)
        o_hat = l_hat + lw_hat + neg_body_part
        c_hat = o_hat + b_hat

        min_oc = np.minimum(o_hat, c_hat)
        max_oc = np.maximum(o_hat, c_hat)
        low_violation = np.maximum(0.0, l_hat - min_oc)
        high_violation = np.maximum(0.0, max_oc - h_hat)
        range_inversion = np.maximum(0.0, l_hat - h_hat)

        diag = {
            "low_violation_rate": float(np.mean(low_violation > 1e-6)),
            "max_low_violation": float(np.max(low_violation)),
            "high_violation_rate": float(np.mean(high_violation > 1e-6)),
            "max_high_violation": float(np.max(high_violation)),
            "range_inversion_rate": float(np.mean(range_inversion > 1e-6)),
        }

        if enforce_constraints:
            l_hat_final = np.minimum(l_hat, min_oc)
            h_hat_final = np.maximum(h_hat, max_oc)
            h_hat_final = np.maximum(h_hat_final, l_hat_final)
        else:
            l_hat_final = l_hat
            h_hat_final = h_hat

        recon = np.column_stack([o_hat, h_hat_final, l_hat_final, c_hat])
        return recon, diag


class CandlestickFeatureScaler:
    """
    Invertible standard scaler for 5D candlestick features.
    
    Fits mean and standard deviation on training features, and normalizes
    features for stable neural network training while preserving all parameters
    for exact inverse transformation.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, features: np.ndarray) -> "CandlestickFeatureScaler":
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(f"Expected feature array of shape (N, 5), got {arr.shape}")
        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0)
        # Avoid division by zero for any constant feature
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.is_fitted = True
        return self

    def transform(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("CandlestickFeatureScaler is not fitted yet.")
        if isinstance(features, torch.Tensor):
            mean_t = torch.tensor(self.mean, dtype=features.dtype, device=features.device)
            std_t = torch.tensor(self.std, dtype=features.dtype, device=features.device)
            return (features - mean_t) / std_t
        else:
            arr = np.asarray(features, dtype=np.float64)
            return (arr - self.mean) / self.std

    def inverse_transform(self, features_norm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("CandlestickFeatureScaler is not fitted yet.")
        if isinstance(features_norm, torch.Tensor):
            mean_t = torch.tensor(self.mean, dtype=features_norm.dtype, device=features_norm.device)
            std_t = torch.tensor(self.std, dtype=features_norm.dtype, device=features_norm.device)
            return features_norm * std_t + mean_t
        else:
            arr = np.asarray(features_norm, dtype=np.float64)
            return arr * self.std + self.mean

    def to_dict(self) -> Dict[str, list]:
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted.")
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CandlestickFeatureScaler":
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        scaler = cls(eps=d.get("eps", 1e-8))
        scaler.mean = np.array(d["mean"], dtype=np.float64)
        scaler.std = np.array(d["std"], dtype=np.float64)
        scaler.is_fitted = True
        return scaler


class CandlestickFeatureEngineer:
    """
    High-level manager for end-to-end feature extraction, normalization, and reconstruction.
    """

    def __init__(self, scaler: Optional[CandlestickFeatureScaler] = None):
        self.scaler = scaler or CandlestickFeatureScaler()

    def fit_transform(self, ohlc: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features, o_prev, curr_ohlc = extract_features(ohlc)
        self.scaler.fit(features)
        features_norm = self.scaler.transform(features)
        return features_norm, o_prev, curr_ohlc

    def transform(self, ohlc: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features, o_prev, curr_ohlc = extract_features(ohlc)
        features_norm = self.scaler.transform(features)
        return features_norm, o_prev, curr_ohlc

    def inverse_transform_and_reconstruct(
        self,
        features_norm: Union[np.ndarray, torch.Tensor],
        o_prev: Union[np.ndarray, torch.Tensor],
        enforce_constraints: bool = True
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, float]]:
        features = self.scaler.inverse_transform(features_norm)
        return reconstruct_ohlc(features, o_prev, enforce_constraints=enforce_constraints)
