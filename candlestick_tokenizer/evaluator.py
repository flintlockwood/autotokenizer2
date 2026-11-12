"""
Comprehensive Candlestick Autoencoder Evaluator.

Calculates all 9 specified metrics:
1. Feature reconstruction MSE (overall and per feature f1-f5)
2. OHLC reconstruction MAE (overall and per price O, H, L, C)
3. OHLC reconstruction RMSE (overall and per price O, H, L, C)
4. Maximum absolute reconstruction error
5. Percentage reconstruction error (MAPE)
6. Binary token utilization (mean entropy across bits)
7. Number of unique binary tokens generated (and diversity ratio)
8. Percentage of active bits across the dataset
9. Reconstruction accuracy segmented by candle type (Bullish, Bearish, Doji)
   plus candle direction classification accuracy
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from .features import reconstruct_ohlc, CandlestickFeatureScaler
from .model import CandlestickAutoencoder, BinaryBottleneck


class CandlestickEvaluator:
    """
    Evaluates a trained CandlestickAutoencoder on validation or test datasets.
    """

    def __init__(
        self,
        model: CandlestickAutoencoder,
        scaler: CandlestickFeatureScaler,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.scaler = scaler
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        data_loader: DataLoader,
        doji_threshold_rel: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Runs full evaluation across all batches and returns structured metrics.
        """
        self.model.eval()

        all_target_norm = []
        all_recon_norm = []
        all_z = []
        all_p = []
        all_o_prev = []
        all_target_ohlc = []

        with torch.no_grad():
            for batch_feat_norm, batch_o_prev, batch_ohlc in data_loader:
                batch_feat_norm = batch_feat_norm.to(self.device)
                recon_norm, z, p, _ = self.model(batch_feat_norm)

                all_target_norm.append(batch_feat_norm.cpu().numpy())
                all_recon_norm.append(recon_norm.cpu().numpy())
                all_z.append(z.cpu().numpy())
                all_p.append(p.cpu().numpy())
                all_o_prev.append(batch_o_prev.cpu().numpy())
                all_target_ohlc.append(batch_ohlc.cpu().numpy())

        # Concatenate arrays
        target_norm = np.concatenate(all_target_norm, axis=0)
        recon_norm = np.concatenate(all_recon_norm, axis=0)
        z = np.concatenate(all_z, axis=0)
        p = np.concatenate(all_p, axis=0)
        o_prev = np.concatenate(all_o_prev, axis=0)
        target_ohlc = np.concatenate(all_target_ohlc, axis=0)
        N = len(target_norm)
        B = z.shape[1]

        # Denormalize features
        target_features = self.scaler.inverse_transform(target_norm)
        recon_features = self.scaler.inverse_transform(recon_norm)

        # 1. Feature Reconstruction MSE (in original unscaled feature space)
        feat_diff = recon_features - target_features
        feat_mse_per_feature = np.mean(feat_diff ** 2, axis=0).tolist()
        feat_mse_overall = float(np.mean(feat_diff ** 2))

        # Reconstruct OHLC prices from discrete binary tokens
        recon_ohlc, raw_diag = reconstruct_ohlc(recon_features, o_prev, enforce_constraints=True)
        recon_ohlc_raw, _ = reconstruct_ohlc(recon_features, o_prev, enforce_constraints=False)

        # 2. OHLC Reconstruction MAE
        price_diff = recon_ohlc - target_ohlc
        abs_diff = np.abs(price_diff)
        ohlc_mae_per_price = np.mean(abs_diff, axis=0).tolist()  # [O, H, L, C]
        ohlc_mae_overall = float(np.mean(abs_diff))

        # 3. OHLC Reconstruction RMSE
        sq_diff = price_diff ** 2
        ohlc_rmse_per_price = np.sqrt(np.mean(sq_diff, axis=0)).tolist()
        ohlc_rmse_overall = float(np.sqrt(np.mean(sq_diff)))

        # 4. Maximum Absolute Reconstruction Error
        max_abs_error_overall = float(np.max(abs_diff))
        max_abs_error_per_price = np.max(abs_diff, axis=0).tolist()

        # 5. Percentage Reconstruction Error (MAPE)
        rel_diff = abs_diff / np.maximum(target_ohlc, 1e-8) * 100.0
        mape_per_price = np.mean(rel_diff, axis=0).tolist()
        mape_overall = float(np.mean(rel_diff))

        # 6. Binary Token Utilization (mean entropy per bit)
        # Average probability of bit being 1 across the dataset
        bit_mean_p = np.mean(p, axis=0)
        eps = 1e-12
        bit_mean_p_clipped = np.clip(bit_mean_p, eps, 1.0 - eps)
        # Binary entropy in bits: - (p * log2(p) + (1-p) * log2(1-p))
        bit_entropy = -(
            bit_mean_p_clipped * np.log2(bit_mean_p_clipped)
            + (1.0 - bit_mean_p_clipped) * np.log2(1.0 - bit_mean_p_clipped)
        )
        mean_bit_entropy = float(np.mean(bit_entropy))
        entropy_utilization_pct = mean_bit_entropy * 100.0  # Max entropy is 1.0 bit

        # 7. Number of Unique Tokens Generated & Token Diversity
        bitstrings = BinaryBottleneck.to_bitstring(z)
        unique_tokens = set(bitstrings)
        num_unique_tokens = len(unique_tokens)
        token_diversity_ratio = float(num_unique_tokens / max(N, 1))

        # 8. Percentage of Active Bits Across Dataset
        # An active bit is defined as having mean probability in [0.05, 0.95]
        active_bits_mask = (bit_mean_p >= 0.05) & (bit_mean_p <= 0.95)
        active_bit_count = int(np.sum(active_bits_mask))
        active_bit_pct = float(active_bit_count / B * 100.0)

        # 9. Reconstruction Accuracy for Bullish, Bearish, and Doji Candles
        # Target candle categorization
        target_o = target_ohlc[:, 0]
        target_c = target_ohlc[:, 3]
        candle_body = target_c - target_o
        body_rel = np.abs(candle_body) / np.maximum(target_o, 1e-8)

        doji_mask = body_rel <= doji_threshold_rel
        bullish_mask = (candle_body > 0) & (~doji_mask)
        bearish_mask = (candle_body < 0) & (~doji_mask)

        def compute_subset_metrics(mask: np.ndarray) -> Dict[str, Any]:
            count = int(np.sum(mask))
            if count == 0:
                return {"count": 0, "mae": 0.0, "rmse": 0.0, "mape": 0.0}
            sub_abs = abs_diff[mask]
            sub_sq = sq_diff[mask]
            sub_rel = rel_diff[mask]
            return {
                "count": count,
                "mae": float(np.mean(sub_abs)),
                "rmse": float(np.sqrt(np.mean(sub_sq))),
                "mape": float(np.mean(sub_rel)),
            }

        bullish_metrics = compute_subset_metrics(bullish_mask)
        bearish_metrics = compute_subset_metrics(bearish_mask)
        doji_metrics = compute_subset_metrics(doji_mask)

        # Candle Direction Classification Accuracy
        # Target direction: +1 (bullish), -1 (bearish), 0 (doji)
        target_dir = np.where(doji_mask, 0, np.where(candle_body > 0, 1, -1))
        recon_o = recon_ohlc[:, 0]
        recon_c = recon_ohlc[:, 3]
        recon_body = recon_c - recon_o
        recon_body_rel = np.abs(recon_body) / np.maximum(recon_o, 1e-8)
        recon_doji = recon_body_rel <= doji_threshold_rel
        recon_dir = np.where(recon_doji, 0, np.where(recon_body > 0, 1, -1))
        direction_accuracy = float(np.mean(target_dir == recon_dir) * 100.0)

        # Binary quantization exactness check
        # Verify that all elements in z are strictly 0.0 or 1.0
        is_strictly_binary = bool(np.all((z == 0.0) | (z == 1.0)))

        metrics = {
            "num_samples": N,
            "bottleneck_dim": B,
            "is_strictly_binary": is_strictly_binary,
            # Metric 1: Feature MSE
            "feature_mse_overall": feat_mse_overall,
            "feature_mse_per_feature": feat_mse_per_feature,  # [f1, f2, f3, f4, f5]
            # Metric 2: OHLC MAE
            "ohlc_mae_overall": ohlc_mae_overall,
            "ohlc_mae_per_price": {
                "Open": ohlc_mae_per_price[0],
                "High": ohlc_mae_per_price[1],
                "Low": ohlc_mae_per_price[2],
                "Close": ohlc_mae_per_price[3],
            },
            # Metric 3: OHLC RMSE
            "ohlc_rmse_overall": ohlc_rmse_overall,
            "ohlc_rmse_per_price": {
                "Open": ohlc_rmse_per_price[0],
                "High": ohlc_rmse_per_price[1],
                "Low": ohlc_rmse_per_price[2],
                "Close": ohlc_rmse_per_price[3],
            },
            # Metric 4: Maximum Absolute Error
            "max_abs_error_overall": max_abs_error_overall,
            "max_abs_error_per_price": {
                "Open": max_abs_error_per_price[0],
                "High": max_abs_error_per_price[1],
                "Low": max_abs_error_per_price[2],
                "Close": max_abs_error_per_price[3],
            },
            # Metric 5: Percentage Reconstruction Error (MAPE)
            "mape_overall": mape_overall,
            "mape_per_price": {
                "Open": mape_per_price[0],
                "High": mape_per_price[1],
                "Low": mape_per_price[2],
                "Close": mape_per_price[3],
            },
            # Metric 6: Binary Token Utilization
            "mean_bit_entropy": mean_bit_entropy,
            "entropy_utilization_pct": entropy_utilization_pct,
            "bit_mean_probabilities": bit_mean_p.tolist(),
            # Metric 7: Unique Tokens & Diversity
            "num_unique_tokens": num_unique_tokens,
            "token_diversity_ratio": token_diversity_ratio,
            "theoretical_max_tokens": 2 ** min(B, 30),
            # Metric 8: Active Bits %
            "active_bit_count": active_bit_count,
            "active_bit_pct": active_bit_pct,
            # Metric 9: Segmented Candle Accuracy
            "bullish_candles": bullish_metrics,
            "bearish_candles": bearish_metrics,
            "doji_candles": doji_metrics,
            "direction_accuracy_pct": direction_accuracy,
            # Physical Constraint Diagnostics
            "constraint_diagnostics": raw_diag,
        }

        return metrics
