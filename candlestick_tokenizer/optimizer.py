"""
Adaptive Bottleneck Dimension Optimizer.

Treats the discrete binary bottleneck size as an empirical hyperparameter to optimize.
Systematically trains, evaluates, and compares autoencoders across candidate dimensions:
e.g. [8, 16, 24, 32, 48, 64] bits.

Identifies the Pareto-optimal token size that balances compactness, bit utilization,
and OHLC reconstruction fidelity.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import CandlestickAutoencoder
from .losses import CandlestickLoss
from .trainer import CandlestickTrainer
from .evaluator import CandlestickEvaluator
from .features import CandlestickFeatureScaler


class BottleneckOptimizer:
    """
    Automates multi-dimension bottleneck search and trade-off analysis.
    """

    def __init__(
        self,
        candidate_dimensions: Optional[List[int]] = None,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        epochs: int = 35,
        learning_rate: float = 1e-3,
        lambda_recon: float = 1.0,
        lambda_binary: float = 0.05,
        lambda_entropy: float = 0.1,
        lambda_price: float = 0.0,
        early_stopping_patience: int = 10,
        mae_threshold: Optional[float] = None,
        min_active_bit_pct: float = 80.0,
        device: Optional[torch.device] = None,
    ):
        self.candidate_dimensions = candidate_dimensions or [8, 16, 24, 32, 48, 64]
        self.hidden_dims = hidden_dims or [64, 128, 64]
        self.activation = activation
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_recon = lambda_recon
        self.lambda_binary = lambda_binary
        self.lambda_entropy = lambda_entropy
        self.lambda_price = lambda_price
        self.early_stopping_patience = early_stopping_patience
        self.mae_threshold = mae_threshold
        self.min_active_bit_pct = min_active_bit_pct
        self.device = device or torch.device("cpu")

        self.results: List[Dict[str, Any]] = []
        self.trained_models: Dict[int, CandlestickAutoencoder] = {}
        self.histories: Dict[int, Dict[str, List[float]]] = {}
        self.optimal_dimension: Optional[int] = None

    def run_search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        scaler: CandlestickFeatureScaler,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Executes training and evaluation for each candidate bottleneck dimension.
        """
        if verbose:
            print("\n" + "=" * 75)
            print(f"ADAPTIVE BOTTLENECK OPTIMIZATION: Evaluating dimensions {self.candidate_dimensions}")
            print("=" * 75)

        for b_dim in self.candidate_dimensions:
            if verbose:
                print(f"\n---> Training CandlestickAutoencoder with {b_dim}-bit binary bottleneck...")

            # 1. Instantiate model
            model = CandlestickAutoencoder(
                input_dim=5,
                bottleneck_dim=b_dim,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                dropout=self.dropout,
            ).to(self.device)

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # 2. Instantiate loss and trainer
            loss_fn = CandlestickLoss(
                lambda_recon=self.lambda_recon,
                lambda_binary=self.lambda_binary,
                lambda_entropy=self.lambda_entropy,
                lambda_price=self.lambda_price,
                scaler=scaler,
            )

            trainer = CandlestickTrainer(
                model=model,
                loss_fn=loss_fn,
                learning_rate=self.learning_rate,
                device=self.device,
                early_stopping_patience=self.early_stopping_patience,
            )

            # 3. Train
            t0 = time.time()
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.epochs,
                verbose=False
            )
            train_duration = time.time() - t0

            # 4. Evaluate on unseen Test Set
            evaluator = CandlestickEvaluator(model=model, scaler=scaler, device=self.device)
            test_metrics = evaluator.evaluate(test_loader)

            # 5. Record result
            res = {
                "bottleneck_dim": b_dim,
                "param_count": param_count,
                "train_time_sec": round(train_duration, 2),
                "best_epoch": trainer.best_epoch,
                "val_recon_mse": round(trainer.best_val_loss, 6),
                "test_feat_mse": round(test_metrics["feature_mse_overall"], 6),
                "test_ohlc_mae": round(test_metrics["ohlc_mae_overall"], 4),
                "test_ohlc_rmse": round(test_metrics["ohlc_rmse_overall"], 4),
                "test_ohlc_mape": round(test_metrics["mape_overall"], 3),
                "test_max_abs_err": round(test_metrics["max_abs_error_overall"], 4),
                "bit_utilization_pct": round(test_metrics["entropy_utilization_pct"], 2),
                "active_bit_pct": round(test_metrics["active_bit_pct"], 1),
                "unique_tokens": test_metrics["num_unique_tokens"],
                "token_diversity_pct": round(test_metrics["token_diversity_ratio"] * 100.0, 2),
                "direction_acc_pct": round(test_metrics["direction_accuracy_pct"], 2),
            }

            self.results.append(res)
            self.trained_models[b_dim] = model
            self.histories[b_dim] = history

            if verbose:
                print(
                    f"Result [{b_dim:2d} bits]: OHLC MAE = {res['test_ohlc_mae']:<8.4f} | "
                    f"MAPE = {res['test_ohlc_mape']:<5.2f}% | "
                    f"Active Bits = {res['active_bit_pct']:<5.1f}% | "
                    f"Unique Tokens = {res['unique_tokens']:<5d} | "
                    f"Direction Acc = {res['direction_acc_pct']}% | "
                    f"Time = {res['train_time_sec']}s"
                )

        df_results = pd.DataFrame(self.results)
        self.optimal_dimension = self._select_optimal_dimension(df_results)
        if verbose:
            print("\n" + "=" * 75)
            print(f"SELECTION DECISION: Optimal Bottleneck Dimension = {self.optimal_dimension} bits")
            print("=" * 75)

        return df_results

    def _select_optimal_dimension(self, df_results: pd.DataFrame) -> int:
        """
        Applies Pareto selection logic:
        Finds the smallest token size that achieves acceptable reconstruction accuracy
        without significant bit collapse.
        """
        # Filter for candidates with acceptable bit utilization
        viable = df_results[df_results["active_bit_pct"] >= self.min_active_bit_pct]
        if viable.empty:
            viable = df_results

        if self.mae_threshold is not None:
            threshold_met = viable[viable["test_ohlc_mae"] <= self.mae_threshold]
            if not threshold_met.empty:
                # Return the smallest dimension meeting the threshold
                return int(threshold_met.iloc[0]["bottleneck_dim"])

        # Default multi-criterion trade-off:
        # Score = (Relative MAE / Min MAE) + 0.5 * (Bits / Max Bits) - 0.2 * (Active Bit % / 100)
        min_mae = df_results["test_ohlc_mae"].min()
        max_bits = df_results["bottleneck_dim"].max()

        scores = []
        for _, row in viable.iterrows():
            mae_penalty = row["test_ohlc_mae"] / max(min_mae, 1e-6)
            size_penalty = row["bottleneck_dim"] / max_bits
            diversity_bonus = (row["active_bit_pct"] / 100.0) * 0.3
            score = mae_penalty + 0.4 * size_penalty - diversity_bonus
            scores.append((score, int(row["bottleneck_dim"])))

        scores.sort(key=lambda x: x[0])
        return scores[0][1]

    def get_optimal_model(self) -> Tuple[int, CandlestickAutoencoder]:
        if self.optimal_dimension is None:
            raise RuntimeError("Run search before retrieving optimal model.")
        return self.optimal_dimension, self.trained_models[self.optimal_dimension]
