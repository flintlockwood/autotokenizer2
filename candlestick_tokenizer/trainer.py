"""
Training and Validation Loop for Candlestick Autoencoder.

Supports:
- Differentiable training with Straight-Through Estimator (STE)
- Multi-component loss tracking (Reconstruction, Binary Commitment, Bit Entropy, Price)
- Learning rate scheduling (Cosine Annealing)
- Early stopping based on validation feature reconstruction
- Best model checkpointing and history recording
"""

import copy
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import CandlestickLoss
from .model import CandlestickAutoencoder


class CandlestickTrainer:
    """
    Manages the training, validation, scheduling, and early stopping of CandlestickAutoencoder.
    """

    def __init__(
        self,
        model: CandlestickAutoencoder,
        loss_fn: CandlestickLoss,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[torch.device] = None,
        scheduler_type: str = "cosine",
        early_stopping_patience: int = 15,
        min_delta: float = 1e-5
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_total_loss": [],
            "train_recon_loss": [],
            "train_binary_loss": [],
            "train_entropy_loss": [],
            "train_price_loss": [],
            "val_total_loss": [],
            "val_recon_loss": [],
            "lr": [],
            "epoch_time_sec": []
        }

        self.best_val_loss = float("inf")
        self.best_state_dict: Optional[dict] = None
        self.best_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss_accum = 0.0
        recon_accum = 0.0
        binary_accum = 0.0
        entropy_accum = 0.0
        price_accum = 0.0
        n_batches = len(train_loader)

        for batch_feat, batch_o_prev, batch_ohlc in train_loader:
            batch_feat = batch_feat.to(self.device)
            batch_o_prev = batch_o_prev.to(self.device)
            batch_ohlc = batch_ohlc.to(self.device)

            self.optimizer.zero_grad()
            recon_x, z, p, logits = self.model(batch_feat)

            loss, loss_dict = self.loss_fn(
                target_norm=batch_feat,
                recon_norm=recon_x,
                z=z,
                p=p,
                o_prev=batch_o_prev,
                target_ohlc=batch_ohlc
            )

            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss_accum += loss_dict["total_loss"]
            recon_accum += loss_dict["recon_loss"]
            binary_accum += loss_dict["binary_loss"]
            entropy_accum += loss_dict["entropy_loss"]
            price_accum += loss_dict["price_loss"]

        return {
            "total_loss": total_loss_accum / max(n_batches, 1),
            "recon_loss": recon_accum / max(n_batches, 1),
            "binary_loss": binary_accum / max(n_batches, 1),
            "entropy_loss": entropy_accum / max(n_batches, 1),
            "price_loss": price_accum / max(n_batches, 1),
        }

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss_accum = 0.0
        recon_accum = 0.0
        n_batches = len(val_loader)

        with torch.no_grad():
            for batch_feat, batch_o_prev, batch_ohlc in val_loader:
                batch_feat = batch_feat.to(self.device)
                batch_o_prev = batch_o_prev.to(self.device)
                batch_ohlc = batch_ohlc.to(self.device)

                recon_x, z, p, logits = self.model(batch_feat)
                loss, loss_dict = self.loss_fn(
                    target_norm=batch_feat,
                    recon_norm=recon_x,
                    z=z,
                    p=p,
                    o_prev=batch_o_prev,
                    target_ohlc=batch_ohlc
                )
                total_loss_accum += loss_dict["total_loss"]
                recon_accum += loss_dict["recon_loss"]

        return {
            "total_loss": total_loss_accum / max(n_batches, 1),
            "recon_loss": recon_accum / max(n_batches, 1),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Executes the full training loop with validation and early stopping.
        """
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        else:
            scheduler = None

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            elapsed = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_total_loss"].append(train_metrics["total_loss"])
            self.history["train_recon_loss"].append(train_metrics["recon_loss"])
            self.history["train_binary_loss"].append(train_metrics["binary_loss"])
            self.history["train_entropy_loss"].append(train_metrics["entropy_loss"])
            self.history["train_price_loss"].append(train_metrics["price_loss"])
            self.history["val_total_loss"].append(val_metrics["total_loss"])
            self.history["val_recon_loss"].append(val_metrics["recon_loss"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time_sec"].append(elapsed)

            val_recon = val_metrics["recon_loss"]

            # Check for best validation reconstruction loss
            if val_recon < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_recon
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs or patience_counter == 0):
                print(
                    f"Epoch {epoch:3d}/{epochs:3d} | "
                    f"Train Loss: {train_metrics['total_loss']:.5f} (Recon: {train_metrics['recon_loss']:.5f}) | "
                    f"Val Recon: {val_metrics['recon_loss']:.5f} | "
                    f"LR: {current_lr:.1e} | {elapsed:.2f}s"
                )

            # Early stopping check
            if patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch was {self.best_epoch} with val_recon {self.best_val_loss:.5f}")
                break

        # Restore best model state
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history
