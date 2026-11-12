"""
High-Quality Visualization Module for Candlestick Autoencoder Analysis.

Generates:
1. Candlestick comparison chart (Ground Truth vs Reconstructed OHLC with error residuals)
2. Training and validation loss curves (Reconstruction, Binary, Entropy)
3. Bit utilization and entropy distribution chart
4. Bottleneck dimension optimization comparison plot
"""

import os
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class CandlestickVisualizer:
    """
    Renders diagnostic charts and saves them to disk.
    """

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Style settings
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training and Validation Curves",
        filename: str = "training_curves.png"
    ) -> str:
        """
        Plots total loss, reconstruction loss, binary loss, and bit entropy loss across epochs.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = history["epoch"]

        # Subplot 1: Total Loss
        axes[0, 0].plot(epochs, history["train_total_loss"], label="Train Total Loss", color="#1f77b4", lw=2)
        axes[0, 0].plot(epochs, history["val_total_loss"], label="Val Total Loss", color="#ff7f0e", lw=2, linestyle="--")
        axes[0, 0].set_title("Total Loss", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Subplot 2: Reconstruction Loss
        axes[0, 1].plot(epochs, history["train_recon_loss"], label="Train Recon Loss (MSE)", color="#2ca02c", lw=2)
        axes[0, 1].plot(epochs, history["val_recon_loss"], label="Val Recon Loss (MSE)", color="#d62728", lw=2, linestyle="--")
        axes[0, 1].set_title("Feature Reconstruction Loss", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Subplot 3: Binary Commitment Regularization Loss
        axes[1, 0].plot(epochs, history["train_binary_loss"], label="Binary Commitment (4p(1-p))", color="#9467bd", lw=2)
        axes[1, 0].set_title("Binary Commitment Regularization", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss (0=Sharp Binary, 1=Ambiguous)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Subplot 4: Bit Entropy / Utilization Loss
        axes[1, 1].plot(epochs, history["train_entropy_loss"], label="Entropy Regularization (mean(p) - 0.5)^2", color="#8c564b", lw=2)
        axes[1, 1].set_title("Bit Utilization / Entropy Loss", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss (0=Balanced 50/50)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_candlestick_reconstruction(
        self,
        true_ohlc: np.ndarray,
        recon_ohlc: np.ndarray,
        start_idx: int = 0,
        n_bars: int = 50,
        title: str = "Original vs Reconstructed Candlesticks",
        filename: str = "candlestick_comparison.png"
    ) -> str:
        """
        Draws financial candlesticks comparing original Ground Truth vs Autoencoder Reconstructed OHLC
        with an error residual panel below.
        """
        end_idx = min(start_idx + n_bars, len(true_ohlc))
        t_sub = true_ohlc[start_idx:end_idx]
        r_sub = recon_ohlc[start_idx:end_idx]
        N = len(t_sub)

        fig, (ax_main, ax_err) = plt.subplots(
            2, 1, figsize=(16, 9),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )

        width = 0.35
        # Colors: Green for bullish, Red for bearish
        c_bull_true = "#26a69a"
        c_bear_true = "#ef5350"
        c_bull_rec = "#004d40"
        c_bear_rec = "#b71c1c"

        for i in range(N):
            # 1. Ground truth candlestick (shifted left by width/2)
            o, h, l, c = t_sub[i]
            x_t = i - width / 2.0
            color_t = c_bull_true if c >= o else c_bear_true
            # Wick
            ax_main.plot([x_t, x_t], [l, h], color=color_t, lw=1.2, alpha=0.85)
            # Body
            body_bottom = min(o, c)
            body_height = max(abs(c - o), 1e-4)
            rect_t = patches.Rectangle(
                (x_t - width * 0.4, body_bottom),
                width * 0.8, body_height,
                linewidth=1, edgecolor=color_t, facecolor=color_t, alpha=0.85
            )
            ax_main.add_patch(rect_t)

            # 2. Reconstructed candlestick (shifted right by width/2)
            ro, rh, rl, rc = r_sub[i]
            x_r = i + width / 2.0
            color_r = c_bull_rec if rc >= ro else c_bear_rec
            # Wick
            ax_main.plot([x_r, x_r], [rl, rh], color=color_r, lw=1.2, linestyle="--")
            # Body
            rbody_bottom = min(ro, rc)
            rbody_height = max(abs(rc - ro), 1e-4)
            rect_r = patches.Rectangle(
                (x_r - width * 0.4, rbody_bottom),
                width * 0.8, rbody_height,
                linewidth=1.2, edgecolor=color_r, facecolor="none", linestyle="--"
            )
            ax_main.add_patch(rect_r)

        ax_main.set_title(f"{title} (Bars {start_idx} to {end_idx})", fontsize=14, fontweight="bold")
        ax_main.set_ylabel("Price", fontsize=12)
        ax_main.set_xlim(-1, N)

        # Custom legend
        h1 = patches.Patch(color=c_bull_true, label="Ground Truth Bullish")
        h2 = patches.Patch(color=c_bear_true, label="Ground Truth Bearish")
        h3 = patches.Patch(facecolor="none", edgecolor=c_bull_rec, linestyle="--", label="Reconstructed Bullish")
        h4 = patches.Patch(facecolor="none", edgecolor=c_bear_rec, linestyle="--", label="Reconstructed Bearish")
        ax_main.legend(handles=[h1, h2, h3, h4], loc="upper left", frameon=True)
        ax_main.grid(True, alpha=0.3)

        # Error panel: Close price error and Range error
        close_error = r_sub[:, 3] - t_sub[:, 3]
        range_error = (r_sub[:, 1] - r_sub[:, 2]) - (t_sub[:, 1] - t_sub[:, 2])

        ax_err.bar(np.arange(N) - 0.2, close_error, width=0.4, label="Close Price Residual (Recon - True)", color="#3f51b5", alpha=0.7)
        ax_err.bar(np.arange(N) + 0.2, range_error, width=0.4, label="Range Residual (Recon - True)", color="#ff9800", alpha=0.7)
        ax_err.axhline(0, color="black", lw=0.8, linestyle=":")
        ax_err.set_ylabel("Residual Error", fontsize=11)
        ax_err.set_xlabel("Time Index (Bar #)", fontsize=11)
        ax_err.legend(loc="upper left")
        ax_err.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_bit_utilization(
        self,
        bit_mean_probs: List[float],
        title: str = "Binary Bottleneck Bit Utilization",
        filename: str = "bit_utilization.png"
    ) -> str:
        """
        Plots bit activation probabilities bar chart, highlighting active bits vs collapsed bits.
        """
        probs = np.array(bit_mean_probs)
        B = len(probs)
        indices = np.arange(1, B + 1)

        fig, ax = plt.subplots(figsize=(12, 5))

        # Color: green if in [0.1, 0.9], yellow if in [0.05, 0.1) or (0.9, 0.95], red if collapsed (<0.05 or >0.95)
        colors = []
        for p in probs:
            if 0.1 <= p <= 0.9:
                colors.append("#2ca02c")
            elif 0.05 <= p <= 0.95:
                colors.append("#ff7f0e")
            else:
                colors.append("#d62728")

        ax.bar(indices, probs, color=colors, edgecolor="black", alpha=0.8, width=0.6)
        ax.axhline(0.5, color="black", linestyle="--", lw=1.5, label="Optimal Balanced Activation (50%)")
        ax.axhline(0.05, color="red", linestyle=":", lw=1.0, label="Dead / Collapse Threshold (5% / 95%)")
        ax.axhline(0.95, color="red", linestyle=":", lw=1.0)

        ax.set_title(f"{title} ({B}-Bit Bottleneck)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Bit Index (1 to B)", fontsize=11)
        ax.set_ylabel("Activation Probability P(z_i = 1)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(indices)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_bottleneck_comparison(
        self,
        dimensions: List[int],
        ohlc_maes: List[float],
        feat_mses: List[float],
        unique_tokens: List[int],
        active_bit_pcts: List[float],
        selected_dim: Optional[int] = None,
        title: str = "Adaptive Bottleneck Dimension Optimization",
        filename: str = "bottleneck_comparison.png"
    ) -> str:
        """
        Plots trade-off curves across evaluated bottleneck dimensions.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. OHLC MAE vs Bottleneck Dimension
        axes[0, 0].plot(dimensions, ohlc_maes, marker="o", color="#d62728", lw=2, markersize=8)
        if selected_dim:
            sel_idx = dimensions.index(selected_dim)
            axes[0, 0].plot(selected_dim, ohlc_maes[sel_idx], marker="*", color="gold", markersize=18, markeredgecolor="black", label=f"Selected ({selected_dim}-bit)")
            axes[0, 0].legend()
        axes[0, 0].set_title("OHLC Reconstruction MAE vs Token Size", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Bottleneck Dimension (Bits)")
        axes[0, 0].set_ylabel("MAE (Price Units)")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Feature MSE vs Bottleneck Dimension
        axes[0, 1].plot(dimensions, feat_mses, marker="s", color="#1f77b4", lw=2, markersize=8)
        if selected_dim:
            sel_idx = dimensions.index(selected_dim)
            axes[0, 1].plot(selected_dim, feat_mses[sel_idx], marker="*", color="gold", markersize=18, markeredgecolor="black")
        axes[0, 1].set_title("Feature Reconstruction MSE vs Token Size", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Bottleneck Dimension (Bits)")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Unique Tokens vs Bottleneck Dimension
        axes[1, 0].plot(dimensions, unique_tokens, marker="^", color="#2ca02c", lw=2, markersize=8)
        if selected_dim:
            sel_idx = dimensions.index(selected_dim)
            axes[1, 0].plot(selected_dim, unique_tokens[sel_idx], marker="*", color="gold", markersize=18, markeredgecolor="black")
        axes[1, 0].set_title("Generated Unique Tokens vs Token Size", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Bottleneck Dimension (Bits)")
        axes[1, 0].set_ylabel("Count of Unique Tokens")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Active Bit % vs Bottleneck Dimension
        axes[1, 1].plot(dimensions, active_bit_pcts, marker="d", color="#9467bd", lw=2, markersize=8)
        axes[1, 1].axhline(100.0, color="gray", linestyle=":", lw=1.2)
        if selected_dim:
            sel_idx = dimensions.index(selected_dim)
            axes[1, 1].plot(selected_dim, active_bit_pcts[sel_idx], marker="*", color="gold", markersize=18, markeredgecolor="black")
        axes[1, 1].set_title("Bit Utilization (% Active Bits) vs Token Size", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Bottleneck Dimension (Bits)")
        axes[1, 1].set_ylabel("Active Bits (%)")
        axes[1, 1].set_ylim(0, 105)
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path
