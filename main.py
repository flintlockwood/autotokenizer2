"""
End-to-End Candlestick Tokenizer: Main Execution, Training, Evaluation, and Optimization Script.

Runs the complete 3-iteration development and verification process:
- Iteration 1: Initial PyTorch autoencoder implementation and baseline training (e.g. 16-bit)
- Iteration 2: Deep metric analysis (reconstruction fidelity, bit utilization, reversibility)
- Iteration 3: Adaptive Bottleneck Optimization across dimensions (8, 16, 24, 32, 48, 64 bits)
- Interactive Demonstration: Discrete encoding and reverse decoding of sample candlesticks
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

from candlestick_tokenizer.features import (
    CandlestickFeatureEngineer,
    CandlestickFeatureScaler,
    extract_features,
    reconstruct_ohlc,
    validate_ohlc_data,
)
from candlestick_tokenizer.model import (
    CandlestickAutoencoder,
    BinaryBottleneck,
)
from candlestick_tokenizer.losses import CandlestickLoss
from candlestick_tokenizer.dataset import (
    load_ohlc_csv,
    generate_synthetic_ohlc,
    create_dataloaders,
)
from candlestick_tokenizer.evaluator import CandlestickEvaluator
from candlestick_tokenizer.trainer import CandlestickTrainer
from candlestick_tokenizer.visualizer import CandlestickVisualizer
from candlestick_tokenizer.optimizer import BottleneckOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Reversible Binary Candlestick Tokenizer")
    parser.add_argument(
        "--data",
        type=str,
        default="btc_1h",
        help="Data source: 'btc_1h', 'spy_daily', 'synthetic', or path to a CSV file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--bottleneck_dim",
        type=int,
        default=16,
        help="Initial / baseline bottleneck dimension (bits)"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 128, 64],
        help="Hidden layer dimensions for encoder and decoder"
    )
    parser.add_argument(
        "--search_bottlenecks",
        action="store_true",
        default=True,
        help="Run adaptive multi-dimension bottleneck search across [8, 16, 24, 32, 48, 64] bits"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save models, metrics, and visualization figures"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training ('cpu' or 'cuda')"
    )
    return parser.parse_args()


def load_dataset(data_arg: str) -> pd.DataFrame:
    """
    Loads dataset based on name or file path.
    """
    if data_arg == "btc_1h":
        path = os.path.join("data", "btc_1h.csv")
        if os.path.exists(path):
            print(f"Loading cached BTC 1-hour dataset from {path}...")
            return load_ohlc_csv(path)
        else:
            print("BTC 1-hour dataset not found locally. Generating synthetic fallback...")
            return generate_synthetic_ohlc(n_bars=5000, seed=42)

    elif data_arg == "spy_daily":
        path = os.path.join("data", "spy_daily.csv")
        if os.path.exists(path):
            print(f"Loading cached SPY daily dataset from {path}...")
            return load_ohlc_csv(path)
        else:
            print("SPY daily dataset not found locally. Generating synthetic fallback...")
            return generate_synthetic_ohlc(n_bars=3000, seed=42)

    elif data_arg == "synthetic":
        print("Generating 5,000 synthetic candlesticks using intraday GBM + jumps...")
        return generate_synthetic_ohlc(n_bars=5000, seed=42)

    elif os.path.exists(data_arg):
        print(f"Loading custom dataset from {data_arg}...")
        return load_ohlc_csv(data_arg)
    else:
        raise FileNotFoundError(f"Cannot resolve data source: '{data_arg}'")


def demo_sample_reconstruction(
    model: CandlestickAutoencoder,
    scaler: CandlestickFeatureScaler,
    test_splits: dict,
    n_samples: int = 5
):
    """
    Demonstrates discrete encoding and decoding on actual sample candlesticks from the test set.
    """
    print("\n" + "=" * 90)
    print("DEMONSTRATION: Discrete Binary Candlestick Encoding & Reverse Decoding")
    print("=" * 90)

    feat_norm = test_splits["test_features_norm"][:n_samples]
    o_prev = test_splits["test_o_prev"][:n_samples]
    true_ohlc = test_splits["test_ohlc"][:n_samples]

    device = next(model.parameters()).device
    t_feat = torch.tensor(feat_norm, dtype=torch.float32, device=device)

    # 1. Discrete Encoding
    z_tensor = model.encode(t_feat, output_format="tensor")
    bitstrings = model.encode(t_feat, output_format="bitstring")
    packed_bytes = model.encode(t_feat, output_format="packed")

    # 2. Reverse Decoding from discrete tokens
    recon_feat_norm = model.decode(z_tensor, input_format="tensor")
    recon_features = scaler.inverse_transform(recon_feat_norm.cpu().numpy())
    recon_ohlc, diag = reconstruct_ohlc(recon_features, o_prev, enforce_constraints=True)

    for i in range(n_samples):
        o_t, h_t, l_t, c_t = true_ohlc[i]
        ro, rh, rl, rc = recon_ohlc[i]
        candle_type = "Bullish" if c_t > o_t else ("Bearish" if c_t < o_t else "Doji")
        token_str = bitstrings[i]
        mae = float(np.mean(np.abs(true_ohlc[i] - recon_ohlc[i])))
        mape = float(np.mean(np.abs(true_ohlc[i] - recon_ohlc[i]) / true_ohlc[i])) * 100.0

        print(f"\n--- Candlestick #{i + 1} [{candle_type}] ---")
        print(f"  Reference Prev Open: {o_prev[i]:.2f}")
        print(f"  Binary Token ({len(token_str)} bits) : {token_str}")
        print(f"  Packed Hex Bytes         : {packed_bytes[i].tobytes().hex()}")
        print(f"  Target OHLC              : O={o_t:8.2f}, H={h_t:8.2f}, L={l_t:8.2f}, C={c_t:8.2f}")
        print(f"  Reconstructed OHLC       : O={ro:8.2f}, H={rh:8.2f}, L={rl:8.2f}, C={rc:8.2f}")
        print(f"  Absolute Errors          : dO={abs(ro-o_t):6.2f}, dH={abs(rh-h_t):6.2f}, dL={abs(rl-l_t):6.2f}, dC={abs(rc-c_t):6.2f}")
        print(f"  Bar MAE: {mae:6.2f} | MAPE: {mape:5.2f}%")

    print("\n" + "=" * 90)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"\n>>> Running Candlestick Tokenizer on device: {device}")
    print(f">>> Configuration: Data={args.data}, Epochs={args.epochs}, BatchSize={args.batch_size}, InitialBits={args.bottleneck_dim}")

    # 1. Load data
    df_ohlc = load_dataset(args.data)
    print(f"Dataset loaded: {len(df_ohlc)} candlestick bars.")

    # 2. Extract features, split chronologically, fit scaler
    print("\nExtracting 5D stationary features and creating chronological DataLoaders (70% train / 15% val / 15% test)...")
    train_loader, val_loader, test_loader, scaler, splits_data = create_dataloaders(
        ohlc_df_or_arr=df_ohlc,
        batch_size=args.batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
    )
    scaler_path = os.path.join(args.output_dir, "feature_scaler.json")
    scaler.save(scaler_path)
    print(f"Feature scaler parameters saved to: {scaler_path}")
    print(f"Train samples: {len(splits_data['train_features_norm'])}, Val: {len(splits_data['val_features_norm'])}, Test: {len(splits_data['test_features_norm'])}")

    visualizer = CandlestickVisualizer(output_dir=vis_dir)

    # =========================================================================
    # ITERATION 1: Initial Baseline Autoencoder (e.g. 16-bit)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"ITERATION 1: Initial Baseline Implementation ({args.bottleneck_dim}-Bit Binary Autoencoder)")
    print("=" * 80)

    model_baseline = CandlestickAutoencoder(
        input_dim=5,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dims=args.hidden_dims,
        activation="leaky_relu",
        dropout=0.0,
    ).to(device)

    loss_fn = CandlestickLoss(
        lambda_recon=1.0,
        lambda_binary=0.05,
        lambda_entropy=0.1,
        lambda_price=0.0,
        scaler=scaler,
    )

    trainer_baseline = CandlestickTrainer(
        model=model_baseline,
        loss_fn=loss_fn,
        learning_rate=args.lr,
        device=device,
        early_stopping_patience=12,
    )

    print(f"Training {args.bottleneck_dim}-bit baseline model for up to {args.epochs} epochs...")
    history_baseline = trainer_baseline.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        verbose=True
    )

    # Save baseline model
    baseline_path = os.path.join(args.output_dir, f"model_baseline_{args.bottleneck_dim}bit.pt")
    model_baseline.save(baseline_path)
    print(f"Baseline model checkpoint saved to: {baseline_path}")

    # =========================================================================
    # ITERATION 2: In-Depth Evaluation and Analysis of Baseline Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITERATION 2: In-Depth Evaluation & Analysis of Baseline Model")
    print("=" * 80)

    evaluator_baseline = CandlestickEvaluator(model=model_baseline, scaler=scaler, device=device)
    metrics_baseline = evaluator_baseline.evaluate(test_loader)

    print("\n--- Baseline Model Evaluation Results (Test Set) ---")
    print(f"  Bottleneck Size              : {metrics_baseline['bottleneck_dim']} bits")
    print(f"  Strictly Discrete Binary     : {metrics_baseline['is_strictly_binary']}")
    print(f"  Feature Reconstruction MSE   : {metrics_baseline['feature_mse_overall']:.6f}")
    print(f"  OHLC Reconstruction MAE      : {metrics_baseline['ohlc_mae_overall']:.4f}")
    print(f"  OHLC Reconstruction RMSE     : {metrics_baseline['ohlc_rmse_overall']:.4f}")
    print(f"  OHLC Mean Absolute % Error   : {metrics_baseline['mape_overall']:.2f}%")
    print(f"  Max Absolute Price Error     : {metrics_baseline['max_abs_error_overall']:.4f}")
    print(f"  Binary Bit Entropy / Util.   : {metrics_baseline['entropy_utilization_pct']:.2f}%")
    print(f"  Active Bits Count            : {metrics_baseline['active_bit_count']}/{metrics_baseline['bottleneck_dim']} ({metrics_baseline['active_bit_pct']:.1f}%)")
    print(f"  Unique Binary Tokens         : {metrics_baseline['num_unique_tokens']} (Diversity: {metrics_baseline['token_diversity_ratio']*100:.2f}%)")
    print(f"  Candle Direction Accuracy    : {metrics_baseline['direction_accuracy_pct']:.2f}%")
    print(f"  Bullish Candles MAE          : {metrics_baseline['bullish_candles']['mae']:.4f} (count: {metrics_baseline['bullish_candles']['count']})")
    print(f"  Bearish Candles MAE          : {metrics_baseline['bearish_candles']['mae']:.4f} (count: {metrics_baseline['bearish_candles']['count']})")
    print(f"  Doji Candles MAE             : {metrics_baseline['doji_candles']['mae']:.4f} (count: {metrics_baseline['doji_candles']['count']})")

    # Generate baseline charts
    curves_file = visualizer.plot_training_curves(
        history_baseline,
        title=f"Training Curves ({args.bottleneck_dim}-Bit Baseline)",
        filename="baseline_training_curves.png"
    )
    bit_file = visualizer.plot_bit_utilization(
        metrics_baseline["bit_mean_probabilities"],
        title=f"Bit Activation Rates ({args.bottleneck_dim}-Bit)",
        filename="baseline_bit_utilization.png"
    )

    # Generate reconstructed OHLC for visualization
    test_feat_t = torch.tensor(splits_data["test_features_norm"], dtype=torch.float32, device=device)
    z_base = model_baseline.encode(test_feat_t, output_format="tensor")
    recon_feat_base = model_baseline.decode(z_base, input_format="tensor")
    recon_feat_np = scaler.inverse_transform(recon_feat_base.cpu().numpy())
    recon_ohlc_base, _ = reconstruct_ohlc(recon_feat_np, splits_data["test_o_prev"], enforce_constraints=True)

    candle_file = visualizer.plot_candlestick_reconstruction(
        true_ohlc=splits_data["test_ohlc"],
        recon_ohlc=recon_ohlc_base,
        start_idx=0,
        n_bars=60,
        title=f"Original vs Reconstructed Candlesticks ({args.bottleneck_dim}-Bit)",
        filename="baseline_candlestick_comparison.png"
    )
    print(f"Baseline visualization saved to: {vis_dir}")

    # =========================================================================
    # ITERATION 3: Adaptive Bottleneck Optimization [8, 16, 24, 32, 48, 64]
    # =========================================================================
    candidate_dims = [8, 16, 24, 32, 48, 64]
    print("\n" + "=" * 80)
    print(f"ITERATION 3: Adaptive Bottleneck Optimization across {candidate_dims} bits")
    print("=" * 80)

    optimizer = BottleneckOptimizer(
        candidate_dimensions=candidate_dims,
        hidden_dims=args.hidden_dims,
        activation="leaky_relu",
        epochs=args.epochs,
        learning_rate=args.lr,
        lambda_recon=1.0,
        lambda_binary=0.05,
        lambda_entropy=0.1,
        early_stopping_patience=10,
        device=device,
    )

    df_results = optimizer.run_search(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        verbose=True
    )

    # Save comparison table
    results_csv = os.path.join(args.output_dir, "bottleneck_comparison.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\nOptimization summary table saved to: {results_csv}")
    print("\n" + df_results.to_string(index=False))

    optimal_dim, optimal_model = optimizer.get_optimal_model()
    print(f"\n>>> Selected Optimal Token Size: {optimal_dim} bits")

    # Save optimal model
    optimal_path = os.path.join(args.output_dir, f"model_optimal_{optimal_dim}bit.pt")
    optimal_model.save(optimal_path)
    print(f"Optimal model checkpoint saved to: {optimal_path}")

    # Plot multi-dimension comparison curves
    comp_plot = visualizer.plot_bottleneck_comparison(
        dimensions=df_results["bottleneck_dim"].tolist(),
        ohlc_maes=df_results["test_ohlc_mae"].tolist(),
        feat_mses=df_results["test_feat_mse"].tolist(),
        unique_tokens=df_results["unique_tokens"].tolist(),
        active_bit_pcts=df_results["active_bit_pct"].tolist(),
        selected_dim=optimal_dim,
        filename="bottleneck_tradeoff_curves.png"
    )
    print(f"Bottleneck comparison plot saved to: {comp_plot}")

    # Reconstruct with optimal model for visualization
    z_opt = optimal_model.encode(test_feat_t, output_format="tensor")
    recon_feat_opt = optimal_model.decode(z_opt, input_format="tensor")
    recon_feat_opt_np = scaler.inverse_transform(recon_feat_opt.cpu().numpy())
    recon_ohlc_opt, _ = reconstruct_ohlc(recon_feat_opt_np, splits_data["test_o_prev"], enforce_constraints=True)

    opt_candle_file = visualizer.plot_candlestick_reconstruction(
        true_ohlc=splits_data["test_ohlc"],
        recon_ohlc=recon_ohlc_opt,
        start_idx=0,
        n_bars=60,
        title=f"Original vs Reconstructed Candlesticks (Optimal {optimal_dim}-Bit Tokenizer)",
        filename="optimal_candlestick_comparison.png"
    )

    # Evaluate optimal model comprehensively
    evaluator_opt = CandlestickEvaluator(model=optimal_model, scaler=scaler, device=device)
    metrics_opt = evaluator_opt.evaluate(test_loader)

    # Save detailed JSON metrics
    metrics_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_16bit": metrics_baseline,
            "optimal_model": metrics_opt,
            "search_summary": df_results.to_dict(orient="records"),
        }, f, indent=2)
    print(f"Complete metrics JSON saved to: {metrics_path}")

    # Sample Candlestick Demonstration
    demo_sample_reconstruction(
        model=optimal_model,
        scaler=scaler,
        test_splits=splits_data,
        n_samples=5
    )

    print("\n" + "=" * 80)
    print("TOKENIZER PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
