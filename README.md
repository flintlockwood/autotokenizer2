# Reversible Binary Candlestick Tokenizer using PyTorch Autoencoder

A production-grade, mathematically reversible candlestick tokenizer implemented in Python and PyTorch. It transforms raw financial candlestick data (OHLC: Open, High, Low, Close) into compact, discrete binary tokens suitable for Transformer sequence modeling, while preserving sufficient geometric and stationary price information to reconstruct the original OHLC candles with minimal loss.

---

## 1. Key Highlights

1. **Strictly Discrete Binary Bottleneck**:
   - Discrete tokens in $\{0, 1\}^B$ generated directly by a Straight-Through Estimator (STE) binary bottleneck layer.
   - Zero continuous latent rounding after the fact; during both training forward passes and evaluation/inference, the decoder receives pure binary vectors.
2. **5D Stationary Feature Representation**:
   - Uses the previous candlestick's Open price ($O_{t-1}$) as the stationary reference.
   - Preserves candle direction via the **signed real body ratio** $f_5 \in [-1, 1]$.
   - Invertible scaling (`CandlestickFeatureScaler`) fits exclusively on training sets to eliminate forward lookahead leakage.
3. **Exact Mathematical Reversibility**:
   - Complete analytical inverse formulas reconstruct $\hat{H}_t, \hat{L}_t, \hat{O}_t, \hat{C}_t$ and enforce physical OHLC constraints ($\hat{L} \le \min(\hat{O}, \hat{C})$ and $\hat{H} \ge \max(\hat{O}, \hat{C})$).
4. **Adaptive Bottleneck Optimization**:
   - Evaluates multi-dimension candidate bottlenecks ($B \in [8, 16, 24, 32, 48, 64]$ bits).
   - Entropy and binary commitment regularization prevent bit collapse, maintaining **100% active bits** and **>95% entropy utilization** across all dimensions.
5. **Multi-Format Token Serialization**:
   - Supports export to binary float tensors, ASCII bitstrings (`"01101001..."`), packed hex bytes (`b98f64dd...`), and 64-bit integer token IDs for direct `nn.Embedding` lookup.

---

## 2. Mathematical Formulation

### 2.1 Feature Engineering (5 Stationary Features)

For candlestick at time $t$ with reference to previous candlestick Open $O_{t-1}$:

1. **High Price Movement (%)**:
   $$f_1 = \frac{H_t - O_{t-1}}{O_{t-1}} \times 100$$
2. **Low Price Movement (%)**:
   $$f_2 = \frac{L_t - O_{t-1}}{O_{t-1}} \times 100$$
3. **Upper Wick Ratio**:
   $$f_3 = \frac{H_t - \max(O_t, C_t)}{H_t - L_t}$$
4. **Lower Wick Ratio**:
   $$f_4 = \frac{\min(O_t, C_t) - L_t}{H_t - L_t}$$
5. **Signed Body Ratio** (Direction & Magnitude):
   $$f_5 = \frac{C_t - O_t}{H_t - L_t}$$
   - $f_5 > 0$: Bullish candle ($C_t > O_t$)
   - $f_5 < 0$: Bearish candle ($C_t < O_t$)
   - $f_5 = 0$: Doji candle ($C_t = O_t$)

*(Zero-range edge cases where $H_t = L_t$ are safely clamped with $f_3 = f_4 = f_5 = 0$.)*

### 2.2 Inverse OHLC Reconstruction

Given reconstructed features $\hat{X}_t = [\hat{f}_1, \hat{f}_2, \hat{f}_3, \hat{f}_4, \hat{f}_5]$ and reference price $O_{t-1}$:

$$\hat{H}_t = O_{t-1} \left(1 + \frac{\hat{f}_1}{100}\right)$$
$$\hat{L}_t = O_{t-1} \left(1 + \frac{\hat{f}_2}{100}\right)$$
$$R_t = \hat{H}_t - \hat{L}_t$$
$$B_t = \hat{f}_5 R_t, \quad LW_t = \hat{f}_4 R_t, \quad UW_t = \hat{f}_3 R_t$$
$$\hat{O}_t = \hat{L}_t + LW_t + \max(0, -B_t)$$
$$\hat{C}_t = \hat{O}_t + B_t$$

Physical constraints are enforced as:
$$\hat{L}_{\text{valid}} = \min(\hat{L}_t, \hat{O}_t, \hat{C}_t), \quad \hat{H}_{\text{valid}} = \max(\hat{H}_t, \hat{O}_t, \hat{C}_t)$$

---

## 3. Directory Structure

```
autotokenizer/
├── candlestick_tokenizer/
│   ├── __init__.py           # Package exports
│   ├── features.py           # 5D feature extraction, invertible scaler, and OHLC reconstruction
│   ├── model.py              # PyTorch Encoder, STE BinaryBottleneck, Decoder, Autoencoder
│   ├── losses.py             # Feature MSE, Binary Commitment, Bit Entropy, and Price Loss
│   ├── dataset.py            # CSV Loader, GBM Synthetic Generator, Chronological DataLoaders
│   ├── evaluator.py          # 9 required metrics, segmented candle accuracy, bit utilization
│   ├── trainer.py            # AdamW training loop with cosine scheduling and early stopping
│   ├── visualizer.py         # Matplotlib candlestick comparison, curves, and bottleneck trade-offs
│   └── optimizer.py          # Multi-dimension bottleneck search [8, 16, 24, 32, 48, 64 bits]
├── data/
│   ├── btc_1h.csv            # 16,756 bars (BTC-USD 1-hour)
│   ├── spy_daily.csv         # 1,258 bars (SPY ETF daily)
│   └── synthetic_daily.csv   # 5,000 bars (GBM intraday jump diffusion)
├── outputs/
│   ├── feature_scaler.json   # Saved normalization parameters (mean & std)
│   ├── bottleneck_comparison.csv
│   ├── final_metrics.json    # Full evaluation metrics across iterations
│   ├── model_baseline_16bit.pt
│   ├── model_optimal_64bit.pt
│   └── visualizations/
│       ├── baseline_training_curves.png
│       ├── baseline_candlestick_comparison.png
│       ├── baseline_bit_utilization.png
│       ├── bottleneck_tradeoff_curves.png
│       └── optimal_candlestick_comparison.png
├── test_pipeline.py          # Automated unit & integration tests
├── main.py                   # Unified end-to-end training & evaluation CLI
└── README.md
```

---

## 4. Quickstart & Usage

### 4.1 Running Unit Tests
```bash
python test_pipeline.py
```

### 4.2 Running Full Training & Optimization Pipeline
```bash
python main.py --data btc_1h --epochs 35 --batch_size 256 --bottleneck_dim 16 --output_dir outputs
```

### 4.3 Training on Other Datasets
- **SPY Equity Daily Data**:
  ```bash
  python main.py --data spy_daily --epochs 40 --batch_size 64
  ```
- **Synthetic Candlestick Data**:
  ```bash
  python main.py --data synthetic --epochs 35 --batch_size 128
  ```
- **Custom CSV File**:
  ```bash
  python main.py --data path/to/your_data.csv --epochs 35
  ```

---

## 5. Python API Usage

```python
import torch
import numpy as np
from candlestick_tokenizer import (
    CandlestickAutoencoder,
    CandlestickFeatureEngineer,
    load_ohlc_csv,
    reconstruct_ohlc
)

# 1. Load data & initialize feature engineer
df = load_ohlc_csv("data/btc_1h.csv")
fe = CandlestickFeatureEngineer()
features_norm, o_prev, curr_ohlc = fe.fit_transform(df)

# 2. Load trained optimal model
model = CandlestickAutoencoder.load("outputs/model_optimal_64bit.pt")
model.eval()

# 3. Discrete Tokenization (Encoding)
t_feat = torch.tensor(features_norm[:5], dtype=torch.float32)
# Export to bitstrings: e.g. ['1001100100000111...', ...]
bitstrings = model.encode(t_feat, output_format="bitstring")
# Export to packed hex bytes: e.g. ['990774c0d8d2cce6', ...]
packed_bytes = model.encode(t_feat, output_format="packed")
print("Discrete Binary Tokens:", bitstrings)

# 4. Reverse Decoding from Tokens
z_tensor = model.encode(t_feat, output_format="tensor") # Discrete {0, 1}
recon_norm = model.decode(z_tensor, input_format="tensor")

# 5. Exact OHLC Price Reconstruction
recon_ohlc, diag = fe.inverse_transform_and_reconstruct(
    recon_norm, 
    o_prev[:5], 
    enforce_constraints=True
)
print("Reconstructed OHLC:\n", recon_ohlc)
```

---

## 6. Downstream Transformer Integration

The binary tokens produced by this tokenizer can be plugged directly into a Transformer sequence architecture:

1. **Continuous Linear Embedding**:
   Feed the $B$-dimensional binary vector $Z_t \in \{0, 1\}^B$ into an `nn.Linear(B, d_model)` layer.
2. **Byte-Level Embedding**:
   Pack a 64-bit token into eight 8-bit bytes (each byte $\in [0, 255]$) and embed using a standard byte vocabulary table `nn.Embedding(256, d_model)`.
3. **Discrete Integer Token Lookup**:
   For $B \le 16$, map the binary vector directly to an integer token $k \in [0, 2^B - 1]$ for an embedding table `nn.Embedding(2**B, d_model)`.
