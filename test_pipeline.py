"""
Unit and integration tests for candlestick tokenizer.
"""
import numpy as np
import torch
from candlestick_tokenizer import (
    CandlestickFeatureEngineer,
    CandlestickFeatureScaler,
    CandlestickAutoencoder,
    CandlestickLoss,
    CandlestickEvaluator,
    CandlestickTrainer,
    BinaryBottleneck,
    extract_features,
    reconstruct_ohlc,
    generate_synthetic_ohlc,
    create_dataloaders
)

def test_features_and_reversibility():
    print("Testing Feature Extraction and Exact Reversibility...")
    df = generate_synthetic_ohlc(n_bars=100, seed=123)
    features, o_prev, curr_ohlc = extract_features(df)
    assert features.shape == (99, 5), f"Expected (99, 5), got {features.shape}"
    
    # Exact reconstruction test
    rec_ohlc, diag = reconstruct_ohlc(features, o_prev, enforce_constraints=True)
    assert np.allclose(curr_ohlc, rec_ohlc, atol=1e-5), "Exact reconstruction failed!"
    print("-> Exact reversibility without autoencoder passes 100%!")

def test_model_and_ste():
    print("Testing Binary Autoencoder and Straight-Through Estimator...")
    model = CandlestickAutoencoder(input_dim=5, bottleneck_dim=16)
    x = torch.randn(10, 5)
    recon_x, z, p, logits = model(x)
    assert recon_x.shape == (10, 5)
    assert z.shape == (10, 16)
    assert p.shape == (10, 16)
    # Check that z contains strictly 0.0 or 1.0
    assert torch.all((z == 0.0) | (z == 1.0)), "Bottleneck output z is not strictly binary!"
    
    # Check gradient flow to model weights
    loss = recon_x.sum()
    loss.backward()
    grad_exists = model.encoder.backbone[0].weight.grad is not None
    assert grad_exists, "Gradient did not flow back to encoder weights through STE!"
    print("-> Model forward and STE backward pass verified!")

def test_token_conversions():
    print("Testing Binary Token Formats (Tensors, Bitstrings, Packed Bytes, Integers)...")
    z = torch.tensor([
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    ])
    bitstrings = BinaryBottleneck.to_bitstring(z)
    assert bitstrings == ["10110010", "01001101"], f"Unexpected bitstrings: {bitstrings}"
    
    z_from_str = BinaryBottleneck.from_bitstring(bitstrings)
    assert torch.all(z == z_from_str), "Bitstring roundtrip failed!"
    
    packed = BinaryBottleneck.to_packed_bytes(z)
    z_from_packed = BinaryBottleneck.from_packed_bytes(packed, bottleneck_dim=8)
    assert torch.all(z == z_from_packed), "Packed bytes roundtrip failed!"
    
    int_tokens = BinaryBottleneck.to_integer_tokens(z)
    z_from_int = BinaryBottleneck.from_integer_tokens(int_tokens, bottleneck_dim=8)
    assert torch.all(z == z_from_int), "Integer tokens roundtrip failed!"
    print("-> All binary token formats roundtrip successfully!")

if __name__ == "__main__":
    test_features_and_reversibility()
    test_model_and_ste()
    test_token_conversions()
    print("\nALL UNIT TESTS PASSED!")
