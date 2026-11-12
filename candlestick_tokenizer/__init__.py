"""
Reversible Binary Candlestick Tokenizer using PyTorch Autoencoder.

Transforms financial candlestick data (OHLC) into compact binary tokens
suitable for Transformer sequence modeling, with exact mathematical reversibility.
"""

from .features import (
    CandlestickFeatureEngineer,
    CandlestickFeatureScaler,
    extract_features,
    reconstruct_ohlc,
    validate_ohlc_data
)
from .model import (
    BinaryBottleneck,
    BinaryQuantizerSTE,
    CandlestickEncoder,
    CandlestickDecoder,
    CandlestickAutoencoder
)
from .losses import CandlestickLoss
from .dataset import (
    CandlestickDataset,
    load_ohlc_csv,
    generate_synthetic_ohlc,
    create_dataloaders
)
from .evaluator import CandlestickEvaluator
from .trainer import CandlestickTrainer
from .visualizer import CandlestickVisualizer
from .optimizer import BottleneckOptimizer

__all__ = [
    "CandlestickFeatureEngineer",
    "CandlestickFeatureScaler",
    "extract_features",
    "reconstruct_ohlc",
    "validate_ohlc_data",
    "BinaryBottleneck",
    "BinaryQuantizerSTE",
    "CandlestickEncoder",
    "CandlestickDecoder",
    "CandlestickAutoencoder",
    "CandlestickLoss",
    "CandlestickDataset",
    "load_ohlc_csv",
    "generate_synthetic_ohlc",
    "create_dataloaders",
    "CandlestickEvaluator",
    "CandlestickTrainer",
    "CandlestickVisualizer",
    "BottleneckOptimizer",
]
