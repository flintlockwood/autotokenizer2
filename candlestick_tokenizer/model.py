"""
PyTorch Autoencoder Architecture with Discrete Binary Bottleneck.

Implements:
- Straight-Through Estimator (STE) differentiable binary quantization
- Configurable Encoder and Decoder networks with LayerNorm and non-linearities
- Multi-format token representation (tensor, bitstrings, packed bytes, integer IDs)
- Full serializability and inspection utilities
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryQuantizerSTE(nn.Module):
    """
    Differentiable Binary Quantizer using the Straight-Through Estimator (STE).
    
    Forward pass:
        Computes probabilities p = sigmoid(logits / temperature).
        Discretizes p to binary values z_hard in {0.0, 1.0} using threshold 0.5
        (or stochastic sampling if training with stochastic=True).
        Returns z = p + (z_hard - p).detach()
    
    Backward pass:
        d(z) / d(p) = 1, so gradients flow directly through to logits and encoder.
    """

    def __init__(self, temperature: float = 1.0, stochastic: bool = False):
        super().__init__()
        self.temperature = temperature
        self.stochastic = stochastic

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: (batch_size, bottleneck_dim) continuous pre-activations
            
        Returns:
            z: (batch_size, bottleneck_dim) discrete binary values in {0.0, 1.0}
            p: (batch_size, bottleneck_dim) continuous probabilities in (0, 1)
        """
        p = torch.sigmoid(logits / self.temperature)

        if self.training and self.stochastic:
            # Stochastic Bernoulli sampling during training
            rand = torch.rand_like(p)
            z_hard = (p > rand).float()
        else:
            # Deterministic thresholding at 0.5
            z_hard = (p >= 0.5).float()

        # Straight-Through Estimator trick:
        # Forward pass output is exactly z_hard
        # Backward pass gradient is dL/dz * dp/dlogits
        z = p + (z_hard - p).detach()
        return z, p


class BinaryBottleneck(nn.Module):
    """
    Discrete Binary Bottleneck Layer.
    
    Projects latent representations into binary tokens of size `bottleneck_dim`,
    and provides conversion utilities between binary tensors, bitstrings,
    integer tokens, and packed bit arrays.
    """

    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int,
        temperature: float = 1.0,
        stochastic: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        self.fc_latent = nn.Linear(in_features, bottleneck_dim)
        self.quantizer = BinaryQuantizerSTE(temperature=temperature, stochastic=stochastic)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, in_features)
            
        Returns:
            z: (batch_size, bottleneck_dim) discrete binary tensor in {0.0, 1.0}
            p: (batch_size, bottleneck_dim) continuous probabilities
            logits: (batch_size, bottleneck_dim) pre-sigmoid logits
        """
        logits = self.fc_latent(x)
        z, p = self.quantizer(logits)
        return z, p, logits

    @staticmethod
    def to_bitstring(z: Union[torch.Tensor, np.ndarray]) -> List[str]:
        """
        Converts a batch of binary vectors into a list of bitstrings, e.g. ['01101...', ...].
        """
        if isinstance(z, torch.Tensor):
            arr = (z >= 0.5).cpu().to(torch.uint8).numpy()
        else:
            arr = (np.asarray(z) >= 0.5).astype(np.uint8)

        bitstrings = ["".join(map(str, row)) for row in arr]
        return bitstrings

    @staticmethod
    def from_bitstring(bitstrings: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Converts a list of bitstrings back to a binary float tensor.
        """
        data = [[float(c) for c in s] for s in bitstrings]
        return torch.tensor(data, dtype=torch.float32, device=device)

    @staticmethod
    def to_packed_bytes(z: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Packs binary tokens into compact uint8 arrays via numpy.packbits.
        """
        if isinstance(z, torch.Tensor):
            arr = (z >= 0.5).cpu().to(torch.uint8).numpy()
        else:
            arr = (np.asarray(z) >= 0.5).astype(np.uint8)
        return np.packbits(arr, axis=-1)

    @staticmethod
    def from_packed_bytes(
        packed: np.ndarray,
        bottleneck_dim: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Unpacks packed uint8 arrays back to binary float tensors.
        """
        unpacked = np.unpackbits(packed, axis=-1)[..., :bottleneck_dim]
        return torch.tensor(unpacked, dtype=torch.float32, device=device)

    @staticmethod
    def to_integer_tokens(z: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Converts binary tokens of size B <= 64 into 64-bit integer token IDs.
        Useful for Transformer vocabulary lookup.
        """
        if isinstance(z, torch.Tensor):
            arr = (z >= 0.5).cpu().to(torch.int64).numpy()
        else:
            arr = (np.asarray(z) >= 0.5).astype(np.int64)

        B = arr.shape[1]
        if B > 64:
            raise ValueError(f"Direct integer token conversion requires B <= 64, got B={B}")

        powers = (1 << np.arange(B, dtype=np.int64))[::-1]
        tokens = np.dot(arr, powers)
        return tokens

    @staticmethod
    def from_integer_tokens(
        tokens: Union[List[int], np.ndarray],
        bottleneck_dim: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Converts integer tokens back into a binary float tensor of shape (N, bottleneck_dim).
        """
        tokens_arr = np.asarray(tokens, dtype=np.int64)
        powers = (1 << np.arange(bottleneck_dim, dtype=np.int64))[::-1]
        # (N, 1) & (1, B) != 0
        bits = ((tokens_arr[:, None] & powers[None, :]) != 0).astype(np.float32)
        return torch.tensor(bits, dtype=torch.float32, device=device)


class CandlestickEncoder(nn.Module):
    """
    Encoder network that maps 5D candlestick features into a discrete binary latent vector.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Optional[List[int]] = None,
        bottleneck_dim: int = 16,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        temperature: float = 1.0,
        stochastic: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dims = hidden_dims or [64, 128, 64]

        # Select activation function
        if activation == "leaky_relu":
            act_fn = lambda: nn.LeakyReLU(0.1)
        elif activation == "gelu":
            act_fn = lambda: nn.GELU()
        elif activation == "relu":
            act_fn = lambda: nn.ReLU()
        else:
            act_fn = lambda: nn.LeakyReLU(0.1)

        layers = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.bottleneck = BinaryBottleneck(
            in_features=prev_dim,
            bottleneck_dim=bottleneck_dim,
            temperature=temperature,
            stochastic=stochastic
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        z, p, logits = self.bottleneck(features)
        return z, p, logits


class CandlestickDecoder(nn.Module):
    """
    Decoder network that maps a discrete binary token in {0, 1}^B back to
    the 5D reconstructed candlestick feature vector.
    """

    def __init__(
        self,
        bottleneck_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 5,
        activation: str = "leaky_relu",
        dropout: float = 0.0
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 128, 64]

        if activation == "leaky_relu":
            act_fn = lambda: nn.LeakyReLU(0.1)
        elif activation == "gelu":
            act_fn = lambda: nn.GELU()
        elif activation == "relu":
            act_fn = lambda: nn.ReLU()
        else:
            act_fn = lambda: nn.LeakyReLU(0.1)

        layers = []
        prev_dim = bottleneck_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection back to 5D feature space
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Receives binary token z in {0, 1}^B and reconstructs 5D features.
        """
        return self.network(z)


class CandlestickAutoencoder(nn.Module):
    """
    End-to-End Reversible Candlestick Autoencoder with Discrete Binary Bottleneck.
    """

    def __init__(
        self,
        input_dim: int = 5,
        bottleneck_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        temperature: float = 1.0,
        stochastic: bool = False
    ):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "bottleneck_dim": bottleneck_dim,
            "hidden_dims": hidden_dims or [64, 128, 64],
            "activation": activation,
            "dropout": dropout,
            "temperature": temperature,
            "stochastic": stochastic,
        }

        self.encoder = CandlestickEncoder(
            input_dim=input_dim,
            hidden_dims=self.config["hidden_dims"],
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dropout=dropout,
            temperature=temperature,
            stochastic=stochastic
        )

        # Decoder uses symmetric hidden layers by default
        decoder_hidden_dims = list(reversed(self.config["hidden_dims"]))
        self.decoder = CandlestickDecoder(
            bottleneck_dim=bottleneck_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            activation=activation,
            dropout=dropout
        )

    @property
    def bottleneck_dim(self) -> int:
        return self.config["bottleneck_dim"]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 5) normalized input features
            
        Returns:
            recon_x: (batch_size, 5) reconstructed normalized features
            z: (batch_size, bottleneck_dim) discrete binary token in {0.0, 1.0}
            p: (batch_size, bottleneck_dim) continuous latent probabilities
            logits: (batch_size, bottleneck_dim) pre-sigmoid logits
        """
        z, p, logits = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z, p, logits

    def encode(
        self,
        x: torch.Tensor,
        output_format: str = "tensor"
    ) -> Union[torch.Tensor, List[str], np.ndarray]:
        """
        Encodes candlestick features into binary tokens.
        
        output_format options:
            'tensor': PyTorch float tensor of {0.0, 1.0}
            'bitstring': list of string representations e.g. ['01101...', ...]
            'int': numpy array of 64-bit integer token IDs (for B <= 64)
            'packed': packed uint8 numpy array
        """
        self.eval()
        with torch.no_grad():
            z, _, _ = self.encoder(x)

        if output_format == "tensor":
            return z
        elif output_format == "bitstring":
            return BinaryBottleneck.to_bitstring(z)
        elif output_format == "int":
            return BinaryBottleneck.to_integer_tokens(z)
        elif output_format == "packed":
            return BinaryBottleneck.to_packed_bytes(z)
        else:
            raise ValueError(f"Unknown output_format: {output_format}")

    def decode(
        self,
        tokens: Union[torch.Tensor, List[str], np.ndarray],
        input_format: str = "tensor"
    ) -> torch.Tensor:
        """
        Decodes binary tokens back to reconstructed 5D features.
        """
        self.eval()
        device = next(self.parameters()).device

        if input_format == "tensor":
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.float32, device=device)
            z = tokens.to(device)
        elif input_format == "bitstring":
            z = BinaryBottleneck.from_bitstring(tokens, device=device)
        elif input_format == "int":
            z = BinaryBottleneck.from_integer_tokens(tokens, self.bottleneck_dim, device=device)
        elif input_format == "packed":
            z = BinaryBottleneck.from_packed_bytes(tokens, self.bottleneck_dim, device=device)
        else:
            raise ValueError(f"Unknown input_format: {input_format}")

        with torch.no_grad():
            recon_x = self.decoder(z)
        return recon_x

    def save(self, filepath: str) -> None:
        """
        Saves model weights and full configuration dictionary.
        """
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, filepath)

    @classmethod
    def load(cls, filepath: str, map_location: Optional[torch.device] = None) -> "CandlestickAutoencoder":
        """
        Loads saved model and restores its architecture.
        """
        checkpoint = torch.load(filepath, map_location=map_location or torch.device("cpu"), weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
