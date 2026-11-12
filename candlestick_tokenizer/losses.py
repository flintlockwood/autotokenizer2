"""
Loss Functions and Regularization for Discrete Binary Candlestick Autoencoder.

Includes:
- Feature Reconstruction Loss (MSE or Smooth L1)
- Binary Commitment Regularization (penalizing non-binary latent probabilities)
- Bit Utilization / Entropy Regularization (preventing bit collapse)
- Price Reconstruction Loss (direct OHLC price error relative to reference price)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import reconstruct_ohlc, CandlestickFeatureScaler


class CandlestickLoss(nn.Module):
    """
    Composite Loss for Candlestick Autoencoder with Discrete Binary Bottleneck.
    
    L_total = lambda_recon * L_recon
            + lambda_binary * L_binary
            + lambda_entropy * L_entropy
            + lambda_price * L_price
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_binary: float = 0.05,
        lambda_entropy: float = 0.1,
        lambda_price: float = 0.0,
        recon_loss_type: str = "mse",
        scaler: Optional[CandlestickFeatureScaler] = None,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_binary = lambda_binary
        self.lambda_entropy = lambda_entropy
        self.lambda_price = lambda_price
        self.recon_loss_type = recon_loss_type
        self.scaler = scaler

    def forward(
        self,
        target_norm: torch.Tensor,
        recon_norm: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        o_prev: Optional[torch.Tensor] = None,
        target_ohlc: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            target_norm: (batch_size, 5) ground truth normalized features
            recon_norm: (batch_size, 5) reconstructed normalized features
            z: (batch_size, bottleneck_dim) discrete binary activations
            p: (batch_size, bottleneck_dim) continuous latent probabilities
            o_prev: (batch_size,) previous Open prices (optional for price loss)
            target_ohlc: (batch_size, 4) target current OHLC (optional for price loss)
            
        Returns:
            total_loss: scalar tensor for backpropagation
            loss_dict: dictionary of individual loss components
        """
        # 1. Feature Reconstruction Loss
        if self.recon_loss_type == "mse":
            l_recon = F.mse_loss(recon_norm, target_norm)
        elif self.recon_loss_type == "smooth_l1":
            l_recon = F.smooth_l1_loss(recon_norm, target_norm, beta=0.1)
        else:
            l_recon = F.mse_loss(recon_norm, target_norm)

        # 2. Binary Regularization Loss (pushes individual bit probabilities to 0 or 1)
        # 4 * p * (1 - p) is in [0, 1], maximum at p = 0.5, 0 at p = 0 or 1
        l_binary = torch.mean(4.0 * p * (1.0 - p))

        # 3. Bit Utilization / Entropy Regularization Loss
        # Prevents bit collapse by driving the batch-mean of each bit towards 0.5 (maximum entropy)
        batch_p_mean = torch.mean(p, dim=0)  # (bottleneck_dim,)
        # MSE from balanced 0.5 activation
        l_entropy = torch.mean((batch_p_mean - 0.5) ** 2)

        # 4. Optional Price Reconstruction Loss
        l_price = torch.tensor(0.0, device=target_norm.device)
        if self.lambda_price > 0.0 and self.scaler is not None and o_prev is not None and target_ohlc is not None:
            # Denormalize reconstructed features differentiably
            recon_features = self.scaler.inverse_transform(recon_norm)
            recon_ohlc_t, _ = reconstruct_ohlc(recon_features, o_prev, enforce_constraints=False)
            # Relative price error normalized by previous Open
            price_diff = torch.abs(recon_ohlc_t - target_ohlc) / (o_prev.view(-1, 1) + 1e-8)
            l_price = torch.mean(price_diff)

        # Combined loss
        total_loss = (
            self.lambda_recon * l_recon
            + self.lambda_binary * l_binary
            + self.lambda_entropy * l_entropy
            + self.lambda_price * l_price
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": l_recon.item(),
            "binary_loss": l_binary.item(),
            "entropy_loss": l_entropy.item(),
            "price_loss": l_price.item(),
        }

        return total_loss, loss_dict
