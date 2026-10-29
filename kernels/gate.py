# GDN gates as component chains (Qwen3.5+ gated delta net)

import torch

from components.adder import Adder
from components.multiplier import Multiplier
from components.scaler import Scaler
from components.softplus import Softplus
from components.translinear import TranslinearExp


class Gate:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset_std: float = 0.01,
        noise_std: float = 0.005,
    ):
        self.exp = TranslinearExp(gain_mismatch, offset_std, noise_std)
        self.sp = Softplus(gain_mismatch, offset_std, noise_std)
        self.add = Adder()
        self.neg = Scaler(-1.0, gain_mismatch)
        self.mul = Multiplier(gain_mismatch, offset_std, noise_std)

    def beta(self, b: torch.Tensor) -> torch.Tensor:
        return self.exp.sigmoid(b)

    def decay(
        self, a: torch.Tensor, dt_bias: torch.Tensor, ealog: torch.Tensor | None = None
    ):
        """g = exp(-ealog * softplus(a + dt_bias)); ealog defaults to 1."""
        s = self.sp(self.add(a, dt_bias))
        if ealog is not None:
            s = self.mul(ealog, s)
        return self.exp(self.neg(s))
