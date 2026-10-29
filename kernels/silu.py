# SiLU: translinear sigmoid into a Gilbert multiplier.

import torch

from components.silu import SiLU


class SiLUKernel:
    def __init__(self, noiseless: bool = False):
        self.act = SiLU(
            0.0 if noiseless else 0.01,
            0.0 if noiseless else 0.01,
            0.0 if noiseless else 0.005,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
