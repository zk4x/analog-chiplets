# Elementwise multiply chiplet (Hadamard): one Gilbert multiplier per lane
# pair. Purely analog — converters live at the tape-out border.

import torch

from components.multiplier import Multiplier


class Multiply:
    def __init__(self, noiseless: bool = False):
        self.mul = Multiplier(
            0.0 if noiseless else 0.01,
            0.0 if noiseless else 0.01,
            0.0 if noiseless else 0.005,
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.mul(a, b)
