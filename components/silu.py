# SiLU activation
#
# silu(x) = x * sigmoid(x). Translinear sigmoid into one
# input of a Gilbert multiplier (the other input is the signal itself).
#
# param        | circuit element                    | error class
# --------------------------------------------------------------------------------
# gain_mismatch| multiplier gain spread             | systematic, per instance
# offset_std   | input-referred offset of sigmoid   | systematic, per instance
# noise_std    | thermal noise at the output        | zero-mean, per call
#

import torch

from .multiplier import Multiplier
from .translinear import TranslinearExp


class SiLU:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset_std: float = 0.01,
        noise_std: float = 0.005,
    ):
        self.exp = TranslinearExp(gain_mismatch, offset_std, noise_std)
        self.mul = Multiplier(gain_mismatch, offset_std, noise_std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.mul(x, self.exp.sigmoid(x))
