# DAC (input driver)
#
# param | circuit element            | error class
# ------------------------------------------------------------------------------
# bits  | input level quantization   | quantization, per apply
#

import torch

from .crossbar import round_to_levels


class DAC:
    def __init__(
        self,
        bits: int,
        in_range: tuple[float, float],
        out_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.levels = 2**bits
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        in_min, in_max = self.in_range
        x_norm = (x.float() - in_min) / (in_max - in_min + 1e-8)
        x_q = round_to_levels(x_norm.clamp(0.0, 1.0), self.levels)
        out_min, out_max = self.out_range
        return x_q * (out_max - out_min) + out_min
