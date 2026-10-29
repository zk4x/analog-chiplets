# ADC (output converter)
#
# param | circuit element            | error class
# ------------------------------------------------------------------------------
# bits  | output level quantization  | quantization, per read
#

import torch

from .crossbar import round_to_levels


class ADC:
    def __init__(
        self,
        bits: int,
        in_range: tuple[float, float],
        out_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.levels = 2**bits
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, i: torch.Tensor) -> torch.Tensor:
        in_min, in_max = self.in_range
        i_norm = (i.float() - in_min) / (in_max - in_min + 1e-8)
        i_q = round_to_levels(i_norm.clamp(0.0, 1.0), self.levels)
        out_min, out_max = self.out_range
        return i_q * (out_max - out_min) + out_min
