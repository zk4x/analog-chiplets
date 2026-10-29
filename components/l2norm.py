# L2 norm (vector normalizer)
#
# x / ||x||: sum-of-squares lane, analog sqrt, translinear divide. Used
# inside the GDN core to condition q/k lanes before the state update.
#
# param        | circuit element              | error class
# ---------------------------------------------------------------------------
# gain_mismatch| squarer/sqrt/divider gains   | systematic, per instance
# offset_std   | input-referred offsets       | systematic, per instance
# noise_std    | thermal noise at the output  | zero-mean, per call
# eps          | dither floor on the divider  | constant
#

import math

import torch

from .divider import Divider
from .sqrt import Sqrt
from .squarer import SumSquarer


class L2Norm:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset_std: float = 0.01,
        noise_std: float = 0.005,
        eps: float = 1e-6,
    ):
        self.squarer = SumSquarer(gain_mismatch)
        # precision norm loop: the sqrt node's input-referred offset/noise
        # must sit well below the eps floor (generic 0.01/0.005 swamps the
        # floor input by ~1e4 and collapses the divider onto b_min), and the
        # divider's offset/noise are scaled to the floor operating point
        # b = sqrt(eps) (its noise gain is 1/b)
        self.sqrt = Sqrt(
            gain_mismatch,
            0.0 if offset_std == 0 else eps * 1e-2,
            0.0 if noise_std == 0 else eps * 1e-2,
        )
        self.div = Divider(
            gain_mismatch,
            offset=0.0 if offset_std == 0 else offset_std * math.sqrt(eps),
            noise_std=0.0 if noise_std == 0 else noise_std * math.sqrt(eps),
            b_min=1e-6,
        )
        self.eps = eps

    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        b = self.sqrt(self.squarer(x, dim) + self.eps)
        if b.dim() < x.dim():  # per-row divisor onto the lane bus
            b = b.reshape(*x.shape[:-1], 1)
        return self.div(x, b)
