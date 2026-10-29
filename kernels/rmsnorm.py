# sumsq -> /dim -> +dither floor -> sqrt -> divide -> gain.
#
# eps is the dither floor at the mean node (squarer leakage/offset
# current): the sqrt/divide loop cannot resolve means below it, just
# like the reference's sqrt(mean + eps)."""
#

import math

import torch

from components.divider import Divider
from components.multiplier import Multiplier
from components.rerouter import Rerouter
from components.scaler import Scaler
from components.sqrt import Sqrt
from components.squarer import SumSquarer


class RMSNorm:
    def __init__(
        self,
        dim: int,
        gain_mismatch: float = 0.01,
        offset_std: float = 0.01,
        noise_std: float = 0.005,
        eps: float = 1e-6,
    ):
        self.squarer = SumSquarer(gain_mismatch)
        self.mean = Scaler(1.0 / dim, gain_mismatch)
        # precision norm loop: the mean/sqrt node must resolve the eps floor
        # (input eps at the mean node = sqrt(eps) rms). Its input-referred
        # offset/noise are therefore specced two decades below eps — the
        # generic 0.01/0.005 logic-level defaults would swamp the floor
        # input by ~1e4 and collapse the divider onto b_min.
        self.sqrt = Sqrt(
            gain_mismatch,
            0.0 if offset_std == 0 else eps * 1e-2,
            0.0 if noise_std == 0 else eps * 1e-2,
        )
        # precision divider spec: divider noise is output-referred with 1/b
        # gain, so at the floor operating point (b = sqrt(eps)) the generic
        # noise_std/offset would swamp the readout (sigma = noise_std/b = 5).
        # Scale both to the floor operating point.
        self.div = Divider(
            gain_mismatch,
            offset=0.0 if offset_std == 0 else offset_std * math.sqrt(eps),
            noise_std=0.0 if noise_std == 0 else noise_std * math.sqrt(eps),
            b_min=1e-6,
        )
        self.gain = Multiplier(gain_mismatch, offset_std, noise_std)
        self.g = torch.ones(dim)
        self.eps = eps
        self.to_lane = Rerouter((-1, 1))  # per-row divisor onto the lane bus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x
        p = self.squarer(v, dim=-1)
        rms = self.to_lane(self.sqrt(self.mean(p) + self.eps))
        return self.gain(self.div(v, rms), self.g)
