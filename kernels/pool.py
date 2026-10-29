# AvgPool: spatial average by current summing.
#
# [B,C,H,W] -> [B,C,1,1]. The N-input add is one summing line (KCL); the
# 1/N readout is a programmable mirror ratio (the program is per call, the
# mismatch of the mirror is a fixed property of the instance).
#

import torch

from components.rerouter import Rerouter
from components.scaler import Scaler
from components.summer import Summer


class AvgPool:
    def __init__(self, gain_mismatch: float = 0.01):
        self.flat = Rerouter((0, 0, -1))  # [B,C,H,W] -> [B,C,H*W]
        self.sum = Summer()
        self.gain_err = 1.0 + torch.randn(1).item() * gain_mismatch

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.sum(self.flat(x), dim=-1)
        # programmable mirror ratio: the program (1/N) changes per call, the
        # mirror mismatch is the instance's (gain_err), so the per-call
        # Scaler is constructed exact
        y = Scaler((1.0 / (h * w)) * self.gain_err, gain_mismatch=0.0)(s)
        return Rerouter((b, c, 1, 1))(y)  # bus framing
