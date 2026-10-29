# Gilbert multiplier
#
# elementwise product of two signals (four-quadrant translinear core)
#
# param         | circuit element                  | error class
# ------------------------------------------------------------------------------------------
# gain_mismatch | core bias current ratio          | systematic, per instance
# offset        | head/tail current mismatch       | systematic, per instance, output offset
#               |                                  | (fraction of full scale)
# noise_std     | shot/thermal noise on the tail   | zero-mean, per call (fraction of full scale)
#

import torch


class Multiplier:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset: float = 0.01,
        noise_std: float = 0.005,
    ):
        self.gain = 1.0 + torch.randn(1).item() * gain_mismatch
        self.offset = torch.randn(1).item() * offset
        self.noise_std = noise_std

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.gain * a * b + self.offset + torch.randn_like(a) * self.noise_std
