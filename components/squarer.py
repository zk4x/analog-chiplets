# Squarer
#
# square of the current I
#
# param               | circuit element                 | error class
# --------------------------------------------------------------------------------------------------------------
# gain_mismatch       | common square-law coefficient   | uniform scale, absorbed downstream
# lane_mismatch       | per-lane squarer device spread  | systematic, per lane
# linear_feedthrough  | imperfect square law            | directional bias on small signals
# cubic_compression   | square-law sag at large drive   | compresses large x
# offset              | output offset current           | overestimates all norms
# noise_std           | shot/thermal current noise      | zero-mean, per read
#

import torch


class Squarer:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        linear_feedthrough: float = 0.0,
        cubic_compression: float = 0.0,
        offset: float = 0.0,
        noise_std: float = 0.0,
    ):
        self.gain = 1.0 + (
            torch.randn(1).item() * gain_mismatch if gain_mismatch > 0 else 0.0
        )
        self.linear = (
            torch.randn(1).item() * linear_feedthrough
            if linear_feedthrough > 0
            else 0.0
        )
        self.cubic = (
            torch.randn(1).item() * cubic_compression if cubic_compression > 0 else 0.0
        )
        self.offset = torch.randn(1).item() * offset if offset > 0 else 0.0
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gain * x * x + self.linear * x + self.cubic * x * x * x + self.offset
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        return y


class SumSquarer:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        lane_mismatch: float = 0.0,
        linear_feedthrough: float = 0.0,
        cubic_compression: float = 0.0,
        offset: float = 0.0,
        noise_std: float = 0.0,
    ):
        self.squarer = Squarer(
            gain_mismatch,
            linear_feedthrough,
            cubic_compression,
            offset,
            noise_std,
        )
        self.lane_mismatch = lane_mismatch
        self._lane_gain = None

    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.lane_mismatch > 0:
            n = x.shape[dim]
            if self._lane_gain is None or self._lane_gain.shape[0] != n:
                self._lane_gain = 1.0 + torch.randn(n) * self.lane_mismatch
            x = x * self._lane_gain
        return self.squarer(x).sum(dim=dim)
