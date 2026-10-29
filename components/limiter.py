# Limiter (activation circuit)
#
# Clamps the signal between lo and hi; ideal-diode / current-limiter pair.
# ReLU6 = Limiter(0, 6); ReLU = Limiter(0, inf).
#
# param              | circuit element              | error class
# -------------------------------------------------------------------------------------------
# lo                 | low-side limiter threshold   | nominal value
# hi                 | high-side limiter threshold  | nominal value
# threshold_mismatch | diode turn-on offset         | systematic, per instance, drawn as a
#                    | (V_on mismatch)              | fraction of the clamp span
#

import math

import torch


class Limiter:
    def __init__(
        self, lo: float = 0.0, hi: float = 6.0, threshold_mismatch: float = 0.01
    ):
        span = (hi - lo) if math.isfinite(hi) else max(abs(lo), 1.0)
        self.lo = lo + torch.randn(1).item() * threshold_mismatch * span
        self.hi = hi + torch.randn(1).item() * threshold_mismatch * span

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(self.lo, self.hi)
