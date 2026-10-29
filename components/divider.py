# Translinear divider
#
# elementwise a / b: Gilbert multiplier in a feedback loop — the op-amp forces
# the multiplier output to equal a, so the loop solves out * b ∝ a
#
# param           | circuit element                   | error class
# ---------------------------------------------------------------------------------------------
# gain_mismatch   | loop transistor area ratio,       | systematic, per instance
#                 | finite loop gain                  |
# offset          | input-referred offset of the      | systematic, per instance
#                 | feedback op-amp (error on out)    |
# noise_std       | output noise; noise gain is 1/b,  | stochastic, per call
#                 | blows up for small denominators   |
# b_min           | minimum denominator of the        | systematic, deterministic
#                 | operating range (loop runs out of |
#                 | gain below it); b clamped to this |
#

import torch


class Divider:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset: float = 0.01,
        noise_std: float = 0.0,
        b_min: float = 0.05,
    ):
        self.gain = 1.0 + (
            torch.randn(1).item() * gain_mismatch if gain_mismatch > 0 else 0.0
        )
        self.offset = torch.randn(1).item() * offset if offset > 0 else 0.0
        self.noise_std = noise_std
        self.b_min = b_min

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b_c = b.clamp_min(self.b_min)
        y = a / (b_c * self.gain + 1e-8) + self.offset
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std / b_c
        return y
