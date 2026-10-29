# Translinear square-rooter
#
# sqrt of the input current in a translinear loop
#
# param         | circuit element                  | error class
# ----------------------------------------------------------------------------------------------
# gain_mismatch | loop transistor area ratio       | systematic, per instance (output scale)
# offset_std    | loop input offset current        | systematic, per instance, input-referred
# noise_std     | input current noise              | zero-mean, per call, input-referred
#

import torch


class Sqrt:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset_std: float = 0.005,
        noise_std: float = 0.002,
    ):
        self.gain = 1.0 + torch.randn(1).item() * gain_mismatch
        self.offset = torch.randn(1).item() * offset_std
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xi = (x + self.offset + torch.randn_like(x) * self.noise_std).clamp_min(0.0)
        return self.gain * xi.sqrt()
