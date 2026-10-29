# Softplus
#
# log(1 + exp(x)): one subthreshold exp plus one translinear log in a loop
#
# param         | circuit element                  | error class
# -----------------------------------------------------------------------------------------------
# gain_mismatch | log-stage transistor ratio       | systematic, per instance (output scale)
# offset_std    | exp-stage input offset           | systematic, per instance, input-referred
#               | (Vbe mismatch)                   |
# noise_std     | subthreshold current noise       | zero-mean, per call, input-referred
#               |                                  | (the exp turns it into lognormal-like
#               |                                  | current noise)
#

import torch


class Softplus:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset_std: float = 0.01,
        noise_std: float = 0.005,
    ):
        self.gain = 1.0 + torch.randn(1).item() * gain_mismatch
        self.offset = torch.randn(1).item() * offset_std
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xi = x + self.offset + torch.randn_like(x) * self.noise_std
        return self.gain * torch.nn.functional.softplus(xi)
