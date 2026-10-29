# Sigmoid and sofplus
#
# param      | circuit element                          | error class
# --------------------------------------------------------------------------------------------------------
# gain       | k/V_T slope, bias-point and temperature  | systematic, per instance, drifts over time
# offset_std | input-referred offset voltage            | systematic, per instance
# noise_std  | shot/thermal on the output current       | zero-mean, per call
# clip       | finite dynamic range of the exp          | guard rail (saturation regime)
#

import torch


class TranslinearExp:
    def __init__(
        self,
        gain_mismatch: float = 0.01,
        offset_std: float = 0.0,
        noise_std: float = 0.0,
        clip: float = 20.0,
    ):
        self.gain = 1.0 + (
            torch.randn(1).item() * gain_mismatch if gain_mismatch > 0 else 0.0
        )
        self.offset = torch.randn(1).item() * offset_std if offset_std > 0 else 0.0
        self.noise_std = noise_std
        self.clip = clip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.gain + self.offset
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        return torch.exp(y.clamp(-self.clip, self.clip))

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + self(-x))

    def softplus(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(self(x))
