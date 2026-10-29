# Attenuator / gain stage
#
# Scales the signal between stages; this is where activation storage units
# (e.g. relu6/6 into the DAC domain [0,1]) are realized in hardware.
# Unlike AGCScaler, this one is static.
#
# param         | circuit element                    | error class
# --------------------------------------------------------------------------------------------
# factor        | resistive/capacitive divider ratio | nominal value
# gain_mismatch | divider ratio mismatch             | systematic, per instance
#

import torch


class Scaler:
    def __init__(self, factor: float, gain_mismatch: float = 0.01):
        self.factor = factor * (1.0 + torch.randn(1).item() * gain_mismatch)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor
