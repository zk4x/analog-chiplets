# Bias current injector
#
# Constant current added per output channel (reference DAC / resistor ladder)
#
# param | circuit element                    | error class
# ------------------------------------------------------------------------------------
# value | reference current (V_ref / R)      | systematic, per instance
#

import torch


class Bias:
    def __init__(self, value: torch.Tensor):
        self.value = value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.value
