# Current summing junction
#
# a + b on a shared wire (KCL); the physical add is free, only the readout
# adds error downstream.
#

import torch


class Adder:
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b
