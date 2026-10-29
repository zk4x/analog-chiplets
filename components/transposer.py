# Transposer
#
# Rewires tensor axes, bus permutation between blocks.
#
# param | circuit element  | error class
# --------------------------------------------------------------
# perm  | bus crossbar     | none
#

import torch


class Transposer:
    def __init__(self, perm: tuple[int, ...]):
        self.perm = perm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.perm)
