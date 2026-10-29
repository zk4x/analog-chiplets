# Current summing node
#
# N input currents on one shared wire (KCL); the physical N-input add is
# error free, only the readout adds error downstream. Generalizes the two-input
# Adder to a summing line (avgpool, row reductions).
#
# param | circuit element        | error class
# -----------------------------------------------------------------------
# -     | shared wire / node     | none
#

import torch


class Summer:
    def __call__(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x.sum(dim=dim)
