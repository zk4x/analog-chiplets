# Rerouter
#
# Regroups tensor elements (reshape); in hardware this is fanout/fanin
# wiring, no state, no arithmetic.
#
# param | circuit element | error class
# -----------------------------------------------------------------
# dims  | bus grouping    | none (-1 = infer)
#

import torch


class Rerouter:
    def __init__(self, dims: tuple[int, ...]):
        self.dims = dims

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(x.shape[i] if d == 0 else d for i, d in enumerate(self.dims))
        return x.reshape(*dims)
