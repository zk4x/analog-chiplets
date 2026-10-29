# Column array
#
# C independent crossbar columns sharing one patch bus: the depthwise
# compute fabric. Fanout/fanin wiring and per-column readout live here.
#
# param             | circuit element            | error class
# ------------------------------------------------------------------------------------------
# weights           | [C, 1, kh, kw] conductances| -
# weight_bits       | conductance lattice        | quantization, at programming
# device_noise_std  | per-column device spread   | systematic, per column (via Crossbar)
# cycle_noise_std   | per-column read noise      | zero-mean, per read (via Crossbar)
# drift_coeff       | per-column conductance     | systematic, accumulates (via Crossbar)
# drift_comp        | per-column reference pair  | via Crossbar (note: 1 extra
#                   |                            | column per column — 100% overhead
#                   |                            | on 1-row lattices; a shared global
#                   |                            | ref is future work)
# xbs               | C crossbar columns         | (composed from the above)
# rows/flat         | readout bus permutation    | none
#

import torch

from components.crossbar import Crossbar
from components.rerouter import Rerouter
from components.transposer import Transposer


class ColumnArray:
    def __init__(
        self,
        weights: torch.Tensor,
        weight_bits: int = 4,
        device_noise_std: float = 0.05,
        cycle_noise_std: float = 0.05,
        drift_coeff: float = 0.03,
        n_reads: int = 1,
        drift_comp: bool = False,
        n_devices: int = 1,
        dev_mult: float = 1.0,
    ):
        # weights [C, 1, kh, kw]
        self.c = weights.shape[0]
        self.k = weights[0].numel()
        self.xbs = [
            Crossbar(
                weights[i].reshape(1, -1),
                weight_bits=weight_bits,
                device_noise_std=device_noise_std * dev_mult,
                cycle_noise_std=cycle_noise_std,
                drift_coeff=drift_coeff,
                n_reads=n_reads,
                drift_comp=drift_comp,
                n_devices=n_devices,
            )
            for i in range(self.c)
        ]
        self.rows = Transposer((0, 2, 3, 1))  # [B, K, oh, ow] -> [B, oh, ow, K]
        self.flat = Rerouter((0, -1, self.k))  # [B, oh, ow, K] -> [B, oh*ow, K]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        # [B, C, K, oh, ow] -> [B, C, oh, ow]
        # All columns read in parallel in hardware: ONE logical drift tick
        # per call (same rule as n_reads — sibling bug class; the per-column
        # loop is simulation scaffolding, not 10240 time steps).
        b, _, _, oh, ow = u.shape
        for xb in self.xbs:
            xb.hold = True
        out = torch.empty(b, self.c, oh, ow)
        for ci in range(self.c):
            out[:, ci] = (
                self.xbs[ci].value(self.flat(self.rows(u[:, ci]))).reshape(b, oh, ow)
            )
        for xb in self.xbs:
            xb.hold = False
            xb._read_drift()
        return out
