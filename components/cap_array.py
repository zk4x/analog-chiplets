# Capacitor array
#
# Useful for SGD like update inside gated delta net (linear attention blocks)
#
# param              | circuit element                                    | error class
# ------------------------------------------------------------------------------------------------------
# read_noise_std     | kT/C sampling noise of the access switch           | zero-mean, per read
# read_gain_mismatch | C_bus/C_cell ratio in the charge-shared readout    | systematic, per array
# write_gain_mismatch| reference packet (cap/V_ref) for charge injection  | systematic, per array
# injection_bits     | granularity of the injected charge packet          | quantization, per write
# alpha_mismatch     | cap ratio matching of the charge-share decay       | systematic, per array
# leak_per_step      | access-transistor subthreshold leakage             | parasitic, per step (refresh)
#

import torch

from .crossbar import round_to_levels


class CapArray:
    def __init__(
        self,
        state: torch.Tensor,
        injection_bits: int = 4,
        read_noise_std: float = 0.01,
        leak_per_step: float = 0.0,
        read_gain_mismatch: float = 0.0,
        write_gain_mismatch: float = 0.0,
        alpha_mismatch: float = 0.0,
    ):
        self.S = state.clone()
        self.injection_bits = injection_bits
        self.read_noise_std = read_noise_std
        self.leak_per_step = leak_per_step
        self.read_gain = 1.0 + read_gain_mismatch
        self.write_gain = 1.0 + write_gain_mismatch
        self.alpha_gain = 1.0 + alpha_mismatch

    def hold(self) -> None:
        if self.leak_per_step > 0:
            self.S = self.S * (1.0 - self.leak_per_step)

    def read_t(self, x: torch.Tensor) -> torch.Tensor:
        S_eff = self.S * self.read_gain
        if self.read_noise_std > 0:
            S_eff = S_eff + torch.randn_like(S_eff) * self.read_noise_std
        return S_eff.T @ x

    def write_rank1(self, k: torch.Tensor, e: torch.Tensor, beta: torch.Tensor) -> None:
        update = k.unsqueeze(-1) * (beta * e).unsqueeze(-2) * self.write_gain
        levels = 2**self.injection_bits
        s = update.abs().max() + 1e-8
        q = round_to_levels(update / s, levels) * s
        self.S = self.S + q

    def decay(self, alpha) -> None:
        self.S = self.S * (alpha * self.alpha_gain)

    def share(self, other: "CapArray", ratio: float) -> None:
        self.S = ratio * self.S + (1.0 - ratio) * other.S
