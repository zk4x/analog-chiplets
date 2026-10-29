# Delta-sigma ADC (equivalent-noise approximation)
#
# One-bit feedback modulator: an integrator accumulates the difference between
# the input and a 1-bit charge packet, a comparator thresholds it, and the
# packet is fed back; running many cycles resolves the input mean far beyond
# the comparator's own resolution.
#
# This is NOT a cycle-by-cycle simulation of the loop. A first-order modulator
# is modeled by its output statistics over N cycles:
#   - first-order noise shaping: the in-band error rms falls as N^-1.5, i.e.
#     the loop gains ~1.5 bits of resolution per doubling of N
#   - comparator dither is white (the loop does not shape it): its share of
#     the resolved mean falls as 1/sqrt(N)
#   - the digital output word is the final quantization at `bits` resolution
#
# param                 | circuit element                          | error class
# ------------------------------------------------------------------------------------------------------------
# cycles                | feedback cycles per conversion           | sets the effective resolution
# comparator_noise_std  | thermal dither on the threshold          | white, per conversion (unshaped)
# comparator_offset     | comparator input offset voltage          | systematic, per instance
# feedback_gain         | feedback packet cap/V_ref mismatch       | systematic, per instance (scale)
# integrator_leak       | integrator cap leakage between ticks     | systematic: dead zone of half-width ~ leak
#                       |                                          | (input-referred, set by the charge lost per
#                       |                                          | cycle); small inputs under-report, with
#                       |                                          | exponential recovery away from zero
#
# Signals are unipolar [0,1] (the storage-unit domain).
#

import math

import torch

from .crossbar import round_to_levels


class DeltaSigma:
    def __init__(
        self,
        bits: int = 8,
        cycles: int = 45,
        comparator_noise_std: float = 0.01,
        comparator_offset: float = 0.0,
        feedback_gain: float = 1.0,
        integrator_leak: float = 0.0,
        in_range: tuple[float, float] | None = None,
        out_range: tuple[float, float] | None = None,
    ):
        self.bits = bits
        self.levels = 2**bits
        # first-order shaping: analog resolution ENOB = 1.5 * log2(N); when it
        # exceeds the output word, the word quantization dominates and the
        # output reduces to an ideal `bits` quantizer
        self.enob = 1.5 * math.log2(max(cycles, 2.0))
        self.cycles = cycles
        self.comparator_noise_std = comparator_noise_std
        self.comparator_offset = comparator_offset
        self.feedback_gain = feedback_gain
        self.integrator_leak = integrator_leak
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, x: torch.Tensor, act=None) -> torch.Tensor:
        if act is not None:
            x = act(x)
        if self.in_range is not None:
            lo, hi = self.in_range
            x = (x - lo) / (hi - lo + 1e-8)
        x = x * self.feedback_gain + self.comparator_offset
        if self.integrator_leak > 0:
            # leaky integrator: inputs below ~leak never accumulate enough
            # charge to flip the comparator; recovery away from zero is
            # exponential in units of the dead-zone width
            x = x - self.integrator_leak * torch.exp(
                -x.clamp_min(0.0) / self.integrator_leak
            )
        # shaped quantization noise (input-referred) + white comparator dither
        step = 1.0 / (2.0**self.enob - 1.0)
        sig_q = step / math.sqrt(12.0)
        sig_th = self.comparator_noise_std / math.sqrt(self.cycles)
        x = x + torch.randn_like(x) * math.sqrt(sig_q**2 + sig_th**2)
        q = round_to_levels(x.clamp(0.0, 1.0), self.levels)
        if self.out_range is not None:
            lo, hi = self.out_range  # map back to storage units
            return q * (hi - lo) + lo
        return q
