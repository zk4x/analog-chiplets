# Crossbar (differential pair per weight)
#
# param            | circuit element                              | error class
# -----------------------------------------------------------------------------------------------------------
# weight_bits      | conductance lattice granularity              | quantization, at programming
# device_noise_std | device-to-device conductance spread          | systematic, per device
# cycle_noise_std  | read thermal noise on conductance            | zero-mean, per read
# drift_coeff      | power-law conductance relaxation             | systematic, accumulates over time
# n_reads          | repeated reads, averaged (software knob)     | averaging: cycle noise / sqrt(N)
# n_devices        | M devices per synapse, currents summed       | averaging: device spread /
#                  | (Boybat multi-memristive; statistical        | sqrt(M). Implemented as the
#                  | equivalence: mean of M iid N(0,s^2) draws =  | exact same distribution at
#                  | one N(0,s^2/M) draw — zero memory cost.      | zero memory cost. Area cost
#                  | Area/silicon cost NOT simulated (Mx lattice).| (Mx) noted, not modeled.
# update_step      | pulse granularity of conductance change      | quantization (stochastic-rounded)
# drift_comp       | extra full-scale reference column + DIGITAL  | kills common-mode gain shrink
#                  | ratio (reference read through the same ADC,  | (residual: ref-device spread
#                  | divided digitally at the tape-out border)    | ~0.1%, per-device exponent
#                  |                                              | spread NOT modeled)
#

import math

import torch


def round_to_levels(x: torch.Tensor, levels: int) -> torch.Tensor:
    return torch.round(x * (levels - 1)) / (levels - 1)


class Crossbar:
    def __init__(
        self,
        weights: torch.Tensor,
        weight_bits: int = 4,
        weight_range: float = 1.0,
        input_std: float = 0.5,
        device_noise_std: float = 0.05,
        cycle_noise_std: float = 0.05,
        drift_coeff: float = 0.03,
        n_reads: int = 1,
        norm: str = "row",
        drift_comp: bool = False,
        n_devices: int = 1,
    ):
        g_min, g_max = 0.0, 1.0
        self.device_noise_std = device_noise_std
        self.cycle_noise_std = cycle_noise_std
        self.drift_coeff = drift_coeff
        self.n_reads = max(1, int(n_reads))
        self.n_devices = max(1, int(n_devices))
        dev = device_noise_std / math.sqrt(self.n_devices)
        self.update_step = 0.01
        self.p = 1.0
        self.weight_range = weight_range
        self.weight_levels = 2**weight_bits
        self.norm = norm

        # Per-element affine programming: w = w_min + w_norm * (w_max - w_min).
        # 'row'    = per output row (per lattice column: own TIA gain / reference)
        # 'column' = per input (per word-line driver swing)
        # 'tensor' = one range for the whole lattice
        if norm == "row":
            w_min = weights.min(dim=1, keepdim=True).values
            w_max = weights.max(dim=1, keepdim=True).values
        elif norm == "column":
            w_min = weights.min(dim=0, keepdim=True).values
            w_max = weights.max(dim=0, keepdim=True).values
        else:
            w_min = weights.min()
            w_max = weights.max()
        w_norm = (weights - w_min) / (w_max - w_min + 1e-8)
        w_q = round_to_levels(w_norm, self.weight_levels)

        buf = (1.0 - weight_range) / 2
        g_plus = buf + w_q * (1.0 - 2 * buf)
        g_minus = (1.0 - buf) - w_q * (1.0 - 2 * buf)
        self.buf2 = 1.0 - 2 * buf  # lattice differential swing: the programmed
        # current is buf2 x the full-swing current — value() must divide the
        # measured current back out (was missing: only ever validated at
        # weight_range=1.0 where buf2=1; caught by the range sweep).

        if self.device_noise_std > 0:
            g_plus = g_plus + torch.randn_like(g_plus) * dev
            g_minus = g_minus + torch.randn_like(g_minus) * dev

        self.g_plus = g_plus.clamp(g_min, g_max)
        self.g_minus = g_minus.clamp(g_min, g_max)
        self.g_min = g_min
        self.g_max = g_max
        self.t = 1
        self.hold = False  # when True, _read_drift skips (columns read in
        # parallel share one logical tick; ColumnArray manages it)
        # Drift reference: one full-scale differential column pair
        # (W_ref = 1.0), same devices/physics, fixed all-ones drive.
        # Read alongside the array; the digital ratio kills common-mode
        # relaxation (the modeled drift is a global gain shrink).
        self.drift_comp = drift_comp
        ck = weights.shape[1]
        if drift_comp:
            rp = torch.ones(ck)
            rm = torch.zeros(ck)
            if self.device_noise_std > 0:
                rp = rp + torch.randn_like(rp) * dev
                rm = rm + torch.randn_like(rm) * dev
            self.ref_plus = rp.clamp(g_min, g_max)
            self.ref_minus = rm.clamp(g_min, g_max)
            # Factory calibration: the AS-PROGRAMMED reference sum (rail
            # clamp biases device spread systematically; the ideal CK
            # would bake that bias into every gain estimate). Measured
            # once fresh, stored digitally — standard practice.
            self.ref_true = float((self.ref_plus - self.ref_minus).sum())
            self.ref_drive = torch.ones(ck)

        n_inputs = weights.shape[1]
        W = self.g_plus - self.g_minus
        weight_std = W.std().item() + 1e-8
        peak_factor = 15
        max_i = peak_factor * math.sqrt(n_inputs) * weight_std * input_std
        self.current_range = (-max_i, max_i)

        # Programming calibration for value(): reconstructs the signal-domain
        # dot product from the lattice current. Kept as tensors so broadcasting
        # covers all norms.
        self.w_min = w_min
        self.span = w_max - w_min
        # Effective programmed weights and their per-row sum: the mid-rail
        # reference word line contributes sum(w)/2 to every dot product
        # (v = 2x - 1 => x @ w = (v @ w + sum(w)) / 2).
        k = weights.shape[1]
        buf2 = 1.0 - 2 * buf  # lattice differential swing
        wn_q = ((self.g_plus - self.g_minus) + buf2) / (2 * buf2)
        w_eff = self.w_min + self.span * wn_q
        self.w_sum = w_eff.sum(-1)

        n_inputs_bwd = weights.shape[0]
        max_i_bwd = peak_factor * math.sqrt(n_inputs_bwd) * weight_std * 0.5
        self.bwd_current_range = (-max_i_bwd, max_i_bwd)

    def _noisy(self):
        g_plus, g_minus = self.g_plus, self.g_minus
        if self.cycle_noise_std > 0:
            g_plus = g_plus + torch.randn_like(g_plus) * self.cycle_noise_std
            g_minus = g_minus + torch.randn_like(g_minus) * self.cycle_noise_std
        return g_plus, g_minus

    def _read_drift(self):
        if self.hold:
            return  # parallel read: no time passes, managed by the array
        if self.drift_coeff > 0:
            ratio = (self.t / (self.t + 1)) ** self.drift_coeff
            self.g_plus *= ratio
            self.g_minus *= ratio
            if self.drift_comp:
                self.ref_plus *= ratio
                self.ref_minus *= ratio
        self.t += 1

    def matmul(self, v: torch.Tensor) -> torch.Tensor:
        # read averaging: N physical reads, analog accumulation; zero-mean
        # cycle noise falls as 1/sqrt(N), device spread and drift do not.
        # Energy and latency scale with N — the software knob. Drift is a
        # slow relaxation: one logical readout advances its clock once (the
        # N sub-reads share the same drifted conductances). With drift_comp,
        # the reference column rides every sub-read and the averaged digital
        # ratio removes the common-mode shrink.
        self._read_drift()
        acc = None
        ref_acc = None
        for _ in range(self.n_reads):
            g_plus, g_minus = self._noisy()
            r = v @ g_plus.T - v @ g_minus.T
            acc = r if acc is None else acc + r
            if self.drift_comp:
                rp = self.ref_plus
                rm = self.ref_minus
                if self.cycle_noise_std > 0:
                    rp = rp + torch.randn_like(rp) * self.cycle_noise_std
                    rm = rm + torch.randn_like(rm) * self.cycle_noise_std
                rr = (self.ref_drive * rp).sum() - (self.ref_drive * rm).sum()
                ref_acc = rr if ref_acc is None else ref_acc + rr
        out = acc / self.n_reads
        if self.drift_comp:
            gain = (ref_acc / self.n_reads) / self.ref_true
            out = out / gain.clamp_min(1e-6)
        return out

    def matmul_batched(self, v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # Row-wise dots against GATHERED lattice rows: out[j] = v[j] @ W[idx[j]].
        # Same physics/read model as matmul() (cycle noise per sub-read,
        # one drift tick per call, reference ratio when drift_comp), but
        # O(R*K) memory instead of O(R*rows): the depthwise fast path, where
        # R = C*P drive rows share C lattice rows and the full R x C outer
        # product would be 6+ GB of all-but-discarded outputs.
        self._read_drift()
        acc = None
        ref_acc = None
        for _ in range(self.n_reads):
            g_plus, g_minus = self._noisy()
            r = (v * g_plus[idx] - v * g_minus[idx]).sum(-1)
            acc = r if acc is None else acc + r
            if self.drift_comp:
                rp = self.ref_plus
                rm = self.ref_minus
                if self.cycle_noise_std > 0:
                    rp = rp + torch.randn_like(rp) * self.cycle_noise_std
                    rm = rm + torch.randn_like(rm) * self.cycle_noise_std
                rr = (self.ref_drive * rp).sum() - (self.ref_drive * rm).sum()
                ref_acc = rr if ref_acc is None else ref_acc + rr
        out = acc / self.n_reads
        if self.drift_comp:
            gain = (ref_acc / self.n_reads) / self.ref_true
            out = out / gain.clamp_min(1e-6)
        return out

    def value(self, v: torch.Tensor) -> torch.Tensor:
        """Signal-domain readout: v is the bipolar word-line drive (v = 2x-1)
        of a [0,1] signal x; returns x @ w including the mid-rail reference
        contribution sum(w)/2."""
        if self.norm == "column":
            v_eff = v * self.span  # span absorbed by the word-line drivers
            i = self.matmul(v_eff)
            lat = (v * self.w_min).sum(-1, keepdim=True) + (
                i / self.buf2 + v_eff.sum(-1, keepdim=True)
            ) / 2
        else:
            i = self.matmul(v)
            s = v.sum(-1, keepdim=True)
            lat = self.w_min.reshape(-1) * s + self.span.reshape(-1) * (i / self.buf2 + s) / 2
        return (lat + self.w_sum) / 2

    def matmul_transpose(self, i: torch.Tensor) -> torch.Tensor:
        self._read_drift()
        acc = None
        for _ in range(self.n_reads):
            g_plus, g_minus = self._noisy()
            r = i @ g_plus - i @ g_minus
            acc = r if acc is None else acc + r
        return acc / self.n_reads

    def _refresh_w_sum(self):
        # Re-derive the mid-rail reference contribution from the RESIDENT
        # conductances. Programming-time w_sum goes stale after update() or
        # prolonged drift; value() reads this, so refresh after any write.
        wn_q = ((self.g_plus - self.g_minus) + self.buf2) / (2 * self.buf2)
        self.w_sum = (self.w_min + self.span * wn_q).sum(-1)

    def read_weights(self):
        # Clean resident readback in weight domain, no noise — bookkeeping
        # for train-inside loops (the clean path's master). Sampling noise
        # belongs in the error measurement, not here.
        wn_q = ((self.g_plus - self.g_minus) + self.buf2) / (2 * self.buf2)
        return self.w_min + self.span * wn_q

    def update(self, grad: torch.Tensor, lr: float = 0.01):
        # grad lives in WEIGHT domain (dL/dw). The lattice is affine-mapped
        # (w = w_min + span * wn_q, differential swing buf2), so the weight
        # delta converts to a conductance delta per element BEFORE pulse
        # rounding (the old version pulsed raw conductance units — wrong
        # effective lr whenever span != 1 or weight_range < 1).
        # Stochastic trunc + probabilistic fraction, +/-2 split across the
        # pair (total conductance conserved), p=1 boundary scales stall the
        # tails (a weight parked at a rail barely moves — the ugly
        # range-tails update physics), rails clamp.
        delta_w = -lr * grad
        scale_lat = (2 * self.buf2) / (self.span + 1e-8)  # dW_lat/dw
        delta = delta_w * scale_lat
        n_pulses = delta / self.update_step
        n_trunc = n_pulses.trunc()
        frac = (n_pulses - n_trunc).abs()
        stochastic = n_trunc + (torch.rand_like(frac) < frac).float() * n_pulses.sign()
        pulse = stochastic * self.update_step / 2

        g_range = self.g_max - self.g_min
        scale_plus = torch.where(
            pulse > 0,
            ((self.g_max - self.g_plus) / g_range) ** self.p,
            ((self.g_plus - self.g_min) / g_range) ** self.p,
        )
        scale_minus = torch.where(
            pulse > 0,
            ((self.g_minus - self.g_min) / g_range) ** self.p,
            ((self.g_max - self.g_minus) / g_range) ** self.p,
        )

        self.g_plus += pulse * scale_plus
        self.g_minus -= pulse * scale_minus
        self.g_plus.clamp_(self.g_min, self.g_max)
        self.g_minus.clamp_(self.g_min, self.g_max)
        self._refresh_w_sum()
