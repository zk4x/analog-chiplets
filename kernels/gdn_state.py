# Gated Delta Net core from Qwen3.5+: per-head CapArray state elements.
#
# sequence(): the chiplet consumes whole qkv streams — the q/k lane
# conditioning (L2 norm) and the per-head addressing (q/k head grouping,
# v lane slicing) are wiring inside the chiplet; the model only chains it.
#
# step(): single-lane primitive (kept for tests).
#
# State convention: S per head is [DK, DV] = S_gdn^T, matching
#   u = S^T k, e = v - alpha*u, S = alpha*S + k (beta*e)^T, y = S^T q.

import math

import torch

from components.adder import Adder
from components.agc import AGCScaler
from components.cap_array import CapArray
from components.deltasigma import DeltaSigma
from components.l2norm import L2Norm
from components.multiplier import Multiplier
from components.scaler import Scaler


class GDNState:
    def __init__(self, state: torch.Tensor | None = None,
                 injection_bits: int = 8, read_noise_std: float = 0.001,
                 gain_mismatch: float = 0.01, offset_std: float = 0.01,
                 noise_std: float = 0.001,
                 noiseless: bool = False,
                 n_heads: int = 1, key_dim: int = 128, val_dim: int = 128,
                 q_groups: int = 16,
                 rescale_after_decay: bool = False, target: float = 0.353,
                 ds_readout: bool = False, ds_bits: int = 12,
                 ds_cycles: int = 256,
                 ds_range_u: tuple[float, float] = (-4.0, 4.0),
                 ds_range_y: tuple[float, float] = (-1.0, 1.0),
                 write_gains: list[float] | None = None,
                 head_mask: list[int] | None = None):
        # write_gains: per-head static write gain G_h (hot-state window trial:
        # S += G_h*outer(beta*e, k); heat-only, calibrated at boot from ideal
        # ||y||; exact multiply here, 1%-mismatch Scaler in silicon).
        # head_mask: statically depopulated heads (squelch) — output zeros,
        # state untouched.
        # target = median ideal S elem-absmax (measured 0.073 min / 0.353
        # median / 3.1e10 max across 48 heads): one reference per stage,
        # per RULES calibration; per-head AGC servos each lane's peak to it.
        # ds_readout pairs with rescale: AGC bounds the scaled domain so the
        # DS ranges below cover it (S peak 0.353, k L1<=11.3 -> u in (-4,4);
        # qn L1<=1 -> y in (-1,1)). DS replaces the charge-share sense amp,
        # so CapArray read noise is zeroed on that path (assumption).
        if noiseless:
            # long charge-packet integration: quantization and noise vanish
            injection_bits, read_noise_std = 24, 0.0
            gain_mismatch = offset_std = noise_std = 0.0
        if ds_readout and not noiseless:
            # DS replaces the sense amp on this path
            read_noise_std = 0.0
        if state is None:
            state = torch.zeros(n_heads, key_dim, val_dim)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        self.n_heads = state.shape[0]
        self.key_dim = state.shape[1]
        self.val_dim = state.shape[2]
        self.q_groups = q_groups
        self.caps = [
            CapArray(state[h], injection_bits=injection_bits,
                     read_noise_std=read_noise_std)
            for h in range(self.n_heads)
        ]
        self.mul = Multiplier(gain_mismatch, offset_std, noise_std)
        self.add = Adder()
        self.neg = Scaler(-1.0, gain_mismatch)
        # q/k lane conditioning (per-head L2 norm; q carries the extra
        # 1/sqrt(KD) attenuator from the reference)
        self.q_norm = L2Norm(gain_mismatch, offset_std, noise_std)
        self.k_norm = L2Norm(gain_mismatch, offset_std, noise_std)
        self.q_atten = Scaler(1.0 / math.sqrt(self.key_dim), gain_mismatch)
        # rescale-after-decay trial: per-head AGCScaler keeps each CapArray
        # at the calibrated ideal peak; true magnitude carried in 1/g_eff
        # and reapplied to u/y (feedforward nets absorb gain downstream,
        # the recurrence cannot — e = v - g*u mixes scales).
        self.rescale_after_decay = rescale_after_decay
        self.target = target
        self.agcs = [
            AGCScaler(target=target, loop_gain=1.0,
                      detector_offset=0.0 if noiseless else 0.005,
                      detector_noise_std=0.0 if noiseless else 0.005,
                      gain_offset=0.0 if noiseless else 0.01, g0=1.0)
            for _ in range(self.n_heads)
        ]
        self.ds_readout = ds_readout
        self.ds_u = DeltaSigma(bits=ds_bits, cycles=ds_cycles,
                               comparator_noise_std=0.0 if noiseless else 0.01,
                               in_range=ds_range_u, out_range=ds_range_u)
        self.ds_y = DeltaSigma(bits=ds_bits, cycles=ds_cycles,
                               comparator_noise_std=0.0 if noiseless else 0.01,
                               in_range=ds_range_y, out_range=ds_range_y)
        self.write_gains = list(write_gains) if write_gains is not None \
            else [1.0] * self.n_heads
        self.head_mask = set(head_mask) if head_mask is not None else set()

    def _step(self, h: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              beta: torch.Tensor, alpha) -> torch.Tensor:
        if h in self.head_mask:
            return torch.zeros_like(v)
        wg = self.write_gains[h]
        if not self.rescale_after_decay:
            u = self.caps[h].read_t(k)
            e = self.add(v, self.neg(self.mul(alpha, u)))  # v - alpha * u
            self.caps[h].decay(alpha)
            self.caps[h].write_rank1(k, e * wg, beta)
            return self.caps[h].read_t(q)
        # AGC-tracked path: S_stored = S_true * g_eff, c = 1/g_eff
        agc = self.agcs[h]
        g_old = agc.g * (1.0 + agc.gain_offset)
        u_scaled = self.caps[h].read_t(k)
        if self.ds_readout:
            u_scaled = self.ds_u(u_scaled)
        u = u_scaled / g_old
        e = self.add(v, self.neg(self.mul(alpha, u)))
        self.caps[h].decay(alpha / g_old)
        self.caps[h].write_rank1(k, e, beta)
        # S_stored now holds S_true'; servo peak to calibrated target
        self.caps[h].S = agc(self.caps[h].S)
        g_new = agc.g * (1.0 + agc.gain_offset)
        y_scaled = self.caps[h].read_t(q)
        if self.ds_readout:
            y_scaled = self.ds_y(y_scaled)
        return y_scaled / g_new

    def step(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             beta: torch.Tensor, alpha) -> torch.Tensor:
        return self._step(0, q, k, v, beta, alpha)

    def sequence(self, mixed: torch.Tensor, beta: torch.Tensor,
                 g: torch.Tensor) -> torch.Tensor:
        """mixed [B, T, 2*q_groups*KD + H*VD] (q|k|v layout), beta/g [B, T, H]
        -> core readouts [B, T, H*VD]
        A bit ugly, but it's just "routing", no conversion of the signal,
        thus fast and relatively low power (hopefully)
        """
        B, T, _ = mixed.shape
        if B != 1:
            raise NotImplementedError("state is per-sample; batch=1 only")
        kd, vd, qg = self.key_dim, self.val_dim, self.q_groups
        voff = 2 * kd * qg
        out = torch.zeros(B, T, self.n_heads * vd)
        for t in range(T):
            for h in range(self.n_heads):
                kh = h * qg // self.n_heads
                q = self.q_atten(self.q_norm(mixed[0, t, kh * kd:(kh + 1) * kd], dim=-1))
                k = self.k_norm(mixed[0, t, qg * kd + kh * kd: qg * kd + (kh + 1) * kd], dim=-1)
                v = mixed[0, t, voff + h * vd: voff + (h + 1) * vd]
                out[0, t, h * vd:(h + 1) * vd] = self._step(
                    h, q, k, v, beta[0, t, h], g[0, t, h])
        return out
