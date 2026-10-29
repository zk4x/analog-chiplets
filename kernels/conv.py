# Conv: dense and depthwise MVM, feature-map domain.
#
# Conv:   extract (auto SAME / causal) -> word-line drive -> crossbar.value -> bias -> act -> scale
# DWConv: extract -> word-line drive -> column array -> bias -> act -> scale
#
# [B,Cin,H,W] -> [B,Cout,oh,ow]. All addressing and bus routing is wiring
# inside the components; models only chain components.
#
# The word-line drive v = 2x - 1 is an analog level shift (driver circuit), not
# a converter: it is linear, exact, and maps signal 0 to -1 (pad rings stay
# signal 0 — extraction happens before the drive).
#

import torch

from components.bias import Bias
from components.column_array import ColumnArray
from components.crossbar import Crossbar
from components.patch_extract import ColumnExtractor
from components.rerouter import Rerouter
from components.scaler import Scaler
from components.transposer import Transposer


class Conv:
    def __init__(
        self,
        weights,
        bias,
        stride=(1, 1),
        act=None,
        weight_bits=4,
        noiseless: bool = False,
        n_reads: int = 1,
        drift_comp: bool = False,
        n_devices: int = 1,
        dev_mult: float = 1.0,
    ):
        # weights: [Cout, Cin, kh, kw]
        cout, cin, kh, kw = weights.shape
        ck = cin * kh * kw
        self.extract = ColumnExtractor((kh, kw), stride)  # auto TF-SAME
        # patch bus: [B,C,K,oh,ow] -> rows [B*oh*ow, C*K]
        self.unl = Transposer((0, 3, 4, 1, 2))
        self.r1 = Rerouter((0, -1, ck))
        self.r2 = Rerouter((-1, ck))
        self.xb = Crossbar(
            weights.reshape(cout, ck),
            weight_bits=weight_bits,
            norm="row",
            device_noise_std=(0.0 if noiseless else 0.05 * dev_mult),
            cycle_noise_std=0.0 if noiseless else 0.05,
            drift_coeff=0.0 if noiseless else 0.03,
            n_reads=n_reads,
            drift_comp=drift_comp,
            n_devices=n_devices,
        )
        self.bias = Bias(bias)
        self.act = act
        self.store = Scaler(1.0 / 6.0, gain_mismatch=0.0 if noiseless else 0.01)

    def forward(self, x):
        # [B,Cin,H,W] -> [B,Cout,oh,ow]
        u = self.extract(x)
        b, oh, ow = u.shape[0], u.shape[-2], u.shape[-1]
        rows = self.r2(self.r1(self.unl(u)))  # [B*oh*ow, Cin*kh*kw]
        pre = self.bias(self.xb.value(2 * rows - 1))
        out = self.fold(pre, b, oh, ow)
        if self.act is not None:
            return self.store(self.act(out))
        return out

    def fold(self, pre, b, oh, ow):
        # fanout wiring [B*oh*ow, Cout] -> [B,Cout,oh,ow]; geometry is
        # runtime framing, realized as reconfigurable routing fabric
        out = Rerouter((b, oh, ow, -1))(pre)
        return Transposer((0, 3, 1, 2))(out)


class DWConv:
    def __init__(
        self,
        weights,
        bias,
        stride=(1, 1),
        act=None,
        weight_bits=4,
        padding=None,
        noiseless: bool = False,
        storage: bool = True,
        n_reads: int = 1,
        drift_comp: bool = False,
        n_devices: int = 1,
        dev_mult: float = 1.0,
    ):
        # weights: [C, 1, kh, kw]
        kh, kw = weights.shape[2], weights.shape[3]
        self.extract = ColumnExtractor((kh, kw), stride, padding=padding)
        self.columns = ColumnArray(
            weights,
            weight_bits=weight_bits,
            device_noise_std=(0.0 if noiseless else 0.05 * dev_mult),
            cycle_noise_std=0.0 if noiseless else 0.05,
            drift_coeff=0.0 if noiseless else 0.03,
            n_reads=n_reads,
            drift_comp=drift_comp,
            n_devices=n_devices,
            dev_mult=dev_mult,
        )
        self.bias = Bias(bias.reshape(1, -1, 1, 1))
        self.act = act
        # storage convention (relu6/6) is a per-model choice; raw wires when
        # the model does not use storage units (e.g. the GDN block)
        self.store = (
            Scaler(1.0 / 6.0, gain_mismatch=0.0 if noiseless else 0.01)
            if storage
            else None
        )

    def forward(self, x):
        # extract in the signal domain (pad ring = signal 0), then the
        # word-line drivers apply the bipolar level shift
        u = self.extract(x)
        out = self.bias(self.columns(2 * u - 1))
        if self.act is not None:
            out = self.act(out)
        return self.store(out) if self.store is not None else out


class DWConvStacked:
    """Depthwise conv, channel-stacked fast path: the C independent [1, K]
    column lattices are exactly one [C, K] row-norm Crossbar (row-affine
    programming is per-row already, so per-channel physics is unchanged).
    One matmul call = one drift tick = the ColumnArray parallel rule.
    update(grad[C, K]) writes all channels in one call; read_weights()
    returns the [C, K] master directly. The affine reconstruction inverts
    Crossbar.value() row-wise (matmul() exposes the raw current, no
    per-row state) — bit-exact vs C separate value() calls up to RNG
    draw order (same distribution; noiseless: exact)."""

    def __init__(
        self,
        weights,
        bias,
        stride=(1, 1),
        act=None,
        weight_bits=4,
        padding=None,
        noiseless: bool = False,
        storage: bool = True,
        n_reads: int = 1,
        drift_comp: bool = False,
        n_devices: int = 1,
        dev_mult: float = 1.0,
    ):
        # weights: [C, 1, kh, kw]
        c, kh, kw = weights.shape[0], weights.shape[2], weights.shape[3]
        self.c = c
        self.k = kh * kw
        self.extract = ColumnExtractor((kh, kw), stride, padding=padding)
        self.xb = Crossbar(
            weights.reshape(c, kh * kw),
            weight_bits=weight_bits,
            norm="row",
            device_noise_std=(0.0 if noiseless else 0.05 * dev_mult),
            cycle_noise_std=0.0 if noiseless else 0.05,
            drift_coeff=0.0 if noiseless else 0.03,
            n_reads=n_reads,
            drift_comp=drift_comp,
            n_devices=n_devices,
        )
        self.bias = Bias(bias.reshape(1, -1, 1, 1))
        self.act = act
        self.store = (
            Scaler(1.0 / 6.0, gain_mismatch=0.0 if noiseless else 0.01)
            if storage
            else None
        )

    def forward(self, x):
        # [B, C, H, W] -> extract [B, C, K, oh, ow] -> channel-major rows
        # [C*P, K] (P = B*oh*ow) -> batched row-wise dots (O(R*K) memory;
        # the full R x C outer product would be 6+ GB of discarded outputs)
        # -> fold back. Affine reconstruction inverts value() row-wise.
        u = self.extract(x)
        b, c, k, oh, ow = u.shape
        p = b * oh * ow
        rows = u.permute(0, 1, 3, 4, 2).reshape(c * p, k)
        v = 2 * rows - 1
        idx = torch.arange(c).repeat_interleave(p)
        i = self.xb.matmul_batched(v, idx)
        s = v.sum(-1)
        wmin_r = self.xb.w_min.reshape(-1).repeat_interleave(p)
        span_r = self.xb.span.reshape(-1).repeat_interleave(p)
        wsum_r = self.xb.w_sum.repeat_interleave(p)
        lat = wmin_r * s + span_r * (i / self.xb.buf2 + s) / 2
        pre = ((lat + wsum_r) / 2).reshape(c, b, oh, ow).permute(1, 0, 2, 3)
        out = self.bias(pre)
        if self.act is not None:
            out = self.act(out)
        return self.store(out) if self.store is not None else out
