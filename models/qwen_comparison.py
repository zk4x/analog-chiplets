# Qwen3.8-27B blk.0: orig vs finetuned weights, full-stack analog config.
#
# Standalone: cd analog_decomposition && python3 models/qwen_comparison.py
# No environment, no arguments — reads the two weights files from ../data
# and prints ideal-vs-golden plus noiseless/default chip metrics vs golden.
#
# Stack (linear CapArray GDN = the hardware the sheet sizes,
# drift_comp, N_READS=16, M=4 devices, squelch mask); WEIGHT_BITS=16.
# Block mapping (layer-level chiplets; this simulation uses one converter pair at block I/O):
#   DAC -> [Conv(w_qkv), Conv(w_z), Conv(w_b), Conv(w_a)]      1x1 GEMMs
#       -> DWConv(conv_w, causal, act=SiLU, storage=False)      ssm_conv1d
#       -> Gate (sigmoid(beta), decay g)                        from b, a rows
#       -> GDNState.sequence (L2Norm q/k inside, multi-head)     delta core
#       -> RMSNorm(gain=ss_norm_w, per 128-lane head)
#       -> SiLUKernel -> Multiply                               gated norm
#       -> Conv(w_o) -> ADC
#

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.adc import ADC
from components.dac import DAC
from components.silu import SiLU
from kernels.conv import Conv, DWConv
from kernels.gate import Gate
from kernels.gdn_state import GDNState
from kernels.multiply import Multiply
from kernels.rmsnorm import RMSNorm
from kernels.silu import SiLUKernel

# Static squelch mask: heads whose ideal ||y|| sits under the RMSNorm floor
# (boot-calibrated lane depopulation; +0.007 noiseless, -20% default).
HEAD_MASK = [20, 32, 36, 37, 45]

S = 6
HIDDEN = 5120
DT_RANK = 48
VH = 48
VD = 128
KEY_DIM = 2048
VAL_DIM = 6144
CONV_DIM = 10240  # KEY_DIM*2 + VAL_DIM
M_PAD = 16
EPS = 1e-6
KD = 128  # per-head key dim

DATA = (
    Path(__file__).parent.parent / "data" / "qwen3_linear_attention_block.safetensors"
)
TUNED = (
    Path(__file__).parent.parent
    / "data"
    / "qwen3_linear_attention_block_finetuned.safetensors"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHT_BITS = 16  # device lattice levels (this block's decision)
ADC_BITS = 12  # output converter resolution (see note in build_net)
N_RUNS = 3
# Deployment stack: linear CapArray GDN (the hardware the sheet sizes),
# drift comp, 16 reads, 4 devices, mask.
CFG = dict(drift_comp=True, n_reads=16,
           head_mask=HEAD_MASK, n_devices=4, dev_mult=1.0)


def load(path=None):
    W = load_file(path or DATA)
    return {k: v.to(DEVICE) for k, v in W.items()}


def ideal(x, W):
    """Op-for-op mirror of the golden reference script. x: [S, HIDDEN]."""
    pinned = torch.zeros(M_PAD, HIDDEN, dtype=torch.float32)
    pinned[:S] = x
    pinned = pinned.half().float()  # simulate F16 conversion

    qkv = pinned @ W["w_qkv"].T  # [M_PAD, CONV_DIM]
    z = pinned @ W["w_z"].T  # [M_PAD, VAL_DIM]
    b = pinned @ W["w_b"].T  # [M_PAD, DT_RANK]
    a = pinned @ W["w_a"].T  # [M_PAD, DT_RANK]

    conv_w = W["conv_w"]
    mixed = torch.zeros(M_PAD, CONV_DIM, dtype=torch.float32)
    for t in range(M_PAD):
        for k in range(4):
            src_t = t + k - 3
            if src_t >= 0:
                mixed[t] += conv_w[:, k] * qkv[src_t]
    mixed = mixed / (1.0 + torch.exp(-mixed))  # silu

    ss_a, ss_dt = W["ss_a"], W["ss_dt_bias"]
    core = torch.zeros(VH, S, VD, dtype=torch.float32)
    for h in range(VH):
        S_state = torch.zeros(VD, KD, dtype=torch.float32)
        for t in range(S):
            kh = h // (VH // 16)
            q_row = mixed[t, kh * KD : (kh + 1) * KD]
            k_row = mixed[t, KEY_DIM + kh * KD : KEY_DIM + (kh + 1) * KD]
            v_row = mixed[t, 2 * KEY_DIM + h * VD : 2 * KEY_DIM + (h + 1) * VD]
            q_ss = (q_row * q_row).sum()
            k_ss = (k_row * k_row).sum()
            qn = q_row / torch.sqrt(q_ss + EPS) / math.sqrt(KD)
            kn = k_row / torch.sqrt(k_ss + EPS)
            beta = torch.sigmoid(b[t, h])
            adt = a[t, h] + ss_dt[h]
            sp = torch.where(adt > 20, adt, torch.log(1.0 + torch.exp(adt)))
            g = torch.exp(-ss_a[h] * sp)
            kv = S_state @ kn
            v_new = v_row - g * kv
            delta = beta * v_new
            S_state = g * S_state + torch.outer(delta, kn)
            core[h, t] = S_state @ qn

    mean = (core * core).sum(dim=-1, keepdim=True) / KD
    scale = 1.0 / torch.sqrt(mean + EPS)
    normed_core = core * scale * W["ss_norm_w"].view(1, 1, VD)
    gz = (z[:S] / (1.0 + torch.exp(-z[:S]))).reshape(S, VH, VD)
    normed = (normed_core * gz.permute(1, 0, 2)).reshape(S, VAL_DIM)

    normed_p = torch.zeros(M_PAD, VAL_DIM, dtype=torch.float32)
    normed_p[:S] = normed
    normed_p = normed_p.half().float()

    out_padded = normed_p @ W["w_o"].T  # [M_PAD, HIDDEN]
    return out_padded[:S]


def build_net(
    noiseless=False,
    weight_bits=WEIGHT_BITS,
    drift_comp=False,
    n_reads=1,
    head_mask=None,
    n_devices=1,
    dev_mult=1.0,
    weights_path=None,
):
    """Simulation framing: one DAC at the token bus and one ADC at the output
    bus; the chiplet partition is layer-level, but this comparison evaluates the
    block in one pass. Everything between the converters is analog. Signed
    voltages throughout — no affine tricks.
    GDN core is the linear CapArray cell (the hardware the sheet sizes).

    Options (all default off = baseline behavior): drift_comp enables
    reference-column ratio on every crossbar; n_reads averages cycle noise;
    head_mask statically depopulates sub-floor heads (squelch).

    Framing (reconfigurable routing fabric, not compute): per-token GEMMs run
    with T on the batch axis; the temporal conv1d runs as [1, C, 1, T]."""

    def p3():
        return (0.0, 0.0, 0.0) if noiseless else (0.01, 0.01, 0.005)

    W = load(weights_path)

    def gemm(w):
        # [Cout, Cin] -> 1x1 Conv chiplet (T rides the batch axis)
        return Conv(
            w.unsqueeze(-1).unsqueeze(-1),
            torch.zeros(1),
            act=None,
            weight_bits=weight_bits,
            noiseless=noiseless,
            n_reads=n_reads,
            drift_comp=drift_comp,
            n_devices=n_devices,
            dev_mult=dev_mult,
        )

    pk = W["input"].abs().max().item()
    # ADC spans the full output peak: the runaway heads (negative ss_a ->
    # g > 1) put a few outlier lanes near +-19 whose energy the rel-L2 metric
    # cannot afford to clip; the block therefore needs a high-resolution
    # output converter (12-bit -> 0.5% ADC floor, 10-bit -> 1.8%, 8-bit -> 7.4%)
    po = ideal(W["input"], W).abs().max().item()
    silu_c = SiLU(*p3())
    gate = Gate(*p3())
    rms = RMSNorm(128, *p3())
    rms.g = W["ss_norm_w"].clone()

    _net = (
        DAC(8, in_range=(-pk, pk), out_range=(-pk, pk)),
        gemm(W["w_qkv"]),
        gemm(W["w_z"]),
        gemm(W["w_b"]),
        gemm(W["w_a"]),
        DWConv(
            W["conv_w"].unsqueeze(1).unsqueeze(1),
            torch.zeros(1),
            padding="causal",
            act=silu_c,
            weight_bits=weight_bits,
            noiseless=noiseless,
            storage=False,
            n_reads=n_reads,
            # NO drift_comp here: depthwise lattice is K=4 wide, so a
            # reference read carries ~3.5% cycle+device noise against a
            # 2% drift signal — compensation adds more than it removes.
            # Rule: reference needs CK >> 1 (the 5 dense GEMMs qualify).
            drift_comp=False,
            n_devices=n_devices,
            dev_mult=dev_mult,
        ),
        gate,
        W["ss_a"],
        W["ss_dt_bias"],
        GDNState(
            noiseless=noiseless, n_heads=VH, key_dim=KD, val_dim=VD, q_groups=16
        ),
        rms,
        SiLUKernel(noiseless),
        Multiply(noiseless),
        gemm(W["w_o"]),
        ADC(ADC_BITS, in_range=(-po, po), out_range=(-po, po)),
    )
    if head_mask:
        _net[9].head_mask = set(head_mask)
    return _net


def forward(net, x):
    (
        dac,
        proj_qkv,
        proj_z,
        proj_b,
        proj_a,
        conv1d,
        gate,
        ss_a,
        dt_bias,
        gdn,
        rms,
        silu,
        mul,
        oproj,
        adc,
    ) = net

    v = torch.zeros(M_PAD, HIDDEN)
    v[:S] = x  # zero-pad ring to the kernel tile
    v = dac(v)  # [M_PAD, HIDDEN] analog token bus
    v4 = v.unsqueeze(2).unsqueeze(3)  # [T, C=H, 1, 1]: T rides the batch axis

    qkv = proj_qkv.forward(v4).reshape(M_PAD, CONV_DIM)
    z = proj_z.forward(v4).reshape(M_PAD, VAL_DIM)
    b = proj_b.forward(v4).reshape(M_PAD, DT_RANK)
    a = proj_a.forward(v4).reshape(M_PAD, DT_RANK)

    # temporal conv + SiLU: [M_PAD, C] -> [1, C, 1, T] -> [M_PAD, C]
    m = conv1d.forward(qkv.T.unsqueeze(0).unsqueeze(2)).squeeze(0).squeeze(1).T

    beta = gate.beta(b).unsqueeze(0)  # [1, T, H]
    g = gate.decay(a, dt_bias, ealog=ss_a).unsqueeze(0)  # [1, T, H]
    core = gdn.sequence(m.unsqueeze(0), beta, g)[:, :S]  # [1, S, H*VD]
    core = core.reshape(S, VH, VD)

    normed_3d = torch.stack(
        [rms.forward(core[:, h]) for h in range(VH)], dim=1
    )  # [S, VH, VD]
    # lane pack must replicate ideal/golden: product lives in [VH, S, VD]
    # and is then interleaved via .reshape(S, VAL_DIM) (VH-major buffer
    # reinterpreted as S-major). Head-major .reshape would shuffle lanes
    # and gives the 119% deterministic error.
    gz_flat = silu.forward(z[:S])  # [S, VAL_DIM] head-major
    gz_3d = gz_flat.reshape(S, VH, VD)
    normed_flat_interleaved = normed_3d.permute(1, 0, 2).reshape(S, VAL_DIM)
    gz_flat_interleaved = gz_3d.permute(1, 0, 2).reshape(S, VAL_DIM)
    normed = mul.forward(normed_flat_interleaved, gz_flat_interleaved)

    return adc(oproj.forward(normed.unsqueeze(2).unsqueeze(3)).reshape(S, HIDDEN))


def dist_metrics(y, ref):
    rel = ((y - ref).norm() / ref.norm()).item()
    corr = (
        sum(
            torch.corrcoef(torch.stack([yr, rr]))[0, 1].item() for yr, rr in zip(y, ref)
        )
        / y.shape[0]
    )
    return rel, corr


if torch.cuda.is_available():
    torch.set_default_device("cuda")

FILES = {"orig": DATA, "tuned": TUNED}

for tag, path in FILES.items():
    W = load(path)
    x, golden = W["input"], W["output"]
    ideal_out = ideal(x, W)
    ri, ci = dist_metrics(ideal_out, golden)
    print(f"[{tag}] ideal-vs-golden: rel={ri:.6f} corr={ci:.6f}", flush=True)
    for label, nl in (("noiseless", True), ("default", False)):
        rels, corrs = [], []
        for _ in range(N_RUNS):
            net = build_net(noiseless=nl, weights_path=path, **CFG)
            y = forward(net, x)
            r, c = dist_metrics(y, golden)
            rels.append(r)
            corrs.append(c)
        r = torch.tensor(rels)
        c = torch.tensor(corrs)
        print(
            f"[{tag} {label}] rel={r.mean():.4f}+-{r.std():.4f} corr={c.mean():.4f}+-{c.std():.4f}",
            flush=True,
        )
print("done")
