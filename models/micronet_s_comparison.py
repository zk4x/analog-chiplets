# MicroNet-S KWS: chiplet chain vs pytorch ideal.
#
# Real weights from ARM ML-zoo kws_micronet_s.tflite (per-channel dequantized,
# BN pre-folded). The chip path is a pure chain of chiplets (Conv, DWConv,
# AvgPool) in feature-map domain — no tensor ops, addressing/pooling live
# inside the chiplets. The torch ideal is the test bench.
#
# Converters are calibrated per chiplet (boot-time calibration: N_CAL random
# inputs through the ideal pass; per-tensor range, lo = hard min,
# hi = PERCENTILE; DAC gets the input range, the output converter
# gets the chiplet's output range).
#

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.adc import ADC
from components.agc import AGCScaler
from components.dac import DAC
from components.limiter import Limiter
from kernels.conv import Conv, DWConv
from kernels.pool import AvgPool
from kernels.softmax import Softmax

torch.manual_seed(42)

W = load_file(Path(__file__).parent.parent / "data/kws_micronet_s.safetensors")

# converter range percentile: per-model tuning knob
PERCENTILE = 99.5
N_CAL = 16
N_RUNS = 10
WEIGHT_BITS = 8  # int8, matching the tflite
N_READS = 1  # read-averaging knob: energy ~ N, cycle noise ~ 1/sqrt(N)

OPS = [  # (kind, idx, kernel_hw, stride_hw)
    ("conv", 0, (10, 4), (1, 1)),
    ("dw", 1, (3, 3), (2, 2)),
    ("pw", 2, (1, 1), (1, 1)),
    ("dw", 3, (3, 3), (1, 1)),
    ("pw", 4, (1, 1), (1, 1)),
    ("dw", 5, (3, 3), (1, 1)),
    ("pw", 6, (1, 1), (1, 1)),
    ("dw", 7, (3, 3), (1, 1)),
    ("pw", 8, (1, 1), (1, 1)),
    ("dw", 9, (3, 3), (1, 1)),
    ("pw", 10, (1, 1), (1, 1)),
    ("head", 11, (1, 1), (1, 1)),
]


def spec():
    """Chiplet sequence (config only). The head splits into AvgPool + 1x1 Conv."""
    s = []
    for kind, i, (kh, kw), (sh, sw) in OPS:
        if kind == "head":
            s.append(("pool", None, None, None))
        s.append((kind, i, (kh, kw), (sh, sw)))
    return s


SPEC = spec()


def tf_same_pad(x, kh, kw, sh, sw):
    h, w = x.shape[-2:]
    ph = max((-(h // -sh) - 1) * sh + kh - h, 0)
    pw = max((-(w // -sw) - 1) * sw + kw - w, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def ideal_step(entry, y):
    kind, i, k, st = entry
    if kind == "pool":
        return y.mean(dim=(2, 3), keepdim=True)
    kh, kw = k
    sh, sw = st
    w, b = W[f"op{i}_w"], W[f"op{i}_b"]
    y = tf_same_pad(y, kh, kw, sh, sw)
    if kind == "dw":
        y = F.conv2d(y, w, b, stride=(sh, sw), groups=y.shape[1])
    else:
        y = F.conv2d(y, w, b, stride=(sh, sw))
    if kind == "head":
        return y
    return F.relu6(y) / 6.0


def ideal(x):
    """Torch ideal in storage units: activations divided by 6 after each relu6."""
    y = x
    for e in SPEC:
        y = ideal_step(e, y)
    return y.flatten(1)


def calibrate(n=N_CAL, percentile=PERCENTILE):
    """Per-chiplet converter ranges from random inputs through the ideal pass."""
    ins = [[] for _ in SPEC]
    outs = [[] for _ in SPEC]
    for _ in range(n):
        y = torch.rand(1, 1, 49, 10)
        for s, e in enumerate(SPEC):
            ins[s].append(y.flatten())
            y = ideal_step(e, y)
            outs[s].append(y.flatten())

    def rng(bufs):
        v = torch.cat(bufs)
        return (
            v.min().item(),
            torch.quantile(v, percentile / 100.0).item(),
            v.abs().max().item(),
        )

    return [rng(b) for b in ins], [rng(b) for b in outs]


def build_net(rin, rout, noiseless=False, weight_bits=4, n_reads=N_READS):
    """Converters (DAC at the input, ADC at the output) live here, at the
    tape-out border — per RULES.md the whole net is one analog chiplet for
    this demo. Per-stage Scalers: not needed, the relu6/6 storage convention
    keeps every stage on the [0,1] rail."""

    def agc(s):
        # per-stage dynamic rescaling: servo the stage output peak to the
        # ideal stage's peak (measured on the wire, absorbs upstream physics)
        return AGCScaler(
            target=rout[s][2],
            loop_gain=1.0,
            detector_offset=0.0 if noiseless else 0.005,
            detector_noise_std=0.0 if noiseless else 0.005,
            gain_offset=0.0 if noiseless else 0.01,
        )

    net = [DAC(8, in_range=(0.0, 1.0), out_range=(0.0, 1.0))]
    for s, (kind, i, k, st) in enumerate(SPEC):
        if kind == "pool":
            net.append(AvgPool(gain_mismatch=0.0 if noiseless else 0.01))
        elif kind == "dw":
            net.append(
                DWConv(
                    W[f"op{i}_w"],
                    W[f"op{i}_b"],
                    st,
                    act=Limiter(
                        0.0, 6.0, threshold_mismatch=0.0 if noiseless else 0.01
                    ),
                    noiseless=noiseless,
                    weight_bits=weight_bits,
                    n_reads=n_reads,
                )
            )
        else:
            net.append(
                Conv(
                    W[f"op{i}_w"],
                    W[f"op{i}_b"],
                    st,
                    act=None
                    if kind == "head"
                    else Limiter(
                        0.0, 6.0, threshold_mismatch=0.0 if noiseless else 0.01
                    ),
                    noiseless=noiseless,
                    weight_bits=weight_bits,
                    n_reads=n_reads,
                )
            )
        if kind != "head":
            net.append(agc(s))
    net.append(ADC(8, in_range=rout[-1][:2], out_range=rout[-1][:2]))
    return net


def forward(net, x):
    y = x
    for kern in net:
        y = kern.forward(y) if hasattr(kern, "forward") else kern(y)
    return y


def dist_metrics(y, ref):
    """Classifier-style comparison: cross-entropy and KL of the analog
    softmax against the ideal softmax (labels unavailable in this harness)."""
    p = torch.softmax(ref, -1)
    q = torch.softmax(y, -1)
    ce = -(p * q.clamp_min(1e-12).log()).sum(-1).item()
    h = -(p * p.clamp_min(1e-12).log()).sum(-1).item()
    return ce, ce - h


def main():
    x = torch.rand(1, 1, 49, 10)  # MFCC-shaped input already in [0,1]

    ref = ideal(x)
    print(f"ref logits: argmax={ref.argmax().item()} std={ref.std():.4f}")

    rin, rout = calibrate()
    net = build_net(rin, rout)

    for label in ("noiseless", "default"):
        net = build_net(
            rin, rout, noiseless=label == "noiseless", weight_bits=WEIGHT_BITS
        )
        errs, ces, kls = [], [], []
        for _ in range(N_RUNS):
            y = forward(net, x).flatten(1)
            errs.append(((y - ref).norm() / (ref.norm() + 1e-8)).item())
            ce, kl = dist_metrics(y, ref)
            ces.append(ce)
            kls.append(kl)
        errs, ces, kls = torch.tensor(errs), torch.tensor(ces), torch.tensor(kls)
        print(
            f"[{label}] rel_L2 = {errs.mean():.4f}+-{errs.std():.4f}  "
            f"CE = {ces.mean():.4f}+-{ces.std():.4f}  "
            f"KL = {kls.mean():.5f}+-{kls.std():.5f}  "
            f"argmax {'==' if (y.argmax() == ref.argmax()).item() else '!='} ideal"
        )

    votes = Softmax(n_samples=2000).forward(ref.flatten())
    print(
        f"arbiter argmax={votes.argmax().item()} vs ideal argmax={ref.argmax().item()}"
    )


if __name__ == "__main__":
    main()
