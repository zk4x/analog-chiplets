# MobileNetV1 (1.0/224): chiplet chain vs pytorch ideal.
#
# Real weights from HF google/mobilenet_v1_1.0_224 (safetensors), BN folded
# into the convs at load time.
#
# Activations divided by 6 after each
# relu6. Head = analog AvgPool + 1x1 Conv classifier (act=None) -> ADC ->
# raw logits.
#

import sys
import time
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
W = load_file(Path(__file__).parent.parent / "data/mobilenet_v1_1.0_224.safetensors")

# converter range percentile: per-model tuning knob
PERCENTILE = 99.95  # heavy-tailed relu6/6 activations: needs near-max range
N_CAL = 16
N_RUNS = 10
WEIGHT_BITS = 8
N_READS = 1  # read-averaging knob: energy ~ N, cycle noise ~ 1/sqrt(N)

# dw layers (even HF indices) with stride 2; odd indices are pointwise.
DW_STRIDE2 = {2, 6, 10, 14}


def fold(prefix):
    """Fold BN running stats into conv weight/bias."""
    w = W[f"{prefix}.convolution.weight"].float()
    g = W[f"{prefix}.normalization.weight"].float()
    b = W[f"{prefix}.normalization.bias"].float()
    mu = W[f"{prefix}.normalization.running_mean"].float()
    var = W[f"{prefix}.normalization.running_var"].float()
    rstd = (var + 1e-5).rsqrt()
    s = (g * rstd).view(-1, 1, 1, 1)
    return w * s, b - mu * g * rstd


def spec():
    s = [("stem", -1, (3, 3), (2, 2))]
    for i in range(26):
        if i % 2 == 0:
            st = (2, 2) if i in DW_STRIDE2 else (1, 1)
            s.append(("dw", i, (3, 3), st))
        else:
            s.append(("pw", i, (1, 1), (1, 1)))
    s.append(("pool", None, None, None))
    s.append(("head", None, (1, 1), (1, 1)))
    return s


SPEC = spec()


def weights_of(entry):
    kind, i, _, _ = entry
    if kind == "stem":
        return fold("mobilenet_v1.conv_stem")
    if kind == "head":
        return W["classifier.weight"].float().unsqueeze(-1).unsqueeze(-1), W[
            "classifier.bias"
        ].float()
    return fold(f"mobilenet_v1.layer.{i}")


def tf_same_pad(x, kh, kw, sh, sw):
    h, w = x.shape[-2:]
    ph = max((-(h // -sh) - 1) * sh + kh - h, 0)
    pw = max((-(w // -sw) - 1) * sw + kw - w, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def ideal_step(entry, y):
    kind, i, k, st = entry
    if kind == "pool":
        return y.mean(dim=(2, 3), keepdim=True)
    w, b = weights_of(entry)
    y = tf_same_pad(y, k[0], k[1], st[0], st[1])
    if kind == "dw":
        y = F.conv2d(y, w, b, stride=st, groups=y.shape[1])
    else:
        y = F.conv2d(y, w, b, stride=st)
    if kind == "head":
        return y
    return F.relu6(y) / 6.0


def ideal(x):
    y = x
    for e in SPEC:
        y = ideal_step(e, y)
    return y.flatten(1)


def calibrate(n=N_CAL, percentile=PERCENTILE):
    """Per-chiplet converter ranges from random inputs through the ideal pass."""
    ins = [[] for _ in SPEC]
    outs = [[] for _ in SPEC]
    for _ in range(n):
        y = torch.rand(1, 3, 224, 224)
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


def build_net(rin, rout, noiseless=False, weight_bits=WEIGHT_BITS, n_reads=N_READS):
    """Converters (DAC in, ADC out) live here at the tape-out border; per-stage
    AGC servos each stage's output peak to the ideal peak. relu6/6 storage
    keeps every stage on the [0,1] rail, so no static per-stage Scalers."""
    tm = 0.0 if noiseless else 0.01

    def agc(s):
        return AGCScaler(
            target=rout[s][2],
            loop_gain=1.0,
            detector_offset=0.0 if noiseless else 0.005,
            detector_noise_std=0.0 if noiseless else 0.005,
            gain_offset=0.0 if noiseless else 0.01,
        )

    net = [DAC(8, in_range=(0.0, 1.0), out_range=(0.0, 1.0))]
    for s, e in enumerate(SPEC):
        kind, i, k, st = e
        if kind == "pool":
            net.append(AvgPool(gain_mismatch=0.0 if noiseless else 0.01))
        elif kind == "dw":
            w, b = weights_of(e)
            net.append(
                DWConv(
                    w,
                    b,
                    st,
                    act=Limiter(0.0, 6.0, threshold_mismatch=tm),
                    noiseless=noiseless,
                    weight_bits=weight_bits,
                    n_reads=n_reads,
                )
            )
        else:
            w, b = weights_of(e)
            act = None if kind == "head" else Limiter(0.0, 6.0, threshold_mismatch=tm)
            net.append(
                Conv(
                    w,
                    b,
                    st,
                    act=act,
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
    p = torch.softmax(ref, -1)
    q = torch.softmax(y, -1)
    ce = -(p * q.clamp_min(1e-12).log()).sum(-1).item()
    h = -(p * p.clamp_min(1e-12).log()).sum(-1).item()
    return ce, ce - h


def main():
    x = torch.rand(1, 3, 224, 224)

    t0 = time.time()
    ref = ideal(x)
    print(
        f"ref logits: shape={tuple(ref.shape)} argmax={ref.argmax().item()} "
        f"std={ref.std():.4f}  (ideal {time.time() - t0:.1f}s)"
    )

    t0 = time.time()
    rin, rout = calibrate()
    print(f"calibration {time.time() - t0:.1f}s")

    for label in ("noiseless", "default"):
        net = build_net(rin, rout, noiseless=label == "noiseless")
        errs, ces, kls = [], [], []
        for _ in range(N_RUNS):
            t0 = time.time()
            y = forward(net, x).flatten(1)
            dt = time.time() - t0
            errs.append(((y - ref).norm() / (ref.norm() + 1e-8)).item())
            ce, kl = dist_metrics(y, ref)
            ces.append(ce)
            kls.append(kl)
        errs, ces, kls = torch.tensor(errs), torch.tensor(ces), torch.tensor(kls)
        print(
            f"[{label}] rel_L2 = {errs.mean():.4f}+-{errs.std():.4f}  "
            f"CE = {ces.mean():.4f}+-{ces.std():.4f}  "
            f"KL = {kls.mean():.5f}+-{kls.std():.5f}  "
            f"argmax {'==' if (y.argmax() == ref.argmax()).item() else '!='} ideal  "
            f"({dt:.1f}s/forward)"
        )

    votes = Softmax(n_samples=2000).forward(ref.flatten())
    print(
        f"arbiter argmax={votes.argmax().item()} vs ideal argmax={ref.argmax().item()}"
    )


if __name__ == "__main__":
    main()
