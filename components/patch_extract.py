# Column extractor (patch bus)
#
# Sliding-window extraction of kernel-sized columns from a feature map:
# patch registers + tap wiring. Owns the output geometry (oh, ow) since it
# walks the map with kernel/stride/padding, no arithmetic.
#
# with padding=None the TF-SAME pads are derived automatically from the
# input geometry — asymmetric pads land the extra row/column at the
# bottom/right, as TF does (e.g. 10x4 on 49x10 -> pads 4/5 and 1/2).
# The pad ring is signal 0: extraction happens in the signal domain,
# before any DAC (in the bipolar DAC domain a zero pad would wrongly be
# the mid-rail 0.5).
#
# param       | circuit element               | error class
# -------------------------------------------------------------------------------------
# kernel_hw   | tap window size               | -
# stride      | tap advance per step          | -
# padding     | border ring (same-pad ring)   | None = automatic TF-SAME,
#             |                               | "causal" = left-pad k-1 (only
#             |                               |   along the last axis)
#

import torch
import torch.nn.functional as F


class ColumnExtractor:
    def __init__(
        self,
        kernel_hw: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int] | str | None = None,
    ):
        self.kernel = kernel_hw
        self.stride = stride
        self.padding = padding

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C, K, oh, ow]
        b, c = x.shape[0], x.shape[1]
        kh, kw = self.kernel
        sh, sw = self.stride
        if self.padding == "causal":
            # sequence conv: only the past is visible (pad k-1 at the left of
            # the time axis, 0 at the right; time = last feature axis)
            x = F.pad(x, (kw - 1, 0, 0, 0))
            pad = (0, 0)
        elif self.padding is None:
            h, w = x.shape[-2:]
            pt = max((-(h // -sh) - 1) * sh + kh - h, 0)
            pl = max((-(w // -sw) - 1) * sw + kw - w, 0)
            x = F.pad(x, (pl // 2, pl - pl // 2, pt // 2, pt - pt // 2))
            pad = (0, 0)
        else:
            pad = self.padding
        u = F.unfold(x, self.kernel, stride=self.stride, padding=pad)
        ph, pw = pad
        oh = (x.shape[-2] + 2 * ph - kh) // sh + 1
        ow = (x.shape[-1] + 2 * pw - kw) // sw + 1
        return u.reshape(b, c, kh * kw, oh, ow)
