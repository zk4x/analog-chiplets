# Rate arbiter
#
# Acts kinda like a softmax, but more like an argmax, especially at low temps
#
# The exp is represented with one subthreshold transistor (so no need for LUT :) ),
# spike generator (elementwise poisson sample) is one capacitor, one comparator and a reset switch
#
# Arbitration and counting is digital
#
# param         | circuit element                              | error class
# --------------------------------------------------------------------------------------------------------
# n_samples     | counting window T (spikes collected)         | statistical, ~1/sqrt(N)
# rate_mismatch | per-class translinear exp gain mismatch      | systematic, per class (the enemy)
# refractory    | absolute max firing rate (reset dead time)   | saturating, flattens top band
# temperature   | bias voltage on the exp (gain knob)          | -
#

import torch


class RateArbiter:
    def __init__(
        self,
        n_samples: int = 64,
        rate_mismatch: float = 0.0,
        refractory: float = 0.0,
        temperature: float = 1.0,
    ):
        self.n_samples = n_samples
        self.rate_mismatch = rate_mismatch
        self.refractory = refractory
        self.temperature = temperature

    def rates(self, scores: torch.Tensor) -> torch.Tensor:
        r = torch.exp(scores / self.temperature)
        if self.refractory > 0:
            r_max = r.max(dim=-1, keepdim=True).values
            r = torch.clamp(r, max=r_max * (1.0 - self.refractory))
        if self.rate_mismatch > 0:
            r = r * (1.0 + torch.randn_like(r) * self.rate_mismatch)
        return r / r.sum(dim=-1, keepdim=True)

    def __call__(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        rates = self.rates(scores)
        shape = rates.shape
        flat = rates.reshape(-1, shape[-1]).clamp(0.0)
        probs = torch.zeros_like(flat)
        counts = torch.multinomial(flat, self.n_samples, replacement=True)
        probs.scatter_add_(-1, counts, torch.ones_like(counts, dtype=probs.dtype))
        return (probs / self.n_samples).reshape(shape)
