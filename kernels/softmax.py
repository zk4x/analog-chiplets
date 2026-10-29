# Softmax

import torch

from components.rate_arbiter import RateArbiter


class Softmax:
    def __init__(self, n_samples: int = 64):
        self.arbiter = RateArbiter(n_samples)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return self.arbiter(scores, dim=-1)
