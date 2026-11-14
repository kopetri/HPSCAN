"""Training losses (Section 3.3 of the paper).

`PairEnum`, `Class2Simi` and `MCL` follow the meta classification likelihood of
Hsu et al., adapted from https://github.com/GT-RIPL/L2C (MIT License,
Copyright (c) 2018 GT-RIPL), see THIRD_PARTY_NOTICES.md. The cluster-size and
negative-pair weighting (omega_D, Eq. 3) is specific to HPSCAN.
"""
import torch
import torch.nn as nn


def PairEnum(x):
    """Enumerate all point pairs. x: (B, N, C) -> two tensors of shape (B, N, N, C)."""
    assert x.ndimension() == 3, "Input dimension must be 3"
    B, N, C = x.shape
    x1 = x.repeat(1, N, 1).view(B, N, N, C)
    x2 = x.repeat(1, 1, N).view(B, N, N, C)
    return x1, x2


def Class2Simi(x, dissimilar_weight):
    """Convert per-point cluster ids into a pairwise similarity target and weights (Eqs. 1 and 3).

    x: (B, N) integer labels.
    Returns `simi` (B, N, N) with +1 for pairs in the same cluster and -1 otherwise, and the
    weight matrix omega_D: 1 / w_c^2 for similar pairs (w_c = relative size of the cluster of
    the first point) and D (`dissimilar_weight`) for dissimilar pairs.
    """
    B, N = x.shape
    expand1 = x.reshape(B, N, 1).repeat(1, 1, N)
    expand2 = x.reshape(B, 1, N).repeat(1, N, 1)

    weights = torch.zeros_like(x).float()
    for c in torch.unique(x):
        mask = x == c
        w = torch.sum(mask, dim=1) / N
        w[w == 0] = 1.0
        w = 1.0 / w ** 2
        weights += mask.float() * w.unsqueeze(1).repeat(1, N)
    weights = weights.reshape(B, N, 1).repeat(1, 1, N)

    out = expand1 - expand2
    simi = out == 0
    out[~simi] = -1
    out[simi] = 1
    weights[~simi] = dissimilar_weight
    return out.float(), weights.float()


class MCL(nn.Module):
    """Weighted meta classification likelihood (contrastive loss L_mcl, Eq. 2)."""

    eps = 1e-7  # avoid log(0)

    def forward(self, prob1, prob2, simi, weights=None):
        assert len(prob1) == len(prob2) == len(simi), "Wrong input size"
        P = prob1.mul_(prob2)
        P = P.sum(-1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(self.eps).log_()
        if weights is not None:
            neglogP *= weights
        return neglogP.mean()


class NoiseLoss(nn.Module):
    """Weighted binary cross entropy for NOISE vs. CLUSTER (L_noise, Eq. 4)."""

    def __init__(self, noise_weights=(0.9, 0.1)):
        super().__init__()
        self.noise_weights = list(noise_weights)

    def forward(self, pred, gt):
        cel = torch.nn.CrossEntropyLoss(torch.tensor(self.noise_weights), reduction="none").to(pred)
        return cel(pred, gt).mean()
