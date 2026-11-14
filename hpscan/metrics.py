"""Evaluation metrics (Section 5.1 of the paper).

* kappa_alpha  (`CrossValidDifference`): change of the group agreement when one rater is
  replaced by the prediction (reported x -100 in the paper, higher is better).
* kappa_v      (`VanbelleKappaIndex`): Vanbelle kappa index between the prediction and the
  group of raters on the pairwise co-membership matrices. If the prediction contains a
  single label only, the positive agreement is used instead (see `group_kappa_v`).
* kappa_n      (`NoiseIou`): Jaccard index of the predicted NOISE points, averaged over raters.
* Silhouette / Calinski-Harabasz scores (supplemental material).
* `point_agreement_scores`: per-point agreement score A (Eq. 6), the regression target of HPSCAN.
"""
import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score, silhouette_score


# --------------------------------------------------------------------------- similarity / agreement
def to_similarity_matrix(rating):
    """(B, N) cluster labels -> (B, N, N) co-membership matrix (1 = same cluster)."""
    B, N = rating.shape[0], rating.shape[1]
    rating0 = rating.reshape(B, 1, N).repeat(1, N, 1)
    rating1 = rating.reshape(B, N, 1).repeat(1, 1, N)
    return ((rating0 - rating1) == 0).long()


def point_agreement_scores(ratings):
    """Per-point agreement score A (Eq. 6) of a group of raters.

    ratings: (B, N, R) -> (B, N), averaged over all pairs of raters.
    """
    (B, N, R) = ratings.shape
    agreement_accum = torch.zeros((B, N, R)).to(ratings.device)
    for r in range(R):
        simi = to_similarity_matrix(ratings[..., r])
        agreement = []
        for r2 in range(R):
            if r == r2:
                continue
            simi2 = to_similarity_matrix(ratings[..., r2])
            score = (simi == simi2).float()
            score *= 1 - torch.eye(N).float().to(ratings.device)
            score = torch.sum(score, dim=1) / (N - 1)
            agreement.append(score.view(B, N, 1))
        agreement_accum[..., r] = torch.cat(agreement, dim=2).mean(dim=2)
    return agreement_accum.mean(dim=2)


def group_agreement(ratings, reduction=True):
    """Agreement A(G) of a group of raters: mean pairwise co-membership agreement.

    ratings: (B, N, R) tensor or (N, R) numpy array.
    """
    if isinstance(ratings, np.ndarray):
        ratings = torch.from_numpy(ratings).squeeze()
        ratings = ratings.view(1, ratings.shape[0], -1)

    def _simi(rating):
        (B, N, M, I) = rating.shape
        rating0 = rating.reshape(B, 1, N, M, I).repeat(1, N, 1, 1, 1)
        rating1 = rating.reshape(B, N, 1, M, I).repeat(1, 1, N, 1, 1)
        return ((rating0 - rating1) == 0).long()

    def _score(rating):
        N = rating.shape[1]
        simi = _simi(rating)
        return torch.sum(torch.sum(simi[..., 0] == simi[..., 1], dim=1), dim=1) / N ** 2

    R = ratings.shape[-1]
    pairs = torch.combinations(torch.arange(R), 2)
    score = _score(ratings[..., pairs])  # (B, n_pairs)
    if reduction:
        return score.mean()
    return score.mean(dim=1)


class CrossValidDifference(torch.nn.Module):
    """kappa_alpha: group agreement minus the mean agreement after replacing one rater by the prediction."""

    def __init__(self, return_individual_scores=False, reduction=True):
        super().__init__()
        self.return_individual_scores = return_individual_scores
        self.reduction = reduction

    def forward(self, pred, gt_raters):
        (B, N, R) = gt_raters.shape
        agreement = group_agreement(gt_raters, False)
        scores = []
        for fold in range(R):
            ratings = gt_raters.clone()
            ratings[..., fold] = pred
            scores.append(group_agreement(ratings, False).reshape(-1, 1))
        scores = torch.cat(scores, dim=-1)
        difference = agreement - torch.mean(scores, dim=-1)
        if self.return_individual_scores:
            individual = agreement.view(-1, 1).repeat(1, scores.shape[-1]) - scores
            return difference.mean(dim=0), individual.mean(dim=0)
        return difference.mean() if self.reduction else difference


# --------------------------------------------------------------------------- Vanbelle kappa
def _proportion_group_rater(group_rating):
    return torch.sum(group_rating, dim=2) / group_rating.shape[2]


def _observed_agreement(group_rating, isolated_rating):
    N = group_rating.shape[1]
    P = _proportion_group_rater(group_rating)
    return torch.sum(P * isolated_rating + (1.0 - P) * (1.0 - isolated_rating), dim=1) / N


def _chance_agreement(group_rating, isolated_rating):
    P = _proportion_group_rater(group_rating)
    p = torch.sum(P, dim=1) / P.shape[1]
    y = torch.sum(isolated_rating, dim=1) / isolated_rating.shape[1]
    return p * y + (1.0 - p) * (1.0 - y)


def _maximum_agreement(group_rating):
    P = _proportion_group_rater(group_rating)
    N = group_rating.shape[1]
    return torch.sum(torch.max(P, 1.0 - P), dim=1) / N


def vanbelle_kappa_binary(group_rating, isolated_rating):
    """Vanbelle kappa index of an isolated rater vs. a group of raters (binary ratings).

    group_rating: (B, I, R), isolated_rating: (B, I) -> (B,)
    """
    assert isolated_rating.shape == group_rating.shape[0:2], isolated_rating.shape
    assert torch.max(group_rating) <= 1 and torch.max(isolated_rating) <= 1, "Ratings must be binary!"
    p_o = _observed_agreement(group_rating, isolated_rating)
    p_e = _chance_agreement(group_rating, isolated_rating)
    p_m = _maximum_agreement(group_rating)
    index = (p_o - p_e) / ((p_m - p_e) + 1e-6)
    # no variability and perfect agreement -> 1 (Vanbelle and Albert)
    perfect = torch.bitwise_and(p_o == p_e, p_m == 1.0)
    index[perfect] = 1.0
    return index


def positive_agreement(group_rating, isolated_rating):
    """Fraction of identical ratings between an isolated rater and each rater of the group."""
    [B, I] = isolated_rating.shape
    R = group_rating.shape[-1]
    isolated_rating = isolated_rating.view((B, I, 1)).repeat(1, 1, R)
    correct = (isolated_rating == group_rating).float()
    correct = torch.sum(correct, dim=1) / I
    return torch.sum(correct, dim=1) / R


class IsolateGroupRating(torch.nn.Module):
    """Apply an isolated-vs-group index on the pairwise co-membership matrices."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, pred, gt):
        """pred: (B, N) labels, gt: (B, N, R) labels of the group of raters."""
        N = pred.shape[1]
        R = gt.shape[2]
        pred_simi = to_similarity_matrix(pred)
        gt0 = gt.reshape(-1, N, 1, R).repeat(1, 1, N, 1)
        gt1 = gt.reshape(-1, 1, N, R).repeat(1, N, 1, 1)
        gt_simi = ((gt0 - gt1) == 0).long()
        return self.index(gt_simi.flatten(1, 2), pred_simi.flatten(1)).mean()


class VanbelleKappaIndex(IsolateGroupRating):
    def __init__(self):
        super().__init__(vanbelle_kappa_binary)


class PositiveAgreement(IsolateGroupRating):
    def __init__(self):
        super().__init__(positive_agreement)


_VANBELLE = VanbelleKappaIndex()
_POSITIVE = PositiveAgreement()


def group_kappa_v(pred, gt_raters):
    """kappa_v as reported in the paper.

    The Vanbelle index is undefined for a prediction without variability (a single label),
    in that case the positive agreement is reported (Section 5.1).
    """
    if torch.unique(pred).shape[0] == 1:
        return _POSITIVE(pred, gt_raters)
    return _VANBELLE(pred, gt_raters)


# --------------------------------------------------------------------------- noise IoU
class NoiseIou(torch.nn.Module):
    """kappa_n: Jaccard index of NOISE points between the prediction and each rater (Eq. 11)."""

    @staticmethod
    def jaccard(pred, gt):
        # inputs: 0 = noise, 1 = cluster. Flip so that noise is the positive class.
        pred = 1 - pred
        gt = 1 - gt
        intersection = torch.sum(pred * gt)
        union = torch.sum(torch.clip(pred + gt, 0, 1))
        no_noise_pred = torch.sum(pred) == 0
        noise_fraction = torch.sum(gt) / gt.shape[0]
        if no_noise_pred and noise_fraction == 0:
            return 1.0
        if no_noise_pred:
            return 1 - noise_fraction
        return intersection / union

    def forward(self, pred, gt_raters):
        """pred: (B, N) binary noise prediction, gt_raters: (B, N, R) binary rater noise labels."""
        (B, _, R) = gt_raters.shape
        scores = torch.zeros((B))
        for b in range(B):
            rater_scores = torch.zeros((R))
            for r in range(R):
                rater_scores[r] = self.jaccard(pred[b], gt_raters[b, :, r])
            scores[b] = rater_scores.mean()
        return scores.mean()


# --------------------------------------------------------------------------- cluster validity indices
class ClusterScore(torch.nn.Module):
    """Average an sklearn cluster validity index over samples (and raters); NaN if undefined."""

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def compute_score(self, x, label):
        scores = []
        for v, l in zip(x, label):
            if len(torch.unique(l)) > 1:
                scores.append(self.metric(v.cpu(), l.cpu()))
        return torch.tensor(scores).mean()

    def forward(self, x, labels):
        if len(labels.shape) == 2:
            return self.compute_score(x, labels)
        scores = [self.compute_score(x, labels[..., r]) for r in range(labels.shape[-1])]
        return torch.tensor([s for s in scores if not torch.isnan(s)]).mean()


class SilhouetteIndex(ClusterScore):
    def __init__(self):
        super().__init__(silhouette_score)


class CalinskiHarabaszScore(ClusterScore):
    def __init__(self):
        super().__init__(calinski_harabasz_score)
