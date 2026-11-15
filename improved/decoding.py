"""Similarity decoding with test-time augmentation.

HPSCAN is trained with a pairwise loss (MCL): for two points i, j the network is asked to predict
whether they belong to the same cluster, P_ij = p_i . p_j, where p are the softmax outputs over the
C = 20 cluster channels. The paper reads out clusters per point (argmax over the channels), which can
spread one perceived cluster over several channels. Here the clusters are read out from P directly:

1. P, the NOISE probability and the agreement are averaged over the 8 rotations / mirror images of
   the [-1, 1] box (test-time augmentation; P does not depend on which channel a cluster uses).
2. A point is NOISE if its NOISE probability exceeds `noise_threshold`.
3. The remaining points are grouped by average-linkage clustering of the distance 1 - P, cut at
   `tau` (or into exactly `n_clusters` clusters, the prior given to the starred baselines).
4. Optionally, clusters with fewer than `min_fraction` of all points are merged into the large
   cluster they are most similar to.
"""
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from hpscan.data import relabel, sort_points

# rotations by k * 90 degrees, without and with mirroring: all map the [-1, 1] box onto itself
DIHEDRAL = [(k, flip) for flip in (False, True) for k in range(4)]


def dihedral(points, k, flip):
    p = np.array(points, dtype=np.float32)
    if flip:
        p[:, 0] = -p[:, 0]
    for _ in range(k):
        p = np.stack([-p[:, 1], p[:, 0]], axis=1)
    return p


@torch.no_grad()
def comembership(model, points, n_views=8, device="cpu"):
    """Average model outputs over `n_views` dihedral views of a scatterplot.

    model: HPSCAN module (`hpscan.module.HPSCANModule`) in eval mode, points: (N, 2) in [-1, 1].
    Returns the NOISE probability (N,), the co-membership matrix P (N, N) and the predicted agreement
    (N,), all in the order of `points`.
    """
    views, orders = [], []
    for k, flip in DIHEDRAL[:n_views]:
        view = dihedral(points, k, flip)
        order = sort_points(view)  # the model expects sorted points (deterministic sampling)
        views.append(view[order])
        orders.append(order)
    x = torch.from_numpy(np.stack(views)).to(device)
    x = torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)
    noise, cluster, agree = model.infer({"point_clouds": x})
    inverse = torch.from_numpy(np.stack([np.argsort(o) for o in orders])).to(device)
    prob = cluster.float().softmax(dim=1)
    prob = prob.gather(2, inverse.unsqueeze(1).expand(-1, prob.shape[1], -1))
    P = torch.bmm(prob.transpose(1, 2), prob).mean(dim=0)
    p_noise = noise.float().softmax(dim=1)[:, 0].gather(1, inverse).mean(dim=0)
    agree = agree.float().sigmoid().gather(1, inverse).mean(dim=0)
    return p_noise.cpu().numpy(), P.cpu().numpy(), agree.cpu().numpy()


def affinity_labels(p_noise, P, noise_threshold=0.7, tau=0.9, min_fraction=0.0, n_clusters=None):
    """Cluster labels (N,) with 0 = NOISE and 1..K = clusters, see the module docstring."""
    N = len(p_noise)
    labels = np.zeros(N, dtype=np.int64)
    idx = np.where(p_noise <= noise_threshold)[0]
    if len(idx) == 1:
        labels[idx] = 1
    elif len(idx) > 1:
        D = 1.0 - P[np.ix_(idx, idx)].astype(np.float64)
        D = np.clip((D + D.T) / 2, 0, None)
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(D, checks=False), method="average")
        if n_clusters is not None:
            labels[idx] = fcluster(Z, t=max(1, int(n_clusters)), criterion="maxclust")
        else:
            labels[idx] = fcluster(Z, t=tau, criterion="distance")
    if min_fraction > 0 and len(idx):
        ids, sizes = np.unique(labels[idx], return_counts=True)
        large = ids[sizes >= min_fraction * N]
        if len(large):
            for small in ids[sizes < min_fraction * N]:
                members = np.where(labels == small)[0]
                similarity = [P[np.ix_(members, np.where(labels == c)[0])].mean() for c in large]
                labels[members] = large[int(np.argmax(similarity))]
    return relabel(labels)


class SimilarityDecoder(torch.nn.Module):
    """Wraps an HPSCAN module so that `forward(batch)` returns cluster ids decoded with the method
    above, with the interface of `HPSCANModule.forward` (cluster, noise mask, probabilities, agreement).
    Expects a batch with a single scatterplot (as `hpscan.evaluation.Evaluator` uses it).
    """

    kind = "hpscan"

    def __init__(self, model, n_views=8, noise_threshold=0.7, tau=0.9, min_fraction=0.0, gt_count=False):
        super().__init__()
        self.model = model
        self.n_views = n_views
        self.noise_threshold = noise_threshold
        self.tau = tau
        self.min_fraction = min_fraction
        self.gt_count = gt_count

    def labels(self, points, n_clusters=None):
        """points (N, 2) in [-1, 1] -> labels (N,) with 0 = NOISE, agreement (N,)."""
        device = next(self.model.parameters()).device
        p_noise, P, agree = comembership(self.model, points, self.n_views, device)
        labels = affinity_labels(p_noise, P, self.noise_threshold, self.tau, self.min_fraction, n_clusters)
        return labels, agree

    def forward(self, batch):
        xyz = batch["point_clouds"]
        assert xyz.shape[0] == 1, "SimilarityDecoder expects one scatterplot per batch"
        n_clusters = int(batch["n_cluster"].reshape(-1)[0]) if self.gt_count else None
        labels, agree = self.labels(xyz[0, :, :2].float().cpu().numpy(), n_clusters)
        cluster = torch.from_numpy(labels).unsqueeze(0).to(xyz.device)
        return cluster, (cluster != 0).int(), None, torch.from_numpy(agree).unsqueeze(0).to(xyz.device)
