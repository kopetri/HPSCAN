"""Training data, augmentation and loss of the improved recipe (improved/train.py).

Same model (hpscan.model.HPSCANNet) and loss terms as the paper (L_noise + 0.1 L_mcl + L_agree, one randomly
chosen rater as target per step). Differences: AdamW + cosine schedule, the MCL term is computed from
P = p^T p (identical value, much less memory than enumerating all point pairs), and more augmentation.
"""
import numpy as np
import torch
import torch.nn.functional as F

from hpscan.data import load_scatterplot, relabel, sort_points


def point_agreement(gt):
    """Per-point agreement A (Eq. 6 of the paper) from contingency counts. gt: (N, R) labels -> (N,).
    Equals hpscan.metrics.point_agreement_scores, in O(N R^2) instead of O(N^2 R^2)."""
    N, R = gt.shape
    size = [np.bincount(gt[:, r])[gt[:, r]] for r in range(R)]
    total, pairs = np.zeros(N), 0
    for r in range(R):
        for q in range(r + 1, R):
            k = int(gt[:, q].max()) + 1
            joint = gt[:, r].astype(np.int64) * k + gt[:, q]
            both = np.bincount(joint)[joint]
            total += (N - size[r] - size[q] + 2 * both - 1) / (N - 1)
            pairs += 1
    return (total / pairs).astype(np.float32)


def cut(points, labels, dim, t):
    """Crop transformation of the paper (margin 1) applied to all raters: points right of t along `dim`
    are moved to the left end and vice versa; every cut cluster is split in two. labels (N, R), 0 = NOISE."""
    right = points[:, dim] > t
    points = points.copy()
    points[right, dim] += -1 - t
    points[~right, dim] += 1 - t
    labels = labels.copy()
    noise = labels == 0
    labels[right] += 100
    for r in range(labels.shape[1]):
        labels[:, r] = relabel(labels[:, r])
    labels[noise] = 0
    return points, labels


class AugmentedScatterplots(torch.utils.data.Dataset):
    """Training stimuli with augmentation. Returns points (N, 3), rater labels (N, 5) with 0 = NOISE and the
    per-point agreement (N,), sorted like at inference."""

    def __init__(self, files, n_raters=5, p_cut=0.5, p_rotate=0.5, p_jitter=0.5, jitter=0.05):
        self.n_raters = n_raters
        self.p_cut, self.p_rotate, self.p_jitter, self.jitter = p_cut, p_rotate, p_jitter, jitter
        self.items = []
        for f in files:
            d = load_scatterplot(f)
            gt = (d[:, 2:] + 1).astype(np.int64)
            # agreement of all raters of the stimulus (recomputed from the selected raters after a cut)
            self.items.append((d[:, 0:2].astype(np.float32), gt, point_agreement(gt)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        points, gt, agree = self.items[i]
        R = gt.shape[1]
        if R > self.n_raters:
            gt = gt[:, np.random.choice(R, self.n_raters, replace=False)]
        elif R < self.n_raters:  # repeat the last rater
            gt = np.concatenate([gt] + [gt[:, -1:]] * (self.n_raters - R), axis=1)
        points = points.copy()
        was_cut = np.random.random() < self.p_cut
        if was_cut:
            t = float(np.clip(np.random.normal(0, 0.2), -0.9, 0.9))
            points, gt = cut(points, gt, np.random.randint(2), t)
        if np.random.random() < self.p_rotate:
            a = (np.random.random() * 2 - 1) * np.pi
            c, s = np.cos(a), np.sin(a)
            points = points @ np.array([[c, s], [-s, c]], dtype=np.float32)
        if np.random.random() < 0.5:
            points[:, 0] *= -1
        if np.random.random() < 0.5:
            points[:, 1] *= -1
        if np.random.random() < 0.5:
            points = points[:, ::-1].copy()
        if self.jitter > 0 and np.random.random() < self.p_jitter:
            points = points + np.random.normal(0, np.random.uniform(0, self.jitter), points.shape).astype(np.float32)
        lo, hi = points.min(0), points.max(0)  # min-max normalisation to [-1, 1], as the data
        points = (points - lo) / np.maximum(hi - lo, 1e-6) * 2 - 1
        order = sort_points(points)
        points, gt, agree = points[order], gt[order], agree[order]
        if was_cut:
            agree = point_agreement(gt)
        xyz = np.concatenate([points, np.zeros((len(points), 1), np.float32)], 1).astype(np.float32)
        return torch.from_numpy(xyz), torch.from_numpy(gt), torch.from_numpy(np.ascontiguousarray(agree, np.float32))


class SubsampleCollate:
    """With probability p, keep a random subset of n in [min_points, N] points of every stimulus of the batch
    (lower point density, the human labels of the remaining points are kept)."""

    def __init__(self, min_points=256, p=0.5):
        self.min_points, self.p = min_points, p

    def __call__(self, items):
        xyz, gt, agree = [torch.stack(t) for t in zip(*items)]
        N = xyz.shape[1]
        if self.min_points < N and np.random.random() < self.p:
            n = np.random.randint(self.min_points, N + 1)
            idx = torch.from_numpy(np.stack([np.sort(np.random.choice(N, n, replace=False)) for _ in range(len(xyz))])).long()
            xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, xyz.shape[2]))
            gt = torch.gather(gt, 1, idx.unsqueeze(-1).expand(-1, -1, gt.shape[2]))
            agree = torch.gather(agree, 1, idx)
        return xyz, gt, agree


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


def mcl_target(labels, dissimilar_weight):
    """Pairwise target (B, N, N) (True = same cluster) and the omega_D weights of the paper: 1 / w_c^2 for
    pairs in the same cluster (w_c = relative size of the cluster), D for pairs in different clusters."""
    B, N = labels.shape
    same = labels.unsqueeze(2) == labels.unsqueeze(1)
    onehot = F.one_hot(labels, num_classes=int(labels.max()) + 1).float()
    fraction = onehot.sum(1) / N
    w_point = (onehot * (1.0 / fraction.clamp(min=1e-6) ** 2).unsqueeze(1)).sum(-1)
    weights = torch.where(same, w_point.unsqueeze(2).expand(B, N, N),
                          torch.full((B, N, N), float(dissimilar_weight), device=labels.device))
    return same, weights


def hpscan_loss(noise_logits, cluster_logits, agree_logits, gt, agree_gt, dissimilar_weight=50.0,
                loss_weights=(1.0, 0.1, 1.0), noise_weights=(0.9, 0.1)):
    """L = L_noise + 0.1 L_mcl + L_agree with one randomly chosen rater (same for the batch) as target."""
    rater = int(np.random.randint(gt.shape[-1]))
    labels = gt[..., rater]
    prob = cluster_logits.float().softmax(dim=1)
    P = torch.bmm(prob.transpose(1, 2), prob).clamp(1e-7, 1 - 1e-7)
    same, weights = mcl_target(labels, dissimilar_weight)
    mcl = (-torch.where(same, P.log(), (1 - P).log()) * weights).mean()
    noise = F.cross_entropy(noise_logits.float(), (labels > 0).long(),
                            weight=torch.tensor(noise_weights, device=gt.device), reduction="none").mean()
    agree = F.l1_loss(agree_logits.float().sigmoid(), agree_gt)
    w = loss_weights
    return w[0] * noise + w[1] * mcl + w[2] * agree, (noise.item(), mcl.item(), agree.item())
