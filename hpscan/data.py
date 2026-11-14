"""Datasets.

A scatterplot is stored as a `.npy` array of shape (N, 2 + R): the x/y position of N points
in [-1, 1] followed by the cluster id of each of the R raters (-1 = NOISE). A dataset folder
contains the `.npy` files and `train.txt`, `valid.txt` and `test.txt` listing the file names
of each split.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .metrics import group_agreement, point_agreement_scores

# --------------------------------------------------------------------------- loading
def load_scatterplot(path):
    """Load an annotated scatterplot: (N, 2 + R) array of x, y and R rater labels (-1 = NOISE)."""
    return np.load(path)


def sort_points(points, pos=(-1, 1)):
    """Order points by distance to the top-left corner; makes inference deterministic."""
    return np.argsort(np.sqrt(np.sum(np.power(points - np.array(pos), 2), axis=1)))


def relabel(labels):
    """Map cluster ids to 1..K while keeping 0 (NOISE)."""
    labels = np.array(labels).astype(int)
    unique = np.unique(labels).tolist()
    if 0 not in unique:
        unique = [0] + unique
    index_map = np.array([unique.index(n) if n in unique else -1 for n in np.arange(np.max(unique) + 1)])
    return index_map[labels]


def perceived_cluster_count(ratings, n=3):
    """Cluster count of a stimulus used to group the test set (Tables 2/3): the count on
    which the first `n` sorted ratings agree, otherwise the rounded mean count."""
    counts = np.sort(ratings)
    if np.all(counts[0] == counts[0:n]):
        return counts[0]
    return np.round(np.mean(ratings))


def cluster_counts_of_raters(ratings):
    return [len(np.unique(ratings[:, i][ratings[:, i] >= 0])) for i in range(ratings.shape[1])]


def raters_with_agreement(ratings, agreement, i=5):
    """Largest subset of raters (>= 2) whose group agreement is >= `agreement` (or None)."""
    from itertools import combinations
    if group_agreement(ratings) >= agreement:
        return np.arange(ratings.shape[1])
    n = i - 1
    if n < 2:
        return None
    scores = sorted([(group_agreement(ratings[:, comb]).item(), comb)
                     for comb in combinations(np.arange(ratings.shape[1]), n)], key=lambda x: x[0], reverse=True)
    best_score, best_comb = scores[0]
    if best_score >= agreement:
        return best_comb
    return raters_with_agreement(ratings, agreement, n)


# --------------------------------------------------------------------------- augmentation
def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation(point_cloud):
    rot_angle = (np.random.random() * 2.0 - 1.0) * np.pi  # [-180, 180] degrees
    point_cloud = np.array(point_cloud)
    point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rotz(rot_angle)))
    return point_cloud


def random_cut(point_cloud, labels, dimension, t=None, margin=1.0, rescale=False):
    """Crop transformation (Section 3.4 / supplement Sec. 6).

    Cut the point set at a random position `t` along `dimension`; points on one side of the
    cutting line are moved to the other side and every cluster that gets cut is split into
    two clusters. margin=1.0, rescale=False is the variant used to train HPSCAN. The image
    baseline was trained with margin=1.2, rescale=True (gap between the two halves and
    min-max rescaling to [-1, 1] afterwards).
    """
    if t is None:
        t = np.random.normal(0, 0.2)
    labels = np.array(labels)
    t = np.clip(-1, 1, t)  # kept as in the original code: equals min(t, 1)
    offset_right = -1 - t
    offset_left = 1 - t
    point_cloud = np.array(point_cloud)
    right = np.where(point_cloud[:, dimension] > t)[0]
    left = np.where(point_cloud[:, dimension] <= t)[0]
    point_cloud[right, dimension] += offset_right * margin
    point_cloud[left, dimension] += offset_left * margin
    if rescale:
        from sklearn.preprocessing import MinMaxScaler
        point_cloud = MinMaxScaler((-1.0, 1.0)).fit_transform(point_cloud)
    noise = np.where(labels == 0)[0]
    if len(right) > 0:
        labels[right] += 100
    labels = relabel(labels)
    labels[noise] = 0
    return point_cloud[:, 0:3], labels


# --------------------------------------------------------------------------- datasets
class ScatterplotDataset(Dataset):
    """Annotated scatterplots (PSC, SDR, ...).

    Args:
        path: dataset folder (or list of folders) with the split files.
        split: 'train', 'valid' or 'test'.
        augment: random rotation and crop (training only).
        n_raters: number of raters per stimulus (5). Fewer raters are padded by repeating the
            last one, more raters are cut (randomly chosen during training).
        n_cluster: keep only stimuli with this perceived cluster count.
        filter_agreement: keep only stimuli whose group agreement is >= this value (T_agree).
        best_agreement: per stimulus, keep the largest group of raters with agreement >= this value.
        render_images: also return the rendered image and targets of the image baseline.
        crop_margin, crop_rescale: crop variant, see `random_cut`.
    """

    def __init__(self, path, split="test", augment=False, n_raters=5, label_idx=-1, n_cluster=None,
                 filter_agreement=-1.0, best_agreement=None, render_images=False, crop_margin=1.0,
                 crop_rescale=False, verbose=True):
        super().__init__()
        self.paths = [path] if isinstance(path, (str, Path)) else list(path)
        self.split = split
        self.augment = augment
        self.n_raters = n_raters
        self.label_idx = label_idx
        self.n_cluster = n_cluster
        self.filter_agreement = filter_agreement
        self.best_agreement = best_agreement
        self.render_images = render_images
        self.crop_margin = crop_margin
        self.crop_rescale = crop_rescale
        self.rater_subsets = None
        self.files = [f for p in self.paths for f in self._read_split(Path(p))]
        if filter_agreement >= 0 or n_cluster is not None or best_agreement is not None:
            self._filter()
        if verbose:
            print(f"Found {len(self.files)} scatterplots for split {split}"
                  + (f" with {n_cluster} cluster(s)" if n_cluster is not None else ""))

    def _read_split(self, path):
        split_file = path / f"{self.split}.txt"
        if not split_file.exists():
            raise ValueError(f"Missing split file {split_file}")
        with open(split_file, "r") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        # the files are next to the split file or in a `data` sub-folder (Hugging Face layout)
        files = [path / n if (path / n).exists() else path / "data" / n for n in names]
        return [f for f in files if f.exists()]

    def _filter(self):
        keep, subsets = [], []
        for file in self.files:
            ratings = load_scatterplot(file)[:, 2:]
            subset = None
            if self.best_agreement is not None:
                subset = raters_with_agreement(ratings, self.best_agreement)
                if subset is None:
                    continue
            if group_agreement(ratings).item() < self.filter_agreement:
                continue
            n = perceived_cluster_count(cluster_counts_of_raters(ratings), 3)
            if self.n_cluster is not None and n != self.n_cluster:
                continue
            keep.append(file)
            subsets.append(subset)
        self.files = keep
        if self.best_agreement is not None:
            self.rater_subsets = subsets

    def _select_raters(self, gt_raters, n_raters):
        n = gt_raters.shape[1]
        if n < n_raters:  # repeat the last rater
            for _ in range(n_raters - n):
                gt_raters = np.concatenate([gt_raters, gt_raters[:, -1].reshape((-1, 1))], axis=1)
            return gt_raters
        if n > n_raters:
            return gt_raters[:, np.random.choice(n, n_raters)] if self.split == "train" else gt_raters[:, 0:n_raters]
        return gt_raters

    def _label_idx(self, n_raters):
        idx = np.random.choice(n_raters, 1)[0] if (self.split == "train" and self.label_idx < 0) else self.label_idx
        return max(idx, 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = load_scatterplot(path)
        n_raters = self.n_raters if self.n_raters >= 1 else data.shape[1] - 2

        indices = sort_points(data[:, 0:2])
        point_cloud = np.concatenate([data[:, 0:2], np.zeros((data.shape[0], 1))], axis=1)  # (x, y, 0)
        gt_raters = data[:, 2:] + 1  # 0 = noise, 1..K = clusters
        if self.rater_subsets is not None:
            gt_raters = gt_raters[:, self.rater_subsets[idx]]
        gt_raters = self._select_raters(gt_raters, n_raters)
        label_idx = self._label_idx(n_raters)
        multi_label = relabel(gt_raters[:, label_idx])  # training target: one rater
        noise_label = np.clip(multi_label, 0, 1)

        point_cloud = point_cloud[indices]
        multi_label = multi_label[indices]
        noise_label = noise_label[indices]
        gt_raters = gt_raters[indices]

        sample = {}
        if self.render_images:
            from .render import project_points, render_input
            stimulus = render_input(data[indices, 0:2])
            pixel_coords = project_points(stimulus, point_cloud * 0.911)

        agreement_score = point_agreement_scores(torch.from_numpy(gt_raters).unsqueeze(0)).squeeze().numpy()

        if self.augment:
            # image baseline only: flips of the input image (as used to train the released
            # baseline; note that `pixel_coords` are not flipped along with the image)
            if self.render_images and np.random.random() > 0.5:
                stimulus = np.fliplr(stimulus)
            if self.render_images and np.random.random() > 0.5:
                stimulus = np.flipud(stimulus)
            if np.random.random() > 0.5:
                point_cloud = random_rotation(point_cloud)
            # The original training code drew two random numbers for horizontal/vertical point
            # flips but discarded the flipped point cloud, i.e. the released models were trained
            # without point flips. The draws are kept so the random sequence matches.
            np.random.random()
            np.random.random()
            if np.random.random() > 0.5:
                dimension = 1 if np.random.random() > 0.5 else 0
                point_cloud, multi_label = random_cut(point_cloud, multi_label, dimension,
                                                      margin=self.crop_margin, rescale=self.crop_rescale)

        sample['point_clouds'] = point_cloud.astype(np.float32)
        sample['multi_label'] = multi_label.astype(np.int64)
        sample['noise_label'] = noise_label.astype(np.int64)
        sample['gt_raters'] = gt_raters.astype(np.int64)
        sample['plot_id'] = str(path.name)
        sample['n_cluster'] = np.array(np.max(multi_label)).astype(np.int64)
        sample['agreement_score'] = agreement_score.astype(np.float32)
        if self.render_images:
            sample['stimulus'] = torch.from_numpy(stimulus.copy()).float().unsqueeze(0) / 255.0
            sample['pixel_coords'] = pixel_coords.astype(np.int32)
        return sample


class ClustMeDataset(Dataset):
    """ClustMe stimuli (Abbas et al. 2019) with the 34 binary judgments per stimulus
    (1 = single cluster, 2 = more than one cluster), converted with tools/convert_clustme.py.

    The points are min-max scaled to `scale` ([-0.95, 0.95], the setting of Table 5).
    """

    def __init__(self, path, split="test", scale=(-0.95, 0.95)):
        self.scale = scale
        path = Path(path)
        with open(path / f"{split}.txt", "r") as f:
            files = [path / line.strip() for line in f.readlines() if line.strip()]
        self.files = [f for f in files if f.exists()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = load_scatterplot(path)
        with open(path.with_suffix(".answer.csv"), "r") as f:
            answers = f.readlines()[1].split(";")[1:]
        answers = np.array([int(a.replace('"', "")) for a in answers], dtype=int)
        assert len(answers) == 34, len(answers)
        points = data[:, 0:2]
        if self.scale is not None:
            from sklearn.preprocessing import MinMaxScaler
            points = MinMaxScaler(tuple(self.scale)).fit_transform(points)
        point_cloud = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
        point_cloud = point_cloud[sort_points(points)]
        return {'point_clouds': point_cloud.astype(np.float32), 'answers': torch.from_numpy(answers)}
