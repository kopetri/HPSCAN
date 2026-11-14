"""scikit-learn clustering baselines of Table 3.

A baseline is specified by a string `<method>_<param1>_<param2>...`, e.g. `dbscan_0.2_10`.
The parameters found by `search_baseline_parameters.py` on the PSC training split are
listed in `BEST_PARAMETERS`. For the methods marked with * in Table 3 the ground-truth
number of clusters is given instead; pass the method name without parameters (e.g. `ward`).
"""
import numpy as np
from sklearn import cluster, mixture

BEST_PARAMETERS = {
    "dbscan": "dbscan_0.2_10",            # eps, min_samples
    "optics": "optics_0.1_0.05_0.1",      # min_samples, xi, min_cluster_size
    "meanshift": "meanshift_0.4",         # bandwidth
    "affinity-propagation": "affinity-propagation_0.9",  # damping
    "average-linkage": "average-linkage_2",  # n_clusters
    "birch": "birch_2",                   # n_clusters
    "ward": "ward_1",                     # n_clusters
    "spectral": "spectral_1",             # n_clusters
    "kmeans": "kmeans_1",                 # n_clusters
    "gmm": "gmm_1",                       # n_components
}

DISPLAY_NAMES = {
    "dbscan": "DBSCAN", "optics": "OPTICS", "ward": "Ward", "meanshift": "Mean Shift",
    "affinity-propagation": "Affinity Propagation", "spectral": "Spectral Clustering",
    "average-linkage": "Agglomerative Clustering", "birch": "BIRCH", "kmeans": "K-means",
    "gmm": "Gaussian Mixture Model",
}

# methods that need the number of clusters
NEEDS_CLUSTER_COUNT = ["ward", "spectral", "average-linkage", "birch", "kmeans", "gmm"]


def is_baseline(name):
    return name.split("_")[0] in DISPLAY_NAMES


class Clustering:
    """Wrap a scikit-learn clustering algorithm. `fit` returns labels with -1 = noise."""

    def __init__(self, spec):
        method, *p = spec.split("_")
        if method == "dbscan":
            self.algorithm = cluster.DBSCAN(eps=float(p[0]), min_samples=int(p[1]))
        elif method == "optics":
            self.algorithm = cluster.OPTICS(min_samples=float(p[0]), xi=float(p[1]), min_cluster_size=float(p[2]))
        elif method == "affinity-propagation":
            self.algorithm = cluster.AffinityPropagation(damping=float(p[0]))
        elif method == "meanshift":
            self.algorithm = cluster.MeanShift(bandwidth=float(p[0]), bin_seeding=True)
        elif method == "ward":
            self.algorithm = cluster.AgglomerativeClustering(n_clusters=int(p[0]), linkage="ward")
        elif method == "spectral":
            self.algorithm = cluster.SpectralClustering(n_clusters=int(p[0]), eigen_solver="arpack", affinity="nearest_neighbors")
        elif method == "average-linkage":
            self.algorithm = cluster.AgglomerativeClustering(linkage="average", metric="cityblock", n_clusters=int(p[0]))
        elif method == "birch":
            self.algorithm = cluster.Birch(n_clusters=int(p[0]))
        elif method == "gmm":
            self.algorithm = mixture.GaussianMixture(n_components=int(p[0]), covariance_type="full")
        elif method == "kmeans":
            self.algorithm = cluster.MiniBatchKMeans(n_clusters=int(p[0]), n_init="auto")
        else:
            raise ValueError(spec)

    def fit(self, points):
        self.algorithm.fit(points)
        if hasattr(self.algorithm, "labels_"):
            labels = self.algorithm.labels_
        else:
            labels = self.algorithm.predict(points)
        return np.asarray(labels).astype(int)


def run_baseline(spec, points, n_cluster=None):
    """Cluster `points` (N, D). If `spec` has no parameters, `n_cluster` (ground truth) is used.

    Returns labels with 0 = noise and 1..K = clusters.
    """
    if "_" not in spec:
        spec = f"{spec}_{max(1, int(n_cluster))}"
    return Clustering(spec).fit(points) + 1
