"""Parameter search for the scikit-learn baselines of Table 3 (on the PSC training split).

For every training stimulus, each parameter setting is scored with kappa_alpha against the
group of raters; the setting that is best for the largest number of stimuli is selected.
The selected settings are listed in hpscan/baselines.py (BEST_PARAMETERS).

Example:
    python search_baseline_parameters.py --data data/PSC --method dbscan optics
"""
import argparse
import itertools
import json
import warnings
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from hpscan.baselines import Clustering
from hpscan.data import ScatterplotDataset
from hpscan.metrics import CrossValidDifference


def _fmt(values):
    return [f"{np.round(v, 4):g}" for v in values]


# search spaces: lists of parameter tuples in the order of the `Clustering` spec
SEARCH_SPACE = {
    "dbscan": [(eps, min_samples) for eps in _fmt(np.arange(0.1, 1.0, 0.1)) for min_samples in _fmt(np.arange(10, 100, 10))],
    "optics": list(itertools.product(_fmt(np.arange(0.1, 1, 0.3)), _fmt(np.arange(0.05, 1.0, 0.2)), _fmt(np.arange(0.1, 1.0, 0.3)))),
    "affinity-propagation": [(d,) for d in _fmt(np.arange(0.5, 1, 0.1))],
    "meanshift": [(b,) for b in _fmt(np.arange(0.1, 1, 0.1))],
    "ward": [(str(n),) for n in range(1, 13)],
    "spectral": [(str(n),) for n in range(1, 13)],
    "average-linkage": [(str(n),) for n in range(1, 13)],
    "birch": [(str(n),) for n in range(2, 13)],
    "kmeans": [(str(n),) for n in range(1, 13)],
    "gmm": [(str(n),) for n in range(1, 13)],
}


def search(method, dataset, device, limit=None, batch_size=10):
    metric = CrossValidDifference(reduction=False)
    parameters = SEARCH_SPACE[method]
    wins = Counter()
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in tqdm(range(n), desc=method):
        sample = dataset[i]
        points = sample['point_clouds']
        group = torch.from_numpy(sample['gt_raters']).unsqueeze(0).to(device)
        preds = torch.stack([torch.from_numpy(Clustering("_".join([method, *p])).fit(points).reshape(-1))
                             for p in parameters]).to(device)
        scores = torch.cat([metric(preds[j:j + batch_size], group.repeat(len(preds[j:j + batch_size]), 1, 1))
                            for j in range(0, len(preds), batch_size)])
        wins[torch.argmin(scores).item()] += 1  # lowest difference = best agreement with the raters
    best = parameters[wins.most_common(1)[0][0]]
    return "_".join([method, *best]), {"_".join(parameters[k]): v for k, v in wins.most_common()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", nargs="+", default=list(SEARCH_SPACE), choices=list(SEARCH_SPACE))
    parser.add_argument("--limit", type=int, default=None, help="use only the first N training stimuli")
    parser.add_argument("--output", default="baseline_parameters.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    np.random.seed(args.seed)

    dataset = ScatterplotDataset(args.data, split="train")
    results = {}
    for method in args.method:
        best, wins = search(method, dataset, args.device, args.limit)
        results[method] = {"best": best, "wins": wins}
        print(f"{method}: {best}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
