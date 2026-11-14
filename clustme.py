"""Cluster counting on ClustMe (Table 5).

Each ClustMe stimulus was judged by 34 raters as showing one (1) or more than one (2) cluster.
HPSCAN's clustering is converted to a count by counting the clusters that contain at least 5%
of the points, clipped to [1, 2]. We report the Vanbelle kappa index of the prediction vs. the
34 raters, and accuracy / F1 score vs. the majority vote.

Example:
    python clustme.py --model checkpoints/hpscan_clustme.ckpt --data data/ClustMe --amp
    python clustme.py --dummy random single multi --data data/ClustMe
"""
import argparse
import contextlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from hpscan.data import ClustMeDataset
from hpscan.metrics import vanbelle_kappa_binary
from hpscan.module import load_model


def count_clusters(cluster, min_fraction=0.05):
    """cluster: (B, N) ids with 0 = noise -> (B,) number of clusters with >= min_fraction of the points, in [1, 2]."""
    counts = []
    for labels in cluster.cpu().numpy():
        ids, sizes = np.unique(labels, return_counts=True)
        n = sum(1 for i, s in zip(ids, sizes) if i != 0 and s / len(labels) >= min_fraction)
        counts.append(int(np.clip(n, 1, 2)))
    return torch.tensor(counts)


def scores(pred, answers):
    """pred: (S,) in {1, 2}, answers: (S, 34) in {1, 2} -> kappa_v, accuracy, F1."""
    kappa = vanbelle_kappa_binary(answers.unsqueeze(0) - 1, pred.view(1, -1) - 1).item()
    majority = torch.round(answers.float().mean(dim=1)).long() - 1  # 0: single, 1: multiple
    p = pred.long() - 1
    accuracy = (p == majority).float().mean().item()
    tp = ((p == 1) & (majority == 1)).sum().item()
    fp = ((p == 1) & (majority == 0)).sum().item()
    fn = ((p == 0) & (majority == 1)).sum().item()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return kappa, accuracy, f1


@torch.no_grad()
def predict_counts(model, dataset, device, amp=False, batch_size=128):
    model = model.to(device).eval()
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
    preds, answers = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        with autocast:
            cluster, _, _, _ = model({"point_clouds": batch["point_clouds"].to(device)})
        preds.append(count_clusters(cluster))
        answers.append(batch["answers"])
    return torch.cat(preds), torch.cat(answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="HPSCAN checkpoint")
    parser.add_argument("--dummy", nargs="*", default=[], choices=["random", "single", "multi"])
    parser.add_argument("--data", required=True, help="converted ClustMe folder (tools/convert_clustme.py)")
    parser.add_argument("--amp", action="store_true", help="float16 autocast inference")
    parser.add_argument("--repetitions", type=int, default=100, help="repetitions of the random dummy model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = ClustMeDataset(args.data, split="test")
    print(f"{len(dataset)} ClustMe stimuli")
    answers = torch.stack([dataset[i]["answers"] for i in range(len(dataset))])
    print(f"{'model':<10} {'kappa_v':>14} {'accuracy':>14} {'F1':>14}")

    if args.model:
        pred, answers = predict_counts(load_model(args.model), dataset, args.device, args.amp)
        k, a, f = scores(pred, answers)
        print(f"{'HPSCAN':<10} {k:14.2f} {a:14.0%} {f:14.0%}")

    generator = torch.Generator().manual_seed(args.seed)
    for dummy in args.dummy:
        if dummy == "random":
            runs = np.array([scores(torch.randint(1, 3, (len(answers),), generator=generator), answers)
                             for _ in range(args.repetitions)])
            m, s = runs.mean(axis=0), runs.std(axis=0)
            print(f"{'random':<10} {round(m[0], 2) + 0.0:8.2f}+-{s[0]:.2f} {m[1]:8.0%}+-{s[1]:.0%} {m[2]:8.0%}+-{s[2]:.0%}")
        else:
            pred = torch.full((len(answers),), 1 if dummy == "single" else 2)
            k, a, f = scores(pred, answers)
            k = round(k, 2) + 0.0
            print(f"{dummy:<10} {k:14.2f} {a:14.0%} {f:14.0%}")


if __name__ == "__main__":
    main()
