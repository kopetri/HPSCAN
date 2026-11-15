"""Evaluate an HPSCAN checkpoint with similarity decoding (improved/decoding.py).

Uses the evaluation protocol and metrics of the paper (hpscan/evaluation.py, hpscan/metrics.py).

    # PSC / SDR (kappa_alpha, kappa_v, kappa_n as in Table 3)
    python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --data data/PSC --max_cluster 6
    python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --data data/SDR --max_cluster 3
    # same prior as the starred baselines: ground-truth number of clusters, no NOISE
    python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --data data/SDR --max_cluster 3 \\
        --gt_count --noise_threshold 1
    # ClustMe cluster counting (Table 5 protocol)
    python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --clustme data/ClustMe
    # validation split of a run trained with --holdout (development protocol)
    python improved/evaluate.py --model runs/improved/epoch0300.ckpt --data data/PSC --split valid runs/improved/holdout.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from hpscan.data import (ClustMeDataset, ScatterplotDataset, cluster_counts_of_raters,  # noqa: E402
                         load_scatterplot, perceived_cluster_count)
from hpscan.evaluation import Evaluator, _batch, format_result, summarize  # noqa: E402
from hpscan.module import load_model  # noqa: E402
from improved.decoding import SimilarityDecoder  # noqa: E402

STANDARD_SPLITS = ("train", "valid", "test")


class PaperMetrics(Evaluator):
    """The metrics of the paper without the cluster validity indices (silhouette / Calinski-Harabasz are not
    reported in the tables and are undefined when every point is its own cluster)."""

    def __init__(self, device):
        super().__init__(device=device)
        self.silhouette = self.calinski = lambda xyz, cluster: float("nan")


def split_files(data, split):
    """Files of a standard split or of a text file listing file names (one per line)."""
    data = Path(data)
    if split in STANDARD_SPLITS:
        return ScatterplotDataset(data, split=split, verbose=False).files
    names = [line.strip() for line in open(split) if line.strip()]
    files = [data / n if (data / n).exists() else data / "data" / n for n in names]
    return [f for f in files if f.exists()]


def evaluate(decoder, data, splits=("test",), max_cluster=6, device="cuda"):
    dataset = ScatterplotDataset(data, split="test" if (Path(data) / "test.txt").exists() else "valid", verbose=False)
    dataset.files = [f for s in splits for f in split_files(data, s)]
    dataset.split = "test"  # deterministic rater selection (first 5 raters)
    evaluator = PaperMetrics(device)
    decoder = decoder.to(device).eval()
    groups = defaultdict(lambda: defaultdict(list))
    for i in tqdm(range(len(dataset)), desc=Path(str(data)).name, leave=False):
        group = int(perceived_cluster_count(cluster_counts_of_raters(load_scatterplot(dataset.files[i])[:, 2:]), 3))
        if group > max_cluster:
            continue
        batch = _batch(dataset[i], device)
        cluster, noise, agree = evaluator.predict(decoder, batch)
        scores = evaluator.score(batch["point_clouds"], cluster, noise, batch["gt_raters"], agree, batch["agreement_score"])
        for key, value in scores.items():
            groups[group][key].append(value)
        groups[group]["plot_id"].append(batch["plot_id"])
    return summarize(groups, max_cluster)


def evaluate_clustme(decoder, data, device="cuda"):
    from clustme import count_clusters, scores
    dataset = ClustMeDataset(data, split="test")
    decoder = decoder.to(device).eval()
    counts, answers = [], []
    for i in tqdm(range(len(dataset)), desc="ClustMe", leave=False):
        item = dataset[i]
        labels, _ = decoder.labels(item["point_clouds"][:, :2])
        counts.append(count_clusters(torch.from_numpy(labels).unsqueeze(0)))
        answers.append(item["answers"])
    kappa, accuracy, f1 = scores(torch.cat(counts), torch.stack(answers))
    return {"kappa_v": kappa, "accuracy": accuracy, "f1": f1, "n": len(dataset)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HPSCAN checkpoint")
    parser.add_argument("--data", default=None, help="dataset folder (PSC, SDR, ...)")
    parser.add_argument("--split", nargs="+", default=["test"], help="train/valid/test or text files with file names")
    parser.add_argument("--max_cluster", type=int, default=6, help="6 for PSC, 3 for SDR")
    parser.add_argument("--clustme", default=None, help="converted ClustMe folder (tools/convert_clustme.py)")
    parser.add_argument("--n_views", type=int, default=8, help="test-time augmentation views (1-8)")
    parser.add_argument("--noise_threshold", type=float, default=0.7, help="1 = never predict NOISE")
    parser.add_argument("--tau", type=float, default=0.9, help="cut of the average-linkage tree (distance 1 - P)")
    parser.add_argument("--min_fraction", type=float, default=0.0, help="merge clusters smaller than this fraction")
    parser.add_argument("--gt_count", action="store_true", help="cut into the ground-truth number of clusters")
    parser.add_argument("--output", default=None, help="append the result to this JSON file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not (args.data or args.clustme):
        parser.error("give --data and/or --clustme")

    model = load_model(args.model)
    if model.kind != "hpscan":
        raise SystemExit("expects a point-based HPSCAN checkpoint")
    decoder = SimilarityDecoder(model, args.n_views, args.noise_threshold, args.tau, args.min_fraction, args.gt_count)
    config = (f"n_views={args.n_views} noise_threshold={args.noise_threshold} tau={args.tau} "
              f"min_fraction={args.min_fraction}" + (" gt_count" if args.gt_count else ""))
    results = {}
    if args.data:
        np.random.seed(0)
        result = evaluate(decoder, args.data, args.split, args.max_cluster, args.device)
        print(format_result(f"{args.model} on {Path(args.data).name} ({' + '.join(args.split)}), {config}", result))
        results[Path(args.data).name] = result
    if args.clustme:
        if args.gt_count:
            parser.error("--gt_count is not defined for ClustMe (the count is the target)")
        r = evaluate_clustme(decoder, args.clustme, args.device)
        print(f"{args.model} on ClustMe ({r['n']} stimuli), {config}:\n"
              f"  kappa_v {r['kappa_v']:.2f} | accuracy {r['accuracy']:.0%} | F1 {r['f1']:.0%}")
        results["ClustMe"] = r
    if args.output:
        path = Path(args.output)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[f"{args.model} | {config}"] = results
        path.write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
