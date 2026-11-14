"""Evaluate HPSCAN, the image baseline or a scikit-learn clustering baseline on a test split.

Examples:
    python evaluate.py --model checkpoints/hpscan.ckpt --data data/PSC
    python evaluate.py --model checkpoints/hpscan_sdr.ckpt --data data/SDR --max_cluster 3 --amp
    python evaluate.py --model dbscan_0.2_10 ward --data data/PSC      # `ward` = ground-truth cluster count

The scores are grouped by the perceived number of clusters and combined with the
1/n weighting described in hpscan/evaluation.py (the numbers reported in the paper).
"""
import argparse
import warnings

import numpy as np
import torch

from hpscan.baselines import is_baseline
from hpscan.evaluation import Evaluator, format_result, save_result
from hpscan.module import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", required=True, help="checkpoint file(s) and/or baseline name(s)")
    parser.add_argument("--data", required=True, help="dataset folder with test.txt")
    parser.add_argument("--max_cluster", type=int, default=6, help="largest perceived cluster count (PSC: 6, SDR: 3)")
    parser.add_argument("--n_raters", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="run inference with float16 autocast (mixed precision)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="append results to this JSON file")
    parser.add_argument("--seed", type=int, default=0, help="seed of the randomised baselines (k-means, GMM, ...)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    evaluator = Evaluator(device=args.device, amp=args.amp)
    for name in args.model:
        np.random.seed(args.seed)
        model = name if is_baseline(name) else load_model(name)
        render = not isinstance(model, str) and model.kind == "image"
        result = evaluator.run(model, args.data, max_cluster=args.max_cluster, n_raters=args.n_raters,
                               render_images=render)
        print(format_result(name, result))
        if args.output:
            save_result(args.output, name, result)


if __name__ == "__main__":
    main()
