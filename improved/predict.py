"""Cluster scatterplots with similarity decoding (post-publication, see improved/README.md).

Same input and output files as predict.py (`<name>.hpscan.csv` with x, y, cluster (0 = NOISE), agreement).

    python improved/predict.py --model checkpoints/hpscan_improved.ckpt --input my_scatterplot.csv --plot
    # the paper model with the decoding parameters selected for it
    python improved/predict.py --model checkpoints/hpscan_psc.ckpt --tau 0.8 --min_fraction 0.05 --input ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root (before improved/)

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hpscan.module import load_model  # noqa: E402
from improved.decoding import SimilarityDecoder  # noqa: E402
from predict import normalize, read_points  # noqa: E402  (predict.py of the repository root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HPSCAN checkpoint")
    parser.add_argument("--input", nargs="+", required=True, help="scatterplot file(s)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_normalize", action="store_true", help="points are already in [-1, 1]")
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--noise_threshold", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--min_fraction", type=float, default=0.0)
    parser.add_argument("--n_clusters", type=int, default=None, help="force this number of clusters")
    parser.add_argument("--plot", action="store_true", help="also save a PNG of the clustering")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.model)
    if model.kind != "hpscan":
        raise SystemExit("expects a point-based HPSCAN checkpoint")
    decoder = SimilarityDecoder(model, args.n_views, args.noise_threshold, args.tau, args.min_fraction).to(args.device).eval()

    for file in args.input:
        file = Path(file)
        points = read_points(file)
        if not args.no_normalize:
            points = normalize(points)
        labels, agree = decoder.labels(points.astype(np.float32), args.n_clusters)
        out_dir = Path(args.output_dir) if args.output_dir else file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{file.stem}.hpscan.csv"
        np.savetxt(out, np.column_stack([points, labels, agree]), delimiter=",",
                   header="x,y,cluster,agreement", comments="", fmt=["%.6f", "%.6f", "%d", "%.4f"])
        n_cluster = len(set(labels.tolist()) - {0})
        print(f"{file.name}: {n_cluster} cluster(s), {np.sum(labels == 0)} noise point(s) -> {out}")
        if args.plot:
            import cv2
            from hpscan.render import draw_scatterplot
            image = draw_scatterplot(points, labels, point_size=20)
            cv2.imwrite(str(out_dir / f"{file.stem}.hpscan.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
