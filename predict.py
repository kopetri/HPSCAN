"""Cluster scatterplots with a trained HPSCAN model.

Input: `.npy` (N, >=2) or `.csv` / `.txt` files; the first two columns are used as x and y.
The points are min-max scaled to [-1, 1] (as in the training data) unless --no_normalize is given.

Output per input file (written next to it, or into --output_dir):
    <name>.hpscan.csv   x, y, cluster (0 = NOISE, 1..K = clusters), agreement (predicted A in [0, 1])
    <name>.hpscan.png   scatterplot colored by the predicted clusters (with --plot)

Example:
    python predict.py --model checkpoints/hpscan_psc.ckpt --input my_scatterplot.csv --plot
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from hpscan.data import relabel, sort_points
from hpscan.module import load_model


def read_points(path):
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        delimiter = ";" if ";" in path.read_text().splitlines()[0] else ","
        data = np.genfromtxt(path, delimiter=delimiter)
        if np.isnan(data[0]).all():  # header
            data = data[1:]
    data = np.asarray(data, dtype=np.float64)[:, 0:2]
    return data[~np.isnan(data).any(axis=1)]


def normalize(points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (points - lo) / np.where(hi - lo > 0, hi - lo, 1.0) * 2.0 - 1.0


@torch.no_grad()
def predict(model, points, device="cpu"):
    """points: (N, 2) in [-1, 1] -> cluster ids (N,) with 0 = NOISE and agreement scores (N,)."""
    order = sort_points(points)
    xyz = np.concatenate([points[order], np.zeros((len(points), 1))], axis=1).astype(np.float32)
    batch = {"point_clouds": torch.from_numpy(xyz).unsqueeze(0).to(device)}
    cluster, _, _, agree = model.to(device).eval()(batch)
    labels = np.zeros(len(points), dtype=int)
    scores = np.zeros(len(points), dtype=np.float32)
    labels[order] = relabel(cluster.squeeze(0).cpu().numpy())
    scores[order] = agree.squeeze(0).float().cpu().numpy()
    return labels, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HPSCAN checkpoint")
    parser.add_argument("--input", nargs="+", required=True, help="scatterplot file(s)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_normalize", action="store_true", help="points are already in [-1, 1]")
    parser.add_argument("--plot", action="store_true", help="also save a PNG of the clustering")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.model)
    if model.kind != "hpscan":
        raise SystemExit("predict.py expects a point-based HPSCAN checkpoint")

    for file in args.input:
        file = Path(file)
        points = read_points(file)
        if not args.no_normalize:
            points = normalize(points)
        labels, agree = predict(model, points, args.device)
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
