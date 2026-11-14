"""Evaluation protocol of Tables 2, 3 and 4.

The test set is grouped by the perceived number of clusters (0 .. max_cluster). Every metric
is averaged over the stimuli of a group, and the group means are combined into the reported
score with weights proportional to 1 / (number of stimuli in the group), i.e. groups with
few stimuli (e.g. NO CLUSTER) are not dominated by frequent cluster counts:

    score = sum_g mean_g * (1 / n_g) / sum_h (1 / n_h)

The MSE / MAE of the agreement regression (Table 4) are the unweighted mean over the groups.
"""
import contextlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .baselines import run_baseline
from .data import ScatterplotDataset, cluster_counts_of_raters, load_scatterplot, perceived_cluster_count
from .metrics import CalinskiHarabaszScore, CrossValidDifference, NoiseIou, SilhouetteIndex, group_kappa_v

METRICS = ["kappa_alpha", "kappa_v", "kappa_n", "silhouette", "calinski", "mse", "mae"]


def _batch(sample, device):
    batch = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if torch.is_tensor(value):
            value = value.unsqueeze(0).to(device)
        batch[key] = value
    return batch


class Evaluator:
    def __init__(self, device="cuda", amp=False):
        self.device = device
        self.amp = amp
        self.cross = CrossValidDifference()
        self.noise_iou = NoiseIou()
        self.silhouette = SilhouetteIndex()
        self.calinski = CalinskiHarabaszScore()

    def autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def score(self, xyz, cluster, noise, gt_raters, agree=None, agree_gt=None):
        out = {
            "kappa_alpha": self.cross(cluster, gt_raters).item() * -100,
            "kappa_v": group_kappa_v(cluster, gt_raters).item(),
            "kappa_n": float(self.noise_iou(noise, torch.clip(gt_raters, 0, 1))) * 100,
            "silhouette": float(self.silhouette(xyz, cluster)),
            "calinski": float(self.calinski(xyz, cluster)) / 1000.0,
        }
        if agree is not None:
            out["mse"] = torch.nn.functional.mse_loss(agree.float(), agree_gt).item()
            out["mae"] = torch.nn.functional.l1_loss(agree.float(), agree_gt).item()
        return out

    def predict(self, model, batch):
        """Returns cluster ids (0 = noise), noise mask (0 = noise) and agreement (or None)."""
        if isinstance(model, str):  # scikit-learn baseline
            xyz = batch['point_clouds'].squeeze(0).cpu().numpy()
            pred = torch.from_numpy(run_baseline(model, xyz, batch['n_cluster'].item())).unsqueeze(0).to(self.device)
            return pred, (pred != 0).int(), None
        with torch.no_grad(), self.autocast():
            cluster, noise, _, agree = model(batch)
        return cluster, noise, agree

    def run(self, model, dataset_path, max_cluster=6, n_raters=5, render_images=False):
        """Evaluate a model (HPSCAN / image baseline module) or a baseline name on the test split."""
        dataset = ScatterplotDataset(dataset_path, split="test", n_raters=n_raters, render_images=render_images,
                                     verbose=False)
        groups = defaultdict(lambda: defaultdict(list))
        if not isinstance(model, str):
            model = model.to(self.device).eval()
        for i in tqdm(range(len(dataset)), desc=Path(str(dataset_path)).name, leave=False):
            ratings = load_scatterplot(dataset.files[i])[:, 2:]
            group = int(perceived_cluster_count(cluster_counts_of_raters(ratings), 3))
            if group > max_cluster:
                continue
            batch = _batch(dataset[i], self.device)
            cluster, noise, agree = self.predict(model, batch)
            with self.autocast():
                scores = self.score(batch['point_clouds'], cluster, noise, batch['gt_raters'], agree,
                                    batch['agreement_score'])
            for key, value in scores.items():
                groups[group][key].append(value)
            groups[group]["plot_id"].append(batch['plot_id'])
        return summarize(groups, max_cluster)


def summarize(groups, max_cluster):
    present = [g for g in range(max_cluster + 1) if g in groups]
    counts = {g: len(groups[g]["plot_id"]) for g in present}
    weights = {g: (1 / counts[g]) / sum(1 / counts[h] for h in present) for g in present}
    result = {"groups": {}, "weighted": {}, "mean": {}, "counts": counts}
    for key in METRICS:
        means = {}
        for g in present:
            values = [v for v in groups[g].get(key, []) if not np.isnan(v)]
            means[g] = float(np.mean(values)) if values else float("nan")
        if all(np.isnan(v) for v in means.values()):
            continue
        result["groups"][key] = means
        valid = [g for g in present if not np.isnan(means[g])]
        result["weighted"][key] = float(sum(means[g] * weights[g] for g in valid))
        result["mean"][key] = float(np.mean([means[g] for g in valid]))
    return result


def format_result(name, result):
    w = result["weighted"]
    lines = [f"{name}:",
             f"  kappa_alpha {w['kappa_alpha']:.2f} | kappa_v {w['kappa_v']:.2f} | kappa_n {w['kappa_n']:.2f}%"]
    if "mse" in result["mean"]:
        lines.append(f"  agreement regression: MSE {result['mean']['mse']:.4f} | MAE {result['mean']['mae']:.4f}")
    header = "  group     " + " ".join(f"{g:>8}" for g in result["counts"]) + "  weighted"
    lines += [header, "  n         " + " ".join(f"{n:>8}" for n in result["counts"].values())]
    for key, means in result["groups"].items():
        lines.append(f"  {key:<10}" + " ".join(f"{v:8.4f}" for v in means.values()) + f"  {result['weighted'][key]:8.4f}")
    return "\n".join(lines)


def save_result(path, name, result):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
