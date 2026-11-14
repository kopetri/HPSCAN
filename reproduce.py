"""Reproduce Tables 2-5 of the paper with the released checkpoints.

    python reproduce.py --table 3            # one table
    python reproduce.py --table all          # everything (~1 h on a single GPU)
    python reproduce.py --table 3 --download # fetch missing checkpoints first

Every row is evaluated with the configuration listed in hpscan/paper.py (checkpoint, test set,
inference precision) and printed next to the published value. Metrics whose rounded value
differs from the paper are listed after "!=", notes explain known differences.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from hpscan.baselines import is_baseline
from hpscan.evaluation import Evaluator
from hpscan.module import load_model
from hpscan.paper import CHECKPOINTS, TABLE_5, TABLES

FORMATS = {"kappa_alpha": "{:7.2f}", "kappa_v": "{:5.2f}", "kappa_n": "{:6.2f}%", "mse": "{:7.4f}", "mae": "{:7.4f}"}


def fmt(key, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " " * len(FORMATS[key].format(0.0))
    return FORMATS[key].format(value)


def same(key, a, b):
    digits = 4 if key in ("mse", "mae") else 2
    return round(a, digits) == round(b, digits)


class Runner:
    def __init__(self, args):
        self.args = args
        self.cache = {}

    def model(self, name):
        if is_baseline(name):
            return name
        path = Path(self.args.checkpoint_dir) / name
        if not path.exists():
            if not self.args.download:
                raise SystemExit(f"missing checkpoint {path} (run with --download or `python download.py --checkpoints`)")
            from download import download_checkpoints
            download_checkpoints(self.args.checkpoint_dir, [name])
        return load_model(path)

    def evaluate(self, row):
        key = (row["model"], row["data"], row["amp"])
        if key not in self.cache:
            np.random.seed(self.args.seed)
            model = self.model(row["model"])
            render = not isinstance(model, str) and model.kind == "image"
            evaluator = Evaluator(device=self.args.device, amp=row["amp"])
            data = Path(self.args.data_root) / row["data"]
            self.cache[key] = evaluator.run(model, data, max_cluster=row["max_cluster"], render_images=render)
        return self.cache[key]

    def table(self, name):
        rows = TABLES[name]
        keys = list(rows[0]["paper"])
        print(f"\nTable {name}")
        header = f"  {'row':<30} {'precision':<9} | " + " ".join(f"{k:>8}" for k in keys) + "  (ours / paper)"
        print(header)
        out = []
        for row in rows:
            result = self.evaluate(row)
            values = {k: (result["mean"] if k in ("mse", "mae") else result["weighted"]).get(k) for k in keys}
            ours = " ".join(f"{fmt(k, values[k]):>8}" for k in keys)
            paper = " ".join(f"{fmt(k, row['paper'][k]):>8}" for k in keys)
            diff = [k for k in keys if row["paper"][k] is not None and not same(k, values[k], row["paper"][k])]
            mark = ("!= " + ",".join(diff)) if diff else "=="
            print(f"  {row['label']:<30} {'fp16-amp' if row['amp'] else 'fp32':<9} | {ours}  ours")
            print(f"  {'':<30} {'':<9} | {paper}  paper {mark}" + (f"  ({row['note']})" if row.get("note") else ""))
            out.append(dict(row, ours=values, differs=diff))
        return out

    def table5(self):
        from clustme import predict_counts, scores
        from hpscan.data import ClustMeDataset
        print("\nTable 5 (ClustMe)")
        dataset = ClustMeDataset(Path(self.args.data_root) / "ClustMe", split="test")
        model = self.model(TABLE_5["model"])
        pred, answers = predict_counts(model, dataset, self.args.device, TABLE_5["amp"])
        results = {"HPSCAN": scores(pred, answers),
                   "single": scores(torch.ones(len(answers), dtype=torch.long), answers),
                   "multi": scores(torch.full((len(answers),), 2), answers)}
        g = torch.Generator().manual_seed(self.args.seed)
        runs = np.array([scores(torch.randint(1, 3, (len(answers),), generator=g), answers) for _ in range(100)])
        results["random"] = tuple(runs.mean(axis=0))
        print(f"  {'model':<10} {'kappa_v':>8} {'acc':>6} {'F1':>6}  (ours / paper)")
        for name in ["HPSCAN", "random", "single", "multi"]:
            k, a, f = results[name]
            pk, pa, pf = TABLE_5["paper"][name]
            print(f"  {name:<10} {round(k, 2) + 0.0:8.2f} {a:6.0%} {f:6.0%}  ours")
            print(f"  {'':<10} {pk:8.2f} {pa:6.0%} {pf:6.0%}  paper" + ("  (100 random runs)" if name == "random" else ""))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", nargs="+", default=["all"], choices=["all", "2", "2a", "2b", "2c", "3", "4", "5"])
    parser.add_argument("--data_root", default="data", help="folder with PSC/, SDR/ and ClustMe/")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--download", action="store_true", help="download missing checkpoints")
    parser.add_argument("--output", default=None, help="write all results to this JSON file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    tables = []
    for t in args.table:
        tables += {"all": ["2a", "2b", "2c", "3", "4", "5"], "2": ["2a", "2b", "2c"]}.get(t, [t])
    runner = Runner(args)
    results = {}
    for t in dict.fromkeys(tables):
        results[t] = runner.table5() if t == "5" else runner.table(t)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1, default=str))
    print("\nCheckpoints: " + ", ".join(f"{k} ({v['run']}, {v['step']})" for k, v in CHECKPOINTS.items()))


if __name__ == "__main__":
    main()
