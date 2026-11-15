"""Train HPSCAN with the improved recipe (post-publication, see improved/README.md).

    # the released checkpoint hpscan_improved.ckpt (all of PSC train, 300 epochs, ~45 min on one GPU)
    python improved/train.py --data data/PSC --output_dir runs/improved
    # development protocol: hold out 200 training stimuli for validation (written to <output_dir>/holdout.txt)
    python improved/train.py --data data/PSC --output_dir runs/improved_dev --holdout 200

Checkpoints are written every --save_every epochs (<output_dir>/epochXXXX.ckpt) in the format of the
released checkpoints; evaluate them with improved/evaluate.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hpscan.data import ScatterplotDataset  # noqa: E402
from hpscan.model import HPSCANNet  # noqa: E402
from improved.training import AugmentedScatterplots, SubsampleCollate, hpscan_loss, seed_worker  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="PSC folder with train.txt")
    p.add_argument("--output_dir", default="runs/improved")
    p.add_argument("--holdout", type=int, default=0, help="hold out this many training stimuli (validation)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=5, help="epochs of linear warm-up before the cosine decay")
    p.add_argument("--dissimilar_weight", type=float, default=50.0, help="D in omega_D")
    p.add_argument("--loss_weights", type=float, nargs=3, default=[1.0, 0.1, 1.0], help="noise, mcl, agreement")
    p.add_argument("--p_cut", type=float, default=0.5)
    p.add_argument("--p_rotate", type=float, default=0.5)
    p.add_argument("--p_jitter", type=float, default=0.5)
    p.add_argument("--jitter", type=float, default=0.05, help="max. std. of the Gaussian jitter")
    p.add_argument("--p_subsample", type=float, default=0.5)
    p.add_argument("--min_points", type=int, default=256, help="smallest random point subset (>= 256)")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    files = ScatterplotDataset(a.data, split="train", verbose=False).files
    if a.holdout > 0:
        idx = np.random.RandomState(0).permutation(len(files))
        (out / "holdout.txt").write_text("\n".join(files[i].name for i in sorted(idx[:a.holdout])) + "\n")
        files = [files[i] for i in sorted(idx[a.holdout:])]
    dataset = AugmentedScatterplots(files, p_cut=a.p_cut, p_rotate=a.p_rotate, p_jitter=a.p_jitter, jitter=a.jitter)
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch_size, shuffle=True, drop_last=True,
                                         num_workers=a.workers, persistent_workers=a.workers > 0,
                                         worker_init_fn=seed_worker, collate_fn=SubsampleCollate(a.min_points, a.p_subsample))
    print(f"training on {len(dataset)} stimuli", flush=True)

    net = HPSCANNet().to(a.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    steps, warmup = a.epochs * len(loader), a.warmup * len(loader)

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    log_file = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "loss", "noise", "mcl", "agree", "lr", "seconds"])
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        net.train()
        total = np.zeros(4)
        for xyz, gt, agree in loader:
            xyz, gt, agree = xyz.to(a.device), gt.to(a.device), agree.to(a.device)
            noise_logits, cluster_logits, agree_logits = net(xyz.permute(0, 2, 1))
            loss, parts = hpscan_loss(noise_logits, cluster_logits, agree_logits, gt, agree,
                                      a.dissimilar_weight, a.loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += np.array([loss.item(), *parts])
        total /= len(loader)
        log.writerow([epoch, *[f"{v:.5f}" for v in total], f"{scheduler.get_last_lr()[0]:.3g}", f"{time.time() - start:.0f}"])
        log_file.flush()
        if epoch % a.save_every == 0 or epoch == a.epochs:
            torch.save({"state_dict": {"model." + k: v for k, v in net.state_dict().items()},
                        "hyper_parameters": {"model": "hpscan"},
                        "info": {"epoch": epoch, "recipe": "improved/train.py", "args": vars(a)}},
                       out / f"epoch{epoch:04d}.ckpt")
            print(f"epoch {epoch}: loss {total[0]:.4f} (noise {total[1]:.3f}, mcl {total[2]:.3f}, "
                  f"agreement {total[3]:.3f})  {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
