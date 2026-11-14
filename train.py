"""Train HPSCAN (or the image-based U-Net baseline) on the PSC dataset.

The defaults are the hyper-parameters of the released models. See README.md, section
"Training", for the exact commands of every released checkpoint.

Examples:
    # Table 2a: contrastive loss weighting D (trained from scratch, 1000 epochs)
    python train.py --data data/PSC --dissimilar_weight 50 --max_epochs 1000

    # Table 2b: train on stimuli with group agreement >= T_agree
    python train.py --data data/PSC --dissimilar_weight 50 --filter_agreement 0.7 --max_epochs 1000

    # Table 2c / final model: fine-tune with a small learning rate and weight D
    python train.py --data data/PSC --finetune runs/<run>/checkpoints/step=147000.ckpt \
        --learning_rate 1e-7 --dissimilar_weight 0.1 --filter_agreement 0.7 --n_cluster 1 --max_epochs 10000

    # image-based baseline
    python train.py --model image --data data/PSC --dissimilar_weight 50 --filter_agreement 0.7 --max_epochs 1000
"""
import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from hpscan.data import ScatterplotDataset
from hpscan.module import HPSCANModule, ImageBaselineModule


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["hpscan", "image"], default="hpscan")
    p.add_argument("--data", required=True, nargs="+", help="training dataset folder(s)")
    p.add_argument("--valid_data", nargs="+", default=None, help="validation dataset folder(s), default: --data")
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--name", default=None, help="run name (also used for W&B)")
    p.add_argument("--wandb_project", default=None, help="log to Weights & Biases (requires `wandb login`)")
    # optimisation
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_patience", type=int, default=100, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--finetune", default=None, help="initialise the weights from this checkpoint")
    # loss
    p.add_argument("--dissimilar_weight", type=float, default=50.0, help="D in Eq. 3")
    p.add_argument("--loss_weights", type=float, nargs=3, default=[1.0, 0.1, 1.0], help="noise, mcl, agreement")
    # data
    p.add_argument("--n_raters", type=int, default=5)
    p.add_argument("--label_idx", type=int, default=-1, help="rater used as target (-1: random rater per sample)")
    p.add_argument("--filter_agreement", type=float, default=-1.0, help="T_agree: minimum group agreement")
    p.add_argument("--best_agreement", type=float, default=None,
                   help="keep the largest group of raters with agreement >= this value (Table 5 model)")
    p.add_argument("--n_cluster", type=int, default=None, help="only stimuli with this perceived cluster count")
    p.add_argument("--no_augmentation", action="store_true")
    p.add_argument("--crop_margin", type=float, default=None, help="default: 1.0 (hpscan), 1.2 (image)")
    p.add_argument("--crop_rescale", type=int, default=None, help="default: 0 (hpscan), 1 (image)")
    p.add_argument("--encoder_name", default="resnet18", help="U-Net encoder of the image baseline")
    # trainer
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--precision", default="32")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ckpt_every_n_steps", type=int, default=1000)
    p.add_argument("--devices", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    image = args.model == "image"
    crop_margin = args.crop_margin if args.crop_margin is not None else (1.2 if image else 1.0)
    crop_rescale = bool(args.crop_rescale) if args.crop_rescale is not None else image

    hparams = dict(learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                   dissimilar_weight=args.dissimilar_weight, loss_weights=args.loss_weights,
                   lr_patience=args.lr_patience, factor=args.factor)
    if image:
        module = ImageBaselineModule(encoder_name=args.encoder_name,
                                     encoder_weights=None if args.finetune else "imagenet", **hparams)
    else:
        module = HPSCANModule(**hparams)
    if args.finetune:
        state = torch.load(args.finetune, map_location="cpu")["state_dict"]
        module.load_state_dict({k: v for k, v in state.items() if k.startswith("model.")})

    common = dict(n_raters=args.n_raters, label_idx=args.label_idx, filter_agreement=args.filter_agreement,
                  best_agreement=args.best_agreement, n_cluster=args.n_cluster, render_images=image)
    train_set = ScatterplotDataset(args.data, split="train", augment=not args.no_augmentation,
                                   crop_margin=crop_margin, crop_rescale=crop_rescale, **common)
    valid_set = ScatterplotDataset(args.valid_data or args.data, split="valid", **common)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)

    name = args.name or f"{args.model}_D{args.dissimilar_weight:g}"
    run_dir = Path(args.output_dir) / name
    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="{epoch}-{valid_loss}",
                                     monitor="valid_loss", mode="min", save_top_k=1),
        pl.callbacks.ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="{epoch}-{valid_cross}",
                                     monitor="valid_cross", mode="min", save_top_k=1),
        pl.callbacks.ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="{step}",
                                     every_n_train_steps=args.ckpt_every_n_steps, save_top_k=-1),
    ]
    if args.wandb_project:
        logger = pl.loggers.WandbLogger(project=args.wandb_project, name=name, save_dir=str(run_dir))
        callbacks.append(pl.callbacks.LearningRateMonitor())
    else:
        logger = pl.loggers.CSVLogger(save_dir=str(run_dir), name="logs")
    logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=args.devices,
                         precision=args.precision, max_epochs=args.max_epochs, max_steps=args.max_steps,
                         min_epochs=10, log_every_n_steps=10, deterministic=True, logger=logger,
                         callbacks=callbacks, default_root_dir=str(run_dir))
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
