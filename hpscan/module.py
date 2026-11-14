"""Lightning modules for training HPSCAN and the image baseline, and checkpoint helpers."""
import lightning.pytorch as pl
import torch

from .loss import MCL, Class2Simi, NoiseLoss, PairEnum
from .metrics import CrossValidDifference, NoiseIou, group_kappa_v
from .model import HPSCANNet, UNetBaseline, decode_prediction


class HPSCANModule(pl.LightningModule):
    """HPSCAN: PointNet++ trained with L_total = L_noise + 0.1 L_mcl + L_agree (Eq. 8)."""

    kind = "hpscan"
    HPARAMS = ("learning_rate", "weight_decay", "dissimilar_weight", "loss_weights", "lr_patience", "factor")

    def __init__(self, learning_rate=1e-5, weight_decay=0.1, dissimilar_weight=50.0,
                 loss_weights=(1.0, 0.1, 1.0), lr_patience=100, factor=0.1, **kwargs):
        super().__init__()
        self.save_hyperparameters(dict(
            model=self.kind, learning_rate=learning_rate, weight_decay=weight_decay,
            dissimilar_weight=dissimilar_weight, loss_weights=list(loss_weights),
            lr_patience=lr_patience, factor=factor, **kwargs))
        self.model = self.build_model()
        self.mcl = MCL()
        self.noise_loss = NoiseLoss()
        self.agree_loss = torch.nn.L1Loss()
        self.cross = CrossValidDifference()
        self.noise_iou = NoiseIou()

    def build_model(self):
        return HPSCANNet()

    def infer(self, batch):
        """Per-point logits: noise (B, 2, N), clusters (B, C, N), agreement (B, N)."""
        return self.model(batch['point_clouds'].permute(0, 2, 1))

    def forward(self, batch):
        """Per-point prediction: cluster ids (0 = noise), noise mask, cluster probabilities, agreement."""
        return decode_prediction(*self.infer(batch))

    # ------------------------------------------------------------------ training
    def _step(self, batch, split):
        noise_logits, cluster_logits, agree_logits = self.infer(batch)
        cluster, noise, prob, agree = decode_prediction(noise_logits, cluster_logits, agree_logits)
        B = cluster.shape[0]
        w = self.hparams.loss_weights

        prob1, prob2 = PairEnum(prob.permute(0, 2, 1))
        simi, weights = Class2Simi(batch["multi_label"], dissimilar_weight=self.hparams.dissimilar_weight)
        mcl_loss = self.mcl(prob1, prob2, simi, weights)
        noise_loss = self.noise_loss(noise_logits, batch["noise_label"])
        agree_loss = self.agree_loss(agree, batch["agreement_score"])
        loss = noise_loss * w[0] + mcl_loss * w[1] + agree_loss * w[2]

        log = dict(loss=loss, noise_loss=noise_loss, mcl_loss=mcl_loss, agree_loss=agree_loss)
        if split == "valid":
            gt_raters = batch["gt_raters"]
            log.update(cross=self.cross(cluster, gt_raters),
                       vanbelle=group_kappa_v(cluster, gt_raters),
                       noise_iou=self.noise_iou(noise, torch.clip(gt_raters, 0, 1)),
                       mae_agree=torch.nn.functional.l1_loss(agree, batch["agreement_score"]))
        for key, value in log.items():
            self.log(f"{split}_{key}", value, prog_bar=key in ("loss", "cross"), batch_size=B,
                     on_step=split == "train", on_epoch=split != "train")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.hparams.lr_patience,
                                                               factor=self.hparams.factor)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'valid_loss',
                              'frequency': 1, 'strict': True}]


class ImageBaselineModule(HPSCANModule):
    """Image-based baseline: U-Net on the rendered scatterplot, read out at the pixel of each point."""

    kind = "image"
    HPARAMS = HPSCANModule.HPARAMS + ("encoder_name", "encoder_weights")

    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", **kwargs):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs)

    def build_model(self):
        return UNetBaseline(self.hparams.encoder_name, self.hparams.encoder_weights)

    @staticmethod
    def restore_points(pixel_coords, maps):
        """maps: (B, H, W, M), pixel_coords: (B, N, 2) -> (B, N, M).

        Indexes maps[b, coord_0, coord_1] exactly as during training of the released baseline.
        """
        B, N = pixel_coords.shape[0], pixel_coords.shape[1]
        batch = torch.arange(B, device=maps.device).view(B, 1).expand(B, N)
        pc = pixel_coords.long()
        return maps[batch, pc[..., 0], pc[..., 1]]

    def infer(self, batch):
        noise, cluster, agree = self.model(batch['stimulus'])
        pc = batch['pixel_coords']
        noise = self.restore_points(pc, noise.permute(0, 2, 3, 1)).permute(0, 2, 1)
        cluster = self.restore_points(pc, cluster.permute(0, 2, 3, 1)).permute(0, 2, 1)
        agree = self.restore_points(pc, agree.unsqueeze(-1)).squeeze(-1)
        return noise, cluster, agree


# --------------------------------------------------------------------------- checkpoints
MODULES = {HPSCANModule.kind: HPSCANModule, ImageBaselineModule.kind: ImageBaselineModule}


def _module_class(hparams):
    kind = hparams.get("model") or ("image" if hparams.get("encoder_name") else "hpscan")
    return MODULES[kind]


def load_model(path, map_location="cpu", **overrides):
    """Load a released checkpoint (or an original training checkpoint) for inference / fine-tuning."""
    ckpt = torch.load(path, map_location=map_location)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    cls = _module_class(hparams)
    kwargs = {k: hparams[k] for k in cls.HPARAMS if k in hparams}
    if cls is ImageBaselineModule:
        kwargs["encoder_weights"] = None  # weights come from the checkpoint
    kwargs.update(overrides)
    module = cls(**kwargs)
    module.load_state_dict(ckpt["state_dict"])
    return module


def export_checkpoint(src, dst, **info):
    """Strip a training checkpoint down to weights + model hyper-parameters (+ provenance `info`)."""
    ckpt = torch.load(src, map_location="cpu")
    hparams = dict(ckpt.get("hyper_parameters", {}))
    cls = _module_class(hparams)
    slim = {
        "state_dict": {k: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")},
        "hyper_parameters": {"model": cls.kind, **{k: hparams[k] for k in cls.HPARAMS if k in hparams}},
        "info": {"epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"), **info},
    }
    torch.save(slim, dst)
    return slim
