"""Network architectures: the point-based HPSCAN model and the image-based U-Net baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

MAX_CLUSTERS = 20  # C in the paper: maximum number of clusters HPSCAN can predict


class HPSCANNet(nn.Module):
    """PointNet++ segmentation network with C + 3 output channels per point.

    Channels 0-1: noise logits, channel 2: agreement logit, channels 3..C+2: cluster logits.
    """

    def __init__(self, max_cluster=MAX_CLUSTERS, input_feature_dim=0):
        super().__init__()
        self.sa1 = PointNetSetAbstractionMsg(256, [0.05, 0.1], [16, 32], 3 + input_feature_dim, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(64, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(32, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(8, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, max_cluster + 3, 1)

    def forward(self, xyz):
        """xyz: (B, 3, N) -> noise (B, 2, N), cluster (B, C, N), agreement (B, N) logits."""
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        noise = x[:, 0:2, :]
        agree = x[:, 2, :]
        cluster = x[:, 3:, :]
        return noise, cluster, agree


class UNetBaseline(nn.Module):
    """Image-based baseline: a U-Net that segments a rendered scatterplot image (Table 3, "Baseline")."""

    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", max_cluster=MAX_CLUSTERS):
        super().__init__()
        import segmentation_models_pytorch as smp  # only needed for the baseline
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=max_cluster + 3,
        )

    def forward(self, x):
        """x: (B, 1, H, W) -> noise (B, 2, H, W), cluster (B, C, H, W), agreement (B, H, W) logits."""
        x = self.model(x)
        noise = x[:, 0:2]
        agree = x[:, 2]
        cluster = x[:, 3:]
        return noise, cluster, agree


def decode_prediction(noise_logits, cluster_logits, agree_logits):
    """Turn network outputs into per-point labels.

    Returns `cluster` (0 = noise, 1..C = cluster ids), `noise` (0 = noise, 1 = clustered),
    the cluster probabilities and the predicted agreement score in [0, 1].
    """
    prob = cluster_logits.softmax(dim=1)
    cluster = torch.argmax(prob, dim=1) + 1
    noise = torch.argmax(noise_logits.softmax(dim=1), dim=1)
    cluster[noise == 0] = 0
    return cluster, noise, prob, agree_logits.sigmoid()
