from dataclasses import dataclass
import torch.nn as nn
import torch
from einops import rearrange
from torchvision.ops import StochasticDepth
from collections import defaultdict
import numpy as np
from typing import List, Literal

arch = Literal["B0", "B1", "B2", "B3", "B4", "B5"]

@dataclass
class SegformerConfig:
    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    channels: List[int]
    reduction_ratio: List[int]
    num_heads: List[int]
    expansion_ratio: List[int]
    num_encoders: List[int]

class LayerNorm2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class OverlappedPatchMerging(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.patches = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = LayerNorm2D(out_channels)

    def forward(self, x):
        return self.norm(self.patches(x))


class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, num_heads, reduction_ratio):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2D(channels),
        )
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        _, _, h, w = x.shape

        reduced_x = self.reducer(x)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")

        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.attention(x, reduced_x, reduced_x)[0]

        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out


class MixFFNBlock(nn.Module):
    def __init__(self, channels: int, expansion_ratio: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            channels,
            channels * expansion_ratio,
            kernel_size=3,
            groups=channels,
            padding=1,
        )
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv2d(channels * expansion_ratio, channels, kernel_size=1)

    def forward(self, x):
        return self.conv3(self.gelu(self.conv2(self.conv1(x))))


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class MixFFNEncoderLayer(nn.Module):
    def __init__(
        self,
        channels,
        reduction_ratio=1,
        num_heads=8,
        expansion_ratio=4,
        drop_path_prob=0.0,
    ):
        super().__init__()
        self.layer1 = ResidualBlock(
            nn.Sequential(
                LayerNorm2D(channels),
                EfficientSelfAttention(channels, reduction_ratio, num_heads),
            )
        )
        self.layer2 = ResidualBlock(
            nn.Sequential(
                LayerNorm2D(channels),
                MixFFNBlock(channels, expansion_ratio=expansion_ratio),
                StochasticDepth(p=drop_path_prob, mode="batch"),
            )
        )

    def forward(self, x):
        return self.layer2(self.layer1(x))


class MixFFNEncoder(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        in_channels,
        out_channels,
        n_layers,
        reduction_ratio,
        num_heads,
        expansion_ratio,
    ):
        super().__init__()
        self.patches = OverlappedPatchMerging(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.mixffn = nn.Sequential(
            *[
                MixFFNEncoderLayer(
                    channels=out_channels,
                    reduction_ratio=reduction_ratio,
                    num_heads=num_heads,
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm2D(out_channels)

    def forward(self, x):
        return self.norm(self.mixffn(self.patches(x)))


class MLPDecoderLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, out_dims: tuple):
        super().__init__(
            nn.UpsamplingBilinear2d(size=out_dims),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class MLPDecoder(nn.Module):
    def __init__(self, in_channels, embed_channels, out_dims, num_classes):
        super().__init__()
        self.scale_factors = [1, 2, 4, 8]
        self.stages = nn.ModuleList(
            [
                MLPDecoderLayer(in_channel, embed_channels, out_dims)
                for in_channel in in_channels
            ]
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_channels * 4, embed_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(embed_channels),
        )
        self.predict = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

    def forward(self, x):
        new_features = []
        for feature, stage in zip(x, self.stages):
            out = stage(feature)
            new_features.append(out)

        return self.predict(self.fuse(torch.cat(new_features, dim=1)))


class Segformer(nn.Module):
    def __init__(self, model_config: SegformerConfig):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                MixFFNEncoder(
                    in_channels=3 if i == 0 else model_config.channels[i - 1],
                    out_channels=model_config.channels[i],
                    kernel_size=model_config.kernel_size[i],
                    stride=model_config.stride[i],
                    padding=model_config.padding[i],
                    reduction_ratio=model_config.reduction_ratio[i],
                    num_heads=model_config.num_heads[i],
                    expansion_ratio=model_config.expansion_ratio[i],
                    n_layers=model_config.num_encoders[i],
                )
                for i in range(4)
            ]
        )
        self.decoder = MLPDecoder([32, 64, 160, 256], 256, (64, 64), 4)
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = 4
        self.miou_list = []
        self.miou_dict = defaultdict(lambda: [])

        self.loss_list = []
        self.loss_dict = defaultdict(lambda: [])

    def forward(self, images):
        embeds = [images]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return self.decoder(embeds[1:])

    def miou(self, prediction, targets):
        preds = torch.argmax(prediction, dim=1) # gives me the class number
        ious = []

        for i in range(4):
          preds_mask = torch.where(preds == i, 1.0, 0.0)
          targets_mask = torch.where(targets == i, 1.0, 0.0)

          intersection = torch.sum(targets_mask * preds_mask)
          union = torch.sum(targets_mask) + torch.sum(preds_mask) - intersection
          iou = intersection.to(torch.float) / union.to(torch.float)
          ious.append(iou.cpu().data.numpy())

        return np.mean(ious)
