import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange



@dataclass
class SegformerConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

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
    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.layer_norm = LayerNorm2D(config.in_channels)
        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
        )

    def forward(self, x):
        return self.proj(self.norm(x))

class MixFFNEncoder(nn.Module):
    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.patch_embedding = OverlappedPatchMerging(config)
        self.block = nn.Sequential(
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
        self.layer_norm = LayerNorm2D(out_channels)

class Segformer(nn.Module):
    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.encoder = None
        self.decoder = None

if __name__ == "__main__":
    config = SegformerConfig(1, 1, 1, 1, 1)
    model = OverlappedPatchMerging(config)
    # print(model.state_dict().keys())
    with open("apna.txt", "a") as text_file:
        text_file.write("\n".join(list(model.state_dict().keys())))
