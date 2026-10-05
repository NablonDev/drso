"""LinearDecoder + GroupNorm -- same idea as LinearDecoderBN (Meta's
use_batchnorm recipe), but GroupNorm normalizes per-sample instead of across
the batch, so it's immune to the batch-statistics instability BatchNorm hits
at batch_size=4 on a 91-image training set (diagnosed cause of v13's
underperformance vs. v8). Standard substitution for BatchNorm on small data
(see e.g. Wu & He, "Group Normalization", 2018).

num_groups=32 on embed_dim=1024 (ViT-L hidden size) -> 32 channels/group,
the standard default from the GroupNorm paper and most implementations
using it in place of BatchNorm (Mask R-CNN/FPN, ConvNeXt, etc).

USAGE (config): decoder: LinearDecoderGN (same kwargs as LinearDecoder:
decoder_upsampling_size, decoder_num_classes).
"""
from terratorch.registry import TERRATORCH_DECODER_REGISTRY
from torch import Tensor, nn

NUM_GROUPS = 32


@TERRATORCH_DECODER_REGISTRY.register
class LinearDecoderGN(nn.Module):
    includes_head: bool = True

    def __init__(self, embed_dim: list[int], num_classes: int, upsampling_size: int, in_index: int = -1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_index = in_index
        self.embed_dim = embed_dim[in_index]

        self.gn = nn.GroupNorm(NUM_GROUPS, self.embed_dim)
        self.conv = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.num_classes,
            kernel_size=upsampling_size,
            stride=upsampling_size,
            padding=0,
            output_padding=0,
        )

    @property
    def out_channels(self) -> int:
        return self.num_classes

    def forward(self, x: list[Tensor]) -> Tensor:
        return self.conv(self.gn(x[self.in_index]))
