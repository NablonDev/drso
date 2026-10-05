"""LinearDecoder + BatchNorm, matching Meta's official DINOv3 ADE20K linear
segmentation recipe (dinov3/eval/segmentation/configs/config-ade20k-linear-
training.yaml: use_batchnorm=True). TerraTorch's own LinearDecoder is a bare
nn.ConvTranspose2d with no normalization layer at all -- this is the one
missing piece to match Meta's validated architecture exactly, isolated as a
single variable against v8 (which uses the un-normalized LinearDecoder).

USAGE (config): decoder: LinearDecoderBN (same kwargs as LinearDecoder:
decoder_upsampling_size, decoder_num_classes).
"""
from terratorch.registry import TERRATORCH_DECODER_REGISTRY
from torch import Tensor, nn


@TERRATORCH_DECODER_REGISTRY.register
class LinearDecoderBN(nn.Module):
    includes_head: bool = True

    def __init__(self, embed_dim: list[int], num_classes: int, upsampling_size: int, in_index: int = -1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_index = in_index
        self.embed_dim = embed_dim[in_index]

        self.bn = nn.BatchNorm2d(self.embed_dim)
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
        return self.conv(self.bn(x[self.in_index]))
