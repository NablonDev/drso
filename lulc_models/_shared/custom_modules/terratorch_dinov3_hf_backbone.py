"""Custom TerraTorch backbone: DINOv3 ViT-L/16 via the already-approved HF
`transformers` checkpoint (facebook/dinov3-vitl16-pretrain-lvd1689m), instead
of TerraTorch's native `DinoV3Wrapper` (which requires a separate gated
download from Meta's own site in Meta's native checkpoint format).

Exposes every transformer layer's hidden state as [CLS, patch_tokens],
register tokens dropped, matching the single-prefix-token shape that
TerraTorch's `ReshapeTokensToImage`/`SelectIndices` necks already expect
(they only know how to strip one CLS token, not DINOv3's 4 register tokens).

Import this module before referencing `dinov3_hf_vitl16` in a config so the
registry decorator runs (e.g. `custom_modules_path` in the YAML, or
`import terratorch_dinov3_hf_backbone` at the top of a training script).
"""

import torch
from torch import nn
from transformers import AutoModel

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

DEFAULT_CHECKPOINT = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class DinoV3HFWrapper(nn.Module):
    def __init__(self, pretrained_name: str = DEFAULT_CHECKPOINT, freeze: bool = True, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_name)
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 4)
        n_layers = self.model.config.num_hidden_layers
        self.out_channels = [self.model.config.hidden_size] * (n_layers + 1)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.model(x, output_hidden_states=True)
        feats = []
        for h in out.hidden_states:
            cls = h[:, :1, :]
            patches = h[:, 1 + self.num_register_tokens :, :]
            feats.append(torch.cat([cls, patches], dim=1))
        return feats


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_hf_vitl16(pretrained: bool = True, freeze: bool = True, **kwargs) -> DinoV3HFWrapper:
    pretrained_name = kwargs.pop("pretrained_name", DEFAULT_CHECKPOINT)
    return DinoV3HFWrapper(pretrained_name=pretrained_name, freeze=freeze, **kwargs)
