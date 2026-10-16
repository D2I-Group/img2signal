import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import logging
from timm.models.vision_transformer import PatchEmbed

logger = logging.getLogger(__name__)

default_linear = {
    'img_size': 224,
    'patch_size': 16,
    'in_chans': 3,
    'embed_dim': 768,
}

class LinearClassification(nn.Module):
    def __init__(self, config):
        super(LinearClassification, self).__init__()

        self.patch_embed = PatchEmbed(img_size=default_linear['img_size'], patch_size=default_linear['patch_size'], \
                                      in_chans=default_linear['in_chans'], embed_dim=default_linear['embed_dim'])
        self.head = nn.LazyLinear(out_features=config.num_class)

        self.flatten = nn.Flatten(start_dim=-2)
        

    def forward(self, x): # x of size (b, n, h, w)
        B = x.shape[0]
        x = einops.rearrange(x, 'b n h w -> (b n) h w')
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = self.patch_embed(x)
        x = self.flatten(x)
        x = einops.rearrange(x, '(b n) d -> b n d', b=B)
        x = x.flatten(1)
        output = self.head(x)
        return output