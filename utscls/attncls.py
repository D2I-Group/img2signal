import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import logging
from timm.models.vision_transformer import PatchEmbed
from timm.layers import use_fused_attn
from typing import Type

logger = logging.getLogger(__name__)

default_linear = {
    'img_size': 224,
    'patch_size': 16,
    'in_chans': 3,
    'embed_dim': 768,
    'num_heads': 16,
}

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttnClassification(nn.Module):
    def __init__(self, config):
        super(AttnClassification, self).__init__()

        self.patch_embed = PatchEmbed(img_size=default_linear['img_size'], patch_size=default_linear['patch_size'], \
                                      in_chans=default_linear['in_chans'], embed_dim=default_linear['embed_dim'])
        self.head = nn.LazyLinear(out_features=config.num_class)

        self.attn = Attention(
            dim=default_linear['embed_dim'],
            num_heads=default_linear['num_heads']
        )

        self.flatten = nn.Flatten(start_dim=-2)
        

    def forward(self, x): # x of size (b, n, h, w)
        B = x.shape[0]
        x = einops.rearrange(x, 'b n h w -> (b n) h w')
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = self.patch_embed(x) # bn, p, d
        x = self.attn(x)
        x = self.flatten(x)
        x = einops.rearrange(x, '(b n) d -> b n d', b=B)
        x = x.flatten(1)
        output = self.head(x)
        return output