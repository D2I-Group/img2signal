
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .mim_swin import SwinTransformer


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        # return_x = x.clone().detach()
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        # return_x_rec = x_rec.clone().detach()

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        # return_mask = mask.clone().detach()
        return x_rec, mask
    
    def forward_features(self, x):
        return self.encoder.forward_features(x)


    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def build_simmim(config):
    model_type = config.mim_model_type
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.mim_image_size,
            patch_size=config.mim_swin_patch_size,
            in_chans=config.mim_swin_in_chans,
            num_classes=0,
            embed_dim=config.mim_swin_embed_dim,
            depths=config.mim_swin_depths,
            num_heads=config.mim_swin_num_heads,
            window_size=config.mim_swin_window_size,
            mlp_ratio=config.mim_swin_mlp_ratio,
            qkv_bias=config.mim_swin_qkv_bias,
            qk_scale=config.mim_swin_qk_scale,
            drop_rate=config.mim_swin_drop_rate,
            drop_path_rate=config.mim_swin_drop_path_rate,
            ape=config.mim_swin_ape,
            patch_norm=config.mim_swin_patch_norm,
            use_checkpoint=config.mim_swin_use_checkpoint)
        encoder_stride = 32
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    return model