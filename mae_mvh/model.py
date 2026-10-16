import torch
import os
from . import models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

class MAE_MVH(nn.Module):
    """
    MAE model specifed for MVH imaging for RQ3
    """

    def __init__(self, context_len, pred_len, n_vars, norm_const, align_const, interpolation,\
                 arch, finetune_type, ckpt_dir, load_ckpt):
        super(MAE_MVH, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        self.vision_model = MAE_ARCH[arch][0]()

        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.vision_model.load_state_dict(checkpoint['model'], strict=True)
            except:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
        
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n
                else:
                    raise ValueError(f'unknown --ft_type {finetune_type}')

        self.image_size = self.vision_model.patch_embed.img_size[0] # 192
        self.patch_size = self.vision_model.patch_embed.patch_size[0] # 4
        self.num_patch = self.image_size // self.patch_size # 48

        self.context_len = context_len # 336
        self.pred_len = pred_len # 96
        self.n_vars = n_vars
        
        input_ratio = self.context_len / (self.context_len + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const) # 13
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input # 48-13=35
        adjust_input_ratio = self.num_patch_input / self.num_patch # 13/48

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = self.context_len / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.n_vars, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const
        
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()

    def forward(self, x, fp64=False): # x: b l n
        means = x.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev

        x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]
        x_resize = self.input_resize(x_enc).unsqueeze(1) # [bs x 1 x nvars x seq_len]
        masked = torch.zeros((x_resize.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_resize.device, dtype=x_resize.dtype)

        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)


        _, y, mask = self.vision_model(
            image_input, 
            mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
        )

        image_reconstructed = self.vision_model.unpatchify(y) # b * 3 * h * w
        
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) 
        y_segmentations = self.output_resize(y_grey) # b, 1, n, l

        y_flatten = einops.rearrange(
            y_segmentations, 
            'b 1 n l -> b l n', 
            b=x_enc.shape[0]
        )

        y = y_flatten[:, self.context_len: self.context_len + self.pred_len, :] # extract the forecasting window

        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        return y