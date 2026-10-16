import torch
import os
from . import models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util
from .util import safe_resize

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

class Flatten_Head(nn.Module):
    def __init__(
        self,
        n_vars,
        head_dim,
        pred_len,
        head_dropout=0,
        vm_dim=768,
        patch_num=197
    ):
        super().__init__()

        self.n_vars = n_vars

        self.dimension_reduction = nn.Linear(vm_dim, head_dim)

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()
        for i in range(self.n_vars):
            self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(head_dim * patch_num, pred_len))
            self.dropouts.append(nn.Dropout(head_dropout))

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.dimension_reduction(x)  
        x_out = []
        x = einops.rearrange(x, 'b n d p -> n b d p')
        for i in range(self.n_vars):
            z = x[i]
            z = self.flattens[i](z)  # z: [bs x d_model * patch_num]
            z = self.linears[i](z)  # z: [bs x target_window]
            z = self.dropouts[i](z)
            x_out.append(z)
        x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        x = einops.rearrange(x, 'b n l -> b l n')
        return x
    
class IMG_MAE(nn.Module):
    """
    MAE model specified for non-heatmap time series images. 
    Encoder + Linear head with no decoder reconstruction.
    """
    def __init__(self, pred_len, num_features, head_dim, head_dropout, vm_dim, interpolation,\
                 arch, finetune_type, ckpt_dir, load_ckpt):
        super(IMG_MAE, self).__init__()

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

        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.resize = safe_resize(self.vision_model.patch_embed.img_size, interpolation=self.interpolation)

        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.head = Flatten_Head(
            n_vars=num_features,
            head_dim=head_dim,
            pred_len=pred_len,
            head_dropout=head_dropout,
            vm_dim=vm_dim,
            patch_num=self.num_patch * self.num_patch + 1
        )

    def _vm_forward(self, x):
        # x of size (B, C, H, W)
        x_2d = einops.rearrange(x, 'b n h w -> (b n) 1 h w')
        x_resize = self.resize(x_2d)
        x_resize = einops.repeat(x_resize, 'b 1 h w -> b c h w', c=3)
        x = self.vision_model.forward_features(x_resize)
        return x
    
    def forward(self, x): # x of size (b, n, h, w)
        B = x.shape[0]
        x = self._vm_forward(x) # (bn, image_patch, vm_dim)

        x = einops.rearrange(x, "(b n) p d -> b n p d", b=B)
        x = self.head(x) # b l n
        return x