import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import models_vit
from .utils import download_file
from .utils import safe_resize
from .RevIN import RevIN
import einops
from PIL import Image
from .swin_transformer import swin_t
from torchvision.models import Swin_T_Weights


class DINOv2RegBackbone(nn.Module):
    """DINOv2 behind the VisionReg backbone interface (Fig. 1(c) linear-head TSF)."""

    def __init__(self, img_size=224, name="vit_base_patch14_dinov2.lvd142m"):
        super().__init__()
        import timm
        self.m = timm.create_model(name, pretrained=True, img_size=img_size, num_classes=0)
        # UTSRegression reads .patch_embed.img_size / .patch_size off the backbone
        self.patch_embed = self.m.patch_embed

    def forward_features(self, x):
        # (B, P, D) with the CLS token first -- matches patch_num = n*n + 1
        return self.m.forward_features(x)

    def named_parameters(self, *a, **kw):
        return self.m.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.m.parameters(*a, **kw)



class _PatchEmbedShim(nn.Module):
    """Carries the two attributes UTSRegression reads off a backbone."""
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)


class QwenVLRegBackbone(nn.Module):
    """Qwen2-VL's vision tower behind the VisionReg backbone interface."""

    def __init__(self, img_size=224, name="Qwen/Qwen2-VL-2B-Instruct"):
        super().__init__()
        from transformers import Qwen2VLForConditionalGeneration
        full = Qwen2VLForConditionalGeneration.from_pretrained(name, dtype=torch.float32)
        self.vm_tower = full.visual
        del full
        cfg = self.vm_tower.config
        self.patch = getattr(cfg, "patch_size", 14)
        self.tps = getattr(cfg, "temporal_patch_size", 2)
        self.merge = getattr(cfg, "spatial_merge_size", 2)
        # effective stride after the spatial merge -- this is what sets token count
        self.patch_embed = _PatchEmbedShim(img_size, self.patch * self.merge)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch
        gh, gw = H // p, W // p
        x = x[:, :, : gh * p, : gw * p]
        x = x.reshape(B, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, gh * gw, C, p, p)
        x = x.unsqueeze(2).repeat(1, 1, self.tps, 1, 1, 1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * gh * gw, C * self.tps * p * p)
        grid = torch.tensor([[1, gh, gw]] * B, device=x.device)
        return x, grid

    def forward_features(self, x):
        B = x.shape[0]
        pv, grid = self._patchify(x)
        out = self.vm_tower(pv.to(self.vm_tower.dtype), grid_thw=grid)  # (B*n_merged, D)
        return out.reshape(B, -1, out.shape[-1]).float()                # (B, P, D)

    def named_parameters(self, *a, **kw):
        return self.vm_tower.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.vm_tower.parameters(*a, **kw)



class MLCDRegBackbone(nn.Module):
    """MLCD behind the VisionReg backbone interface (Fig. 1(c) linear-head TSF)."""

    def __init__(self, img_size=224, name="DeepGlint-AI/mlcd-vit-base-patch32-224"):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name, trust_remote_code=True)
        patch = getattr(self.m.config, "patch_size", 32)
        self.patch_embed = _PatchEmbedShim(img_size, patch)

    def forward_features(self, x):
        # (B, P, D) with the CLS token first -- matches patch_num = n*n + 1
        return self.m(pixel_values=x).last_hidden_state

    def named_parameters(self, *a, **kw):
        return self.m.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.m.parameters(*a, **kw)



class SAMRegBackbone(nn.Module):
    """SAM image encoder behind the VisionReg backbone interface.

    At img_size=224 the 1024-pretrained pos_embed (1, 64, 64, D) and the
    decomposed rel_pos tables (127, d) are interpolated to the 14x14 grid
    (27, d).  Output is the neck feature map, so vm_dim=256 and there is no
    CLS token.
    """

    def __init__(self, img_size=224, name="samvit_base_patch16.sa1b"):
        super().__init__()
        import timm
        if img_size == 1024:
            self.m = timm.create_model(name, pretrained=True, num_classes=0)
        else:
            donor = timm.create_model(name, pretrained=True, num_classes=0)
            self.m = timm.create_model(name, pretrained=False, num_classes=0, img_size=img_size)
            sd, tgt, new = donor.state_dict(), self.m.state_dict(), {}
            for k, v in sd.items():
                if k not in tgt:
                    continue
                t = tgt[k]
                if v.shape == t.shape:
                    new[k] = v
                elif k == "pos_embed":
                    vv = F.interpolate(v.permute(0, 3, 1, 2), size=tuple(t.shape[1:3]),
                                       mode="bicubic", align_corners=False)
                    new[k] = vv.permute(0, 2, 3, 1)
                elif k.endswith("rel_pos_h") or k.endswith("rel_pos_w"):
                    vv = F.interpolate(v.transpose(0, 1).unsqueeze(0), size=t.shape[0],
                                       mode="linear", align_corners=False)
                    new[k] = vv.squeeze(0).transpose(0, 1)
                else:
                    new[k] = t
            bad = self.m.load_state_dict(new, strict=False)
            print("sam: interpolated 1024 checkpoint to img_size=%d; missing=%d unexpected=%d"
                  % (img_size, len(bad.missing_keys), len(bad.unexpected_keys)))
            del donor
        self.patch_embed = _PatchEmbedShim(img_size, 16)

    def forward_features(self, x):
        o = self.m.forward_features(x)        # (B, 256, g, g)
        return o.flatten(2).transpose(1, 2)   # (B, g*g, 256)

    def named_parameters(self, *a, **kw):
        return self.m.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.m.parameters(*a, **kw)


ViT_MODELS = {
    'sam': [SAMRegBackbone, None],
    'mlcd': [MLCDRegBackbone, None],
    'qwenvl': [QwenVLRegBackbone, None],
    'dinov2': [DINOv2RegBackbone, None],
    'vit_base': [models_vit.vit_base_patch16, "mae_pretrain_vit_base.pth"],
    'vit_large': [models_vit.vit_large_patch16, "mae_pretrain_vit_large.pth"],
    'vit_huge': [models_vit.vit_huge_patch14, "mae_pretrain_vit_huge.pth"],
    'swin_tiny': [swin_t, Swin_T_Weights.DEFAULT],
}

DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/"


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

class UTSRegression(nn.Module):
    def __init__(self, config):
        super(UTSRegression, self).__init__()
        self.ckpt = config.ckpt
        self.load_ckpt = config.load_ckpt
        self.vm = ViT_MODELS[config.backbone][0]()
        self.rev_in = RevIN(config.num_features)

        if config.backbone in ('dinov2', 'qwenvl', 'mlcd', 'sam'):
            print(f'{config.backbone}: pretrained weights loaded from its own hub; skipping repo ckpt path')
        elif self.load_ckpt:
            if config.backbone != 'swin_tiny': # vit case
                ckpt_path = os.path.join(self.ckpt, ViT_MODELS[config.backbone][1])
                if not os.path.isfile(ckpt_path):
                    remote_url = DOWNLOAD_URL + ViT_MODELS[config.backbone][1]
                    download_file(remote_url, ckpt_path)
                try:
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    self.vm.load_state_dict(checkpoint['model'], strict=True)
                except:
                    print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
            else: # swin case
                weight = ViT_MODELS[config.backbone][1]
                self.vm.load_state_dict(weight.get_state_dict(progress=True), strict=True)
                self.vm_transform = weight.transforms()
        
        self.pad_left = 0
        if config.context_len % config.periodicity != 0:
            self.pad_left = config.periodicity - config.context_len % config.periodicity
        self.periodicity = config.periodicity

        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[config.interpolation]

        self.resize = safe_resize(self.vm.patch_embed.img_size, interpolation=self.interpolation)

        self.image_size = self.vm.patch_embed.img_size[0]
        self.patch_size = self.vm.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size


        if config.backbone == 'swin_tiny':
            self.head = Flatten_Head(
                n_vars=config.num_features,
                head_dim=config.head_dim,
                pred_len=config.pred_len,
                head_dropout=config.head_dropout,
                vm_dim=config.vm_dim,
                patch_num=self.vm.window_size[0] * self.vm.window_size[1]
            )
        else:
            self.head = Flatten_Head(
                n_vars=config.num_features,
                head_dim=config.head_dim,
                pred_len=config.pred_len,
                head_dropout=config.head_dropout,
                vm_dim=config.vm_dim,
                patch_num=self.num_patch * self.num_patch + (0 if config.backbone in ('qwenvl', 'sam') else 1)
            )

    def _vm_forward(self, x):
        # x of size (B, C, H, W)
        x = self.vm.forward_features(x)
        return x
    
    def _time2img(self, x):
        x = einops.rearrange(x, "b l n -> b n l")
        x_pad = F.pad(x, (self.pad_left, 0), mode='replicate') # [b n l']
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)
        x_resize = self.resize(x_2d)
        x_input = einops.repeat(x_resize, 'b 1 h w -> b c h w', c=3)
        return x_input
        

    
    def forward(self, x): # x of size (b, l, n)
        B = x.shape[0]
        # first nomalize
        x = self.rev_in(x, mode="norm")
        # then resize and interpolate x to 224x224, copy 3 channels
        x = self._time2img(x) # (bn, c, h, w)
        # then forward x to the vision model
        x = self._vm_forward(x) # (bn, image_patch, vm_dim)
        x = einops.rearrange(x, "(b n) p d -> b n p d", b=B)
        # finally use a linear layer to regress the output
        x = self.head(x)
        # denormalize x (b, l, n)
        x = self.rev_in(x, mode="denorm")
        return x