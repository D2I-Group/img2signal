import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import models_vit
from . import models_mae
from .utils import download_file
from .swin_transformer import swin_t
from torchvision.models import Swin_T_Weights
from .simmim import build_simmim
import einops
import logging

logger = logging.getLogger(__name__)


class DINOv2Backbone(nn.Module):
    """DINOv2 adapted to the repo's backbone interface.

    Differs from the MAE/ViT backbones in two ways that matter here:
      * weights come from timm, not from the repo's ckpt/ directory
      * timm's forward_features returns (B, N, D); the repo's classification head
        expects a flat (B, N*D), matching models_vit's global_pool=False path
    """
    def __init__(self, img_size=224, name="vit_base_patch14_dinov2.lvd142m"):
        super().__init__()
        import timm
        self.m = timm.create_model(name, pretrained=True, img_size=img_size, num_classes=0)

    def forward_features(self, x):
        return torch.flatten(self.m.forward_features(x), 1)

    def named_parameters(self, *a, **kw):
        return self.m.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.m.parameters(*a, **kw)



class QwenVLBackbone(nn.Module):
    """Qwen2-VL's vision tower behind the repo's backbone interface."""

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
        self.img_size = img_size

    def _patchify(self, x):
        """(B,3,H,W) -> (B*gh*gw, 3*tps*p*p) flattened patches, plus grid_thw."""
        B, C, H, W = x.shape
        p = self.patch
        gh, gw = H // p, W // p
        # (B, C, gh, p, gw, p) -> (B, gh, gw, C, p, p)
        x = x[:, :, : gh * p, : gw * p]
        x = x.reshape(B, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, gh * gw, C, p, p)
        # a still image is repeated across the temporal patch dimension
        x = x.unsqueeze(2).repeat(1, 1, self.tps, 1, 1, 1)      # (B, N, tps, C, p, p)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * gh * gw, C * self.tps * p * p)
        grid = torch.tensor([[1, gh, gw]] * B, device=x.device)
        return x, grid

    def forward_features(self, x):
        B = x.shape[0]
        pv, grid = self._patchify(x)
        out = self.vm_tower(pv.to(self.vm_tower.dtype), grid_thw=grid)   # (B*n_merged, D)
        return out.reshape(B, -1).float()

    def named_parameters(self, *a, **kw):
        return self.vm_tower.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.vm_tower.parameters(*a, **kw)




class SAMBackbone(nn.Module):
    """SAM's image encoder behind the repo's backbone interface.

    Runs at SAM's native 1024x1024: its decomposed relative position embeddings
    are sized for the 64x64 token grid and cannot be interpolated by timm's
    img_size argument.  forward_features returns (B, 256, 64, 64), which we
    flatten to match the all-token convention used by the other backbones.
    """
    def __init__(self, img_size=1024, name="samvit_base_patch16.sa1b"):
        super().__init__()
        import timm
        self.m = timm.create_model(name, pretrained=True, num_classes=0)
        self.img_size = img_size

    def forward_features(self, x):
        return torch.flatten(self.m.forward_features(x), 1)

    def named_parameters(self, *a, **kw):
        return self.m.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.m.parameters(*a, **kw)



class MLCDBackbone(nn.Module):
    """MLCD's vision encoder behind the repo's backbone interface.

    Shipped as a CLIPVisionModel; last_hidden_state is (B, N, D) with the CLS
    token first, which we flatten to match the all-token convention.  Note the
    checkpoint is patch-32, so it yields 50 tokens at 224 against ViT-B/16's 197.
    """
    def __init__(self, img_size=224, name="DeepGlint-AI/mlcd-vit-base-patch32-224"):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name, trust_remote_code=True)

    def forward_features(self, x):
        return torch.flatten(self.m(pixel_values=x).last_hidden_state, 1)

    def named_parameters(self, *a, **kw):
        return self.m.named_parameters(*a, **kw)

    def parameters(self, *a, **kw):
        return self.m.parameters(*a, **kw)


ViT_MODELS = {
    'mlcd': [MLCDBackbone, None],
    'sam': [SAMBackbone, None],
    'dinov2': [DINOv2Backbone, None],
    'qwenvl': [QwenVLBackbone, None],
    'vit_base': [models_vit.vit_base_patch16, "mae_pretrain_vit_base.pth"],
    'vit_large': [models_vit.vit_large_patch16, "mae_pretrain_vit_large.pth"],
    'vit_huge': [models_vit.vit_huge_patch14, "mae_pretrain_vit_huge.pth"],
    'swin_tiny': [swin_t, Swin_T_Weights.DEFAULT],
    'mae_base': [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    'simmim': [build_simmim, "simmim_pretrain__swin_base__img192_window6__800ep.pth"],
}

# Pretrained weights are downloaded into --ckpt on first use.
# SimMIM weights must be downloaded manually from https://github.com/microsoft/SimMIM
DOWNLOAD_URL = {
    "mae_pretrain_vit_base.pth": "https://dl.fbaipublicfiles.com/mae/pretrain/",
    "mae_pretrain_vit_large.pth": "https://dl.fbaipublicfiles.com/mae/pretrain/",
    "mae_pretrain_vit_huge.pth": "https://dl.fbaipublicfiles.com/mae/pretrain/",
    "mae_visualize_vit_base.pth": "https://dl.fbaipublicfiles.com/mae/visualize/",
}


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r update:  y = Wx + (B A x) * a/r."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A), self.B) * self.scaling


def inject_lora(model, r=8, alpha=16, targets=("qkv", "proj")):
    """Replace attention projections with LoRA-wrapped versions. Returns #wrapped."""
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(mod, name, LoRALinear(child, r, alpha))
                n += 1
    return n


class UTSClassification(nn.Module):
    def __init__(self, config):
        super(UTSClassification, self).__init__()
        self.ckpt = config.ckpt
        self.load_ckpt = config.load_ckpt

        if config.backbone == 'simmim':
            self.vm = ViT_MODELS[config.backbone][0](config)
        else:
            self.vm = ViT_MODELS[config.backbone][0]()

        self.head_tokens = getattr(config, 'head_tokens', 'all')
        self.embed_dim = getattr(getattr(self.vm, 'embed_dim', None), 'real', None) or 768
        self.head = nn.LazyLinear(out_features=config.num_class)

        if self.load_ckpt:
            if config.backbone in ('dinov2', 'qwenvl', 'sam', 'mlcd'):
                print(f'{config.backbone}: pretrained weights loaded from its own hub; skipping repo ckpt path')
            elif config.backbone != 'swin_tiny': # vit & mae
                ckpt_name = ViT_MODELS[config.backbone][1]
                ckpt_path = os.path.join(self.ckpt, ckpt_name)
                if not os.path.isfile(ckpt_path):
                    if ckpt_name not in DOWNLOAD_URL:
                        raise FileNotFoundError(f"{ckpt_path} not found; see README for how to download it")
                    download_file(DOWNLOAD_URL[ckpt_name] + ckpt_name, ckpt_path)
                try:
                    print('loading visual model checkpoint from {}'.format(ckpt_path))
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    self.vm.load_state_dict(checkpoint['model'], strict=True)
                except:
                    print("ERROR - " * 10)
                    print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
                    print("ERROR - " * 10)
            else: # swin case
                weight = ViT_MODELS[config.backbone][1]
                self.vm.load_state_dict(weight.get_state_dict(progress=True), strict=True)
                self.vm_transform = weight.transforms()

        finetune_type = config.ft_type
        print(f"Finetune type: {finetune_type}")
        if finetune_type == 'lora':
            for p in self.vm.parameters():
                p.requires_grad = False
            k = inject_lora(self.vm, r=getattr(config, 'lora_r', 8),
                            alpha=getattr(config, 'lora_alpha', 16))
            n_tr = sum(p.numel() for p in self.vm.parameters() if p.requires_grad)
            print(f'LoRA: wrapped {k} projections, {n_tr/1e6:.3f}M trainable backbone params')

        if finetune_type not in ('full', 'lora'):
            for n, param in self.vm.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'full_wo_cls_mask' == finetune_type:
                    param.requires_grad = 'mask_token' not in n and 'cls_token' not in n
                elif 'mlp+ln' == finetune_type: # without attn
                    param.requires_grad = 'mlp' in n or 'norm' in n or 'patch_embed' in n or 'decoder_embed' in n or 'decoder_pred' in n
                elif 'pos_embed' == finetune_type:
                    param.requires_grad = ('pos_embed' in n) and ('decoder' not in n)
                elif 'patch_embed' == finetune_type:
                    param.requires_grad = ('patch_embed' in n) and ('decoder' not in n)
                elif 'pos+patch' == finetune_type:
                    param.requires_grad = (('pos_embed' in n) or ('patch_embed' in n)) and ('decoder' not in n)
                elif 'zero-shot' == finetune_type:
                    param.requires_grad = False

    def _vm_forward(self, x):
        # x of size (B, C, H, W)
        # x = self.vm_transform(x) if self.vm_transform else x
        x = self.vm.forward_features(x)
        return x
        

    def _select_tokens(self, flat):
        """The backbone returns all tokens flattened. Optionally reduce to
        the CLS token or a mean over patch tokens before the head."""
        mode = getattr(self, 'head_tokens', 'all')
        if mode == 'all':
            return flat
        D = getattr(self, 'embed_dim', 768)
        seq = flat.view(flat.shape[0], -1, D)      # (B, n_tok, D)
        if mode == 'cls':
            return seq[:, 0, :]
        if mode == 'mean':
            return seq[:, 1:, :].mean(dim=1)
        raise ValueError(f'unknown head_tokens={mode}')

    def forward(self, x): # x of size (b, n, h, w)
        B = x.shape[0]
        x = einops.rearrange(x, 'b n h w -> (b n) h w')
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        vm_output = self._vm_forward(x) # (b*n, d)
        vm_output = self._select_tokens(vm_output)
        vm_output = einops.rearrange(vm_output, '(b n) d -> b n d', b=B)
        vm_output = vm_output.flatten(1)
        output = self.head(vm_output)
        return output