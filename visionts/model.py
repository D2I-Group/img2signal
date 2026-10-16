import torch
import os
from . import models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util
import time

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"


class _LoRALinearTSF(torch.nn.Module):
    """Frozen nn.Linear plus a trainable rank-r update: y = Wx + (B A x) * alpha/r."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.A = torch.nn.Linear(base.in_features, r, bias=False)
        self.B = torch.nn.Linear(r, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale


def _inject_lora_tsf(model, r=8, alpha=16):
    """Wrap every attention qkv / proj Linear in the encoder blocks."""
    n = 0
    for blk in getattr(model, 'blocks', []):
        attn = getattr(blk, 'attn', None)
        if attn is None:
            continue
        for name in ('qkv', 'proj'):
            mod = getattr(attn, name, None)
            if isinstance(mod, torch.nn.Linear):
                setattr(attn, name, _LoRALinearTSF(mod, r, alpha))
                n += 1
    return n


class VisionTS(nn.Module):

    def __init__(self, context_len, pred_len, periodicity, norm_const, align_const, interpolation,\
                 arch, finetune_type, ckpt_dir, load_ckpt, sep_enc_dec=None):
        super(VisionTS, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        self.vision_model = MAE_ARCH[arch][0]()

        if sep_enc_dec:
            print(f"Using separate {sep_enc_dec} checkpoint")
            enc_dic = {'enc': 'mae_visualize_vit_base_wo_decoder.pth', 
                       'dec': 'mae_visualize_vit_base_wo_encoder.pth'}
            ckpt_path = os.path.join(ckpt_dir, enc_dic[sep_enc_dec])
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            incompatible_keys = self.vision_model.load_state_dict(checkpoint['model'], strict=False)
            print("Missing keys:", incompatible_keys.missing_keys)
            print("Unexpected keys:", incompatible_keys.unexpected_keys)
        else:
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
                elif 'lora' == finetune_type:
                    param.requires_grad = False
                    
        if finetune_type == 'lora':
            nwrapped = _inject_lora_tsf(self.vision_model)
            tr = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in self.vision_model.parameters())
            print(f'LoRA: wrapped {nwrapped} projections, trainable {tr} / {tot} = {100.0*tr/tot:.2f}%')
        print(f"Finetune type: {finetune_type}")

        self.image_size = self.vision_model.patch_embed.img_size[0] # 192
        self.patch_size = self.vision_model.patch_embed.patch_size[0] # 4
        self.num_patch = self.image_size // self.patch_size # 48

        self.context_len = context_len # 336
        self.pred_len = pred_len # 96
        self.periodicity = periodicity # 24 

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
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
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const
        self.norm_mode = 'instance'
        
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()

    def forward(self, x, fp64=False):
        # cur_time = time.time()
        # 'instance' (default) applies RevIN-style per-instance, per-variate
        # normalization on top of the loader's dataset-level StandardScaler.
        # 'dataset' skips it, leaving only the dataset-level scaling.
        if getattr(self, 'norm_mode', 'instance') == 'dataset':
            means = torch.zeros_like(x[:, :1, :])
            x_enc = x
            stdev = torch.ones_like(x[:, :1, :])
        else:
            means = x.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
            x_enc = x - means
            stdev = torch.sqrt(
                torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate') # [b n s]
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity) # (bs * nvars, 1, period, patch)

        # resize_init = time.time()
        x_resize = self.input_resize(x_2d)
        # resize_end = time.time() - resize_init
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)


        _, y, mask = self.vision_model(
            image_input, 
            mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
        )

        image_reconstructed = self.vision_model.unpatchify(y) # [(bs x nvars) x 3 x h x w]
        
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # [(bs x nvars) x 1 x h x w]

        resize_init = time.time()
        y_segmentations = self.output_resize(y_grey)
        # resize_end = resize_end + time.time() - resize_init
        # print(f"Resize time: {resize_end:.4f}s")
        y_flatten = einops.rearrange(
            y_segmentations, 
            '(b n) 1 f p -> b (p f) n', 
            b=x_enc.shape[0], f=self.periodicity
        )
        y = y_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window

        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        # print(f"Total time: {time.time() - cur_time:.4f}s")
        return y