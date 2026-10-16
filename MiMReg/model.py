import torch
import os
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util

from .simmim import build_simmim
from torchvision.utils import save_image

class MiMreg(nn.Module):

    def __init__(self, context_len, pred_len, periodicity, norm_const, align_const, interpolation,\
                 finetune_type, load_ckpt, args):
        super(MiMreg, self).__init__()

        self.vision_model = build_simmim(args)

        if args.sep_enc_dec:
            print(f"Using separate {args.sep_enc_dec} checkpoint")
            enc_dic = {'enc': 'simmim_wo_decoder.pth', 
                       'dec': 'simmim_wo_encoder.pth'}
            ckpt_path = os.path.join(args.ckpt, enc_dic[args.sep_enc_dec])
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            incompatible_keys = self.vision_model.load_state_dict(checkpoint['model'], strict=False)
            print("Missing keys:", incompatible_keys.missing_keys)
            print("Unexpected keys:", incompatible_keys.unexpected_keys)
        else:
            if load_ckpt:
                ckpt_path = os.path.join(args.ckpt, args.mim_ckpt)
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

        self.image_size = self.vision_model.encoder.patch_embed.img_size[0] # 192
        self.patch_size = self.vision_model.encoder.patch_embed.patch_size[0] # 4
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
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const
        
        mask = torch.ones((self.num_patch, self.num_patch)).to(next(self.vision_model.parameters()).device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float())
        self.mask_ratio = torch.mean(mask).item()

    def forward(self, x, fp64=False):
        means = x.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate') # [b n s]
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity) # (bs * nvars, 1, period, patch)

        x_resize = self.input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)


        image_reconstructed, mask = self.vision_model(
            image_input, 
            mask=einops.repeat(self.mask, 'h w -> n h w', n=image_input.shape[0])
        )

        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # [(bs x nvars) x 1 x h x w]
        y_segmentations = self.output_resize(y_grey)
        y_flatten = einops.rearrange(
            y_segmentations, 
            '(b n) 1 f p -> b (p f) n', 
            b=x_enc.shape[0], f=self.periodicity
        )
        y = y_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window

        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        return y