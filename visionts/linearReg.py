import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from PIL import Image
from .util import safe_resize
from timm.models.vision_transformer import PatchEmbed

default_linear = {
    'img_size': 224,
    'patch_size': 16,
    'in_chans': 3,
    'embed_dim': 768,
}

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

    def forward(self, x):  # x: [bs x nvars x patch_num x d_model]
        x = self.dimension_reduction(x)  
        x_out = []
        x = einops.rearrange(x, 'b n d p -> n b d p')
        for i in range(self.n_vars):
            z = x[i]
            z = self.flattens[0](z)  # z: [bs x d_model * patch_num]
            z = self.linears[0](z)  # z: [bs x target_window]
            z = self.dropouts[0](z)
            x_out.append(z)
        x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        x = einops.rearrange(x, 'b n l -> b l n')
        return x

class LinearRegression(nn.Module):
    def __init__(self, config):
        super(LinearRegression, self).__init__()
        
        self.pad_left = 0
        self.pred_len = config.pred_len
        if config.context_len % config.periodicity != 0:
            self.pad_left = config.periodicity - config.context_len % config.periodicity
        self.periodicity = config.periodicity

        self.patch_num = (default_linear['img_size'] // default_linear['patch_size'])**2

        self.patch_embed = PatchEmbed(img_size=default_linear['img_size'], patch_size=default_linear['patch_size'], \
                                      in_chans=default_linear['in_chans'], embed_dim=default_linear['embed_dim'])

        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[config.interpolation]

        self.resize = safe_resize((default_linear['img_size'], default_linear['img_size']), interpolation=self.interpolation)
        self.head = Flatten_Head(
            n_vars=config.num_features,
            head_dim=config.head_dim,
            pred_len=config.pred_len,
            head_dropout=config.head_dropout,
            vm_dim=default_linear['embed_dim'],
            patch_num=self.patch_num
        )

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _time2img(self, x):
        x = einops.rearrange(x, "b l n -> b n l")
        x_pad = F.pad(x, (self.pad_left, 0), mode='replicate') # [b n l']
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)
        x_resize = self.resize(x_2d)
        x_input = einops.repeat(x_resize, 'b 1 h w -> b c h w', c=3)
        return x_input
        

    
    def forward(self, x, fp64=False): # x of size (b, l, n)
        B = x.shape[0]
        means = x.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        x_enc /= stdev

        x_2d = self._time2img(x_enc) # (bn, c, h, w)
        x_out = self.patch_embed(x_2d)
        x_out = einops.rearrange(x_out, '(b n) p d -> b n p d', b=B)
        x_out = self.head(x_out) # (b, l, n)

        x_out = x_out * (stdev.repeat(1, self.pred_len, 1))
        x_out = x_out + (means.repeat(1, self.pred_len, 1))

        return x_out