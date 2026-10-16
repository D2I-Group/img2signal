"""Build the encoder-only / decoder-only checkpoints used in RQ8 (`--sep_enc_dec enc|dec`).

    python tools/split_checkpoint.py --ckpt_dir ./ckpt

Reads the pretrained MAE and SimMIM checkpoints from --ckpt_dir and writes:
    mae_visualize_vit_base_wo_decoder.pth   (MAE encoder only)
    mae_visualize_vit_base_wo_encoder.pth   (MAE decoder, with cls/mask tokens and positional embeddings)
    simmim_wo_decoder.pth                   (SimMIM encoder only)
    simmim_wo_encoder.pth                   (SimMIM decoder, with encoder.mask_token)
"""
import argparse
import os

import torch

MAE_DECODER = ('decoder_blocks.', 'decoder_embed.', 'decoder_norm.', 'decoder_pred.')
MAE_ENCODER = ('blocks.', 'norm.', 'patch_embed.')


def split(src, dst, drop):
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if not drop(k)}
    torch.save(ckpt, dst)
    print(f'{dst}: kept {len(ckpt["model"])} tensors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='./ckpt')
    d = parser.parse_args().ckpt_dir

    mae = os.path.join(d, 'mae_visualize_vit_base.pth')
    split(mae, os.path.join(d, 'mae_visualize_vit_base_wo_decoder.pth'), lambda k: k.startswith(MAE_DECODER))
    split(mae, os.path.join(d, 'mae_visualize_vit_base_wo_encoder.pth'), lambda k: k.startswith(MAE_ENCODER))

    simmim = os.path.join(d, 'simmim_pretrain__swin_base__img192_window6__800ep.pth')
    split(simmim, os.path.join(d, 'simmim_wo_decoder.pth'), lambda k: k.startswith('decoder.'))
    split(simmim, os.path.join(d, 'simmim_wo_encoder.pth'),
          lambda k: k.startswith('encoder.') and k != 'encoder.mask_token')
