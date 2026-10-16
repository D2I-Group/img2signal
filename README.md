<div align="center">

# (TMLR 2026) From Images to Signals: Are Image-Pretrained Vision Models Useful for Time Series Analysis?

[![TMLR](https://img.shields.io/badge/TMLR-2026-1f5fbf.svg)](https://openreview.net/forum?id=aWVG2rN4OV)
[![OpenReview](https://img.shields.io/badge/OpenReview-aWVG2rN4OV-8c1b13.svg)](https://openreview.net/forum?id=aWVG2rN4OV)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

**ChengAo Shen<sup>1</sup>, Hanghang Tong<sup>2</sup>, Dongjin Song<sup>3</sup>, Zhigang Deng<sup>1</sup>, Qingsong Wen<sup>4</sup>, Jingchao Ni<sup>1</sup>**

<sup>1</sup>University of Houston &nbsp; <sup>2</sup>University of Illinois Urbana-Champaign &nbsp; <sup>3</sup>University of Connecticut &nbsp; <sup>4</sup>Squirrel Ai Learning

</div>

## About

This is the official code for the TMLR 2026 paper "[From Images to Signals: Are Image-Pretrained Vision Models Useful for Time Series Analysis?](https://openreview.net/forum?id=aWVG2rN4OV)". We benchmark 4 image-pretrained vision models (ViT, Swin, MAE, SimMIM) with 8 imaging methods on 18 datasets against 21 baselines, for time series classification and forecasting.

<div align="center">
<img src="docs/assets/framework.png" width="100%" alt="Framework">
</div>

**Findings**

- LVMs are effective for classification and achieve the best average accuracy on 10 UEA datasets.
- For forecasting, only self-supervised LVMs with UVH imaging (e.g., MAE) are competitive, and their pretrained decoders matter more than their encoders.
- LVM forecasters are biased toward periodic data and make limited use of long look-back windows.

<div align="center">
<img src="docs/assets/classification.png" width="45%" alt="Classification results">
<img src="docs/assets/periodicity_bias.png" width="45%" alt="Periodicity bias">
</div>

## Quick Start

**Environment** ([uv](https://docs.astral.sh/uv/))

```bash
git clone https://github.com/D2I-Group/img2signal.git
cd img2signal
uv sync
source .venv/bin/activate
```

**Data.** Download the datasets from [Time-Series-Library](https://github.com/thuml/Time-Series-Library) into `./dataset/` (e.g., `./dataset/ETT-small/ETTh1.csv`, `./dataset/UWaveGestureLibrary/`).

**Checkpoints.** MAE and ViT weights are downloaded into `./ckpt/` automatically. Download the [SimMIM](https://github.com/microsoft/SimMIM) checkpoint `simmim_pretrain__swin_base__img192_window6__800ep.pth` into `./ckpt/` manually. For the encoder/decoder ablation, run `python tools/split_checkpoint.py --ckpt_dir ./ckpt`.

**Run**

```bash
bash scripts/classification/UWaveGestureLibrary.sh   # classification
bash scripts/forecasting/ETTh1_mae.sh                # forecasting
```

Scripts for the other experiments are in `scripts/classification/`, `scripts/forecasting/`, `scripts/appendix/`, and `scripts/baselines/`.

## Citation

```bibtex
@article{shen2026images,
  title   = {From Images to Signals: Are Image-Pretrained Vision Models Useful for Time Series Analysis?},
  author  = {Shen, ChengAo and Tong, Hanghang and Song, Dongjin and Deng, Zhigang and Wen, Qingsong and Ni, Jingchao},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=aWVG2rN4OV}
}
```
