<div align="center">

# From Images to Signals: Are Image-Pretrained Vision Models Useful for Time Series Analysis?

### TMLR 2026, Featured Certification

[![TMLR 2026 · Featured Certification](docs/assets/badges/tmlr-featured.svg)](https://openreview.net/forum?id=aWVG2rN4OV)
[![OpenReview](docs/assets/badges/openreview.svg)](https://openreview.net/forum?id=aWVG2rN4OV)
[![Project Page](docs/assets/badges/project.svg)](https://d2i-group.github.io/img2signal/)
[![Video](docs/assets/badges/video.svg)](https://www.youtube.com/watch?v=Ykoex0ULap8)
<br>
![Task: Classification | Forecasting](docs/assets/badges/task.svg)
![PyTorch 2.6](docs/assets/badges/pytorch.svg)
![PRs Welcome](docs/assets/badges/prs.svg)
[![License: MIT](docs/assets/badges/license.svg)](LICENSE)

**ChengAo Shen<sup>1</sup>, Hanghang Tong<sup>2</sup>, Dongjin Song<sup>3</sup>, Zhigang Deng<sup>1</sup>, Qingsong Wen<sup>4</sup>, Jingchao Ni<sup>1</sup>**

<sup>1</sup>University of Houston &nbsp; <sup>2</sup>University of Illinois Urbana-Champaign &nbsp; <sup>3</sup>University of Connecticut &nbsp; <sup>4</sup>Squirrel Ai Learning

🔍 [About](#about) · 🧩 [Framework](#framework) · 🚀 [Quick Start](#quick-start) · 📊 [Findings](#findings) · 🔗 [Citation](#citation)

</div>

<a id="about"></a>
## 🔍 About

This is the official repository for the TMLR 2026 paper "[From Images to Signals: Are Image-Pretrained Vision Models Useful for Time Series Analysis?](https://openreview.net/forum?id=aWVG2rN4OV)", which received a **TMLR Featured Certification**. The project page is at [d2i-group.github.io/img2signal](https://d2i-group.github.io/img2signal/).

Image-pretrained large vision models (LVMs) are increasingly used for time series: a series is first *imaged*, then fed into a vision backbone. This work is a benchmark study that asks not only **whether** LVMs work for time series, but also **why** they succeed or fall short, on both a high-level task (classification) and a low-level task (forecasting).

> In this work, "LVM" refers to image-pretrained models with standalone vision backbones. Vision-language and multimodal foundation models are out of scope.

### 🔑 Key Features

- **Large-scale benchmark:** 4 LVMs (ViT, Swin, MAE, SimMIM), 8 imaging methods, 18 datasets, and 21 baselines, with additional backbones (DINOv2, MLCD, LaVin-DiT, and the Qwen-VL vision encoder) in the appendix.
- **Ten research questions:** covering pretraining type, imaging method, fine-tuning strategy, architecture, temporal dependency, computational cost, model components, inductive bias, and look-back windows.
- **Practical guidance:** which LVM, imaging method, and fine-tuning strategy to use for each task, and where current LVM forecasters fall short.

<a id="framework"></a>
## 🧩 Framework

<div align="center">

| [<img src="docs/assets/framework.png" width="100%" alt="Framework overview">](docs/assets/framework.png) |
|:--:|
| **Figure 1.** (a) The eight imaging methods. (b) LVM-based classification with a linear head. (c) LVM-based forecasting with a linear head. (d) LVM encoder-decoder forecasting, which masks the future patches and reconstructs them with the pretrained decoder. |

</div>

The study is organized around ten research questions (RQs) in three groups:

<div align="center">

| [<img src="docs/assets/research_questions.png" width="75%" alt="Research questions">](docs/assets/research_questions.png) |
|:--:|
| **Figure 2.** Structure of the experiments. |

</div>

<a id="quick-start"></a>
## 🚀 Quick Start

### 1. Environment

We use [uv](https://docs.astral.sh/uv/) to manage the environment (PyTorch 2.6.0 with CUDA 12.4):

```bash
git clone https://github.com/D2I-Group/img2signal.git
cd img2signal
uv sync                      # creates .venv from pyproject.toml / uv.lock
source .venv/bin/activate    # or prefix commands with `uv run`
```

Alternatively, `pip install -r requirements.txt`.

Experiments in the paper used PyTorch 2.6.0 and CUDA 12.4 on NVIDIA RTX 6000 Ada GPUs.

### 2. Data

Download the datasets from [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and place them under `./dataset/`:

```
dataset/
├── ETT-small/            # ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
├── weather/  illness/  traffic/  electricity/
└── EthanolConcentration/ FaceDetection/ Handwriting/ Heartbeat/ JapaneseVowels/
    PEMS-SF/ SelfRegulationSCP1/ SelfRegulationSCP2/ SpokenArabicDigits/ UWaveGestureLibrary/   # UEA (.ts)
```

### 3. Pretrained checkpoints

The first three checkpoints are stored under `./ckpt/`; the others are cached by torchvision, timm, or Hugging Face.

| Backbone | Checkpoint | Source |
|---|---|---|
| `vit_base` | `mae_pretrain_vit_base.pth` | [MAE](https://github.com/facebookresearch/mae), downloaded automatically |
| `mae_base` | `mae_visualize_vit_base.pth` | [MAE](https://github.com/facebookresearch/mae), downloaded automatically |
| `simmim` | `simmim_pretrain__swin_base__img192_window6__800ep.pth` | [SimMIM](https://github.com/microsoft/SimMIM), download manually |
| `swin_tiny` | `Swin_T_Weights.DEFAULT` | torchvision, downloaded automatically |
| `dinov2` (Appendix) | `vit_base_patch14_dinov2.lvd142m` | timm, downloaded automatically |
| `sam` (Appendix) | `samvit_base_patch16.sa1b` | timm, downloaded automatically |
| `mlcd` (Appendix) | `DeepGlint-AI/mlcd-vit-base-patch32-224` | Hugging Face, downloaded automatically |
| `qwenvl` (Appendix) | `Qwen/Qwen2-VL-2B-Instruct` (vision encoder) | Hugging Face, downloaded automatically |

For the encoder/decoder ablation (RQ8), create the split checkpoints once:

```bash
python tools/split_checkpoint.py --ckpt_dir ./ckpt
```

### 4. Run

All experiments go through `run.py`. For example:

```bash
# Classification: ViT with GAF imaging
bash scripts/classification/UWaveGestureLibrary.sh

# Forecasting: MAE with UVH imaging
bash scripts/forecasting/ETTh1_mae.sh
```

Edit `CUDA_VISIBLE_DEVICES` at the top of each script to choose a GPU.

| Paper | Classification scripts | Forecasting scripts |
|---|---|---|
| RQ1–RQ3 (LVMs, imaging, SOTA) | `scripts/classification/*.sh`, `mae/`, `simmim/` | `scripts/forecasting/*.sh`, `swin/`, `RQ3/` (imaging methods) |
| RQ4 (fine-tuning) | `RQ4/` | `RQ4/` |
| RQ5 (architecture) | `RQ5/` | `RQ5/` |
| RQ6 (temporal perturbation) | `RQ6/` | `RQ6/` |
| RQ7 (cost) | `RQ10/` | `RQ10/` |
| RQ8 (encoder vs. decoder) | – | `RQ7/` |
| RQ9 (UVH segment length) | – | `RQ8/` |
| RQ10 (look-back window) | – | `RQ9/` |
| Baselines | `scripts/baselines/` | `scripts/baselines/` |

#### Appendix experiments

Run from the repository root, e.g. `GPU=0 bash scripts/appendix/lora.sh`.

| Appendix section | Script / tool |
|---|---|
| Wilcoxon signed-rank tests per horizon | `python tools/wilcoxon_per_horizon.py` |
| Computational cost (RQ7) | `python tools/profile_cost.py`, `python tools/imaging_memory.py` |
| Token selection for the classification head | `scripts/appendix/head_tokens.sh` |
| Datasets with weak periodicity (Exchange, Solar) | `scripts/appendix/weak_periodicity.sh` (Solar: run `tools/prepare_solar.py` first) |
| Comparison with SOTA LVMs (DINOv2, MLCD, Qwen2-VL, SAM) | `scripts/appendix/backbones_classification.sh`, `backbones_forecasting.sh` |
| Additional fine-tuning (LoRA, positional encoding, patch embedding) | `scripts/appendix/lora.sh`, `finetune_embeddings.sh` |
| Robustness of MVH to variate permutation | `scripts/appendix/mvh_permutation.sh` |
| Normalization methods | `scripts/appendix/normalization.sh` |


<a id="findings"></a>
## 📊 Findings

### RQ1–RQ3: How do LVMs perform?

**Classification.** With GAF imaging, both the supervised ViT and the self-supervised MAE achieve the best average accuracy on 10 UEA datasets, ahead of all 19 non-LVM baselines.

<div align="center">

| [<img src="docs/assets/classification.png" width="70%" alt="Classification results">](docs/assets/classification.png) |
|:--:|
| **Figure 5.** Average accuracy (%) on 10 UEA datasets. |

</div>

**Forecasting.** Only the self-supervised MAE with UVH imaging is competitive. It wins 9 of 16 dataset-metric cases, while the classification-pretrained ViT is roughly on par with DLinear.

<div align="center">

| [<img src="docs/assets/table_forecasting.png" width="100%" alt="Forecasting results">](docs/assets/table_forecasting.png) |
|:--:|
| **Table 2.** Forecasting results averaged over four horizons. |

</div>

**Imaging methods.** The best imaging method depends on the task: GAF for classification, and UVH (which keeps raw values in pixels) for forecasting.

<div align="center">

| [<img src="docs/assets/imaging_methods.png" width="100%" alt="Imaging method ranks">](docs/assets/imaging_methods.png) |
|:--:|
| **Figure 4.** Average ranks of the eight imaging methods (lower is better). |

</div>

### RQ4–RQ7: Are LVMs truly useful?

| RQ | Question | Finding |
|:--:|---|---|
| RQ4 | Are pretrained parameters useful? | Yes. A frozen LVM beats training from scratch on every classification dataset. Fine-tuning MLP + norm layers is enough for classification; fine-tuning only norm layers works best for forecasting. |
| RQ5 | Is the architecture overly complex? | No. Pretrained LVMs outperform simpler models built from the same components. |
| RQ6 | Do LVMs use temporal order? | Yes. Shuffling or masking time steps hurts LVMs more than simpler models. |
| RQ7 | What does it cost? | Inference is about 4× faster than the best classification baseline, but about 20× slower than the best forecasting baseline. |

### RQ8–RQ10: Why is forecasting hard for LVMs?

**Pretrained decoders matter more than encoders.** Re-initializing the decoder hurts more than re-initializing the encoder, even for SimMIM, whose decoder is a single linear layer. This also explains why supervised LVMs, which lack a pixel decoder, struggle with forecasting.

<div align="center">

| [<img src="docs/assets/encoder_decoder.png" width="60%" alt="Encoder vs decoder ablation">](docs/assets/encoder_decoder.png) |
|:--:|
| **Figure 8.** Forecasting performance drop (%) when only the encoder (blue) or only the decoder (red) keeps its pretrained parameters. |

</div>

**A bias toward periodicity.** MAE with UVH forecasts by "combining past periods". Its error follows the number of segments $n = k/\gcd(i, k)$ before a segment reoccurs (Lemma 4.1), closely on strongly periodic data and less so on weakly periodic data.

<div align="center">

| [<img src="docs/assets/periodicity_bias.png" width="60%" alt="Periodicity bias">](docs/assets/periodicity_bias.png) |
|:--:|
| **Figure 10.** Normalized MSE of MAE vs. UVH segment length; the green curve is $n$. |

</div>

**Limited use of long look-back windows.** Error decreases up to roughly 500–1500 steps and then plateaus or rises, as fixed-size images limit how much of a long series can be encoded.

<div align="center">

| [<img src="docs/assets/lookback.png" width="90%" alt="Look-back window">](docs/assets/lookback.png) |
|:--:|
| **Figure 11.** MSE of MAE (UVH) with varying look-back window lengths. |

</div>

### ✅ Recommended Configurations

| | Classification | Forecasting |
|---|---|---|
| **Model** | ViT or MAE | MAE (self-supervised) with its pretrained decoder |
| **Imaging** | GAF | UVH |
| **Fine-tuning** | MLP and norm layers | Norm layers only |
| **Best suited to** | Pattern-based tasks | Periodic data; look-back up to about 1500 steps |

<a id="citation"></a>
## 🔗 Citation

If you find this work useful, please consider citing:

```bibtex
@article{shen2026images,
  title   = {From Images to Signals: Are Image-Pretrained Vision Models Useful for Time Series Analysis?},
  author  = {Shen, ChengAo and Tong, Hanghang and Song, Dongjin and Deng, Zhigang and Wen, Qingsong and Ni, Jingchao},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=aWVG2rN4OV},
  note    = {Featured Certification}
}
```

## 📚 Related Work from D2I Group

- [Awesome Vision for Time Series](https://github.com/D2I-Group/awesome-vision-time-series): Harnessing Vision Models for Time Series Analysis: A Survey (IJCAI 2025)
- [DMMV](https://github.com/D2I-Group/dmmv): Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting (NeurIPS 2025)
