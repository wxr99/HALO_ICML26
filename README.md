# HALO: Hierarchical Adaptive Learning with Organized Prototypes

[![arXiv](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **HALO**, presented at ICML 2026.

> **Online Continual Learning with Dynamic Label Hierarchies**
>
> Xinrui Wang, Bartłomiej Twardowski, Alexandra Gomez-Villa, Shao-Yuan Li, Songcan Chen
>
> *International Conference on Machine Learning (ICML), 2026*

## Overview

HALO addresses **Dynamic Hierarchical Online Continual Learning (DHOCL)**, where data arrives in a stream with labels at arbitrary hierarchical granularities, and the taxonomy itself evolves over time. Unlike standard continual learning that assumes flat label spaces, DHOCL reflects real-world scenarios such as biodiversity monitoring, where new species are discovered and taxonomic relationships are continuously revised.

HALO is built on two key components:
- **Hierarchical Prototype Regularization (HPR)**: enforces alignment between feature representations and the evolving taxonomy tree through learnable prototypes, with hierarchical structure alignment and temporal saliency consistency.
- **Prediction-Level Adaptive Ensemble (PredLA)**: dynamically balances predictions from multiple classification heads at different hierarchy levels, reconciling rapid adaptation with long-term stability.

## Key Results

HALO consistently outperforms replay-based and regularization-based baselines across four benchmarks (CIFAR-100, FGVC-Aircraft, CUB-200, iNaturalist) and ImageNet-H, achieving:
- Higher accuracy across all hierarchy levels (AAUC, FAUC)
- Lower mistake severity (MS), with errors occurring at semantically closer nodes
- Robustness to backbone choice (ViT, ResNet) and pretraining method (DINO, CLIP, MAE, Supervised)

## Installation

```bash
git clone https://github.com/wxr99/HALO_ICML26.git
cd HALO_ICML26
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.13+, torchvision, timm, numpy, scipy, scikit-learn.

## Quick Start

### Data Preparation

Datasets are downloaded automatically via torchvision/timm where available. For iNaturalist and ImageNet-H, follow the instructions in `data/README.md`.

### Training

```bash
# CIFAR-100 with DHOCL setup (10 splits, default config)
python main.py --dataset cifar100 --num_tasks 10 --buffer_size 1000

# iNaturalist with DHOCL setup (20 splits)
python main.py --dataset inaturalist --num_tasks 20 --buffer_size 5000

# ImageNet-H with DHOCL setup
python main.py --dataset imagenet_h --num_tasks 10 --buffer_size 10000
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name (cifar100, aircraft, cub200, inaturalist, imagenet_h) | cifar100 |
| `--num_tasks` | Number of incremental splits | 10 |
| `--buffer_size` | Replay buffer size | 1000 |
| `--backbone` | Model backbone (vit_small, vit_base, resnet50) | vit_small |
| `--pretrain` | Pretraining method (dino, clip, mae, supervised) | dino |
| `--lambda_hpr` | HPR regularization coefficient | 0.1 |
| `--delta` | PredLA temperature trade-off | 0.5 |
| `--num_prototypes` | Prototypes per class | 5 |
| `--topk` | TopK patches for temporal consistency | 0.1 (fraction) |
| `--margin` | Margin for hierarchical alignment | 0.1 |

## Project Structure

```
HALO_ICML26/
├── main.py                 # Entry point
├── config/                 # Configuration files
├── models/
│   ├── halo.py             # HALO model
│   ├── backbone.py         # Feature extractors
│   ├── prototype.py        # HPR module
│   └── predla.py           # PredLA module
├── data/                   # Data loading & preprocessing
├── utils/
│   ├── buffer.py           # Memory buffer management
│   ├── hierarchy.py        # Dynamic hierarchy construction
│   ├── metrics.py          # Evaluation metrics (AAUC, FAUC, MS)
│   └── maskcut.py          # MaskCut for prototype visualization
├── baselines/              # Baseline implementations
└── scripts/                # Experiment scripts
```

## Citation

```bibtex
@inproceedings{wang2026halo,
  title     = {Online Continual Learning with Dynamic Label Hierarchies},
  author    = {Wang, Xinrui and Twardowski, Bart{\l}omiej and Gomez-Villa, Alexandra and Li, Shao-Yuan and Chen, Songcan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgements

This work was supported by the National Science and Technology Major Project of China (No. 2024YFB3311401), the National Natural Science Foundation of China (NSFC) under Grant No. 62376126, and the Funding for Outstanding Doctoral Dissertation in NUAA (BCXJ25-21). Bartłomiej Twardowski and Alexandra Gomez-Villa acknowledge support from the European Union's Horizon Europe programme (ELLIOT, Grant No. 101214398) and the Spanish Ministry of Science and Innovation (PID2022-143257NB-I00).
