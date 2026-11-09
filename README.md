# ADMFN: Main experiment code

Reference implementation of the main experiment on one-way skeleton-conditioned gated fusion of visual and human-geometry features for outfit compatibility assessment. This repository contains the main training and evaluation script and its documentation only; the manuscript, the dataset, model weights and supporting scripts are not included.

## Contents

| File | Purpose |
| --- | --- |
| `code/model_multimodal_fusion.py` | Model definition, multimodal dataset, training and evaluation used in the main experiment |
| `requirements.txt` | Python dependencies |
| `README.md` | This document |

## Requirements

Python 3.8 or later. Install the Python dependencies with:

```
pip install -r requirements.txt
```

PyTorch is used with CUDA when available and falls back to CPU.

## Data preparation

The DeepFashion2 images and annotations are not redistributed here; obtain them from the official dataset release. The training script consumes pre-extracted features and two label files, all expected under one root directory:

```
<ADMFN_DATA_ROOT>/
    labels/train_match.json
    labels/val_match.json
    51normcharacteristic/{train,validation}_result/features/*.npy
    28normcharacteristic/{train,validation}_result/features/*.npy
    standardResnet50/03_Outputs/resnet_features_engineering/{train,validation}_norm/*/*.pkl
```

The label files are generated from the official annotations following the protocol described in the manuscript. Visual features are 2048-dimensional ResNet-50 descriptors; the 79-dimensional skeleton representation combines 51 keypoint dimensions and 28 geometric descriptors.

Set the root through an environment variable; the original workstation layout is kept as the default:

```
ADMFN_DATA_ROOT      root containing the extracted feature folders and labels/
```

## Reproducing the experiment

```
python code/model_multimodal_fusion.py
```

`RUN_TAG` appends a suffix to the log, curve and checkpoint names so that several configurations can be run without overwriting each other, and `GATE_BIAS` sets the initial gate logit. The seed is fixed at 42. Logs, curves and checkpoints are written to the working directory and excluded from version control.

## Notes

Only the main experiment is provided here. Model checkpoints, raw images, annotations, training logs and auxiliary scripts (label construction, ablations, gate-factor analysis and figure generation) are intentionally not included. A license should be selected when the GitHub repository is created.
