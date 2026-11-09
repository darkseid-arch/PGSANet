# PGSANet

PyTorch implementation derived from the manuscript **PGSANet: Physics-Guided Spectrum-Adaptive Network for Multi-Source Sonar Image Semantic Segmentation**, by Zhen Wang, Ke Wang, Chuanlei Zhang, Jianxin Guo and Feng Wang.


## Included

- SDCB: spectral decomposition of learnable convolution kernels, noise modulation and spatial fusion (Eqs. 1–7).
- ABDA: Sobel boundary cues, confidence prediction, differentiable offset sampling and multiscale attention (Eqs. 8–17).
- PAFE: physical-prior attention, spatial expert routing and semantic-to-physical cross-attention (Eqs. 18–29).
- ASPP, weighted CE, boundary loss, physics consistency, SSS/SAS palettes and metrics.
- Training, checkpoint/resume, validation, held-out evaluation, original-size prediction and CPU tests.

## Installation

Use an isolated Python environment. Install the PyTorch build appropriate for your hardware, then:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Dataset contract

Create separate `train.json`, `val.json`, `test.json` JSON lists. Paths resolve relative to each manifest, not the current working directory:

```json
[
  {"image": "images/001.png", "mask": "masks/001.png", "prior": "priors/001.npy"}
]
```

## Training and evaluation

```bash
python train.py --config configs/sss.json --train data/sss/train.json --val data/sss/val.json --output runs/sss
python train.py --config configs/sas.json --train data/sas/train.json --val data/sas/val.json --output runs/sas
python evaluate.py --checkpoint runs/sss/best.pt --manifest data/sss/test.json --output runs/sss/test_metrics.json
python predict.py --checkpoint runs/sss/best.pt --image data/sss/images/example.png --prior data/sss/priors/example.npy --output predictions
```

`best.pt` uses validation mPA by default; change `monitor` to `mIoU` for a different selection protocol. `last.pt` stores optimizer, scheduler, configuration and random states. Resume with the same config and manifests:

```bash
python train.py --config configs/sss.json --train data/sss/train.json --val data/sss/val.json --output runs/sss --resume runs/sss/last.pt
```
