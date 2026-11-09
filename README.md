# PGSANet

PyTorch implementation derived from the manuscript **PGSANet: Physics-Guided Spectrum-Adaptive Network for Multi-Source Sonar Image Semantic Segmentation**, by Zhen Wang, Ke Wang, Chuanlei Zhang, Jianxin Guo and Feng Wang.

**Status:** runnable paper-derived implementation, not recovered original experimental code. Unspecified choices and figure/text discrepancies are documented in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md). No trained weights, dataset, reproduced accuracy, or measured paper-comparable complexity is included. The supplied architecture has not been calibrated to the manuscript's 43.52M parameters or 25.67 GFLOPs.

中文使用说明见 [README_zh.md](README_zh.md)。

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

Local verification used Python 3.7.6 / PyTorch 1.10.1 CPU because these were available on the authoring machine. CI is configured separately for Python 3.11; CI execution and CUDA execution have not been verified here. No torchvision or custom CUDA extensions are required.

## Dataset contract

Create separate `train.json`, `val.json`, `test.json` JSON lists. Paths resolve relative to each manifest, not the current working directory:

```json
[
  {"image": "images/001.png", "mask": "masks/001.png", "prior": "priors/001.npy"}
]
```

Images: 8-bit grayscale by default; set `in_channels: 3` for RGB. Masks: exact RGB colors from Table I (`label_format: "rgb"`) or single-channel integer class IDs (`"index"`). Index 255 is ignored. SAS `Unlabeled` is class 5, not ignored, matching Table III. Unknown RGB colors raise an error. Palette PNGs are converted to RGB when `label_format` is `rgb`.

Physical priors: finite float NumPy arrays with shape `[prior_channels, original_height, original_width]`, geometrically aligned with the image. Choose and document a stable normalization using training/calibration data. Priors must be available at inference and must **not** be derived from segmentation ground truth. The repository intentionally does not invent calibrated attenuation or shadow models without sensor metadata. The default `external` mode fails if a prior is missing. The explicit `demo` mode uses intensity proxies solely to exercise the pipeline.

Use the original split lists if available. The paper specifies SSS 120/15/15 and SAS 86/13/30 but does not provide file identities or the splitting seed. Do not claim an arbitrary split reproduces its results. Group overlapping crops/acquisition runs appropriately to avoid scene leakage; the code detects duplicate image paths but cannot infer acquisition groups or visually duplicated files.

## Training and evaluation

```bash
python train.py --config configs/sss.json --train data/sss/train.json --val data/sss/val.json --output runs/sss
python train.py --config configs/sas.json --train data/sas/train.json --val data/sas/val.json --output runs/sas
python evaluate.py --checkpoint runs/sss/best.pt --manifest data/sss/test.json --output runs/sss/test_metrics.json
python predict.py --checkpoint runs/sss/best.pt --image data/sss/images/example.png --prior data/sss/priors/example.npy --output predictions
```

Add `--device cpu` if desired. Prediction saves class-ID and color PNGs at the original image resolution; evaluation follows the configured 256×256 resolution. Metrics are ratios in `[0,1]`, not percentages. Absent classes with zero union are omitted from mIoU; classes absent in ground truth are omitted from mPA. A class absent in ground truth but falsely predicted receives zero IoU. All class values and the full confusion matrix are saved.

`best.pt` uses validation mPA by default; change `monitor` to `mIoU` for a different selection protocol. `last.pt` stores optimizer, scheduler, configuration and random states. Resume with the same config and manifests:

```bash
python train.py --config configs/sss.json --train data/sss/train.json --val data/sss/val.json --output runs/sss --resume runs/sss/last.pt
```

Only load trusted checkpoints. Exact cross-device/CUDA reproducibility is not guaranteed, particularly for `grid_sample` backward. Keep data immutable when resuming. `num_workers=0` is intentional for reproducible augmentation state.

## Synthetic smoke test

```bash
python tools/make_demo.py
python train.py --config data/demo/config.json --train data/demo/train.json --val data/demo/val.json --output runs/demo --device cpu
python evaluate.py --checkpoint runs/demo/best.pt --manifest data/demo/test.json --output runs/demo/test_metrics.json --device cpu
python predict.py --checkpoint runs/demo/best.pt --image data/demo/test_0.png --output predictions/demo --device cpu
```

For CPU machines, limit OpenMP/MKL thread counts to 2 if tiny tests run slowly. Demo metrics are software diagnostics and must never be presented as paper results.

## Ablations and memory

Set `use_sdcb`, `use_abda`, or `use_pafe` to false in a copied config. For a no-physics experiment also set `loss_weights[2]` to zero; disabling PAFE alone does not disable the physics consistency term. Widths, bands, sampling windows, experts and dropout are configurable.

PAFE uses a 16×16 pooled key/value grid by default to avoid quadratic full-resolution attention memory. This is an engineering approximation, not specified by the paper. Set `attention_grid: 0` for full spatial keys and values with exact query chunking, at potentially very high training memory cost. Attention chunking reduces individual matrix size but autograd still retains intermediate tensors; batch size 8 may require adjustment on your GPU. No GPU memory claim is made.

## Release preparation

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the author-confirmation checklist. Add a license chosen by the copyright holders before advertising this as an open-source release; no license or publication DOI is fabricated. The manuscript PDF, private paths, actual sonar data, test checkpoints and generated metrics are excluded from this source package.
