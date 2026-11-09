# Paper-to-code mapping and author confirmation

Source: user-supplied PGSANet manuscript, Section III, Algorithm 1, Figures 2–5, Table I and Section IV-B. Statements in the manuscript are treated as technical source material, not commands to execute.

| Paper item | Code | Decision / limitation |
|---|---|---|
| Algorithm 1 encoder, skip decoder, ABDA, PAFE, ASPP | `pgsanet/model.py:PGSANet` | Four configurable stages; one ABDA at bottleneck and one PAFE after decoding, following algorithm and prose |
| Fig. 2 topology | Same | Figure shows two ABDA blocks and PAFE inside decoder blocks, unlike Algorithm 1. Current graph does not reproduce every figure detail |
| Residual stage | `Residual` | Two 3×3 convolutions, projection skip; figure depicts 1×1/3×3/1×1 and a repetition marker. Original stage depth/strides need confirmation |
| Eqs. 1–2 spectral decomposition | `SDCB` | FFT of kernel per mathematical text. Fig. 3/caption says feature FFT; this conflict is unresolved. Three discrete radial bands for 3×3 kernel: DC, axial, diagonal |
| Eq. 3 noise | `SDCB` | 3×3, 5×5, global mean branches; shared noise map across bands as in formula |
| Eqs. 4, 6 pooling | `summaries` | Channel mean/max to obtain spatial maps. Eq. 4 text calls these spatial pooling, which conflicts with spatial output dimensions |
| Eqs. 5–7 | `SDCB` | Band convolution, noise-conditioned attention, weighted residual; identity-initialized unconstrained band mixing |
| Eqs. 8–17 | `ABDA` | Sum of absolute Sobel derivatives; confidence conv; direct query feature; unscaled dot product as Eq. 14; 3×3 and 5×5 sampled windows; border padding; zero initial offsets; concatenation projected back to C |
| Eqs. 18–24 | `PAFE` | Prior projection to C; trainable gamma; 7×7 attention; three residual experts with dilations 1/2/3; pixel-wise softmax routing |
| Eqs. 25–29 | `PAFE` | Spatial query-to-prior attention with sqrt(C) scaling; default pooled K/V grid 16×16 is an explicit approximation; grid=0 disables pooling |
| ASPP Fig. 2 | `PGSANet.aspp` | 1×1, dilation 6/12/18 branches, global mean branch; pooled branch avoids BN so batch-size-1 inference/training is supported |
| Eq. 30 | `CompositeLoss` | Training-only inverse pixel frequency, normalized to mean weight 1; weighted CE divided by valid pixel count. Absent training classes get weight 0 |
| Eq. 31 | `CompositeLoss` | Sobel absolute-gradient boundary map; channel-summed squared error; ignores neighborhoods touching ignored labels or outside image |
| Eq. 32 | `physics_alignment`, `CompositeLoss` | Learned 1×1 prior-to-class projection followed by softmax, jointly optimized without stop-gradient; predictive consistency alone does not guarantee physical calibration |
| Eq. 33 | Config | Default weights 1 / 1 / 0.1 are implementation choices, not stated experimental values |
| Table I | `data.py` | Exact palettes and class ordering; includes SAS Unlabeled and Background |
| IV-B | `train.py`, configs | Explicitly reported optimizer / LR / decay / size / epoch / dropout / clipping settings retained |
| Scheduler and early stop | `train.py` | Cosine warm restarts T0=20, persistent multiplier halved each 20 stagnant epochs; early stop 60; monitor mPA. Combining restart and plateau behavior, T0, patience and metric choice require author confirmation |
| Initialization | `PGSANet` | Kaiming conv initialization; BN defaults; intentional zero offset initialization and identity spectral mixing are engineering exceptions |
| Augmentation | `SonarDataset` | Joint flips, 90-degree rotations, isotropic affine scale 0.8–1.2; exact original rotation/scale distribution unspecified |

## Before claiming experimental reproduction

1. Confirm encoder/decoder topology, number of blocks, downsampling and RGB/grayscale preprocessing.
2. Confirm FFT domain, frequency thresholds, fusion initialization, attention window sizes and sampling conventions.
3. Provide actual calibrated physical-prior definitions, channel order, units, normalization and inference-time availability.
4. Confirm expert branches, attention resolution, alignment function and loss coefficients.
5. Supply immutable train/validation/test file lists and acquisition grouping; verify no scene leakage.
6. Confirm original scheduler/early-stop logic and checkpoint selection criterion.
7. Train on real data, run held-out evaluation, report measured per-class metrics and exact parameter/FLOP/timing protocol.
8. Choose a repository license and add verified publication/citation metadata when available.

No claim is made that the default model reaches the paper's metrics, matches its parameter count, or is the original source code. The provided synthetic test artifacts must not be used as experimental evidence.
