# Local verification

Verified on 2026-11-09 with Python 3.7.6, PyTorch 1.10.1+cpu, on Windows. GPU execution, modern-PyTorch CI and real-dataset accuracy have not been verified.

- 10 tests passed: 5/9-class forward/backward; odd spatial sizes; finite gradients through every trainable parameter; FFT band partition and reconstruction; exact chunked-attention equivalence; hand-computed IoU/mPA; both RGB palettes and invalid labels; all-ignore loss; external prior loading and shape rejection; synchronized transforms; duplicate split rejection; default-width forward and checkpoint prediction equivalence.
- Synthetic integration: two training epochs completed, checkpoint files and logs created, held-out evaluation JSON written, original-size ID/color prediction PNGs written.
- Resume entry point successfully restored completed checkpoint state with the same configuration/manifests. Continuing an interrupted multi-epoch real training job was not tested.
- Local pytest plugin auto-discovery was disabled to avoid unrelated globally installed plugin startup delays: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` (set the environment variable separately in PowerShell).
- Two environment warnings remained: a legacy Windows Python `imp` deprecation, and a PyTorch 1.10 complex-to-real backward warning during FFT gradients. Gradient finiteness tests passed; the FFT output intentionally uses real reconstructed kernels. Modern-version behavior remains for CI verification.

Synthetic data and generated checkpoints/results are not bundled. Recreate them with `tools/make_demo.py` and the README commands. Test numbers are software validation, not sonar segmentation evidence.
