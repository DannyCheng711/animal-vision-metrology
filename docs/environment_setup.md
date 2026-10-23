# Python Environment Setup

This project uses uv for Python environment and dependency management.

1. Install Dependencies

From the project root, run:

```bash
uv sync --locked
uv pip check
```

This automatically creates a .venv virtual environment (if needed) and installs the dependencies defined in pyproject.toml and uv.lock.

2. Run the COCO Dataset Loader

```bash
uv run python -m dataset.dataloader
```

## System Dependencies (Debian/Ubuntu)

```bash
apt-get update
apt-get install -y --no-install-recommends libgl1
```

## Fixed Pose Environment

Target: CPython 3.11, Linux x86_64. The CUDA-enabled packages can also run on CPU.

| Package | Version | Reason |
| --- | --- | --- |
| torch / torchvision | 2.1.0 / 0.16.0, cu118 | Match the compiled MMCV wheel |
| mmcv | 2.1.0 | Official CPython 3.11 Linux wheel |
| mmengine | 0.10.7 | Configuration and checkpoint loading |
| mmpose | 1.3.2 | AP-10K pose models |
| mmdet | 3.2.0 | Required by RTMO imports in the MMPose public API |
| numpy | 1.26.4 | Compatibility with the older PyTorch stack |
| opencv-python | 4.11.0.86 | Compatible with NumPy 1.x |
| setuptools | 69.5.1 | Provides pkg_resources.packaging used by torchvision |

PyTorch uses an explicit index; other packages use PyPI. MMCV uses an exact wheel URL. Keep `uv.lock` in Git. This environment targets the server; macOS needs different compatible sources.

MMPose transitively requires `chumpy`, whose setup script omits build requirements. `[tool.uv.extra-build-dependencies]` supplies pip and setuptools inside chumpy's isolated build environment. Pip is not a direct runtime dependency. Use `uv sync --locked` to reproduce this setup rather than a sequence of manual pip installs.

## Download and Load Pose Models

Run from the project root. These commands download the official config/checkpoint pairs and validate loading on CPU:

```bash
uv run python -m model.eye_localization --architecture w32 --download
uv run python -m model.eye_localization --architecture w48 --download
```

Configs are pinned to MMPose `v1.3.2`; assets are saved under the ignored `weights/` directory. Existing nonempty files are reused. Initial downloads need network access. The installed MMPose package also supplies AP-10K metadata; a warning about using its bundled metadata path is harmless.

After downloading, load again without downloading assets:

```bash
uv run python -m model.eye_localization --architecture w32
```

Output should include `Device: cpu`, `Evaluation mode: True`, and eye indices `left_eye: 0`, `right_eye: 1`. This verifies model loading, not eye localization accuracy. `predict()` is not implemented yet. GPU execution has not been tested.

## References

- [MMPose installation](https://mmpose.readthedocs.io/en/latest/installation.html)
- [Official HRNet AP-10K models](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/animal_2d_keypoint/topdown_heatmap/ap10k/hrnet_ap10k.md)
- [uv build dependencies](https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies)
