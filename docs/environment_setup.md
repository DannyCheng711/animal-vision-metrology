Python Environment Setup

This project uses uv for Python environment and dependency management.

1. Install Dependencies

From the project root, run:

```bash
uv sync
```

This automatically creates a .venv virtual environment (if needed) and installs the dependencies defined in pyproject.toml and uv.lock.

2. Run the COCO Dataset Loader

```bash
uv run python -m dataset.dataloader
```