#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, even when called from another directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This environment requires Linux x86_64 and CPython 3.11." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first; see docs/setup.md." >&2
    exit 1
fi

uv sync --locked --python 3.11
uv pip check --python .venv/bin/python
.venv/bin/python -m run.check_environment

echo "Environment ready. Run: source .venv/bin/activate"
