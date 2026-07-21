#!/usr/bin/env bash
# Build the optional C++ suffix-tree backend in place for the active Python env.

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TREE_DIR="${SGLANG_DIR}/python/sglang/srt/speculative/sri_forest"

python -c "import pybind11" || {
    echo "pybind11 is required; install it in the active environment first." >&2
    exit 1
}
python "${TREE_DIR}/setup.py" build_ext --inplace
PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
from sglang.srt.speculative.sri_forest._sglang_sri_suffix_tree import SuffixTree

tree = SuffixTree(8)
tree.extend(0, [1, 2, 3])
assert tree.speculate([1, 2], 2, 1.0, 0.0, 0.1).token_ids == [3]
print("SRI suffix-tree extension smoke test passed.")
PY
