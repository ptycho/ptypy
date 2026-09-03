#!/bin/bash
#
# One-command check of the ThreePIE multislice engines (CPU / serial / GPU)
# and the near-field anti-aliasing (band-limit + slice_pad).
#
# Runs three things in order:
#   1. crop sampling / slice_pad diagnostic   (pure numpy, instant)
#   2. ground-truth slice separation          (CPU + serial, no GPU needed)
#   3. GPU engine + wave kernel               (needs CuPy + GPU; skipped otherwise)
#
# Usage:
#   bash test/engine_tests/check_threepie.sh
#   bash test/engine_tests/check_threepie.sh --crops 256,512 --slice-thickness 1500e-6
#       (extra args are forwarded to the crop diagnostic only)
#
# Point at a different ptypy checkout:
#   PTYPY_PATH=/path/to/ptypy bash test/engine_tests/check_threepie.sh
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PTYPY_PATH:-$(cd "$HERE/../.." && pwd)}"
PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "check_threepie: using ptypy from $REPO_ROOT"
"$PY" -c "import ptypy, os; print('  ptypy:', os.path.dirname(ptypy.__file__))" || {
    echo "ERROR: cannot import ptypy from $REPO_ROOT"; exit 2; }
echo

rc=0

echo "================ 1/3  crop sampling + slice_pad diagnostic ==============="
"$PY" "$REPO_ROOT/ptypy/debug/diagnose_threepie_crop.py" "$@" || rc=1
echo

echo "================ 2/3  ground-truth slice separation (no GPU) ============="
"$PY" -m pytest -q \
    "$REPO_ROOT/test/engine_tests/threepie_groundtruth_slices_test.py" \
    "$REPO_ROOT/test/util_tests/threepie_crop_diagnostic_test.py" || rc=1
echo

echo "================ 3/3  GPU engine + wave kernel (needs CuPy/GPU) =========="
"$PY" -m pytest -q \
    "$REPO_ROOT/test/accelerate_tests/cuda_cupy_tests/threepie_engine_test.py" \
    "$REPO_ROOT/test/accelerate_tests/cuda_cupy_tests/threepie_wave_kernel_test.py" || rc=1
echo

echo "========================================================================="
if [ "$rc" -eq 0 ]; then
    echo "check_threepie: DONE: all checks passed"
else
    echo "check_threepie: DONE: a check exited non-zero (see output above)"
fi
exit "$rc"
