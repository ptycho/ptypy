"""
NVTX ranges for Nsight Systems, shared by the CPU-side engines.

Switched on with the environment variable ``PTYPY_NVTX=1``; otherwise
:func:`nvtx_push` and :func:`nvtx_pop` are no-ops and cost one attribute
lookup. The ranges are host-side markers only, so they work in a process
that never touches the GPU. The bindings of cupy are used when cupy is
importable, else the ``nvtx`` package; without either the helpers stay
no-ops. The GPU engine carries its own copy of the same two helpers.
"""
import os

__all__ = ["nvtx_push", "nvtx_pop", "NVTX_ENABLED"]

NVTX_ENABLED = os.environ.get("PTYPY_NVTX", "0").lower() not in ("", "0", "false", "no")

_push = _pop = None
if NVTX_ENABLED:
    try:
        import cupy as _cp
        _push, _pop = _cp.cuda.nvtx.RangePush, _cp.cuda.nvtx.RangePop
    except Exception:
        try:
            import nvtx as _nvtx
            _push, _pop = _nvtx.push_range, _nvtx.pop_range
        except Exception:
            pass


def nvtx_push(name):
    """Open an NVTX range called ``name`` (no-op unless enabled)."""
    if _push is not None:
        _push(name)


def nvtx_pop():
    """Close the innermost NVTX range (no-op unless enabled)."""
    if _pop is not None:
        _pop()
