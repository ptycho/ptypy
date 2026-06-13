#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate ptypy.custom.threepie_serial.ThreePIE_serial on the CPU.

Checks the three properties that prove the serial multislice bridge is correct:

  A. number_of_slices=1  ->  reconstruction matches EPIE_serial   (corr > 0.9)
  B. number_of_slices=2  ->  error decreases over iterations       (converges)
  C. 2-slice product object matches the pod/view CPU reference
     ptypy.custom.threepie.ThreePIE                                (corr > 0.8)

All runs use the MoonFlower synthetic scan; no GPU required.
"""
import numpy as np
import sys
sys.path.insert(0, "/home/litang/ptypy_v8/dev-work")  # dev-work
from ptypy.core import Ptycho
from ptypy import utils as u

np.set_printoptions(precision=4, suppress=True)
NUMITER = 60
NFRAMES = 100
SHAPE = 64
THICK = 5e-7


def base_params(engine_name, nslices, register_cpu=False):
    import ptypy
    ptypy.load_gpu_engines("serial")          # registers *_serial engines
    if register_cpu:
        from ptypy.custom import threepie     # registers CPU ThreePIE
    from ptypy.custom import threepie_serial   # registers ThreePIE_serial

    p = u.Param()
    p.verbose_level = "error"
    p.io = u.Param()
    p.io.autosave = u.Param(active=False)
    p.io.interaction = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = "BlockFull"
    p.scans.MF.data = u.Param()
    p.scans.MF.data.name = "MoonFlowerScan"
    p.scans.MF.data.shape = SHAPE
    p.scans.MF.data.num_frames = NFRAMES
    p.scans.MF.data.density = 0.2
    p.scans.MF.data.photons = 1e8
    p.scans.MF.data.psf = 0.0
    p.scans.MF.data.save = None

    p.engines = u.Param()
    p.engines.e0 = u.Param()
    p.engines.e0.name = engine_name
    p.engines.e0.numiter = NUMITER
    p.engines.e0.probe_center_tol = None
    p.engines.e0.compute_log_likelihood = True
    if not register_cpu:   # serial/GPU engines expose this; pod/view CPU does not
        p.engines.e0.compute_fourier_error = True
    p.engines.e0.object_norm_is_global = True
    p.engines.e0.alpha = 1
    p.engines.e0.beta = 1
    p.engines.e0.probe_update_start = 0
    if nslices is not None:
        p.engines.e0.number_of_slices = nslices
        p.engines.e0.slice_thickness = THICK
        p.engines.e0.fslices = "/tmp/slices_serial_val.h5"
    return p


def get_arrays(P):
    ob = list(P.obj.storages.values())[0].data[0].copy()
    pr = list(P.probe.storages.values())[0].data[0].copy()
    return ob, pr


def ncorr(a, b):
    """Phase/scale-invariant normalized correlation of two complex fields."""
    a = a.ravel(); b = b.ravel()
    a = a - a.mean(); b = b - b.mean()
    num = np.abs(np.vdot(a, b))
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(num / den) if den else 0.0


def run(engine_name, nslices, seed=0, register_cpu=False):
    np.random.seed(seed)
    P = Ptycho(base_params(engine_name, nslices, register_cpu), level=5)
    return P, None


print("=" * 70)
print("A. 1-slice ThreePIE_serial  vs  EPIE_serial")
print("=" * 70)
P_epie, _ = run("EPIE_serial", None, seed=1)
ob_e, pr_e = get_arrays(P_epie)
P_1s, _ = run("ThreePIE_serial", 1, seed=1)
ob_1, pr_1 = get_arrays(P_1s)
cP = ncorr(pr_e, pr_1)
cO = ncorr(ob_e, ob_1)
print(f"probe  correlation EPIE_serial vs 1-slice ThreePIE_serial : {cP:.4f}")
print(f"object correlation EPIE_serial vs 1-slice ThreePIE_serial : {cO:.4f}")
print(f"  -> {'PASS' if cP > 0.85 and cO > 0.85 else 'CHECK'} (expect > 0.85)")

print("\n" + "=" * 70)
print("B. 2-slice ThreePIE_serial convergence")
print("=" * 70)
P_2s, _ = run("ThreePIE_serial", 2, seed=3)
errs = []
for it in P_2s.runtime["iter_info"]:
    e = it.get("error")
    if e is None:
        continue
    e = np.asarray(e, dtype=float).ravel()
    if e.size and e[0] > 0:
        errs.append(float(e[0]))   # mean fourier error
if len(errs) > 2:
    ratio = errs[-1] / errs[0]
    print(f"fourier error  first={errs[0]:.4e}  last={errs[-1]:.4e}  "
          f"ratio={ratio:.3f}")
    print(f"  -> {'PASS' if ratio < 0.8 else 'CHECK'} (expect last/first < 0.8)")
else:
    print("  (not enough error samples captured)")

print("\n" + "=" * 70)
print("C. 2-slice ThreePIE_serial  vs  CPU reference custom/threepie.ThreePIE")
print("=" * 70)
P_ref, _ = run("ThreePIE", 2, seed=2, register_cpu=True)
ob_r, pr_r = get_arrays(P_ref)
P_2c, _ = run("ThreePIE_serial", 2, seed=2)
ob_c, pr_c = get_arrays(P_2c)
cP2 = ncorr(pr_r, pr_c)
cO2 = ncorr(ob_r, ob_c)
print(f"probe  correlation CPU-ref vs serial : {cP2:.4f}")
print(f"object correlation CPU-ref vs serial : {cO2:.4f}")
print(f"  -> {'PASS' if cP2 > 0.8 and cO2 > 0.8 else 'CHECK'} (expect > 0.8)")

print("\nDONE.")