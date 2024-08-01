"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

import tempfile
import argparse

ptypy.load_gpu_engines("cupy")
tmpdir = tempfile.gettempdir()

def run_benchmark():
    opt = parse()
    p = get_params(opt)
    P = Ptycho(p,level=5)
    print_results(P)


def parse():
    parser = argparse.ArgumentParser(description="A script to benchmark ptypy using the moonflower simulation")
    parser.add_argument("-n", "--frames", type=int, help="Nr. of data frames")
    parser.add_argument("-s", "--shape", type=int, help="1D shape of each data frame")
    parser.add_argument("-i", "--iterations", type=int, help="Nr. of iterations")
    parser.add_argument("-f", "--fftlib", type=str, default="cupy")
    args = parser.parse_args()
    return args

def get_params(args):

    p = u.Param()

    # for verbose output
    p.verbose_level = "info"

    # set home path
    p.io = u.Param()
    p.io.home = "/".join([tmpdir, "ptypy"])

    # saving intermediate results
    p.io.autosave = u.Param(active=False)

    # opens plotting GUI if interaction set to active)
    p.io.autoplot = u.Param(active=False)
    p.io.interaction = u.Param(active=False)

    # Save benchmark timings
    p.io.benchmark = "all"

    # max 200 frames (128x128px) of diffraction data
    p.scans = u.Param()
    p.scans.MF = u.Param()
    # now you have to specify which ScanModel to use with scans.XX.name,
    # just as you have to give 'name' for engines and PtyScan subclasses.
    p.scans.MF.name = 'BlockVanilla' # or 'BlockFull'
    p.scans.MF.data= u.Param()
    p.scans.MF.data.name = 'MoonFlowerScan'
    p.scans.MF.data.shape = args.shape
    p.scans.MF.data.num_frames = args.frames
    p.scans.MF.data.save = None

    # position distance in fraction of illumination frame
    p.scans.MF.data.density = 0.2
    # total number of photon in empty beam
    p.scans.MF.data.photons = 1e8
    # Gaussian FWHM of possible detector blurring
    p.scans.MF.data.psf = 0.

    # attach a reconstrucion engine
    p.engines = u.Param()
    p.engines.engine00 = u.Param()
    p.engines.engine00.name = 'DM_cupy'
    p.engines.engine00.numiter = args.iterations
    p.engines.engine00.fft_lib = args.fftlib

    return p

def print_results(ptycho):
    # Print benchmarking results
    if (ptycho.p.io.benchmark == "all") and u.parallel.master:
        print("\nBenchmark:")
        print("==========")
        total = 0
        for k,v in ptycho.benchmark.items():
            total += v
            print(f"{k}: {v:.02f} s")
        print(f"Total: {total:.02f} s")

# prepare and run
if __name__ == "__main__":
    run_benchmark()
