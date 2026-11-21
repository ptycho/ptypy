"""
Test that refined positions from all MPI ranks reach the saved reconstruction.

Each rank refines the positions of the views it reconstructs. At the end of
an engine they have to be gathered on the master node, which writes the file.
This needs more than one process, so the test runs position_gather_mpi.py
under mpiexec with 2 ranks. It is skipped when mpi4py or mpiexec is missing.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os
import shutil
import subprocess
import sys
import unittest

try:
    import mpi4py  # noqa: F401
    have_mpi4py = True
except ImportError:
    have_mpi4py = False

MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")
HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "position_gather_mpi.py")


@unittest.skipIf(not have_mpi4py or MPIEXEC is None, "needs mpi4py and mpiexec")
class PositionGatherTest(unittest.TestCase):

    def check(self, *engines):
        out = subprocess.run([MPIEXEC, "-n", "2", sys.executable, HELPER] + list(engines),
                             capture_output=True, text=True, timeout=900)
        self.assertEqual(out.returncode, 0, msg=out.stdout + out.stderr[-3000:])

    def test_DM(self):
        self.check("DM")

    def test_ML(self):
        self.check("ML")

    def test_DM_serial(self):
        self.check("DM_serial")

    def test_ML_serial(self):
        self.check("ML_serial")


if __name__ == "__main__":
    unittest.main()
