import unittest
import numpy as np
from . import CupyCudaTest, have_cupy

if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.kernels import ThreePIEWaveKernel

COMPLEX_TYPE = np.complex64
INT_TYPE = np.int32


class ThreePIEWaveKernelTest(CupyCudaTest):

    def test_probe_aux_roundtrip_uses_serialized_addresses(self):
        probe = (
            np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
            + 1j * np.ones((2, 5, 6), dtype=np.float32)
        ).astype(COMPLEX_TYPE)

        addr = np.zeros((1, 2, 5, 3), dtype=INT_TYPE)
        addr[0, 0] = np.array([[0, 1, 2],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]], dtype=INT_TYPE)
        addr[0, 1] = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [1, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]], dtype=INT_TYPE)

        expected_aux = np.zeros((2, 3, 4), dtype=COMPLEX_TYPE)
        expected_aux[0] = probe[0, 1:4, 2:6]
        expected_aux[1] = probe[1, 0:3, 1:5]

        kernel = ThreePIEWaveKernel(self.stream)
        aux_dev = cp.zeros(expected_aux.shape, dtype=cp.complex64)
        probe_dev = cp.asarray(probe)
        addr_dev = cp.asarray(addr)

        kernel.pr_to_aux(aux_dev, probe_dev, addr_dev)
        self.stream.synchronize()
        np.testing.assert_array_equal(expected_aux, aux_dev.get())

        out_probe_dev = cp.zeros_like(probe_dev)
        kernel.aux_to_pr(out_probe_dev, aux_dev, addr_dev)
        self.stream.synchronize()

        expected_probe = np.zeros_like(probe)
        expected_probe[0, 1:4, 2:6] = expected_aux[0]
        expected_probe[1, 0:3, 1:5] = expected_aux[1]
        np.testing.assert_array_equal(expected_probe, out_probe_dev.get())


if __name__ == '__main__':
    unittest.main()
