'''
'''

import unittest
import numpy as np
from . import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda.tools import DeviceMemoryPool
    from ptypy.engines.DM_pycuda_stream import GpuData, GpuDataManager, GpuStreamData

class GpuDataTest(PyCudaTest):

    def setUp(self):
        super().setUp()
        self.alloc = DeviceMemoryPool()
    
    def tearDown(self):
        self.alloc.stop_holding()
        super().tearDown()
    
    def test_to_gpu_new(self):
        # arrange
        cpu = 2. * np.ones((5,5), dtype=np.float32)
        gdata = GpuData(cpu.nbytes, syncback=False)
        
        # act
        gpu = gdata.to_gpu(cpu, '1', self.stream)
        self.stream.synchronize()

        # assert
        np.testing.assert_array_equal(cpu, gpu.get())
    
    def test_to_gpu_sameid(self):
        # arrange
        cpu = 2. * np.ones((5,5), dtype=np.float32)
        gdata = GpuData(cpu.nbytes, syncback=False)

        # act
        gpu1 = gdata.to_gpu(cpu, '1', self.stream)
        cpu *= 2.
        gpu2 = gdata.to_gpu(cpu, '1', self.stream)
        self.stream.synchronize()

        # assert
        np.testing.assert_array_equal(gpu1.get(), gpu2.get())
        
    def test_to_gpu_new_syncback(self):
        # arrange
        cpu = 2. * np.ones((5,5), dtype=np.float32)
        gdata = GpuData(cpu.nbytes, syncback=True)

        # act
        gpu1 = gdata.to_gpu(cpu, '1', self.stream)
        gpu1.fill(np.float32(3.), self.stream)
        cpu2 = 2. * cpu
        gpu2 = gdata.to_gpu(cpu2, '2', self.stream)
        self.stream.synchronize()

        # assert
        np.testing.assert_array_equal(cpu, 3.)
        np.testing.assert_array_equal(gpu2.get(), cpu2)

    def test_to_gpu_new_nosyncback(self):
        # arrange
        cpu = 2. * np.ones((5,5), dtype=np.float32)
        gdata = GpuData(cpu.nbytes, syncback=False)

        # act
        gpu1 = gdata.to_gpu(cpu, '1', self.stream)
        gpu1.fill(np.float32(3.), self.stream)
        cpu2 = 2. * cpu
        gpu2 = gdata.to_gpu(cpu2, '2', self.stream)
        self.stream.synchronize()

        # assert
        np.testing.assert_array_equal(cpu, 2.)
        np.testing.assert_array_equal(gpu2.get(), cpu2)

    def test_from_gpu(self):
        # arrange
        cpu = 2. * np.ones((5,5), dtype=np.float32)
        gdata = GpuData(cpu.nbytes, syncback=False)

        # act
        gpu1 = gdata.to_gpu(cpu, '1', self.stream)
        gpu1.fill(np.float32(3.), self.stream)
        gdata.from_gpu(self.stream)
        self.stream.synchronize()

    def test_data_variable_size(self):
        # arrange
        cpu = np.ones((2,5), dtype=np.float32)
        cpu2 = 2. * np.ones((1,5), dtype=np.float32)
        gdata = GpuData(cpu.nbytes, syncback=False)

        # act
        gpu = gdata.to_gpu(cpu, '1', self.stream)
        gpu2 = gdata.to_gpu(cpu2, '2', self.stream)
        self.stream.synchronize()

        # assert
        np.testing.assert_array_equal(gpu2.get(), cpu2)
        self.assertEqual(cpu2.nbytes, gpu2.nbytes)
        np.testing.assert_array_equal(gpu.get(), np.array([
            [2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1]
        ], dtype=np.float32))

    def test_data_variable_size_raise(self):
        # arrange
        cpu = np.ones((1,5), dtype=np.float32)
        cpu2 = np.ones((2,4), dtype=np.float32)
        gdata = GpuData(cpu.nbytes, syncback=False)

        # act/assert
        with self.assertRaises(Exception):
            gdata.to_gpu(cpu2, '1', self.stream)
        
    def test_datamanager_newids(self):
        # arrange
        cpu1 = 2. * np.ones((5,5), dtype=np.float32)
        cpu2 = 2. * cpu1  # 4
        cpu3 = 2. * cpu2  # 8
        cpu4 = 2. * cpu3  # 16
        gdm = GpuDataManager(cpu1.nbytes, 4, syncback=False)

        # act
        gpu1 = gdm.to_gpu(cpu1, '1', self.stream)
        gpu2 = gdm.to_gpu(cpu2, '2', self.stream)
        gpu11 = gdm.to_gpu(-1.*cpu1, '1', self.stream)
        gpu21 = gdm.to_gpu(-1.*cpu4, '2', self.stream)
        gpu3 = gdm.to_gpu(cpu3, '3', self.stream)
        gpu31 = gdm.to_gpu(-1.*cpu1, '3', self.stream)
        gpu4 = gdm.to_gpu(cpu4, '4', self.stream)
        gpu41 = gdm.to_gpu(-1.*cpu1, '4', self.stream)
        self.stream.synchronize()

        # assert
        np.testing.assert_array_equal(cpu1, gpu1.get())
        np.testing.assert_array_equal(cpu1, gpu11.get())
        np.testing.assert_array_equal(cpu1, 2.)
        np.testing.assert_array_equal(cpu2, gpu2.get())
        np.testing.assert_array_equal(cpu2, gpu21.get())
        np.testing.assert_array_equal(cpu2, 4.)
        np.testing.assert_array_equal(cpu3, gpu3.get())
        np.testing.assert_array_equal(cpu3, gpu31.get())
        np.testing.assert_array_equal(cpu3, 8.)
        np.testing.assert_array_equal(cpu4, gpu4.get())
        np.testing.assert_array_equal(cpu4, gpu41.get())
        np.testing.assert_array_equal(cpu4, 16.)

    def test_datamanager_syncback(self):
        # arrange
        cpu1 = 2. * np.ones((5,5), dtype=np.float32)
        cpu2 = 2. * cpu1  # 4
        cpu3 = 2. * cpu2  # 8
        cpu4 = 2. * cpu3  # 16
        gdm = GpuDataManager(cpu1.nbytes, 2, syncback=True)

        # act
        gpu1 = gdm.to_gpu(cpu1, '1', self.stream)
        gpu2 = gdm.to_gpu(cpu2, '2', self.stream)
        gpu1.fill(np.float32(3.), self.stream)
        gpu2.fill(np.float32(5.), self.stream)
        gpu3 = gdm.to_gpu(cpu3, '3', self.stream)
        gpu3.fill(np.float32(7.), self.stream)
        gpu4 = gdm.to_gpu(cpu4, '4', self.stream)
        gpu4.fill(np.float32(9.), self.stream)
        gdm.syncback = False
        gpu5 = gdm.to_gpu(cpu4*.2, '5', self.stream)
        gpu6 = gdm.to_gpu(cpu4*.4, '6', self.stream)
        self.stream.synchronize()

        # assert
        np.testing.assert_array_equal(cpu1, 3.)
        np.testing.assert_array_equal(cpu2, 5.)
        np.testing.assert_array_equal(cpu3, 8.)
        np.testing.assert_array_equal(cpu4, 16.)

    def test_data_synctransfer(self):
        # arrange
        sh = (1024, 1024, 1)  # 4MB
        cpu1 = cuda.pagelocked_zeros(sh, np.float32, order="C", mem_flags=0)
        cpu2 = cuda.pagelocked_zeros(sh, np.float32, order="C", mem_flags=0)
        cpu1[:] = 1.
        cpu2[:] = 2.
        gdata = GpuData(cpu1.nbytes, syncback=True)
        # long-running kernel
        knl = """
        extern "C" __global__ void tfill(float* d, int sz, float dval) {
            for (int i = 0; i < sz; ++i) 
                d[i] = dval;
        }
        """
        mod = SourceModule(knl, no_extern_c=True)
        tfill = mod.get_function('tfill')
        
        # act
        s2 = cuda.Stream()
        gpu1 = gdata.to_gpu(cpu1, '1', self.stream)
        tfill(gpu1, np.int32(gpu1.size), np.float32(2.), grid=(1,1,1), block=(1,1,1), stream=self.stream)
        gdata.record_done(self.stream)  # it will fail without this
        gpu2 = gdata.to_gpu(cpu2, '2', s2)
        tfill(gpu1, np.int32(gpu2.size), np.float32(4.), grid=(1,1,1), block=(1,1,1), stream=s2)
        gdata.from_gpu(s2)
        self.stream.synchronize()
        s2.synchronize()
        
        # assert
        np.testing.assert_array_equal(cpu1, 2.)
        np.testing.assert_array_equal(cpu2, 4.)
