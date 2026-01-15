import unittest
import numpy as np
from . import PyCudaTest, have_pycuda
import time

if have_pycuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.fft import FFT as ReiknaFFT
    from ptypy.accelerate.cuda_pycuda.cufft import FFT_cuda as cuFFT
    from ptypy.accelerate.cuda_pycuda.cufft import FFT_skcuda as SkcudaCuFFT

    COMPLEX_TYPE = np.complex64
    FLOAT_TYPE = np.float32
    INT_TYPE = np.int32

class FftSetStreamTest(PyCudaTest):

    def helper(self, FFT):
        f = np.ones(shape=(200, 128, 128), dtype=COMPLEX_TYPE)
        t1 = time.time()
        FW = FFT(f, self.stream, pre_fft=None, post_fft=None, inplace=True,
            symmetric=True)
        self.stream.synchronize()
        t2 = time.time()
        dur1 = t2 - t1
        f_dev = gpuarray.to_gpu(f)
        self.stream.synchronize()

        # measure with events to make sure that something actually 
        # happened in the right stream
        ev1 = cuda.Event()
        ev2 = cuda.Event()
        rt1 = time.time()
        ev1.record(self.stream)
        FW.ft(f_dev, f_dev)
        ev2.record(self.stream)
        ev1.synchronize()
        ev2.synchronize()
        self.stream.synchronize()
        rt2 = time.time()
        cput = rt2-rt1
        gput = ev1.time_till(ev2)*1e-3
        rel = 1-gput/cput

        print('Origial: CPU={}, GPU={}, reldiff={}'.format(cput, gput, rel))

        self.assertEqual(self.stream, FW.queue)
        self.assertLess(rel, 0.3)  # max 30% diff
        
        stream2 = cuda.Stream()

        measure = False # measure time to set the stream
        if measure:
            avg = 100
        else:
            avg = 1
        t1 = time.time()
        for i in range(avg):
            FW.queue = stream2
        stream2.synchronize()
        t2 = time.time()
        dur2 = (t2 - t1)/avg
        
        
        ev1 = cuda.Event()
        ev2 = cuda.Event()
        rt1 = time.time()
        ev1.record(stream2)
        FW.ft(f_dev, f_dev)
        ev2.record(stream2)
        ev1.synchronize()
        ev2.synchronize()
        stream2.synchronize()
        self.stream.synchronize()
        rt2 = time.time()
        cput = rt2-rt1
        gput = ev1.time_till(ev2)*1e-3
        rel = 1 - gput/cput

        print('New: CPU={}, GPU={}, reldiff={}'.format(cput, gput, rel))

        self.assertEqual(stream2, FW.queue)
        self.assertLess(rel, 0.3)  # max 30% diff

        if measure:
            print('initial: {}, set_stream: {}'.format(dur1, dur2))
            assert False 



    def test_set_stream_a_reikna(self):
        self.helper(ReiknaFFT)

    def test_set_stream_b_cufft(self):
        self.helper(cuFFT)

    @unittest.skip("Skcuda is currently broken")
    def test_set_stream_c_skcuda_cufft(self):
        self.helper(SkcudaCuFFT)
