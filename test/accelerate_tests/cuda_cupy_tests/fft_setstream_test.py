import unittest
import numpy as np
from . import CupyCudaTest, have_cupy
import time

if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.cufft import FFT_cuda as cuFFT
    from ptypy.accelerate.cuda_cupy.cufft import FFT_cupy as cupyCuFFT

    COMPLEX_TYPE = np.complex64
    FLOAT_TYPE = np.float32
    INT_TYPE = np.int32

class FftSetStreamTest(CupyCudaTest):

    def helper(self, FFT):
        f = np.ones(shape=(200, 128, 128), dtype=COMPLEX_TYPE)
        t1 = time.time()
        FW = FFT(f, self.stream, pre_fft=None, post_fft=None, inplace=True,
            symmetric=True)
        t2 = time.time()
        dur1 = t2 - t1
        with self.stream:
            f_dev = cp.asarray(f)
            self.stream.synchronize()

        # measure with events to make sure that something actually 
        # happened in the right stream
        with self.stream:
            ev1 = cp.cuda.Event()
            ev2 = cp.cuda.Event()
            rt1 = time.time()
            ev1.record()
        FW.ft(f_dev, f_dev)
        with self.stream:
            ev2.record()
            ev1.synchronize()
            ev2.synchronize()
            self.stream.synchronize()
            gput = cp.cuda.get_elapsed_time(ev1, ev2)*1e-3
        rt2 = time.time()
        cput = rt2-rt1
        rel = 1-gput/cput

        print('Origial: CPU={}, GPU={}, reldiff={}'.format(cput, gput, rel))

        self.assertEqual(self.stream, FW.queue)
        self.assertLess(rel, 0.3)  # max 30% diff
        
        stream2 = cp.cuda.Stream()

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
        
        with stream2:
            ev1 = cp.cuda.Event()
            ev2 = cp.cuda.Event()
            rt1 = time.time()
            ev1.record()
        FW.ft(f_dev, f_dev)
        with stream2:
            ev2.record()
            ev1.synchronize()
            ev2.synchronize()
            stream2.synchronize()
            gput = cp.cuda.get_elapsed_time(ev1, ev2)*1e-3
        self.stream.synchronize()
        rt2 = time.time()
        cput = rt2-rt1
        rel = 1 - gput/cput

        print('New: CPU={}, GPU={}, reldiff={}'.format(cput, gput, rel))

        self.assertEqual(stream2, FW.queue)
        self.assertLess(rel, 0.3)  # max 30% diff

        if measure:
            print('initial: {}, set_stream: {}'.format(dur1, dur2))
            assert False 



    def test_set_stream_b_cufft(self):
        self.helper(cuFFT)

    def test_set_stream_c_cupy_cufft(self):
        self.helper(cupyCuFFT)
