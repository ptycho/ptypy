'''
Load real data and prepare an accuracy report of GPU vs numpy
'''

import h5py
import numpy as np
import csv

import pycuda.driver as cuda
from pycuda import gpuarray

from ptypy.accelerate.cuda_pycuda.kernels import GradientDescentKernel
from ptypy.accelerate.base.kernels import GradientDescentKernel as BaseGradientDescentKernel


class GradientDescentAccuracyTester:

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-%s/"
    rtol = 1e-6
    atol = 1e-6
    headings = ['Kernel', 'Version', 'Iter', 'MATH_TYPE', 'IN/OUT_TYPE',
                'ACC_TYPE', 'Array', 'num_elements', 'num_errors', 'max_relerr', 'max_abserr']

    def __init__(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.stream = cuda.Stream()
        self.results = []

    def __del__(self):
        np.set_printoptions()
        self.ctx.pop()
        self.ctx.detach()

    def test_make_model(self, name, iter,
                        math_type={'float', 'double'},
                        data_type={'float', 'double'}):

        res = []

        # Load data
        with h5py.File(self.datadir % name + "make_model_%04d.h5" % iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]

        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.make_model(aux, addr)
        ref = BGDK.npy.Imodel

        # GPU variants
        addr_dev = gpuarray.to_gpu(addr)
        for d in data_type:
            if d == 'float':
                aux_dev = gpuarray.to_gpu(aux.astype(np.complex64))
            else:
                aux_dev = gpuarray.to_gpu(aux.astype(np.complex128))
            for m in math_type:
                # data type will be determined based on aux_dev data type automatically
                GDK = GradientDescentKernel(
                    aux_dev, addr.shape[1], queue=self.stream, math_type=m)
                GDK.allocate()
                GDK.make_model(aux_dev, addr_dev)
                act = GDK.gpu.Imodel.get()

                num, num_mis, max_abs, max_rel = self._calc_diffs(act, ref)

                line = ['make_model', name, iter, d, m, 'N/A',
                        'Imodel', num, num_mis, max_rel, max_abs]
                print(line)
                res.append(line)

        return res

    def test_floating_intensity(self, name, iter,
                                math_type={'float', 'double'},
                                data_type={'float', 'double'},
                                acc_type={'float', 'double'}):

        # note that this is actually calling 4 kernels:
        # - floating_intensity_cuda_step1
        # - error_reduce_cuda (2x)
        # - floating_intensity_cuda_step2

        res = []

        # Load data
        with h5py.File(self.datadir % name + "floating_intensities_%04d.h5" % iter, "r") as f:
            w = f["w"][:]
            addr = f["addr"][:]
            I = f["I"][:]
            fic = f["fic"][:]
            Imodel = f["Imodel"][:]
        with h5py.File(self.datadir % name + "make_model_%04d.h5" % iter, "r") as f:
            aux = f["aux"][:]

        # CPU Kernel
        ficref = np.copy(fic)
        Iref = np.copy(Imodel)
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.npy.Imodel = Iref
        BGDK.floating_intensity(addr, w, I, ficref)  # modifies fic, Imodel
        Iref = BGDK.npy.Imodel

        addr_dev = gpuarray.to_gpu(addr)
        for d in data_type:
            for m in math_type:
                for a in acc_type:
                    if d == 'float':
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex64))
                        I_dev = gpuarray.to_gpu(I.astype(np.float32))
                        fic_dev = gpuarray.to_gpu(fic.astype(np.float32))
                        w_dev = gpuarray.to_gpu(w.astype(np.float32))
                        Imodel_dev = gpuarray.to_gpu(Imodel.astype(np.float32))
                    else:
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex128))
                        I_dev = gpuarray.to_gpu(I.astype(np.float64))
                        fic_dev = gpuarray.to_gpu(fic.astype(np.float64))
                        w_dev = gpuarray.to_gpu(w.astype(np.float64))
                        Imodel_dev = gpuarray.to_gpu(Imodel.astype(np.float64))

                    # GPU kernel
                    GDK = GradientDescentKernel(
                        aux_dev, addr.shape[1], accumulate_type=a, math_type=m, queue=self.stream)
                    GDK.allocate()
                    GDK.gpu.Imodel = Imodel_dev
                    GDK.floating_intensity(addr_dev, w_dev, I_dev, fic_dev)

                    Iact = GDK.gpu.Imodel.get()
                    fact = fic_dev.get()

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        Iact, Iref)
                    line = ['floating_intensity', name, iter, d, m,
                            a, 'Imodel', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        fact, ficref)
                    line = ['floating_intensity', name, iter, d, m,
                            a, 'fic', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

        return res

    def test_main_and_error_reduce(self, name, iter,
                                   math_type={'float', 'double'},
                                   data_type={'float', 'double'},
                                   acc_type={'float', 'double'}):

        res = []

        # Load data
        with h5py.File(self.datadir % name + "main_%04d.h5" % iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]
            w = f["w"][:]
            I = f["I"][:]
        # Load data
        with h5py.File(self.datadir % name + "error_reduce_%04d.h5" % iter, "r") as f:
            err_phot = f["err_phot"][:]

        # CPU Kernel
        auxref = np.copy(aux)
        errref = np.copy(err_phot)
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.main(auxref, addr, w, I)
        BGDK.error_reduce(addr, errref)
        LLerrref = BGDK.npy.LLerr

        addr_dev = gpuarray.to_gpu(addr)
        for d in data_type:
            for m in math_type:
                for a in acc_type:
                    if d == 'float':
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex64))
                        I_dev = gpuarray.to_gpu(I.astype(np.float32))
                        w_dev = gpuarray.to_gpu(w.astype(np.float32))
                        err_phot_dev = gpuarray.to_gpu(
                            err_phot.astype(np.float32))
                    else:
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex128))
                        I_dev = gpuarray.to_gpu(I.astype(np.float64))
                        w_dev = gpuarray.to_gpu(w.astype(np.float64))
                        err_phot_dev = gpuarray.to_gpu(
                            err_phot.astype(np.float64))

                    # GPU kernel
                    GDK = GradientDescentKernel(
                        aux_dev, addr.shape[1], accumulate_type=a, math_type=m)
                    GDK.allocate()
                    GDK.main(aux_dev, addr_dev, w_dev, I_dev)
                    GDK.error_reduce(addr_dev, err_phot_dev)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        auxref, aux_dev.get())
                    line = ['main_and_error_reduce', name, iter, d,
                            m, a, 'aux', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        LLerrref, GDK.gpu.LLerr.get())
                    line = ['main_and_error_reduce', name, iter, d,
                            m, a, 'LLerr', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        errref, err_phot_dev.get())
                    line = ['main_and_error_reduce', name, iter, d, m,
                            a, 'err_phot', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

        return res

    def test_make_a012(self, name, iter,
                       math_type={'float', 'double'},
                       data_type={'float', 'double'},
                       acc_type={'float', 'double'}):

        # Reduce the array size to make the tests run faster
        Nmax = 10
        Ymax = 128
        Xmax = 128

        res = []

        # Load data
        with h5py.File(self.datadir % name + "make_a012_%04d.h5" % iter, "r") as g:
            addr = g["addr"][:Nmax]
            I = g["I"][:Nmax, :Ymax, :Xmax]
            b_f = g["f"][:Nmax, :Ymax, :Xmax]
            b_a = g["a"][:Nmax, :Ymax, :Xmax]
            b_b = g["b"][:Nmax, :Ymax, :Xmax]
            fic = g["fic"][:Nmax]
        with h5py.File(self.datadir % name + "make_model_%04d.h5" % iter, "r") as h:
            aux = h["aux"][:Nmax, :Ymax, :Xmax]

        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.make_a012(b_f, b_a, b_b, addr, I, fic)
        Imodelref = BGDK.npy.Imodel
        LLerrref = BGDK.npy.LLerr
        LLdenref = BGDK.npy.LLden

        addr_dev = gpuarray.to_gpu(addr)
        for d in data_type:
            for m in math_type:
                for a in acc_type:
                    if d == 'float':
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex64))
                        I_dev = gpuarray.to_gpu(I.astype(np.float32))
                        b_f_dev = gpuarray.to_gpu(b_f.astype(np.complex64))
                        b_a_dev = gpuarray.to_gpu(b_a.astype(np.complex64))
                        b_b_dev = gpuarray.to_gpu(b_b.astype(np.complex64))
                        fic_dev = gpuarray.to_gpu(fic.astype(np.float32))
                    else:
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex128))
                        I_dev = gpuarray.to_gpu(I.astype(np.float64))
                        b_f_dev = gpuarray.to_gpu(b_f.astype(np.complex128))
                        b_a_dev = gpuarray.to_gpu(b_a.astype(np.complex128))
                        b_b_dev = gpuarray.to_gpu(b_b.astype(np.complex128))
                        fic_dev = gpuarray.to_gpu(fic.astype(np.float64))

                    GDK = GradientDescentKernel(aux_dev, addr.shape[1], queue=self.stream,
                                                math_type=m, accumulate_type=a)
                    GDK.allocate()
                    GDK.gpu.Imodel.fill(np.nan)
                    GDK.gpu.LLerr.fill(np.nan)
                    GDK.gpu.LLden.fill(np.nan)
                    GDK.make_a012(b_f_dev, b_a_dev, b_b_dev,
                                  addr_dev, I_dev, fic_dev)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        LLerrref, GDK.gpu.LLerr.get())
                    line = ['make_a012', name, iter, d, m, a,
                            'LLerr', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        LLdenref, GDK.gpu.LLden.get())
                    line = ['make_a012', name, iter, d, m, a,
                            'LLden', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        Imodelref, GDK.gpu.Imodel.get())
                    line = ['make_a012', name, iter, d, m, a,
                            'Imodel', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

        return res

    def test_fill_b(self, name, iter,
                    math_type={'float', 'double'},
                    data_type={'float', 'double'},
                    acc_type={'float', 'double'}):

        res = []

        # Load data

        Nmax = 10
        Ymax = 128
        Xmax = 128

        with h5py.File(self.datadir % name + "fill_b_%04d.h5" % iter, "r") as f:
            w = f["w"][:Nmax, :Ymax, :Xmax]
            addr = f["addr"][:]
            B = f["B"][:]
            Brenorm = f["Brenorm"][...]
            A0 = f["A0"][:Nmax, :Ymax, :Xmax]
            A1 = f["A1"][:Nmax, :Ymax, :Xmax]
            A2 = f["A2"][:Nmax, :Ymax, :Xmax]
        with h5py.File(self.datadir % name + "make_model_%04d.h5" % iter, "r") as f:
            aux = f["aux"][:Nmax, :Ymax, :Xmax]

        # CPU Kernel
        Bref = np.copy(B)
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.npy.Imodel = A0
        BGDK.npy.LLerr = A1
        BGDK.npy.LLden = A2
        BGDK.fill_b(addr, Brenorm, w, Bref)

        addr_dev = gpuarray.to_gpu(addr)
        for d in data_type:
            for m in math_type:
                for a in acc_type:
                    if d == 'float':
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex64))
                        w_dev = gpuarray.to_gpu(w.astype(np.float32))
                        B_dev = gpuarray.to_gpu(B.astype(np.float32))
                        A0_dev = gpuarray.to_gpu(A0.astype(np.float32))
                        A1_dev = gpuarray.to_gpu(A1.astype(np.float32))
                        A2_dev = gpuarray.to_gpu(A2.astype(np.float32))
                    else:
                        aux_dev = gpuarray.to_gpu(aux.astype(np.complex128))
                        w_dev = gpuarray.to_gpu(w.astype(np.float64))
                        B_dev = gpuarray.to_gpu(B.astype(np.float64))
                        A0_dev = gpuarray.to_gpu(A0.astype(np.float64))
                        A1_dev = gpuarray.to_gpu(A1.astype(np.float64))
                        A2_dev = gpuarray.to_gpu(A2.astype(np.float64))

                    GDK = GradientDescentKernel(
                        aux_dev, addr.shape[1], queue=self.stream, math_type=m, accumulate_type=a)
                    GDK.allocate()
                    GDK.gpu.Imodel = A0_dev
                    GDK.gpu.LLerr = A1_dev
                    GDK.gpu.LLden = A2_dev
                    GDK.fill_b(addr_dev, Brenorm, w_dev, B_dev)

                    num, num_mis, max_abs, max_rel = self._calc_diffs(
                        Bref, B_dev.get())
                    line = ['fill_b', name, iter, d, m, a,
                            'B', num, num_mis, max_rel, max_abs]
                    print(line)
                    res.append(line)

        return res

    def _calc_diffs(self, act, ref):
        diffs = np.abs(ref - act)
        max_abs = np.max(diffs[:])
        aref = np.abs(ref[:])
        max_rel = np.max(
            np.divide(diffs[:], aref, out=np.zeros_like(diffs[:]), where=aref > 0))
        num_mis = np.count_nonzero(diffs[:] > self.atol + self.rtol * aref)
        num = np.prod(ref.shape)

        return num, num_mis, max_abs, max_rel


tester = GradientDescentAccuracyTester()
print(tester.headings)

res = [tester.headings]
for ver in [("base", 10), ("regul", 50), ("floating", 0)]:
    res += tester.test_make_model(*ver)
    res += tester.test_floating_intensity(*ver)
    res += tester.test_main_and_error_reduce(*ver)
    res += tester.test_make_a012(*ver)
    res += tester.test_fill_b(*ver)

with open('ML_accuracy_test_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(res)

print('Done.')
