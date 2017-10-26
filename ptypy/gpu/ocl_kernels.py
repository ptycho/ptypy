import pyopencl as cl
from pyopencl import array as cla
import numpy as np
import time
from inspect import getargspec
from collections import OrderedDict

class Adict(object):
    
    def __init__(self):
        pass
        

class Fourier_update_kernel(object):
    
    def __init__(self, queue_thread=None, nmodes = 1, pbound = 0.0):
        
        self.pbound = np.float32(pbound)
        self.nmodes = np.int32(nmodes)
        self.queue = queue_thread
        self.verbose = False
    
    def log(self, x):
            print(x)
            
    def configure(self,I, mask, f):
        
        self.fshape = I.shape
        self.shape = (self.nmodes*I.shape[0],I.shape[1],I.shape[2])
        assert self.shape == f.shape
        self.framesize = np.int32(np.prod(I.shape[-2:]))
        
        self.benchmark = OrderedDict()
        
        self.npy = Adict()
        self.npy.f = f
        self.npy.fmask = mask
        self.npy.mask_sum = mask.sum(-1).sum(-1)
        d = I.copy()
        d[d<0.] = 0.0 # just in case
        d[np.isnan(d)] = 0.0
        self.npy.fmag = np.sqrt(d)
        self.npy.err_fmag = np.zeros((self.fshape[0],),dtype=np.float32)
        # temporary buffer arrays
        self.npy.fdev = np.zeros_like(self.npy.fmag)
        self.npy.ferr = np.zeros_like(self.npy.fmag)
        self.npy.fm = np.zeros_like(self.npy.fmag)
        
        self.kernels = [
            'fourier_error',
            'error_reduce',
            'calc_fm',
            'fmag_update'        
        ]
        
    def sync_ocl(self):
        for key,array in self.npy.__dict__.iteritems():
            self.ocl.__dict__[key].set(array)
            
    def configure_ocl(self):
        self.ocl = Adict()
        for key,array in self.npy.__dict__.iteritems():
            self.ocl.__dict__[key] = cla.to_device(self.queue, array)
             
        assert self.queue is not None
        self.prg = cl.Program(self.queue.context,"""
        #include <pyopencl-complex.h>
        __kernel void fourier_error(int nmodes,
                                   __global cfloat_t *exit,
                                   __global float *fmag,
                                   __global float *fdev,             // fdev = af - fmag 
                                   __global float *ferr,    // fmask*fdev**2  
                                   __global float *fmask,
                                   __global float *mask_sum
                                   //__global cfloat_t *post_fft_g
                                   )
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z_merged = get_global_id(0);
            size_t z_z = z_merged*nmodes;

            __private float loc_f [3];
            __private float loc_af2 [2];
            
            // saves model intensity 
            loc_af2[1] = 0;
            
            for(int i=0; i<nmodes; i++){
                //exit[(z_z+i)*dx*dx + y*dx + x] = cfloat_mul(exit[(z_z+i)*dx*dx + y*dx + x], post_fft_g[y*dx + x]);
                loc_af2[0] = cfloat_abs(exit[(z_z+i)*dx*dx + y*dx + x]);
                loc_af2[1] += loc_af2[0]*loc_af2[0];
            }
            
            loc_f[2] = sqrt(loc_af2[1]) - fmag[z_merged*dx*dx + y*dx + x];
            fdev[z_merged*dx*dx + y*dx + x] = loc_f[2];

            ferr[z_merged*dx*dx + y*dx + x] = fmask[z_merged*dx*dx + y*dx + x] * loc_f[2] * loc_f[2] / mask_sum[z_merged];
        }
        __kernel void calc_fm(float pbound,
                            __global float *fm,
                            __global float *fmask,
                            __global float *fmag,
                            __global float *fdev,
                            __global float *ferr)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z_merged = get_global_id(0);
            size_t lx = get_local_id(2);
            size_t idx = z_merged*dx*dx + y*dx + x;

            __private float a[3];
            
            float error = ferr[z_merged];
            float renorm = sqrt(pbound/error);
            const float eps = 1e-10;
            
            if (renorm < 1.){
                a[0] = fmask[idx]; 
                a[1] = 1. - a[0];
                a[2] = fdev[idx] * renorm + fmag[idx];
                a[2] /= fdev[idx] + fmag[idx] + eps;
                fm[idx] = a[0] * a[2] + a[1];
            }
            else {
                fm[idx] = 1.0;
            }            
        }
        __kernel void fmag_update(int nmodes,
                            __global cfloat_t *f,
                            __global float *fm
                            //__global cfloat_t *pre_ifft_g,
                            )
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t z_merged = z/nmodes;
            
            float fac = fm[z_merged*dx*dx + y*dx + x];
            //cfloat_t ft = cfloat_mul(f[z*dx*dx + y*dx + x], pre_ifft_g[y*dx + x]);
            f[z*dx*dx + y*dx + x] = cfloat_mulr(f[z*dx*dx + y*dx + x] , fac);
            
        }
        __kernel void reduce_one_step(int framesize,
                    __global const float* buffer,
                    __global float* result) {
                    
            __local float scratch[256];
            
            size_t z = get_global_id(0);
            size_t y = get_global_id(1);
      
            size_t ldz = get_local_size(0);
            
            size_t ly = get_local_id(1);
            size_t ldy = get_local_size(1);
            
            
            float accumulator = 0;
            size_t jump = z*framesize + y;
            size_t next = (z+1)*framesize + y;
            
            
            while (jump < next) {
                accumulator += buffer[jump];
                jump += ldy;
            }
               
            scratch[ly] = accumulator;
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            for(int offset = ldy / 2;
                    offset > 0;
                    offset = offset / 2) 
                {
                
                    if (ly < offset) {
                      scratch[ly] += scratch[ly + offset];
                    }
                
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            
            if (ly == 0) {
                result[z] = scratch[0];
            }
        }
        """).build()
    
    def execute_ocl(self, kernel_name=None, compare = False, sync=False):
                
        if kernel_name is None:
            for kernel in self.kernels:
                self.execute_ocl(kernel, compare, sync)
        else:
            self.log("KERNEL " + kernel_name)
            m_ocl = getattr(self,'_ocl_' + kernel_name )
            m_npy = getattr(self,'_npy_' + kernel_name )
            ocl_kernel_args = getargspec(m_ocl).args[1:]
            npy_kernel_args = getargspec(m_npy).args[1:]
            assert ocl_kernel_args == npy_kernel_args
            # OCL
            if sync:
                self.sync_ocl()
            args = [getattr(self.ocl,a).data for a in ocl_kernel_args]
            
            self.benchmark[kernel_name] = -time.time()
            m_ocl(*args)
            self.benchmark[kernel_name]+= time.time()
            
            if compare:
                args = [getattr(self.npy,a) for a in npy_kernel_args]
                m_npy(*args)
                self.verify_ocl()
        
        return self.ocl.err_fmag.get()
        
    def execute_npy(self, kernel_name=None):
        
        if kernel_name is None:
            for kernel in self.kernels:
                self.execute_npy(kernel)
        else:
            self.log("KERNEL " + kernel_name)
            m_npy = getattr(self,'_npy_' + kernel_name )
            npy_kernel_args = getargspec(m_npy).args[1:]
            args = [getattr(self.npy,a) for a in npy_kernel_args]
            m_npy(*args)
               
        return self.npy.err_fmag
    
    
    def _npy_fourier_error(self,f, fmag, fdev, ferr, fmask, mask_sum):
        sh = f.shape
        tf = f.reshape(sh[0]/self.nmodes,self.nmodes,sh[1],sh[2])
    
        af = np.sqrt((np.abs(tf)**2).sum(1))
        
        fdev[:] = af - fmag
        ferr[:] = fmask * np.abs(fdev)**2 / mask_sum.reshape((mask_sum.shape[0],1,1))
        
    def _ocl_fourier_error(self,f, fmag, fdev, ferr, fmask, mask_sum):
        self.prg.fourier_error(self.queue, self.fshape, (1,1,32), self.nmodes, 
                                f, fmag, fdev, ferr, fmask, mask_sum)
        self.queue.finish()       
        
    def _npy_error_reduce(self, ferr, err_fmag):
        err_fmag[:] = ferr.astype(np.double).sum(-1).sum(-1).astype(np.float)
        
    def _ocl_error_reduce(self, ferr, err_fmag):
        shape = (self.fshape[0],64),
        self.prg.reduce_one_step(self.queue, (self.fshape[0],64), (1,64), self.framesize, 
                                ferr, err_fmag)
        self.queue.finish()
        
    def _npy_calc_fm(self,fm, fmask, fmag, fdev, err_fmag):

        renorm = np.ones_like(err_fmag)
        ind = err_fmag > self.pbound
        renorm[ind] = np.sqrt(self.pbound / err_fmag[ind])
        renorm = renorm.reshape((renorm.shape[0],1,1))
        af = fdev + fmag
        fm[:] = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
        """
        # C Amplitude correction           
        if err_fmag > self.pbound:
            # Power bound is applied
            renorm = np.sqrt(pbound / err_fmag)
            fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
        else:
            fm = 1.0
        """
    def _ocl_calc_fm(self,fm, fmask, fmag, fdev, err_fmag):
        self.prg.calc_fm(self.queue, self.fshape, (1,1,32), self.pbound,
                          fm, fmask, fmag, fdev, err_fmag)
        self.queue.finish()
        
    def _npy_fmag_update(self,f,fm):
        sh = f.shape
        tf = f.reshape(sh[0]/self.nmodes,self.nmodes,sh[1],sh[2])
        sh = fm.shape
        tf *= fm.reshape(sh[0],1,sh[1],sh[2])
    
    def _ocl_fmag_update(self,f,fm):
        self.prg.fmag_update(self.queue, self.shape, (1,1,32), 
                    self.nmodes,f,fm)
        self.queue.finish()
        
    def verify_ocl(self, precision=2**(-23)):
        
        for name, val in self.npy.__dict__.iteritems():
            val2 = self.ocl.__dict__[name].get()
            val = val
            if np.allclose(val,val2,atol=precision):
                continue 
            else:
                dev = np.std(val - val2)
                print("Key %s : %.2e std, %.2e mean" % (name, dev, np.mean(val)))        
        
    @classmethod
    def test(cls, shape =  (739,256,256), nmodes = 1, pbound = 0.0):

        L,M,N = shape
        fshape = shape
        shape = (nmodes*L,M,N)
        
        f = np.random.rand(*shape).astype(np.complex64) * 200 
        I = np.random.rand(*fshape).astype(np.float32) * 200**2 * nmodes
        mask = (I > 10).astype(np.float32) 
        
        
        devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
        queue = cl.CommandQueue(cl.Context([devices[0]]))
        
        inst = cls(queue_thread=queue, nmodes = nmodes, pbound = pbound)
        inst.configure(I, mask, f.copy())
        inst.configure_ocl()
               
        #inst.execute_ocl(compare=True,sync=False)
        inst.execute_ocl()
        g = inst.ocl.f.get()
        inst.execute_npy()
        f = inst.npy.f
        """
        g = f.copy()
        inst.configure(I, mask, g)
        err = inst.execute_npy(g)
        inst.verify_ocl()
        """
        print('Error : %.2e' % np.std(f-g))
        for key, val in inst.benchmark.items():
            print('Kernel %s : %.2f ms' % (key,val*1000))

if __name__=='__main__':
    Fourier_update_kernel.test()
    """
    nmodes = 8
    pbound = 0.
    fshape = (50,128,128)
    shape = (nmodes*50,128,128)
    
    f = np.random.rand(*shape).astype(np.complex64) * 200 
    I = np.random.rand(*fshape).astype(np.float32) * 200**2 * nmodes
    mask = (I > 10).astype(np.float32) 
    
    
    devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
    queue = cl.CommandQueue(cl.Context([devices[0]]))
    
    inst = Fourier_update_kernel(queue_thread=queue, nmodes = nmodes, pbound = 0.0)
    inst.configure(I, mask, f.copy())
    inst.configure_ocl()
    
    inst2 = Fourier_update_kernel(queue_thread=queue, nmodes = nmodes, pbound = 0.0)
    inst2.configure(I, mask, f.copy())
    inst2.configure_ocl()
    #err = inst.execute_npy(f)
    #gf = cla.to_device(queue,f)
    #err_ocl = inst.execute_ocl(gf)
    
    inst.execute_ocl_auto(True,False)
    g = inst.ocl.f.get()
    f = inst.npy.f
    print np.std(f-g)
    """
