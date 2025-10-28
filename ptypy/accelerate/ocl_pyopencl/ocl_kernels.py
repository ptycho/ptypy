import pyopencl as cl
from pyopencl import array as cla
import numpy as np
import time

from . import get_ocl_queue
from .npy_kernels_for_block import AuxiliaryWaveKernel as AWK_NPY
from .npy_kernels_for_block import PoUpdateKernel as POK_NPY
from .npy_kernels_for_block import FourierUpdateKernel as FUK_NPY


class Adict(object):

    def __init__(self):
        pass


class OclBase(object):

    def __init__(self, queue_thread=None):

        self.queue = queue_thread if queue_thread is not None else get_ocl_queue()
        self._check_profiling()
        self.benchmark = dict()
        self.ocl_wg_size = (1, 16, 16)

    def _check_profiling(self):
        if self.queue.properties == cl.command_queue_properties.PROFILING_ENABLE:
            self.profile = True
        else:
            self.profile = False


class FourierUpdateKernel(FUK_NPY, OclBase):

    def __init__(self, aux, nmodes=1, queue_thread=None):
        FUK_NPY.__init__(self, aux, nmodes)
        OclBase.__init__(self, queue_thread)

        self.framesize = np.int32(np.prod(aux.shape[-2:]))

        assert self.queue is not None
        self.prg = cl.Program(self.queue.context, """
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
            __private float loc_af2a = 0.;
            __private float loc_af2b = 0.;
            
            // saves model intensity 
            //loc_af2[1] = 0;
            
            #pragma unroll
            
            for(int i=0; i<nmodes; i++){
                //loc_af2a = cfloat_abs(exit[(z_z+i)*dx*dx + y*dx + x]);
                //loc_af2b += loc_af2a * loc_af2a;
                loc_af2b += cfloat_abs_squared(exit[(z_z+i)*dx*dx + y*dx + x]);
            }
            
            loc_f[2] = sqrt(loc_af2b) - fmag[z_merged*dx*dx + y*dx + x];
            fdev[z_merged*dx*dx + y*dx + x] = loc_f[2];

            ferr[z_merged*dx*dx + y*dx + x] = fmask[z_merged*dx*dx + y*dx + x] * loc_f[2] * loc_f[2] / mask_sum[z_merged];
        }
        __kernel void fmag_all_update(int nmodes,
                                      float pbound,
                                    __global cfloat_t *f,
                                    __global float *fmask,
                                    __global float *fmag,
                                    __global float *fdev,
                                    __global float *err)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t z_merged = z/nmodes;
            size_t midx = z_merged*dx*dx + y*dx + x;
            size_t idx = z*dx*dx + y*dx + x;
            
            __private float renorm = sqrt(pbound/err[z_merged]);
            __private float eps = 1e-7;//pow(10.,-10);
            __private float fm=1.;
            
            __private float m=fmask[midx];
            __private float g=fmag[midx];
            __private float d=fdev[midx];
            
            if (renorm < 1.){            
                fm =  (1-m) + m * native_divide(g+d*renorm, d+g+eps);
                //fm =  m * (d*renorm +g) / (d+g+eps) + (1-m);
            }
            f[idx] = cfloat_mulr(f[idx] , fm );
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

    def allocate(self):
        self.npy.fdev = cla.zeros(self.queue, self.fshape, dtype=np.float32)
        self.npy.ferr = cla.zeros(self.queue, self.fshape, dtype=np.float32)

    def fourier_error(self, b_aux, addr, mag, mask, mask_sum):
        fdev = self.npy.fdev
        ferr = self.npy.ferr

        self.prg.fourier_error(self.queue, mag.shape, self.ocl_wg_size,
                               self.nmodes,
                               b_aux.data, mag.data, fdev.data, ferr.data,
                               mask.data, mask_sum.data)
        self.queue.finish()

    def error_reduce(self, addr, err_sum):
        # batch buffers
        ferr = self.npy.ferr

        self.prg.reduce_one_step(self.queue, (err_sum.shape[0], 64), (1, 64),
                                 self.framesize,
                                 ferr.data, err_sum.data)
        self.queue.finish()

    def fmag_all_update(self, b_aux, addr, mag, mask, err_sum, pbound=0.0):
        # maybe cache this?
        pbound = np.float32(pbound)

        sh = mag.shape
        shape = (sh[0] * self.nmodes, sh[1], sh[2])  # could have also used `addr` for this
        fdev = self.npy.fdev

        self.prg.fmag_all_update(self.queue, shape, self.ocl_wg_size,
                                 self.nmodes, pbound,
                                 b_aux.data, mask.data, mag.data, fdev.data,
                                 err_sum.data)
        self.queue.finish()


class AuxiliaryWaveKernel(AWK_NPY, OclBase):

    def __init__(self, queue_thread=None):
        AWK_NPY.__init__(self)
        OclBase.__init__(self, queue_thread)

        self._ob_shape = None
        self._ob_id = None
        # self.ocl_wg_size = (1, 16, 16)

        self.prg = cl.Program(self.queue.context, """
        #include <pyopencl-complex.h>
        
        // Define usable names for buffer access
        
        
        #define pr_dlayer(k) addr[k*15]
        #define ex_dlayer(k) addr[k*15 + 6]
        
        #define obj_dlayer(k) addr[k*15 + 3]
        #define obj_roi_row(k) addr[k*15 + 4]
        #define obj_roi_column(k) addr[k*15 + 5]
        
        // calculates: 
        // aux =  (1+alpha)*pod.probe*pod.object - alpha* pod.exit
        __kernel void build_aux(float alpha,
                            int ob_sh_row,
                            int ob_sh_col,
                            __global cfloat_t *aux,  
                            __global cfloat_t *ob,
                            __global cfloat_t *pr,
                            __global cfloat_t *ex,
                            __global int *addr)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t zb = get_global_id(0);
            
            //size_t obj_idx = obj_dlayer(z)*ob_sh_row*ob_sh_col + (y+obj_roi_row(z))*ob_sh_col + obj_roi_column(z)+x;

            cfloat_t ex0 = cfloat_rmul(alpha,ex[ex_dlayer(z)*dx*dx + y*dx + x]);
            cfloat_t ex1 = cfloat_mul(ob[obj_dlayer(z)*ob_sh_row*ob_sh_col + (y+obj_roi_row(z))*ob_sh_col + obj_roi_column(z)+x],pr[pr_dlayer(z)*dx*dx + y*dx+x]);
            //loc_sub[3] = cfloat_fromreal(1. + loc_sub[0].real);
            
            //cfloat_t ex2 = cfloat_sub(cfloat_rmul(1.+alpha,ex1),ex0);
            aux[zb*dx*dx + y*dx + x] = cfloat_sub(cfloat_rmul(1.+alpha,ex1),ex0);            
        }
        
        __kernel void build_exit(int ob_sh_row,
                            int ob_sh_col,
                            __global cfloat_t *f,
                            __global cfloat_t *ob,
                            __global cfloat_t *pr,
                            __global cfloat_t *ex,
                            __global int *addr)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t zb = get_global_id(0); 
            
            size_t obj_idx = obj_dlayer(z)*ob_sh_row*ob_sh_col + (y+obj_roi_row(z))*ob_sh_col + obj_roi_column(z)+x;
            
            cfloat_t ex1 = cfloat_mul(ob[obj_idx],pr[pr_dlayer(z)*dx*dx + y*dx+x]);
            cfloat_t df = cfloat_sub(f[zb*dx*dx + y*dx + x] , ex1); 
            f[zb*dx*dx + y*dx + x] = df ; // t.b. removed later
            ex[ex_dlayer(z)*dx*dx + y*dx + x] = cfloat_add(ex[ex_dlayer(z)*dx*dx + y*dx + x] , df);
        }

        """).build()

    def build_aux(self, b_aux, addr, ob, pr, ex, alpha=1.0):
        obr, obc = self._cache_object_shape(ob)
        ev = self.prg.build_aux(self.queue, ex.shape, self.ocl_wg_size,
                                np.float32(alpha), obr, obc,
                                b_aux.data, ob.data, pr.data, ex.data, addr.data)
        return ev

    def build_exit(self, b_aux, addr, ob, pr, ex):
        obr, obc = self._cache_object_shape(ob)
        ev = self.prg.build_exit(self.queue, ex.shape, self.ocl_wg_size,
                                 obr, obc,
                                 b_aux.data, ob.data, pr.data, ex.data, addr.data)
        return ev

    def _cache_object_shape(self, ob):
        oid = id(ob)

        if not oid == self._ob_id:
            self._ob_id = oid
            self._ob_shape = (np.int32(ob.shape[-2]), np.int32(ob.shape[-1]))

        return self._ob_shape


class PoUpdateKernel(POK_NPY, OclBase):

    def __init__(self, queue_thread=None):
        POK_NPY.__init__(self)
        OclBase.__init__(self, queue_thread)
        self.ocl_wg_size = (16, 16)
        self.prg = cl.Program(self.queue.context, """
        #include <pyopencl-complex.h>
        
        // Define usable names for buffer access
        
        #define pr_dlayer(k) addr[k*15]
        #define ex_dlayer(k) addr[k*15 + 6]
        
        #define obj_dlayer(k) addr[k*15 + 3]
        #define obj_roi_row(k) addr[k*15 + 4]
        #define obj_roi_column(k) addr[k*15 + 5]
        
        __kernel void ob_update(int pr_sh,
                                int ob_modes,
                                int num_pods,
                                __global cfloat_t *ob_g,
                                __global float *obn_g,
                                __global cfloat_t *pr_g,
                                __global cfloat_t *ex_g,
                                __global int *addr)
        {
            size_t z = get_global_id(1);
            size_t dz = get_global_size(1);
            size_t y = get_global_id(0);
            size_t dy = get_global_size(0);
            __private cfloat_t ob[8];
            __private float obn[8];

            int v1 = 0;
            int v2 = 0;
            size_t x = y*dz + z;
            cfloat_t pr = pr_g[0];
            
            for (int i=0;i<ob_modes;i++){
                ob[i] = ob_g[i*dy*dz + y*dz + z];
                obn[i] = obn_g[i*dy*dz + y*dz + z];
            }
            
            for (int i=0;i<num_pods;i++){
                v1 = (int)y - obj_roi_row(i);
                v2 = (int)z - obj_roi_column(i);
                if ((v1>=0)&&(v1<pr_sh)&&(v2>=0)&&(v2<pr_sh)){ 
                    pr = pr_g[pr_dlayer(i)*pr_sh*pr_sh + v1*pr_sh + v2];
                    ob[obj_dlayer(i)] = cfloat_add(ob[obj_dlayer(i)],cfloat_mul(cfloat_conj(pr),ex_g[ex_dlayer(i)*pr_sh*pr_sh +v1*pr_sh + v2]));
                    //obn[obj_dlayer(i)] = cfloat_add(obn[obj_dlayer(i)],cfloat_mul(pr,cfloat_conj(pr)));
                    obn[obj_dlayer(i)] += cfloat_abs_squared(pr);
                }
            
            }
            for (int i=0;i<ob_modes;i++){
                ob_g[i*dy*dz + y*dz + z] = ob[i];
                obn_g[i*dy*dz + y*dz + z] = obn[i];
            }
        }
        __kernel void pr_update(int pr_sh,
                                int ob_sh_row,
                                int ob_sh_col,
                                int pr_modes,
                                int num_pods,
                                __global cfloat_t *pr_g,
                                __global float *prn_g,
                                __global cfloat_t *ob_g,
                                __global cfloat_t *ex_g,
                                __global int *addr)
        {
            size_t z = get_global_id(1);
            size_t dz = get_global_size(1);
            size_t y = get_global_id(0);
            size_t dy = get_global_size(0);
            __private cfloat_t pr[8];
            __private float prn[8];
            
            int v1 = 0;
            int v2 = 0;
            cfloat_t ob; 
            
            for (int i=0;i<pr_modes;i++){
                pr[i] = pr_g[i*dy*dz + y*dz + z];
                prn[i] = prn_g[i*dy*dz + y*dz + z];
            }

            for (int i=0;i<num_pods;i++){
                v1 = (int)y + obj_roi_row(i);
                v2 = (int)z + obj_roi_column(i);
                if ((v1>=0)&&(v1<ob_sh_row)&&(v2>=0)&&(v2<ob_sh_col)){ 
                    ob = ob_g[obj_dlayer(i)*ob_sh_row*ob_sh_col + v1*ob_sh_col + v2];
                    pr[pr_dlayer(i)] = cfloat_add(pr[pr_dlayer(i)], cfloat_mul(cfloat_conj(ob),ex_g[ex_dlayer(i)*pr_sh*pr_sh +y*pr_sh + z]));
                    //prn[pr_dlayer(i)] = cfloat_add(prn[pr_dlayer(i)],cfloat_mul(ob,cfloat_conj( ob )));
                    prn[pr_dlayer(i)] += cfloat_abs_squared(ob);
                }
            
            }

            for (int i=0;i<pr_modes;i++){
                pr_g[i*dy*dz + y*dz + z] = pr[i];
                prn_g[i*dy*dz + y*dz + z] = prn[i];
            }

        }

        """).build()

    def ob_update(self, addr, ob, obn, pr, ex):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        num_pods = np.int32(addr.shape[0] * addr.shape[1])
        ev = self.prg.ob_update(self.queue, ob.shape[-2:], self.ocl_wg_size,
                                prsh[-1],
                                obsh[0], num_pods,
                                ob.data, obn.data, pr.data, ex.data, addr.data)
        return ev

    def pr_update(self, addr, pr, prn, ob, ex):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        num_pods = np.int32(addr.shape[0] * addr.shape[1])

        ev = self.prg.pr_update(self.queue, pr.shape[-2:], self.ocl_wg_size,
                                prsh[-1], obsh[-2], obsh[-1],
                                prsh[0], num_pods,
                                pr.data, prn.data, ob.data, ex.data, addr.data)
        return ev
