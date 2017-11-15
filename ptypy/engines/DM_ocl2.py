# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

#from .. import core
from __future__ import division
import os.path
from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
#from utils import basic_fourier_update
from . import BaseEngine
import numpy as np
import time
import pyopencl as cl

from pyopencl import array as cla
from pyopencl import clmath as clm
from .. import gpu

# queue = gpu.get_ocl_queue()

### TODOS 
# 
# - Make DM_ocl a subclass of DM
# - Transfer remaining kernel code in a ocl_kernels.py
# - The Propagator needs to be made somewhere else
# - Get it running faster with MPI
# - implement "batching" when processing frames to lower the pressure on memory

## for debugging
from matplotlib import pyplot as plt

__all__=['DM']

parallel = u.parallel


DEFAULT = u.Param(
    fourier_relax_factor = 0.05,
    alpha = 1,
    update_object_first = True,
    overlap_converge_factor = .1,
    overlap_max_iterations = 10,
    probe_inertia = 1e-9,               # Portion of probe that is kept from iteraiton to iteration, formally cfact
    object_inertia = 1e-4,              # Portion of object that is kept from iteraiton to iteration, formally DM_smooth_amplitude
    obj_smooth_std = None,              # Standard deviation for smoothing of object between iterations
    clip_object = None,                 # None or tuple(min,max) of desired limits of the object modulus
)

        
def gaussian_kernel(sigma, size=None, sigma_y=None, size_y=None):
    size = int(size)
    sigma = np.float(sigma)
    if not size_y:
        size_y = size
    if not sigma_y:
        sigma_y = sigma
    
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    
    g = np.exp(-(x**2/(2*sigma**2)+y**2/(2*sigma_y**2)))
    return g / g.sum()
    
                
class DM_ocl(BaseEngine):
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        if pars is None:
            pars = DEFAULT.copy()
            
        super(DM_ocl,self).__init__(ptycho_parent,pars)
          
        self.queue = gpu.get_ocl_queue()
        
        # allocator for READ only buffers
        #self.const_allocator = cl.tools.ImmediateAllocator(queue, cl.mem_flags.READ_ONLY)
        ## gaussian filter
        # dummy kernel
        if not self.p.obj_smooth_std: 
            gauss_kernel = gaussian_kernel(1,1).astype(np.float32)
        else:
            gauss_kernel = gaussian_kernel(self.p.obj_smooth_std,self.p.obj_smooth_std).astype(np.float32)
        kernel_pars = {'kernel_sh_x' : gauss_kernel.shape[0], 'kernel_sh_y': gauss_kernel.shape[1]}
        
        self.gauss_kernel_gpu = cla.to_device(self.queue,gauss_kernel)
        
        ## GPU Kernel
        self.prg = cl.Program(self.queue.context, """
        #include <pyopencl-complex.h>
        
        // filter shape
        #define KERNEL_SHAPE_X %(kernel_sh_x)d
        #define KERNEL_SHAPE_Y %(kernel_sh_y)d
        
        // Define usable names for buffer access
        
        #define obj_dlayer(k) addr[k*15 + 3]
        #define pr_dlayer(k) addr[k*15]
        #define ex_dlayer(k) addr[k*15 + 6]
        
        #define obj_roi_row(k) addr[k*15 + 4]
        #define obj_roi_column(k) addr[k*15 + 5]
        
        #define obj_sh_row info[3]
        #define obj_sh_column info[4]
        
        #define pr_sh info[5]
        
        #define N_pods info[0]*info[1]
        
        #define nmodes info[1]
                       
        __kernel void build_aux(__global int *info,
                            __global float *DM_info,
                            __global cfloat_t *ob_g,
                            __global cfloat_t *pr_g,
                            __global cfloat_t *ex_g,
                            __global cfloat_t *f_g,     // calculate: (1+alpha)*pod.probe*pod.object - alpha* pod.exit
                            //__global float *af2,
                            __global cfloat_t *pre_fft_g,
                            __global int *addr)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            //__private cfloat_t loc_sub [4];
            //__private cfloat_t loc_res [1];
            __private float alpha = DM_info[0];
            
            //loc_sub[0] = cfloat_fromreal(DM_info[0]);
            cfloat_t ex0 = cfloat_rmul(alpha,ex_g[ex_dlayer(z)*dx*dx + y*dx + x]);
            cfloat_t ex1 = cfloat_mul(ob_g[obj_dlayer(z)*obj_sh_row*obj_sh_column + (y+obj_roi_row(z))*obj_sh_column + obj_roi_column(z)+x],pr_g[pr_dlayer(z)*dx*dx + y*dx+x]);
            //loc_sub[3] = cfloat_fromreal(1. + loc_sub[0].real);
            
            cfloat_t ex2 = cfloat_sub(cfloat_rmul(1.+alpha,ex1),ex0);
            f_g[z*dx*dx + y*dx + x] = cfloat_mul(ex2,pre_fft_g[y*dx+x]);
            //af2[(z/nmodes)*dx*dx + y*dx + x] = 0;          //maybe better with if z < af2_size
            
        }
        
        
        __kernel void post_fft(__global int *info,
                            __global cfloat_t *f_g,
                            __global float *af2,
                            __global cfloat_t *post_fft_g)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t z_z = z*nmodes;
            __private float loc_f[2];
            loc_f[1] = 0;
            
            
            for(int i=0; i<nmodes; i++){
                f_g[(z_z+i)*dx*dx + y*dx + x] = cfloat_mul(f_g[(z_z+i)*dx*dx + y*dx + x], post_fft_g[y*dx + x]);
                loc_f[0] = cfloat_abs(f_g[(z_z+i)*dx*dx + y*dx + x]);
                loc_f[1] += loc_f[0]*loc_f[0];
            }
            af2[z*dx*dx + y*dx + x] = loc_f[1];
            
        }
        
        __kernel void dev(__global cfloat_t *f_g,
                            __global float *af,
                            __global float *af2,
                            __global float *fmag,
                            __global float *fdev,             // fdev = af - fmag 
                            __global float *I,
                            __global float *err_fmag_t,    // fmask*fdev**2  
                            __global float *mask_g,
                            __global float *mask_sum_g,
                            __global float *err_ph_temp)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z_merged = get_global_id(0);
            __private float loc_err_phot[5];
            __private float loc_f [3];
            
            loc_f[1] = af2[z_merged*dx*dx + y*dx + x];
            af[z_merged*dx*dx + y*dx + x] = sqrt(loc_f[1]);
            loc_f[2] = af[z_merged*dx*dx + y*dx + x] - fmag[z_merged*dx*dx + y*dx + x];
            fdev[z_merged*dx*dx + y*dx + x] = loc_f[2];
            
            loc_err_phot[0] = mask_g[z_merged*dx*dx + y*dx + x];
            loc_err_phot[1] = loc_f[1] - I[z_merged*dx*dx + y*dx + x];     // af2 - I
            loc_err_phot[2] = loc_err_phot[1]*loc_err_phot[1];                     // (af2 - I)**2
            loc_err_phot[3] = loc_err_phot[2] / (I[z_merged*dx*dx + y*dx + x]+1.);
            loc_err_phot[4] = loc_err_phot[3] / mask_sum_g[z_merged];
            
            err_ph_temp[z_merged*dx*dx + y*dx + x] = loc_err_phot[4];
    
            err_fmag_t[z_merged*dx*dx + y*dx + x] = loc_err_phot[0] * loc_f[2] * loc_f[2] / mask_sum_g[z_merged];
        }
        __kernel void dev_post_fft(__global int *info,
                                   __global cfloat_t *f_g,
                                   __global cfloat_t *post_fft_g,
                                   __global float *fmag,
                                   __global float *fdev,             // fdev = af - fmag 
                                   __global float *err_fmag,    // fmask*fdev**2  
                                   __global float *mask_g,
                                   __global float *mask_sum_g)
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
                f_g[(z_z+i)*dx*dx + y*dx + x] = cfloat_mul(f_g[(z_z+i)*dx*dx + y*dx + x], post_fft_g[y*dx + x]);
                loc_af2[0] = cfloat_abs(f_g[(z_z+i)*dx*dx + y*dx + x]);
                loc_af2[1] += loc_af2[0]*loc_af2[0];
            }
            
            loc_f[2] = sqrt(loc_af2[1]) - fmag[z_merged*dx*dx + y*dx + x];
            fdev[z_merged*dx*dx + y*dx + x] = loc_f[2];

            err_fmag[z_merged*dx*dx + y*dx + x] = mask_g[z_merged*dx*dx + y*dx + x] * loc_f[2] * loc_f[2] / mask_sum_g[z_merged];
        }
        __kernel void calc_fm(__global float *DM_info,
                            __global float *fm,
                            __global float *mask,
                            __global float *fmag,
                            __global float *fdev,
                            //__global float *af,
                            __global float *err_fmag)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z_merged = get_global_id(0);
            size_t lx = get_local_id(2);
            size_t idx = z_merged*dx*dx + y*dx + x;

            __private float a[3];
            
            float pbound = fabs(DM_info[1]);
            float error = err_fmag[z_merged];
            float renorm = sqrt(pbound/error);
            const float eps = 1e-10;
            
            //pbound = fabs(DM_info[1]);
            if (renorm < 1.){
                a[0] = mask[idx]; 
                a[1] = 1. - a[0];
                a[2] = fdev[idx] * renorm + fmag[idx];
                //fm_temp[2] += fmag[z_merged*dx*dx + y*dx + x];
                //fm_temp[3] = af[z_merged*dx*dx + y*dx + x] + pow(10.,-10);
                a[2] /= fdev[idx] + fmag[idx] + eps;
                //fm_temp[2] /= fm_temp[3];
                fm[idx] = a[0] * a[2] + a[1];
            }
            else {
                fm[idx] = 1.0;
            }
            //fm[idx] = mask_g[idx]*fm_t + 1.-mask_g[idx];
            
        }
        __kernel void fmag_update(__global int *info,
                            __global cfloat_t *f,
                            __global float *fm,
                            __global cfloat_t *pre_ifft_g)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t z_merged = z/nmodes;
            
            float fac = fm[z_merged*dx*dx + y*dx + x];
            cfloat_t ft = cfloat_mul(f[z*dx*dx + y*dx + x], pre_ifft_g[y*dx + x]);
            f[z*dx*dx + y*dx + x] = cfloat_mulr(ft , fac);
            
        }
        __kernel void fmag_update2(__global int *info,
                            __global cfloat_t *f,
                            __global float *fm,
                            __global cfloat_t *pre_ifft_g)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t zi = z*nmodes;
            
            float fac = fm[z*dx*dx + y*dx + x];
            for (int i=0;i<nmodes;i++){
                cfloat_t ft = cfloat_mul(f[(zi+i)*dx*dx + y*dx + x], pre_ifft_g[y*dx + x]);
                f[(zi+i)*dx*dx + y*dx + x] = cfloat_mulr(ft , fac);
            }
        }
        __kernel void fmag_all_update(__global int *info,
                                    __global float *DM_info,
                                    __global cfloat_t *f,
                                    __global cfloat_t *pre_ifft_g,
                                    __global float *mask_g,
                                    __global float *fmag,
                                    __global float *fdev,
                                    __global float *err_fmag)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            size_t z_merged = z/nmodes;
            size_t midx = z_merged*dx*dx + y*dx + x;
            size_t idx = z*dx*dx + y*dx + x;
            
            __private float pbound = fabs(DM_info[1]);
            __private float renorm = sqrt(pbound/err_fmag[z_merged]);
            __private float eps = 1e-10;//pow(10.,-10);
            
            __private float m=mask_g[midx];
            __private float g=fmag[midx];
            __private float d=fdev[midx];
                        
            float fm =  m * native_divide(d*renorm +g, d+g+eps) + (1-m);
            
            cfloat_t ft = cfloat_mul(f[idx], pre_ifft_g[y*dx + x]);
            f[idx] = cfloat_mulr(ft , fm );
            
            /*
            fm_temp[0] = mask_g[idx];
            fm_temp[1] = 1. - fm_temp[0];
            fm_temp[2] = fdev[idx] * renorm + fmag[idx];
            fm_temp[2] /= fdev[idx] + fmag[idx] + eps;
            fm_temp[3] = fm_temp[0]*fm_temp[2] + fm_temp[1];
            
            ft = cfloat_mul(f[z*dx*dx + y*dx + x], pre_ifft_g[y*dx + x]);
            f[z*dx*dx + y*dx + x] = cfloat_mulr(ft , fm_temp[3] );
            */
        }
        
        __kernel void post_ifft(__global int *info,
                            __global cfloat_t *ob_g,
                            __global cfloat_t *pr_g,
                            __global cfloat_t *ex_g,
                            __global cfloat_t *f,
                            __global cfloat_t *post_ifft_g,
                            __global int *addr)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0);
            __private cfloat_t ex_df[3];
            
            ex_df[0] = cfloat_mul(f[z*dx*dx + y*dx + x],post_ifft_g[y*dx + x]);
            ex_df[1] = cfloat_mul(ob_g[obj_dlayer(z)*obj_sh_row*obj_sh_column + (y+obj_roi_row(z))*obj_sh_column + obj_roi_column(z)+x],pr_g[pr_dlayer(z)*dx*dx + y*dx+x]);
            ex_df[2] = cfloat_sub(ex_df[0] , ex_df[1]); 
            f[z*dx*dx + y*dx + x] = ex_df[2] ; // t.b. removed later
            ex_g[ex_dlayer(z)*dx*dx + y*dx + x] = cfloat_add(ex_g[ex_dlayer(z)*dx*dx + y*dx + x] , ex_df[2]);
    
        }

        __kernel void new_obj_update(__global int *info,
                                __global cfloat_t *ob_g,
                                __global cfloat_t *pr_g,
                                __global cfloat_t *obn_g,
                                __global cfloat_t *ex_g,
                                __global int *addr)
        {
            size_t z = get_global_id(0);
            size_t dz = get_global_size(1);
            size_t y = get_global_id(1);
            __private cfloat_t ob[8];
            __private cfloat_t obn[8];

            int v1 = 0;
            int v2 = 0;
            size_t x = y*dz + z;
            cfloat_t pr = pr_g[0];
            
            for (int i=0;i<1;i++){
                ob[i] = ob_g[i*dz*dz + y*dz + z];
                obn[i] = obn_g[i*dz*dz + y*dz + z];
            }
            
            for (int i=0;i<N_pods;i++){
                v1 = (int)y - obj_roi_row(i);
                v2 = (int)z - obj_roi_column(i);
                if ((v1>=0)&&(v1<pr_sh)&&(v2>=0)&&(v2<pr_sh)){ 
                    pr = pr_g[pr_dlayer(i)*pr_sh*pr_sh + v1*pr_sh + v2];
                    ob[obj_dlayer(i)] = cfloat_add(ob[obj_dlayer(i)],cfloat_mul(cfloat_conj(pr),ex_g[ex_dlayer(i)*pr_sh*pr_sh +v1*pr_sh + v2]));
                    obn[obj_dlayer(i)] = cfloat_add(obn[obj_dlayer(i)],cfloat_mul(pr,cfloat_conj(pr)));
                }
            
            }
            for (int i=0;i<1;i++){
                ob_g[i*dz*dz + y*dz + z] = ob[i];
                obn_g[i*dz*dz + y*dz + z] = obn[i];
            }
        }
        __kernel void new_pr_update(__global int *info,
                                __global cfloat_t *ob_g,
                                __global cfloat_t *pr_g,
                                __global cfloat_t *prn_g,
                                __global cfloat_t *ex_g,
                                __global int *addr)
        {
            size_t z = get_global_id(0);
            size_t dz = get_global_size(1);
            size_t y = get_global_id(1);
            //size_t dy = get_global_size(1);
            //int off1 = 0;
            //int off2 = 0;
            __private cfloat_t pr[8];
            __private cfloat_t prn[8];
            
            int v1 = 0;
            int v2 = 0;
            cfloat_t ob; 
            
            for (int i=0;i<nmodes;i++){
                pr[i] = pr_g[i*dz*dz + y*dz + z];
                prn[i] = prn_g[i*dz*dz + y*dz + z];
            }

            for (int i=0;i<N_pods;i++){
                v1 = (int)y + obj_roi_row(i);
                v2 = (int)z + obj_roi_column(i);
                if ((v1>=0)&&(v1<obj_sh_row)&&(v2>=0)&&(v2<obj_sh_column)){ 
                    ob = ob_g[obj_dlayer(i)*obj_sh_row*obj_sh_column + v1*obj_sh_column + v2];
                    pr[pr_dlayer(i)] = cfloat_add(pr[pr_dlayer(i)], cfloat_mul(cfloat_conj(ob),ex_g[ex_dlayer(i)*pr_sh*pr_sh +y*pr_sh + z]));
                    prn[pr_dlayer(i)] = cfloat_add(prn[pr_dlayer(i)],cfloat_mul(ob,cfloat_conj( ob )));
                }
            
            }

            for (int i=0;i<nmodes;i++){
                pr_g[i*dz*dz + y*dz + z] = pr[i];
                prn_g[i*dz*dz + y*dz + z] = prn[i];
            }

        }
        
        
        __kernel void sum(__global float *err_fmag_t, __global float *err_fmag, __global float *err_ph_t, __global float *err_ph)
        {
            size_t z = get_global_id(0);
            size_t dx = 256;
            int x;
            int y; 
            for (y=0; y<256; y++){
                for (x=0; x<256; x++){
                    err_fmag[z] += err_fmag_t[z*dx*dx + y*dx + x];
                    err_ph[z] += err_ph_t[z*dx*dx + y*dx + x];
                }
            }
        }

        __kernel void reduce_one_step(__global const int* info,
                    __global const float* buffer,
                    __global float* result) {
                    
            __local float scratch[256];
            
            size_t z = get_global_id(0);
            size_t y = get_global_id(1);
      
            size_t ldz = get_local_size(0);
            
            size_t ly = get_local_id(1);
            size_t ldy = get_local_size(1);
            
            
            float accumulator = 0;
            size_t jump = z*pr_sh*pr_sh + y;
            size_t next = (z+1)*pr_sh*pr_sh + y;
            
            
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
        
        __kernel void sum2(__global int *info, __global float *buffer, __global float *result)
        {
            size_t z = get_global_id(0);
            size_t dz = get_global_size(0);
            size_t lz = get_local_id(0);
            
            for (int y=0; y<pr_sh/16; y++){
                for (int x=0; x<pr_sh/16; x++){
                    result[z] += buffer[z*pr_sh*pr_sh + y*pr_sh + x]; 
                }
            }      
        }
        __kernel void reduce(__global int *info,
                    __global float* buffer,
                    __global float* result) {
                    
            __local float loc_sum[256];
            size_t z = get_global_id(0);
            size_t y = get_global_id(1);
    
            
            size_t dz = get_global_size(0);
            size_t dy = get_global_size(1);
    
            
            size_t lz = get_local_id(0);
            size_t ldz = get_local_size(0);
            
            size_t ly = get_local_id(1);
            size_t ldy = get_local_size(1);
            
            
            size_t gidz = get_group_id(0);
            size_t gidy = get_group_id(1);
            
            int groups = dy/ldy;
            size_t lidx = lz*ldz +ly;
            
            for(int i=0; i<info[0]; i++){
                size_t idx = i*dz*dz + z*dz + y;
                size_t gid = gidz*groups + gidy;
                
                loc_sum[lidx] = buffer[idx];
                buffer[idx] = 0;
                for (int stride=ldy*ldz/2; stride>0; stride/=2){
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if(lidx < stride){
                        loc_sum[lidx] += loc_sum[lidx + stride];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (lidx==0){
                    result[i*dz*dz + gid] = loc_sum[0];
                }
            }
    
            
    
        }
        
        __kernel void gaussian_filter(__global cfloat_t *buffer, __global float *gauss_kernel){
            
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
            size_t dx = get_global_size(0);
            size_t dy = get_global_size(1);
            
            int ksx = KERNEL_SHAPE_X;
            int ksy = KERNEL_SHAPE_Y;
            __private cfloat_t sum;
            sum = cfloat_fromreal(0.0);
            cfloat_t img = buffer[x*dx + y];
            for(int kidx=0; kidx<ksx; kidx++){
                for(int kidy=0; kidy<ksy; kidy++){
                    if(x-kidx+(ksx-1)/2>=0 && y-kidy+(ksy-1)/2>=0 && x-kidx+(ksx-1)/2<dx && y-kidy+(ksy-1)/2<dy){
                        sum = cfloat_add(sum,cfloat_mulr(buffer[(x-kidx+(ksx-1)/2)*dx+(y-kidy+(ksy-1)/2)] , gauss_kernel[kidx*ksy + kidy]));
                        }
                }
            }
            buffer[x*dx + y] = sum;
        }
    
        """% kernel_pars).build()
        
        
        
    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        #assert os.path.exists('/tmp/gpus.lock'), 'No GPU lock found! This instance will be reported.'
        self.dattype=np.complex64
        
        def constbuffer(nbytes):
            return cl.Buffer(self.queue.context,cl.mem_flags.READ_ONLY,size=nbytes)
            
        self.error = []
        # object padding on high side (due to 16x16 wg size)
        for s in self.ob.S.values():
            pad = (32-np.asarray(s.shape[-2:]) % 32)
            s.data = u.crop_pad(s.data,[[0,pad[0]],[0,pad[1]]],axes=[-2,-1],filltype='project')
            s.shape = s.data.shape
                
        self.probe_fourier_support = {}

        supp = self.p.get('probe_fourier_support')
        if supp is not None:
            for name, s in self.pr.S.iteritems():
                sh = s.data.shape
                ll, xx, yy = u.grids(sh, center='fft',FFTlike=True)
                support = (np.pi * (xx**2 + yy**2) < supp * sh[1] * sh[2])
                self.probe_fourier_support[name] = support
                

            
        ## Container copies ##
        self.r = u.Param()
        # object-like
        #self.r.ob_buf = self.ob.copy(self.ob.ID+'_alt',fill=0.) 
        self.r.ob_nrm = self.ob.copy(self.ob.ID+'_nrm',fill=0.)
        self.r.ob_cfact = self.ob.copy(self.ob.ID+'_cfact',fill=0.)
        
        # Fill object with coverage of views
        for name, s in self.r.ob_cfact.S.iteritems():
            s.fill((s.get_view_coverage()+1) * self.p.object_inertia)

        
        # probe-like
        self.r.pr_buf = self.pr.copy(self.pr.ID+'_alt',fill=0.)
        self.r.pr_nrm = self.pr.copy(self.pr.ID+'_nrm',fill=0.)
        
        # exit-like
        self.r.f = self.ex.copy(self.ex.ID+'_fourier',fill=0.)
        
        # diff-like
        self.r.fm = self.di.copy(self.di.ID+'_fm',fill=0.)
        self.r.fmag = self.di.copy(self.di.ID+'_fmag',fill=0.)
        self.r.fdev = self.di.copy(self.di.ID+'_fdev',fill=0.)
        self.r.err_temp = self.di.copy(self.di.ID+'_err_temp',fill=0.)

        # copy Fourier magnitudes
        for name,s in self.r.fmag.S.iteritems():
            d = self.di.S[name].data
            d[d<0.] = 0.0 # just in case
            d[np.isnan(d)] = 0.0
            s.data[:] = np.sqrt(d)
            #s.gpu = cla.to_device(self.queue,s.data, allocator= constbuffer)
            
        ## access to gpu arrays via
        self.err_exit = {}
        self.err_fmag = {}
        for dID in self.di.S.keys():
            self.err_exit[dID] = cla.zeros(self.queue, (self.di.S[dID].shape[0],), np.float32)
            self.err_fmag[dID] = cla.zeros(self.queue, (self.di.S[dID].shape[0],), np.float32)
            self.ma.S[dID].data = (self.ma.S[dID].data).astype(np.float32)
            
        # recursive copy to gpu
        for name,c in self.ptycho.containers.iteritems():
            #if c is self.r.fmag:
            #    continue
            for name,s in c.S.iteritems():
                s.gpu = cla.to_device(self.queue,s.data)
        
        # memory check
        #self.ptycho.print_stats()
               
        self.di_views = {}
        self.DM_info = {}
        self.DM_info_gpu = {}
        self.addr = {}
        self.addr_gpu = {}
        self.info_gpu = {}
        self.info = {} 
        self.mask_sum = {}
        self.mask_sum_gpu = {}
        self.probe_object_exit_ID = {}
        self.shape ={}
        self.info = {}
        self.geometries = {}
        
        self.diff_info = {}
        for dID, diffs in self.di.S.iteritems():
            
            prep = u.Param()
            self.diff_info[dID] = prep
            
            prep.DM_info = []
            prep.addr = []
            prep.mask_sum = []
            
            # Sort views according to layer in diffraction stack 
            views = diffs.views
            dlayers = [view.dlayer for view in views]
            views = [views[i] for i in np.argsort(dlayers)]
            prep.views_sorted = views
            
            # Master pod
            mpod = views[0].pod
            
            # Determine linked storages for probe, object and exit waves
            pr = mpod.pr_view.storage
            ob = mpod.ob_view.storage
            ex = mpod.ex_view.storage
            
            prep.probe_object_exit_ID = (pr.ID,ob.ID,ex.ID)
            
            for view in views:
                address = []
                                
                for pname,pod in view.pods.iteritems():
                    ## store them for each pod
                    # create addresses
                    a = (pod.pr_view.dlayer,pod.pr_view.dlow[0],pod.pr_view.dlow[1])
                    a += (pod.ob_view.dlayer,pod.ob_view.dlow[0],pod.ob_view.dlow[1])
                    a += (pod.ex_view.dlayer,pod.ex_view.dlow[0],pod.ex_view.dlow[1])
                    a += (pod.di_view.dlayer,pod.di_view.dlow[0],pod.di_view.dlow[1])
                    a += (pod.ma_view.dlayer,pod.ma_view.dlow[0],pod.ma_view.dlow[1])
                    address.append(a)
                    
                    if pod.pr_view.storage.ID != pr.ID:
                        log(1, "Splitting probes for one diffraction stack is not supported in " + self.__class__.__name__)
                    if pod.ob_view.storage.ID != ob.ID:
                        log(1, "Splitting objects for one diffraction stack is not supported in " + self.__class__.__name__)
                    if pod.ex_view.storage.ID != ex.ID:
                        log(1, "Splitting exit stacks for one diffraction stack is not supported in " + self.__class__.__name__)
                                            
                ## store data for each view
                # adresses
                prep.addr.append(address)
                # mask sum
                prep.mask_sum.append(pod.mask.sum())
            
                
            # store them for each storage
            arr = np.asarray(prep.addr).astype(np.int32)
            msum = np.asarray(prep.mask_sum)
            
            prep.mask_sum = msum
            self.mask_sum_gpu[dID] = cla.to_device(self.queue, msum, allocator= constbuffer)
            prep.addr = arr
            self.addr_gpu[dID] = cla.to_device(self.queue, arr)
            
            diffs.pbound = .25 *  self.p.fourier_relax_factor**2 * diffs.pbound_stub
            
            prep.DM_info.append(self.p.alpha)
            prep.DM_info.append(diffs.pbound)
            prep.DM_info = np.asarray(prep.DM_info).astype(np.float32)
            prep.DM_info_gpu = cla.to_device(self.queue, prep.DM_info, allocator = constbuffer)
            
            # get shapes for each exit storage
            prep.shape = self.r.f.S[ex.ID].data.shape
            

            # TODO: nlayers for pr_modes & obj_modes
            prep.sh_info = []
            prep.sh_info += [arr.shape[0]]
            prep.sh_info += [arr.shape[1]]
            prep.sh_info += [ob.nlayers]
            prep.sh_info += [ob.data.shape[-2]]
            prep.sh_info += [ob.data.shape[-1]]
            prep.sh_info += [pr.data.shape[-2]]
            
            info = np.asarray(prep.sh_info).astype(np.int32)
            prep.sh_info_gpu = cla.to_device(self.queue, info, allocator = constbuffer)
            self.queue.finish()
            
            ## setup Fourier kernels
            from ptypy.gpu.ocl_kernels import Fourier_update_kernel as FUK
            from ptypy.gpu.ocl_kernels import Auxiliary_wave_kernel as AWK
            prep.fourier_update = FUK(self.queue, nmodes = arr.shape[1], pbound = diffs.pbound)
            mask = self.ma.S[dID].data.astype(np.float32)
            prep.fourier_update.configure(diffs.data, mask, ex.data)
            # put the right reference here
              
            
            ## FFT setup
            # we restrict to single geometry per diffraction storage => incompatible with multiple propagators
            geo = mpod.geometry
                        
            geo.propagator.pre_fft_gpu = cla.to_device(self.queue,np.ones_like(geo.propagator.pre_fft))
            geo.propagator.pre_ifft_gpu = cla.to_device(self.queue,np.ones_like(geo.propagator.pre_fft))
            geo.propagator.post_fft_gpu = cla.to_device(self.queue,np.ones_like(geo.propagator.pre_fft))
            geo.propagator.post_ifft_gpu = cla.to_device(self.queue,np.ones_like(geo.propagator.pre_fft))
            
            prep.geo = mpod.geometry
            from ptypy.gpu.ocl_fft import FFT_2D_ocl_gpyfft as FFT
            prep.geo.transform = FFT(self.queue, ex.data,  
                    pre_fft = geo.propagator.pre_fft, 
                    post_fft = geo.propagator.post_fft, 
                    inplace = True,
                    symmetric = True)
            prep.geo.itransform = FFT(self.queue, ex.data,  
                    pre_fft = geo.propagator.pre_ifft, 
                    post_fft =  geo.propagator.post_ifft, 
                    inplace = True,
                    symmetric = True)
            self.queue.finish()
                
        # finish init queue
        self.queue.finish()
        
        self.benchmark = u.Param()
        self.benchmark.A_assign_arrays = 0.
        self.benchmark.B_build_aux = 0.
        self.benchmark.C_fft = 0.
        self.benchmark.D_post_fft = 0.
        self.benchmark.E_calc_dev = 0.
        self.benchmark.F_error_reduce = 0.
        self.benchmark.G_fmag_update = 0.
        self.benchmark.H_ifft = 0.
        self.benchmark.I_post_ifft = 0.
        self.benchmark.probe_update = 0.
        self.benchmark.object_update = 0.
        self.benchmark.calls_fourier = 0
        self.benchmark.calls_object = 0
        self.benchmark.calls_probe = 0
        
    def engine_prepare(self):
        """
        last minute initialization, everything, that needs to be recalculated, when new data arrives
        """
        #self.err_fmag = cla.zeros(self.queue, (self.N,), np.float32)
        #self.err_phot = cla.zeros(self.queue, (self.N,), np.float32)


    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        
        for it in range(num):

            error_dct = {}
            
            for dID in self.di.S.keys():
                t1 = time.time()
                
                prep = self.diff_info[dID]
                # find probe, object in exit ID in dependence of dID
                pID,oID,eID = prep.probe_object_exit_ID
                
                # get addresses 
                addr_gpu = self.addr_gpu[dID]
                
                # get info arrays
                info = prep.sh_info
                info_gpu = prep.sh_info_gpu
                DM_info_gpu = prep.DM_info_gpu

                # get exit wave shape
                shape = prep.shape
                
                # local references
                mask_gpu = self.ma.S[dID].gpu
                obj_gpu = self.ob.S[oID].gpu
                probe_gpu = self.pr.S[pID].gpu
                
                ex_gpu = self.ex.S[eID].gpu
                f = self.r.f.S[eID].gpu
                prep.fourier_update.ocl.f = f
                
                geo = prep.geo
                queue = self.queue
                shape_merged = self.di.S[dID].shape
                mask_sum = self.mask_sum_gpu[dID]
                
                fm = self.r.fm.S[dID].gpu
                fmag = self.r.fmag.S[dID].gpu
                fdev = self.r.fdev.S[dID].gpu
                err_temp = self.r.err_temp.S[dID].gpu
                #err_fmag_temp1 = self.r.err_fmag_temp1.S[dID].gpu
                err_fmag = self.err_fmag[dID]
                err_exit = self.err_exit[dID]
                
                #err_phot.fill(0.0)
                err_fmag.fill(0.0)
    
                
                self.benchmark.A_assign_arrays += time.time() - t1
                #self.af2 = cla.zeros(self.queue, self.shape_merged, np.float32)
                
                
                t1 = time.time()
                self.prg.build_aux(queue, shape, (1,1,32), info_gpu.data, \
                                            DM_info_gpu.data, \
                                            obj_gpu.data, \
                                            probe_gpu.data, \
                                            ex_gpu.data, \
                                            f.data, \
                                            geo.propagator.pre_fft_gpu.data, \
                                            #geo.ocl_fft.pre_fft.data, \
                                            addr_gpu.data)
                queue.finish()
                
                self.benchmark.B_build_aux += time.time() - t1
    
                ## FFT
                t1 = time.time()
                geo.transform.ft(f)
                queue.finish()
                self.benchmark.C_fft += time.time() - t1
                
                ## Deviation from measured data
                t1 = time.time()
                fourier_error = prep.fourier_update.execute_ocl()
                queue.finish()
                self.benchmark.G_fmag_update += time.time() - t1
                """
                self.prg.dev_post_fft(queue, shape_merged, (1,1,32),
                                            info_gpu.data, 
                                            f.data,
                                            #geo.ocl_fft.post_fft.data,
                                            geo.propagator.post_fft_gpu.data,
                                            fmag.data, 
                                            fdev.data, 
                                            err_temp.data, 
                                            mask_gpu.data, 
                                            mask_sum.data, 
                                            )
                queue.finish()
                self.benchmark.E_calc_dev += time.time() - t1    
                
                t1 = time.time()
                
                self.prg.reduce_one_step(queue, (shape_merged[0],64), (1,64), info_gpu.data, err_temp.data, err_fmag.data)
                queue.finish()
                
                self.benchmark.F_error_reduce += time.time() - t1
                
                t1 = time.time()

                self.prg.fmag_all_update(queue, shape, (1,1,32), 
                                            info_gpu.data,
                                            DM_info_gpu.data,
                                            f.data,
                                            geo.propagator.pre_ifft_gpu.data,
                                            #geo.ocl_ifft.pre_ifft.data,
                                            mask_gpu.data, 
                                            fmag.data, 
                                            fdev.data, 
                                            err_fmag.data)
                
                """
                ## iFFT
                t1 = time.time()
                geo.itransform.ift(f)
                self.queue.finish()
                                
                self.benchmark.H_ifft += time.time() - t1
                                
                ## apply changes #2
                t1 = time.time()
                self.prg.post_ifft(queue, shape, (1,1,32), info_gpu.data, 
                                        obj_gpu.data, 
                                        probe_gpu.data, 
                                        ex_gpu.data, 
                                        f.data, 
                                        geo.propagator.post_ifft_gpu.data, 
                                        #geo.ocl_ifft.post_ifft.data.data, 
                                        addr_gpu.data)
                self.queue.finish()
    
                self.prg.reduce_one_step(queue, (shape_merged[0],64), (1,64), info_gpu.data, err_temp.data, err_exit.data)
                queue.finish()
                self.benchmark.I_post_ifft += time.time() - t1 
                
                viewIDs = [v.ID for v in prep.views_sorted]
                #err_exit = np.zeros(info[0],)
                err_phot = np.zeros(info[0],)
                #errs = np.array(zip(err_fmag.get(),err_phot , err_exit.get()))
                errs = np.array(zip(fourier_error,err_phot , err_exit.get()))
                error = dict(zip(viewIDs, errs))
                #print np.array(error.values()).mean(0)
                
                self.benchmark.calls_fourier +=1
                
            sync = (self.curiter % 1==0)
            #parallel.barrier()
            #log(3,str(self.curiter)+str(sync),True)
            parallel.barrier()
            self.overlap_update(MPI=True)
            parallel.barrier()
            self.curiter += 1
            self.queue.finish()

        for name, s in self.ob.S.iteritems():  
            s.data[:] = s.gpu.get(queue=self.queue)
        for name, s in self.pr.S.iteritems():  
            s.data[:] = s.gpu.get(queue=self.queue)

        self.queue.finish()
        
        self.error = error
        return error

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        self.queue.finish()
        if parallel.master:
            print "----- BENCHMARKS ----"
            acc = 0.
            for name in sorted(self.benchmark.keys()):
                t = self.benchmark[name]
                if name[0] in 'ABCDEFGHI':
                    print '%20s : %1.3f ms per iteration' % (name, t / self.benchmark.calls_fourier *1000)
                    acc +=t
                elif str(name) == 'probe_update':
                    #pass
                    print '%20s : %1.3f ms per call. %d calls' % (name, t / self.benchmark.calls_probe * 1000, self.benchmark.calls_probe)
                elif str(name) == 'object_update':
                    print '%20s : %1.3f ms per call. %d calls' % (name, t / self.benchmark.calls_object *1000, self.benchmark.calls_object)
            
            print '%20s : %1.3f ms per iteration. %d calls' % ('Fourier_total', acc / self.benchmark.calls_fourier *1000, self.benchmark.calls_fourier)
            
            """
            for name, s in self.ob.S.iteritems():
                plt.figure('obj')
                d = s.gpu.get()
                #print np.abs(d[0][300:-300,300:-300]).mean()
                plt.imshow(u.imsave(d[0][400:-400,400:-400]))
            for name, s in self.pr.S.iteritems():
                d = s.gpu.get()
                for l in d:
                    plt.figure()
                    plt.imshow(u.imsave(l))
                #print u.norm2(d)

            plt.show()
            """
            
        for original in [self.pr,self.ob,self.ex,self.di, self.ma]:
            original.delete_copy()
            
        # delete local references to container buffer copies
        del self.r

        
    def overlap_update(self, MPI=True):
        """
        DM overlap constraint update.
        """
        change = 1.
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)
         
        for inner in range(self.p.overlap_max_iterations):
            prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)
            # Update object first
            if self.p.update_object_first or (inner > 0):
                # Update object
                log(4,prestr + '----- object update -----',True)
                self.object_update(MPI=(parallel.size>1 and MPI))
                               
            # Exit if probe should not yet be updated
            if not do_update_probe: break
            
            # Update probe
            log(4,prestr + '----- probe update -----',True)
            change = self.probe_update(MPI=(parallel.size>1 and MPI))
            #change = self.probe_update(MPI=(parallel.size>1 and MPI))

            log(4,prestr + 'change in probe is %.3f' % change,True)
            
            # stop iteration if probe change is small
            if change < self.p.overlap_converge_factor: break
            

    ## object update
    def object_update(self, MPI=False):
        t1 = time.time()
        
        # storage for-loop
        for dID in self.di.S.keys():
            
            queue = self.queue
            
            prep = self.diff_info[dID]
            
            info = prep.sh_info
            info_gpu = prep.sh_info_gpu
            
            # find probe, object in exit ID in dependence of dID
            pID,oID,eID = prep.probe_object_exit_ID
                                   
            # get addresses 
            addr_gpu = self.addr_gpu[dID]
            
            # local references
            ob = self.ob.S[oID]
            pr = self.pr.S[pID]
            ex = self.ex.S[eID]
            obn = self.r.ob_nrm.S[oID]
            
            """
            if self.p.obj_smooth_std is not None:
                logger.info('Smoothing object, cfact is %.2f' % cfact)
                t2 = time.time()
                self.prg.gaussian_filter(queue, (info[3],info[4]), None, obj_gpu.data, self.gauss_kernel_gpu.data)
                queue.finish()
                obj_gpu *= cfact
                print 'gauss: '  + str(time.time()-t2)
            else:
                obj_gpu *= cfact
            """
            cfact = self.r.ob_cfact.S[oID].gpu
            ob.gpu *= cfact
            #obn.gpu.fill(cfact)
            obn.gpu[:] = cfact
            queue.finish()

            # scan for loop
            self.prg.new_obj_update(queue, (info[3],info[4]), (16,16), 
                                                info_gpu.data, 
                                                ob.gpu.data, 
                                                pr.gpu.data, 
                                                obn.gpu.data, 
                                                ex.gpu.data, 
                                                addr_gpu.data)
            queue.finish()
            
            # MPI test
            if MPI:
                ob.data[:]=ob.gpu.get(queue=queue)
                obn.data[:]=obn.gpu.get(queue=queue)
                queue.finish()
                parallel.allreduce(ob.data)
                parallel.allreduce(obn.data)
                ob.data /= obn.data
                
                # Clip object (This call takes like one ms. Not time critical)
                if self.p.clip_object is not None:
                    clip_min, clip_max = self.p.clip_object
                    ampl_obj = np.abs(ob.data)
                    phase_obj = np.exp(1j * np.angle(ob.data))
                    too_high = (ampl_obj > clip_max)
                    too_low = (ampl_obj < clip_min)
                    ob.data[too_high] = clip_max * phase_obj[too_high]
                    ob.data[too_low] = clip_min * phase_obj[too_low]
                ob.gpu.set(ob.data)
            else:
                ob.gpu /= obn.gpu
            
            #self.ob.S[oID].gpu = obj_gpu
            queue.finish()
            
        #print 'object update: ' + str(time.time()-t1)
        self.benchmark.object_update += time.time()-t1
        self.benchmark.calls_object +=1
    
    ## probe update
    def probe_update(self,MPI=False):
        t1 = time.time()
        
        # storage for-loop
        change = 0
        for dID in self.di.S.keys():
            
            queue = self.queue
            
            prep = self.diff_info[dID]
            
            info = prep.sh_info
            info_gpu = prep.sh_info_gpu
            
            cfact = self.p.probe_inertia * info[0]
            
            # find probe, object in exit ID in dependence of dID
            pID,oID,eID = prep.probe_object_exit_ID
            
            # get addresses 
            addr_gpu = self.addr_gpu[dID]
            
            # local references
            ob = self.ob.S[oID]
            pr = self.pr.S[pID]
            ex = self.ex.S[eID]
            prn = self.r.pr_nrm.S[pID]
            buf = self.r.pr_buf.S[pID]
            
            #cfact = self.p.probe_inertia * info[0] #*info[1] / self.pr.S[pID].data.shape[0]
            #prb.data[:] = pr.gpu.get(queue=queue)
            #queue.finish()
            
            pr.gpu *= cfact
            prn.gpu.fill(cfact)
            
            # scan for-loop
            self.prg.new_pr_update(queue, (info[5], info[5]), (16,16),#(16,16), 
                                            info_gpu.data, 
                                            ob.gpu.data, 
                                            pr.gpu.data, 
                                            prn.gpu.data, 
                                            ex.gpu.data, 
                                            addr_gpu.data)
            queue.finish()
            
            # MPI test
            if MPI:
            #if False:
                pr.data[:]=pr.gpu.get(queue=queue)
                prn.data[:]=prn.gpu.get(queue=queue)
                queue.finish()
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data
                
                # Apply probe support if requested
                support = self.probe_support.get(pID)
                if support is not None: 
                    pr.data *= support
    
                # Apply probe support in Fourier space (This could be better done on GPU)
                support = self.probe_fourier_support.get(pID)
                if support is not None: 
                    pr.data[:] = np.fft.ifft2(support * np.fft.fft2(pr.data))
                    
                pr.gpu.set(pr.data)
            else:
                pr.gpu /= prn.gpu

            # ca. 0.3 ms
            #self.pr.S[pID].gpu = probe_gpu
            pr.data[:]=pr.gpu.get(queue=queue)
            ## this should be done on GPU
            queue.finish()
            
            #change += u.norm2(pr[i]-buf_pr[i]) / u.norm2(pr[i])
            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size
                
        #print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time()-t1
        self.benchmark.calls_probe +=1
        
        return np.sqrt(change)
    
