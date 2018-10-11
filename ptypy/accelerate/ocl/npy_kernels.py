import numpy as np
import time
from inspect import getargspec
from collections import OrderedDict

class Adict(object):
    
    def __init__(self):
        pass
        
class BaseKernel(object):
    
    def __init__(self):
        
        self.verbose = False
        self.npy = Adict()
        self.benchmark = OrderedDict()       
           
    def log(self, x):
        if self.verbose:
            print(x)
        
class Fourier_update_kernel(BaseKernel):
    
    def __init__(self, pbound = 0.0):
        
        self.pbound = np.float32(pbound)
            
    def test(self, I, mask, f):
        """
        Test arrays for shape and data type
        """
        assert I.dtype == np.float32
        assert I.shape == self.fshape
        assert mask.dtype == np.float32
        assert mask.shape == self.fshape
        assert f.dtype == np.complex64
        assert f.shape == self.ishape
    
    def allocate(self, shape, nmodes = 1):
        """
        Allocate memory according to the number of modes and
        shape of the diffraction stack.
        """
        assert len(fshape) == 2
        self.nmodes = np.int32(nmodes)
        self.fshape = shape
        self.ishape = (self.nmodes*shape[0],shape[1],shape[2])

        self.framesize = np.int32(np.prod(shape[-2:]))
       
        # temporary buffer arrays
        self.npy.fdev = np.zeros(shape, dtype = np.float32)
        self.npy.ferr = np.zeros(shape, dtype = np.float32)
        
        self.kernels = [
            'fourier_error',
            'error_reduce',
            'fmag_all_update'        
        ]
        
    def npy_fourier_error(self,f, fmag, fdev, ferr, fmask, mask_sum, offset = 0):
        # reference shape (write-to shape)
        sh = self.fshape
        # read from slice for global arrays
        sl = slice(offset,offset+sh[0])
        
        # build model from complex fourier magnitudes, summing up 
        # all modes incoherently
        tf = f.reshape(sh[0],self.nmodes,sh[1],sh[2])
        af = np.sqrt((np.abs(tf)**2).sum(1))
        
        # calculate difference to real data (fmag)
        fdev[:] = af - fmag[sl]
        
        # Calculate error on fourier magnitudes on a per-pixel basis
        ferr[:] = fmask[sl] * np.abs(fdev)**2 / mask_sum[sl].reshape((sh[0],1,1)) 
        
    def npy_error_reduce(self, ferr, err_fmag, offset = 0):
        sh = self.fshape
        sl = slice(offset,offset+sh)
        # Reduceses the Fourier error along the last 2 dimensions.fd
        err_fmag[sl] = ferr.astype(np.double).sum(-1).sum(-1).astype(np.float)
        
    def npy_fmag_all_update(self,f,fmask, fmag, fdev, err_fmag, offset = 0):
        
        # reference shape (write-to shape)
        sh = self.ishape
        sl = slice(offset,offset+sh[0])
        
        # local values
        fm = np.ones(self.fshape, np.float32)
        renorm = np.ones(self.fshape, np.float32)
        
        ## As opposed to DM we use renorm to differentiate the cases.
        
        # pbound >= err_fmag  
        # fm = 1.0 (as renorm = 1, i.e. renorm[~ind])
        # pbound < err_fmag :
        # fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10) 
        # (as renorm in [0,1])
        # pbound == 0.0 
        # fm = (1 - fmask) + fmask * fmag / (af + 1e-10) (as renorm=0)
        
        ind = err_fmag[sl] > self.pbound
        renorm[ind] = np.sqrt(self.pbound / err_fmag[sl][ind])
        renorm = renorm.reshape((renorm.shape[0],1,1))
        
        af = fdev + fmag[sl]
        fm[:] = (1 - fmask[sl]) + fmask[sl] * (fmag[sl] + fdev * renorm) / (af + 1e-10)

        # upcasting
        f[:] = f.reshape(sh[0]/self.nmodes,self.nmodes,sh[1],sh[2]) * fm[:,newaxis,:,:]
   
    @classmethod
    def test(cls, shape =  (739,256,256), nmodes = 1, pbound = 0.0):


        self.npy.f = f
        self.npy.fmask = mask
        self.npy.mask_sum = mask.sum(-1).sum(-1)
        d = I.copy()
        d[d<0.] = 0.0 # just in case
        d[np.isnan(d)] = 0.0
        self.npy.fmag = np.sqrt(d)
        self.npy.err_fmag = np.zeros((self.fshape[0],),dtype=np.float32)
        
        
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
        inst.verbose =  True
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

class Auxiliary_wave_kernel(BaseKernel):
    
    def __init__(self, queue_thread=None):
            
        super(Auxiliary_wave_kernel, self).__init__(queue_thread)        

        self.prg = cl.Program(self.queue.context,"""
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
                            int batch_offset,
                            __global cfloat_t *aux,  
                            __global cfloat_t *ob,
                            __global cfloat_t *pr,
                            __global cfloat_t *ex,
                            __global int *addr)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0) + batch_offset;
            size_t zb = get_global_id(0);
            
            size_t obj_idx = obj_dlayer(z)*ob_sh_row*ob_sh_col + (y+obj_roi_row(z))*ob_sh_col + obj_roi_column(z)+x;

            cfloat_t ex0 = cfloat_rmul(alpha,ex[ex_dlayer(z)*dx*dx + y*dx + x]);
            cfloat_t ex1 = cfloat_mul(ob[obj_dlayer(z)*ob_sh_row*ob_sh_col + (y+obj_roi_row(z))*ob_sh_col + obj_roi_column(z)+x],pr[pr_dlayer(z)*dx*dx + y*dx+x]);
            //loc_sub[3] = cfloat_fromreal(1. + loc_sub[0].real);
            
            //cfloat_t ex2 = cfloat_sub(cfloat_rmul(1.+alpha,ex1),ex0);
            aux[zb*dx*dx + y*dx + x] = cfloat_sub(cfloat_rmul(1.+alpha,ex1),ex0);            
        }
        
        __kernel void build_exit(float alpha,
                            int ob_sh_row,
                            int ob_sh_col,
                            int batch_offset,
                            __global cfloat_t *f,
                            __global cfloat_t *ob,
                            __global cfloat_t *pr,
                            __global cfloat_t *ex,
                            __global int *addr)
        {
            size_t x = get_global_id(2);
            size_t dx = get_global_size(2);
            size_t y = get_global_id(1);
            size_t z = get_global_id(0) + batch_offset;
            size_t zb = get_global_id(0); 
            
            size_t obj_idx = obj_dlayer(z)*ob_sh_row*ob_sh_col + (y+obj_roi_row(z))*ob_sh_col + obj_roi_column(z)+x;
            
            cfloat_t ex1 = cfloat_mul(ob[obj_idx],pr[pr_dlayer(z)*dx*dx + y*dx+x]);
            cfloat_t df = cfloat_sub(f[zb*dx*dx + y*dx + x] , ex1); 
            f[zb*dx*dx + y*dx + x] = df ; // t.b. removed later
            ex[ex_dlayer(z)*dx*dx + y*dx + x] = cfloat_add(ex[ex_dlayer(z)*dx*dx + y*dx + x] , df);
        }

        """).build()
        
        self.kernels = [
            'build_aux',
            'build_exit',    
        ]
            
    def configure(self,ob, addr, alpha = 1.0):
        
        self.batch_offset = 0
        self.alpha = np.float32(alpha)
        self.ob_shape = (np.int32(ob.shape[-2]),np.int32(ob.shape[-1]))
        
        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape        
        self.ocl_wg_size = (1,1,32)
        
    @property
    def batch_offset(self):
        return self._offset
    
    @batch_offset.setter
    def batch_offset(self, x):
        self._offset = np.int32(x)
        
    def load(self, aux, ob, pr, ex, addr):

        assert pr.dtype == np.complex64
        assert ex.dtype == np.complex64
        assert aux.dtype == np.complex64
        assert ob.dtype == np.complex64
        assert addr.dtype == np.int32
        
        self.npy.aux = aux
        self.npy.pr = pr
        self.npy.ob = ob
        self.npy.ex = ex
        self.npy.addr = addr
    
        for key,array in self.npy.__dict__.iteritems():
            self.ocl.__dict__[key] = cla.to_device(self.queue, array)
    
    def sync_ocl(self):
        for key,array in self.npy.__dict__.iteritems():
            self.ocl.__dict__[key].set(array)
                    
    
    def execute_ocl(self, kernel_name=None, compare = False, sync=False):
                
        if kernel_name is None:
            for kernel in self.kernels:
                self.execute_ocl(kernel, compare, sync)
        else:
            self.log("KERNEL " + kernel_name)
            m_ocl = getattr(self,'ocl_' + kernel_name )
            m_npy = getattr(self,'npy_' + kernel_name )
            ocl_kernel_args = getargspec(m_ocl).args[1:]
            npy_kernel_args = getargspec(m_npy).args[1:]
            assert ocl_kernel_args == npy_kernel_args
            # OCL
            if sync:
                self.sync_ocl()
            args = [getattr(self.ocl,a) for a in ocl_kernel_args]
            
            self.benchmark[kernel_name] = -time.time()
            m_ocl(*args)
            self.benchmark[kernel_name]+= time.time()
            
            if compare:
                args = [getattr(self.npy,a) for a in npy_kernel_args]
                m_npy(*args)
                self.verify_ocl()
        
        return 
        
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
               
        return 
    
    
    def ocl_build_aux(self, aux, ob, pr, ex, addr):
        obsh = self.ob_shape
        ev = self.prg.build_aux(self.queue, aux.shape, self.ocl_wg_size,
                           self.alpha, obsh[0], obsh[1], self._offset,
                           aux.data, ob.data, pr.data, ex.data, addr.data)
        return ev
        
    def npy_build_aux(self, aux, ob, pr, ex, addr):
        
        sh = addr.shape
        flat_addr = addr.reshape(sh[0]*sh[1],sh[2],sh[3])
        off = self.batch_offset
        flat_addr = flat_addr[off:off+aux.shape[0]]
        rows, cols = ex.shape[-2:]
        
        for ind, (prc,obc,exc,mac,dic) in enumerate(flat_addr):
            tmp = ob[obc[0],obc[1]:obc[1]+rows,obc[2]:obc[2]+cols] * \
                  pr[prc[0],:,:] * \
                  (1.+self.alpha) - \
                  ex[exc[0],exc[1]:exc[1]+rows,exc[2]:exc[2]+cols] * \
                  self.alpha
            aux[ind,:,:] = tmp
            
    def ocl_build_exit(self, aux, ob, pr, ex, addr):
        obsh = self.ob_shape        
        ev = self.prg.build_exit(self.queue, aux.shape, self.ocl_wg_size,
                            self.alpha, obsh[0], obsh[1], self._offset, 
                            aux.data, ob.data, pr.data, ex.data, addr.data)
        
        return ev
        
    def npy_build_exit(self, aux, ob, pr, ex, addr):
        
        sh = addr.shape
        flat_addr = addr.reshape(sh[0]*sh[1],sh[2],sh[3])
        off = self.batch_offset
        flat_addr = flat_addr[off:off+aux.shape[0]]
        rows, cols = ex.shape[-2:]
        for ind, (prc,obc,exc,mac,dic) in enumerate(flat_addr):
            dex = aux[ind,:,:] - \
                  ob[obc[0],obc[1]:obc[1]+rows,obc[2]:obc[2]+cols] * \
                  pr[prc[0],prc[1]:prc[1]+rows,prc[2]:prc[2]+cols] 
                   
            ex[exc[0],exc[1]:exc[1]+rows,exc[2]:exc[2]+cols] += dex
            aux[ind,:,:] = dex
        
        
    def verify_ocl(self, precision=2**(-23)):
        
        for name, val in self.npy.__dict__.iteritems():
            val2 = self.ocl.__dict__[name].get()
            val = val
            if np.allclose(val,val2,atol=precision):
                continue 
            else:
                dev = np.std(val - val2)
                mn = np.mean(np.abs(val))
                self.log("Key %s : %.2e std, %.2e mean" % (name, dev, mn))        
        
    @classmethod
    def test(cls, ob_shape =  (10,300,300), pr_shape = (1,256,256)):

        nviews,rows,cols = ob_shape
        ex_shape = (nviews,)+pr_shape[-2:]
        addr = np.zeros((nviews,1,5,3),dtype=np.int32)
        for i in range(nviews):
            obc = (0,2*i,i)
            prc = (0,0,0)
            exc = (i,0,0)
            mac = (i,0,0)# unimportant
            dic = (i,0,0)# same here
            addr[i,0,:,:] = np.array([prc,obc,exc,mac,dic],dtype=np.int32)
             
        ob = np.random.rand(*ob_shape).astype(np.complex64) 
        pr = np.random.rand(*pr_shape).astype(np.complex64) 
        ex = np.random.rand(*ex_shape).astype(np.complex64)
        
        devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
        queue = cl.CommandQueue(cl.Context([devices[0]]),properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        inst = cls(queue_thread=queue)
        inst.verbose =True
        bsize = nviews / 2
        batch = np.zeros((bsize,)+pr_shape[-2:],dtype = np.complex64)
        args = (batch, ob, pr, ex, addr)
        ocl_args = tuple([cla.to_device(queue,arg) for arg in args])
        inst.configure(ob,addr)
        #ns = inst._ocl_build_exit(*ocl_args, batch_offset = 0)
        #inst._npy_build_exit(*args, batch_offset = 0)
        
        #print ns
        inst.load(*args)
        inst.bath_offset = 3
        inst.execute_ocl(compare=True,sync=False)

        """
        inst.execute_ocl()
        g = inst.ocl.f.get()
        inst.execute_npy()
        f = inst.npy.f
        
        g = f.copy()
        inst.configure(I, mask, g)
        err = inst.execute_npy(g)
        inst.verify_ocl()
        
        print('Error : %.2e' % np.std(f-g))
        for key, val in inst.benchmark.items():
            print('Kernel %s : %.2f ms' % (key,val*1000))
        """


class PO_update_kernel(BaseKernel):
    
    def __init__(self, queue_thread=None):
            
        super(PO_update_kernel, self).__init__(queue_thread)        

        self.prg = cl.Program(self.queue.context,"""
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
                                __global cfloat_t *obn_g,
                                __global cfloat_t *pr_g,
                                __global cfloat_t *ex_g,
                                __global int *addr)
        {
            size_t z = get_global_id(1);
            size_t dz = get_global_size(1);
            size_t y = get_global_id(0);
            size_t dy = get_global_size(0);
            __private cfloat_t ob[8];
            __private cfloat_t obn[8];

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
                    obn[obj_dlayer(i)] = cfloat_add(obn[obj_dlayer(i)],cfloat_mul(pr,cfloat_conj(pr)));
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
                                __global cfloat_t *prn_g,
                                __global cfloat_t *ob_g,
                                __global cfloat_t *ex_g,
                                __global int *addr)
        {
            size_t z = get_global_id(1);
            size_t dz = get_global_size(1);
            size_t y = get_global_id(0);
            size_t dy = get_global_size(0);
            __private cfloat_t pr[8];
            __private cfloat_t prn[8];
            
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
                    prn[pr_dlayer(i)] = cfloat_add(prn[pr_dlayer(i)],cfloat_mul(ob,cfloat_conj( ob )));
                }
            
            }

            for (int i=0;i<pr_modes;i++){
                pr_g[i*dy*dz + y*dz + z] = pr[i];
                prn_g[i*dy*dz + y*dz + z] = prn[i];
            }

        }

        """).build()
        
        self.kernels = [
            'pr_update',
            'ob_update',    
        ]
            
    def configure(self,ob, pr, addr):
        
        self.batch_offset = 0
        self.ob_shape = tuple([np.int32(ax) for ax in ob.shape])
        self.pr_shape = tuple([np.int32(ax) for ax in pr.shape])
        #self.ob_shape = (np.int32(ob.shape[-2]),np.int32(ob.shape[-1]))
        #self.pr_shape = (np.int32(pr.shape[-2]),np.int32(pr.shape[-1]))
        
        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape 
        self.num_pods = np.int32(self.nviews * self.nmodes)
        self.ocl_wg_size = (16,16)
        
    @property
    def batch_offset(self):
        return self._offset
    
    @batch_offset.setter
    def batch_offset(self, x):
        self._offset = np.int32(x)
        
    def load(self, obn, prn, ob, pr, ex, addr):

        assert pr.dtype == np.complex64
        assert ex.dtype == np.complex64
        assert ob.dtype == np.complex64
        assert addr.dtype == np.int32
        
        self.npy.pr = pr
        self.npy.prn = prn
        self.npy.ob = ob
        self.npy.obn = obn
        self.npy.ex = ex
        self.npy.addr = addr
    
        for key,array in self.npy.__dict__.iteritems():
            self.ocl.__dict__[key] = cla.to_device(self.queue, array)
    
    def sync_ocl(self):
        for key,array in self.npy.__dict__.iteritems():
            self.ocl.__dict__[key].set(array)
                    
    
    def execute_ocl(self, kernel_name=None, compare = False, sync=False):
                
        if kernel_name is None:
            for kernel in self.kernels:
                self.execute_ocl(kernel, compare, sync)
        else:
            self.log("KERNEL " + kernel_name)
            m_ocl = getattr(self,'ocl_' + kernel_name )
            m_npy = getattr(self,'npy_' + kernel_name )
            ocl_kernel_args = getargspec(m_ocl).args[1:]
            npy_kernel_args = getargspec(m_npy).args[1:]
            assert ocl_kernel_args == npy_kernel_args
            # OCL
            if sync:
                self.sync_ocl()
            args = [getattr(self.ocl,a) for a in ocl_kernel_args]
            
            self.benchmark[kernel_name] = -time.time()
            m_ocl(*args)
            self.queue.finish()
            self.benchmark[kernel_name]+= time.time()
            
            if compare:
                args = [getattr(self.npy,a) for a in npy_kernel_args]
                m_npy(*args)
                self.verify_ocl()
        
        return 
        
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
               
        return 
    
    
    def ocl_ob_update(self, ob, obn, pr, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
        ev = self.prg.ob_update(self.queue, ob.shape[-2:], self.ocl_wg_size, 
                      prsh[-1], 
                      obsh[0], self.num_pods,
                      ob.data, obn.data, pr.data, ex.data, addr.data)            
        return ev
        
    def ocl_pr_update(self, pr, prn, ob, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
        ev = self.prg.pr_update(self.queue, pr.shape[-2:], self.ocl_wg_size, 
                        prsh[-1], obsh[-2], obsh[-1], 
                        prsh[0], self.num_pods,
                        pr.data, prn.data, ob.data, ex.data, addr.data)
        return ev
    
    def npy_ob_update(self, ob, obn, pr, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
        sh = addr.shape
        flat_addr = addr.reshape(sh[0]*sh[1],sh[2],sh[3]) 
        rows, cols = ex.shape[-2:]
        for ind, (prc,obc,exc,mac,dic) in enumerate(flat_addr):
            ob[obc[0],obc[1]:obc[1]+rows,obc[2]:obc[2]+cols] += \
            pr[prc[0],prc[1]:prc[1]+rows,prc[2]:prc[2]+cols].conj() * \
            ex[exc[0],exc[1]:exc[1]+rows,exc[2]:exc[2]+cols]
            obn[obc[0],obc[1]:obc[1]+rows,obc[2]:obc[2]+cols] += \
            pr[prc[0],prc[1]:prc[1]+rows,prc[2]:prc[2]+cols].conj() * \
            pr[prc[0],prc[1]:prc[1]+rows,prc[2]:prc[2]+cols]
        return 
        
    def npy_pr_update(self, pr, prn, ob, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
        sh = addr.shape
        flat_addr = addr.reshape(sh[0]*sh[1],sh[2],sh[3]) 
        rows, cols = ex.shape[-2:]
        for ind, (prc,obc,exc,mac,dic) in enumerate(flat_addr):
            pr[prc[0],prc[1]:prc[1]+rows,prc[2]:prc[2]+cols] += \
            ob[obc[0],obc[1]:obc[1]+rows,obc[2]:obc[2]+cols].conj() * \
            ex[exc[0],exc[1]:exc[1]+rows,exc[2]:exc[2]+cols]
            prn[prc[0],prc[1]:prc[1]+rows,prc[2]:prc[2]+cols] += \
            ob[obc[0],obc[1]:obc[1]+rows,obc[2]:obc[2]+cols].conj() * \
            ob[obc[0],obc[1]:obc[1]+rows,obc[2]:obc[2]+cols] 
        return 
        
        
    def verify_ocl(self, precision=2**(-23)):
        
        for name, val in self.npy.__dict__.iteritems():
            val2 = self.ocl.__dict__[name].get()
            val = val
            if np.allclose(val,val2,atol=precision):
                continue 
            else:
                dev = np.std(val - val2)
                mn = np.mean(np.abs(val))
                self.log("Key %s : %.2e std, %.2e mean" % (name, dev, mn))        
        
    @classmethod
    def test(cls, ob_shape =  (1,320,352), pr_shape = (4,256,256)):

        nviews,rows,cols = ob_shape
        nviews = 10
        ex_shape = (nviews,)+pr_shape[-2:]
        addr = np.zeros((1,nviews,5,3),dtype=np.int32)
        for i in range(nviews):
            obc = (0,2*i,i)
            prc = (0,0,0)
            exc = (i,0,0)
            mac = (i,0,0)# unimportant
            dic = (i,0,0)# same here
            addr[0,i,:,:] = np.array([prc,obc,exc,mac,dic],dtype=np.int32)
             
        ob = np.random.rand(*ob_shape).astype(np.complex64) 
        obn = np.random.rand(*ob_shape).astype(np.complex64) 
        pr = np.random.rand(*pr_shape).astype(np.complex64) 
        prn = np.random.rand(*pr_shape).astype(np.complex64) 
        ex = np.random.rand(*ex_shape).astype(np.complex64)
        
        devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
        queue = cl.CommandQueue(cl.Context([devices[0]]),properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        inst = cls(queue_thread=queue)
        inst.verbose =True
        args = ( obn, prn, ob, pr, ex, addr)
        inst.configure(ob, pr, addr)
        inst.load(*args)
        inst.bath_offset = 3
        inst.execute_ocl(compare=True,sync=False)

        """
        inst.execute_ocl()
        g = inst.ocl.f.get()
        inst.execute_npy()
        f = inst.npy.f
        
        g = f.copy()
        inst.configure(I, mask, g)
        err = inst.execute_npy(g)
        inst.verify_ocl()
        
        print('Error : %.2e' % np.std(f-g))
        for key, val in inst.benchmark.items():
            print('Kernel %s : %.2f ms' % (key,val*1000))
        """


if __name__=='__main__':
    #Fourier_update_kernel.test()
    #Auxiliary_wave_kernel.test()
    PO_update_kernel.test()
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
