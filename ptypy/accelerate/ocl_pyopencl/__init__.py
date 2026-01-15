

ocl_context = None
ocl_queue = None

def get_ocl_queue(new_queue=False):
    
    from ptypy.utils import parallel
    import pyopencl as cl
    devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)

    global ocl_context
    global ocl_queue
    
    if ocl_context is None and parallel.rank_local < len(devices):
        #ocl_context = cl.Context([devices[parallel.rank_local]])
        ocl_context = cl.Context([devices[-1]])
        print("parallel.rank:%s, parallel.rank_local:%s" % (str(parallel.rank),
                                                            str(parallel.rank_local)))

    if ocl_context is not None:
        if new_queue or ocl_queue is None:
            ocl_queue = cl.CommandQueue(ocl_context)
        return ocl_queue
    else:
        return None
    
