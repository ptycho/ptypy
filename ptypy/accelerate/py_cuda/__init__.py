import pycuda.driver as cuda
# debug_options = []
# debug_options = ['-O0', '-G', '-g']
debug_options = ['-O3', '-DNDEBUG', '-lineinfo']

context = None
queue = None

def get_context(new_queue=False):

    from ptypy.utils import parallel

    global context
    global queue

    if context is None:
        cuda.init()
        if parallel.rank_local < cuda.Device.count():
            import pyopencl as cl
            context = cuda.Device(parallel.rank_local).make_context()
            context.push()
        # print("made context %s on rank %s" % (str(context), str(parallel.rank)))
        # print("The cuda device count on %s is:%s" % (str(parallel.rank),
        #                                              str(cuda.Device.count())))
        # print("parallel.rank:%s, parallel.rank_local:%s" % (str(parallel.rank),
        #                                                     str(parallel.rank_local)))
    if queue is None:
        queue = cuda.Stream()
    return context, queue



