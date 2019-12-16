context = None
queue = None


def get_queue(new_queue=False):
    from ptypy.utils import parallel
    import pycuda.driver as cuda
    cuda.init()
    global context
    global queue

    if context is None and parallel.rank_local < cuda.Device.count():
        context = cuda.Device(parallel.rank_local).make_context()
        context.push()

    if context is not None:
        if new_queue or queue is None:
            queue = cuda.Stream()
        return context, queue
    else:
        return None
