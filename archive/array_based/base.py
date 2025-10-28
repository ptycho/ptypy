from collections import OrderedDict

class Adict(object):

    def __init__(self):
        pass


class BaseKernel(object):

    def __init__(self, queue_thread=None, verbose=False):

        self.queue = queue_thread
        self.verbose = False
        self.npy = Adict()
        self.ocl = Adict()
        self.benchmark = OrderedDict()


    def log(self, x):
        if self.verbose:
            print(x)