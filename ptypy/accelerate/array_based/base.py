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