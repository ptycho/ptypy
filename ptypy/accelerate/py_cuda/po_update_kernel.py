import numpy as np
from . import load_kernel


from ..array_based import po_update_kernel as ab



class PoUpdateKernel(ab.PoUpdateKernel):

    def __init__(self, queue_thread=None):
        super(PoUpdateKernel, self).__init__()
        # and now initialise the cuda

        self.ob_update_cuda = load_kernel("ob_update")
        self.pr_update_cuda = load_kernel("pr_update")

    def ob_update(self, addr, ob, obn, pr, ex):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        num_pods = np.int32(addr.shape[0] * addr.shape[1])
        self.ob_update_cuda(ex, num_pods, prsh[1], prsh[2],
                            pr, prsh[0], prsh[1], prsh[2],
                            ob, obsh[0], obsh[1], obsh[2],
                            addr,
                            obn,
                            block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)

    def pr_update(self, addr, pr, prn, ob, ex):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        num_pods = np.int32(addr.shape[0] * addr.shape[1])
        self.pr_update_cuda(ex, num_pods, prsh[1], prsh[2],
                               pr, prsh[0], prsh[1], prsh[2],
                               ob, obsh[0], obsh[1], obsh[2],
                               addr,
                               prn,
                               block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)
