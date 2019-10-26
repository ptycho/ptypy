"""
Provides a PtyScan class for reading zmq streamed data from Contrast:
https://github.com/alexbjorling/contrast    
"""

import numpy as np
import zmq
import time

from ..core.data import PtyScan
from .. import utils as u
from . import register
from ..utils import parallel

logger = u.verbose.logger


@register()
class ContrastZmqScan(PtyScan):
    """
	This class parses zmq streams from the Contrast system.

    Defaults:

    [name]
    default = ContrastZmqScan
    type = str
    help =
    doc =

    [host]
    default = 'localhost'
    type = str
    help = Name of the publishing host
    doc =

    [port]
    default = 5556
    type = int
    help = Port number on the publishing host
    doc =

    [xMotor]
    default = sx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = sy
    type = str
    help = Which y motor to use
    doc =

    [detector]
    default = 'diff'
    type = str
    help = Which detector to use

    """

    def __init__(self, *args, **kwargs):
        super(ContrastZmqScan, self).__init__(*args, **kwargs)
        self.context = zmq.Context()
        socket = self.context.socket(zmq.SUB)
        socket.connect("tcp://%s:%u" % (self.info.host, self.info.port))
        socket.setsockopt(zmq.SUBSCRIBE, b"") # subscribe to all topics
        self.socket = socket
        self.latest_index_received = -1
        self.incoming = {}

    def check(self, frames=None, start=None):
        end_of_scan = False
        while True:
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                headers = (msg == {}) or (u'path' in msg.keys())
                if not headers:
                    self.latest_index_received += 1
                    self.incoming[self.latest_index_received] = msg
                if (self.latest_index_received - start + 1) == frames:
                    break
                elif msg == {}:
                    end_of_scan = True
                    break
            except zmq.ZMQError:
                # no more data available - working around bug in ptypy here
                if self.latest_index_received < self.info.min_frames * parallel.size:
                    print 'have %u frames, waiting...' % (self.latest_index_received + 1)
                    time.sleep(.5)
                else:
                    break
        ret = self.latest_index_received - start + 1
        return ret, end_of_scan

    def load(self, indices):
        raw, weight, pos = {}, {}, {}

        # communication
        if parallel.master:
            # send data to each node
            for node in range(1, parallel.size):
                node_inds = parallel.receive(source=node)
                dct = {i:self.incoming[i] for i in node_inds}
                parallel.send(dct, dest=node)
                for i in node_inds:
                    del self.incoming[i]
            # take data for this node
            dct = {i: self.incoming[i] for i in indices}
            for i in indices:
                del self.incoming[i]
        else:
            # receive data from the master node
            parallel.send(indices, dest=0)
            dct = parallel.receive(source=0)

        # repackage data and return
        for i in  indices:
            raw[i] = dct[i][self.info.detector]
            pos[i] = np.array([
                        dct[i][self.info.xMotor],
                        dct[i][self.info.yMotor],
                        ])
            weight[i] = np.ones_like(raw[i])

        return raw, pos, weight
