"""
Provides a PtyScan class for reading zmq streamed data from Contrast:
https://github.com/alexbjorling/contrast    
"""

import numpy as np
import zmq
from zmq.utils import jsonapi as json
import time
import bitshuffle
import struct

from ..core.data import PtyScan
from .. import utils as u
from . import register
from ..utils import parallel

logger = u.verbose.logger


def decompress(data, shape, dtype):
    block_size = struct.unpack('>I', data[8:12])[0] // dtype.itemsize
    data = np.frombuffer(data[12:], np.int8)
    output = bitshuffle.decompress_lz4(arr=data,
                                       shape=shape,
                                       dtype=dtype,
                                       block_size=block_size)
    return output


@register()
class NanomaxZmqScan(PtyScan):
    """
	This class parses zmq streams from the Contrast system.

    Defaults:

    [name]
    default = NanomaxZmqScan
    type = str
    help =
    doc =

    [host]
    default = '172.16.125.18'
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
    help = Which detector from the contrast stream to use

    [detector_host]
    default = '192.168.19.182'
    type = str
    help = Take images from a separate stream - hostname
    doc =

    [detector_port]
    default = 9998
    type = int
    help = Take images from a separate stream - port
    doc =

    """

    def __init__(self, *args, **kwargs):
        super(NanomaxZmqScan, self).__init__(*args, **kwargs)
        self.context = zmq.Context()

        # main socket
        socket = self.context.socket(zmq.SUB)
        socket.connect("tcp://%s:%u" % (self.info.host, self.info.port))
        socket.setsockopt(zmq.SUBSCRIBE, b"") # subscribe to all topics
        self.socket = socket

        # separate detector socket
        self.stream_images = None not in (self.info.detector_host,
                                          self.info.detector_port)
        if self.stream_images and parallel.master:
            det_socket = self.context.socket(zmq.PULL)
            det_socket.connect("tcp://%s:%u" % (self.info.detector_host, self.info.detector_port))
            self.det_socket = det_socket

        self.latest_pos_index_received = -1
        self.latest_det_index_received = -1
        self.incoming = {}
        self.incoming_det = {}

    def check(self, frames=None, start=None):
        """
        Only called on the master node.
        """
        end_of_scan, end_of_det_stream = False, False

        # get all frames from the main socket
        while True:
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                headers = ('path' in msg.keys())
                if not headers:
                    self.latest_pos_index_received += 1
                    self.incoming[self.latest_pos_index_received] = msg
                elif msg['path'] in ('interrupted', 'finished'):
                    end_of_scan = True
                    break
            except zmq.ZMQError:
                # no more data available - working around bug in ptypy here
                if self.latest_pos_index_received < self.info.min_frames * parallel.size:
                    logger.info('have %u frames, waiting...' % (self.latest_pos_index_received + 1))
                    time.sleep(.5)
                else:
                    break

        # get all frames from the detector socket
        while self.stream_images:
            try:
                parts = self.det_socket.recv_multipart(flags=zmq.NOBLOCK)
                info = json.loads(parts[0])
                shape = info['shape']
                dtype = np.dtype(info['type'])
                img = decompress(parts[1], shape, dtype)
                self.latest_det_index_received += 1
                self.incoming_det[self.latest_det_index_received] = img

            except zmq.ZMQError:
                # no more data available - working around bug in ptypy here
                if self.latest_det_index_received < self.info.min_frames * parallel.size:
                    logger.info('have %u detector frames, waiting...' % (self.latest_det_index_received + 1))
                    time.sleep(.5)
                else:
                    break

        logger.info('-------------------------------------')
        logger.info(self.incoming.keys())
        logger.info(self.incoming_det.keys())
        logger.info('-------------------------------------')
        ind = self.latest_pos_index_received
        if self.stream_images:
            ind = min(ind, self.latest_det_index_received)
        return (ind - start + 1), (end_of_scan and end_of_det_stream)

    def load(self, indices):
        raw, weight, pos = {}, {}, {}

        # communication
        if parallel.master:
            # send data to each node
            for node in range(1, parallel.size):
                node_inds = parallel.receive(source=node)
                dct = {i:self.incoming[i] for i in node_inds}
                if self.stream_images:
                    # merge streamed images
                    for i in node_inds:
                        dct[i][self.info.detector] = self.incoming_det[i]
                parallel.send(dct, dest=node)
                for i in node_inds:
                    del self.incoming[i]
                    if self.stream_images:
                        del self.incoming_det[i]

            # take data for this node
            dct = {i: self.incoming[i] for i in indices}
            if self.stream_images:
                # merge streamed images
                for i in indices:
                    dct[i][self.info.detector] = self.incoming_det[i]
            for i in indices:
                del self.incoming[i]
                if self.stream_images:
                    del self.incoming_det[i]
        else:
            # receive data from the master node
            parallel.send(indices, dest=0)
            dct = parallel.receive(source=0)

        # repackage data and return
        for i in  indices:
            raw[i] = dct[i][self.info.detector]
#            pos[i] = np.array([
#                        dct[i][self.info.xMotor],
#                        dct[i][self.info.yMotor],
#                        ]) * 1e-6
            x = dct[i][self.info.xMotor]
            y = dct[i][self.info.yMotor]
            pos[i] = -np.array((y, x)) * 1e-6
            weight[i] = np.ones_like(raw[i])
            weight[i][np.where(raw[i] == 2**32-1)] = 0

        return raw, pos, weight
