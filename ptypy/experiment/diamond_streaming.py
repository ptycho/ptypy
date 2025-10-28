"""\
Description here

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import logging
import zmq
import time
import json
import pickle
import numpy as np
from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy.utils.verbose import log

@register()
class DiamondZMQLoader(PtyScan):
    """

    Defaults:

    [name]
    default = 'DiamondZMQLoader'
    type = str
    help =

    [metadata]
    default = 'tcp://127.0.0.1:5553'
    type = str
    help = Address for metadata socket

    [datastream]
    default = 'tcp://127.0.0.1:5552'
    type = str
    help = Address for datastream socket

    [chunksize]
    default = 50
    type = int
    help = Nr. of frames (chunks) to be pulled at once from the socket

    [logfile]
    default = /tmp/ptypy_streaming_log.json
    type = str
    help = A JSON file for time logging
    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars, in_place_depth=99)
        super().__init__(self.p, **kwargs)

        # ZMQ Context
        self.context = zmq.Context()
        
        # Create socket to request some information and ask for metadata
        self.metadata_socket = self.context.socket(zmq.REQ) 
        self.metadata_socket.connect(self.p.metadata)
        
        # Socket to pull main data
        self.datastream_socket = self.context.socket(zmq.PULL)
        self.datastream_socket.connect(self.p.datastream)
        self.connected = True

        # Meta information
        self.metadata_socket.send(b"Start")
        log(4, 'Waiting for metadata...')
        self.metadata = pickle.loads(self.metadata_socket.recv())
        log(4, "Metadata recieved")
        
        # Setting meta/info parameters
        self.data_dtype = self.metadata["dtype"]
        self.data_shape = self.metadata["shape"]
        self.frame_shape = self.data_shape[1:]
        self.num_frames = self.data_shape[0]
        self.p.shape = self.frame_shape
        self.info.shape = self.p.shape
        self.info.center = None
        self.info.auto_center = self.p.auto_center
        self.meta.energy  = self.p.energy
        self.meta.distance = self.p.distance
        self.info.psize = self.p.psize

        # Create empty memory
        self._data = np.empty(shape=self.data_shape, dtype=self.data_dtype)
        self._pos = np.empty(shape=(self.data_shape[0], 2), dtype=float)
        self.framecount = 0

        # Logging
        self.log = {}
        self.log["start"] = time.time()
        
    def fetch(self,chunksize=1):
        if not self.connected:
            return
        # Fetch data from socket
        for i in range(chunksize):
            databuf, posxbuf, posybuf = self.datastream_socket.recv_multipart()
            self._data[self.framecount] = np.frombuffer(databuf,dtype=self.data_dtype).reshape(self.frame_shape)
            self._pos[self.framecount] = np.array([float(posybuf.decode()), float(posxbuf.decode())])
            self.framecount += 1

    def check(self, frames=None, start=None):
        """
        Check how many frames are available.

        Parameters
        ----------
        frames : int or None
            Number of frames requested.
        start : int or None
            Scanpoint index to start checking from.

        Returns
        -------
        frames_accessible : int
            Number of frames readable.

        end_of_scan : int or None
            is one of the following,
            - 0, end of the scan is not reached
            - 1, end of scan will be reached or is
            - None, can't say
        """

        if start is None:
            start = self.framestart
        
        if frames is None:
            frames = self.min_frames

        if (self.num_frames - self.framecount) > self.p.chunksize:
            chunksize = self.p.chunksize
        else:
            chunksize = 1
        self.fetch(chunksize)
            
        # Check how many frames are available
        available = self.framecount
        new_frames = available - start        
        # not reached expected nr. of frames
        if new_frames <= frames:
            # but its last chunk of scan so load it anyway
            if available == self.num_frames:
                frames_accessible = new_frames    
                end_of_scan = 1
                if self.connected:
                    self.finish()
                    # end all ZMQ communications
                    self.context.destroy()
                    self.connected = False                    
            # otherwise, do nothing
            else:
                end_of_scan = 0
                frames_accessible = 0
        # reached expected nr. of frames
        else:
            end_of_scan = 0
            frames_accessible = frames
        #log(3, f"frames = {frames}, start = {start}, available = {available}, frames_accessible = {frames_accessible}, end_of_scan = {end_of_scan}, new_frames = {new_frames}, num_frames = {self.num_frames}")

        return frames_accessible, end_of_scan
                    
    def load(self, indices):
        """
        return data

        Returns
        -------
        raw, positions, weight : dict
            Dictionaries whose keys are the given scan point `indices`
            and whose values are the respective frame / position according
            to the scan point index. `weight` and `positions` may be empty
        """
        intensities = {}
        positions = {}
        weights = {}
        log(4, "Loading...")
        log(4, f"indices = {indices}")
        for ind in indices:
            intensities[ind] = self._data[ind]
            positions[ind] = self._pos[ind]
            weights[ind] = np.ones(len(intensities[ind]))
            #print(f"Loaded index {ind} with pos {positions[ind]} and data {intensities[ind].sum()}")
                        
        return intensities, positions, weights

    def finish(self):
        with open(self.p.logfile, "w") as f:
            self.log["stop"] = time.time()
            json.dump(self.log,f)
        self.metadata_socket.send(b"Stop")
        # while True:
        #     reply = self.metadata_socket.recv()
        #     print("[Recons] Recevied stop reply ", reply)
        #     break
