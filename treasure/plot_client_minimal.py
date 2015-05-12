"""
A minimal plotting client that doesn't plot.
"""

import time
import numpy as np

import ptypy
from ptypy import utils as u
from ptypy.io.interaction import Client
from ptypy.utils.verbose import logger

ptypy.utils.verbose.set_level(3)


class PlotClientMinimal(object):
    """
    A client that connects and gets the data required for plotting.
    """

    def __init__(self, client_pars=None):
        """
        Create a client and attempt to connect to a running reconstruction server.
        """

        # If client_pars is None, connect with the defaults defined in ptypy.io.interaction
        self.client = Client(client_pars)

        # Data requests are stored in this dictionary:
        # self.cmd_dct['cmd'] = [ticket, buffer, key]
        # When the data associated with a ticket arrives it is places in buffer[key].
        self.cmd_dct = {}

        # This call activates the client thread.
        self.connect()

        # Initialize data containers. Here we use our own "Param" class, which adds attribute access
        # On top of dictionary.
        self.pr = u.Param() # Probe
        self.ob = u.Param() # Object
        self.runtime = u.Param() # Runtime information (contains reconstruction metrics)

        # Here you should initialize plotting capabilities.

    def connect(self):
        """
        Connect to the reconstruction server.
        """
        self.client.activate()
        logger.info('Connecting to server...')

        # Pause until connected
        while not self.client.connected:
            time.sleep(0.1)
        logger.info('Connected.')

    def disconnect(self):
        """
        Disconnect from reconstruction server.
        """
        self.client.stop()

    def initialize(self):
        """
        Wait for reconstruction to start, then grab synchronously basic information
        for initialization.
        """

        # Wait until reconstruction starts.
        logger.info('Waiting for reconstruction to start...')
        ready = self.client.get_now("'start' in Ptycho.runtime")
        while not ready:
            time.sleep(.1)
            ready = self.client.get_now("'start' in Ptycho.runtime")

        logger.info('Ready')

        # Get the list of object IDs
        ob_IDs = self.client.get_now("Ptycho.obj.S.keys()")
        logger.info('1 object to plot.' if len(ob_IDs) == 1 else '%d objects to plot.' % len(ob_IDs))

        # Prepare the data requests
        for ID in ob_IDs:
            S = u.Param()
            self.ob[ID] = S
            self.cmd_dct["Ptycho.obj.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.obj.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.obj.S['%s'].center" % str(ID)] = [None, S, 'center']

        # Get the list of probe IDs
        pr_IDs = self.client.get_now("Ptycho.probe.S.keys()")
        logger.info('1 probe to plot.' if len(pr_IDs) == 1 else '%d probes to plot.' % len(pr_IDs))

        # Prepare the data requests
        for ID in pr_IDs:
            S = u.Param()
            self.pr[ID] = S
            self.cmd_dct["Ptycho.probe.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.probe.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.probe.S['%s'].center" % str(ID)] = [None, S, 'center']

        # Data request for the error.
        self.cmd_dct["Ptycho.runtime['iter_info']"] = [None, self.runtime, 'iter_info']

    def request_data(self):
        """
        Request all data to the server (asynchronous).
        """
        for cmd, item in self.cmd_dct.iteritems():
            item[0] = self.client.get(cmd)
            
    def store_data(self):
        """
        Transfer all data from the client to local attributes.
        """
        for cmd, item in self.cmd_dct.iteritems():
            item[1][item[2]] = self.client.data[item[0]]
            
        self.client.flush()

        # An extra step for the error. This should be handled differently at some point.
        self.error = np.array([info['error'].sum(0) for info in self.runtime.iter_info])

    def loop(self, timeout=0.1):
        self.initialize()
        while True:

            self.request_data()

            self.client.wait()
            # If control has to be returned to the GUI, replace self.client.wait() with
            # while not self.client.wait(timeout):
            #    GUI_specific_process_event()

            self.store_data()

            # New data has arrived. Update plot.

            logger.info('Now I would plot probe and object in self.pr and self.ob')
            logger.info('And the latest error after %d iterations is %s' % (len(self.error), self.error[-1]))


if __name__ == "__main__":
    pcm = PlotClientMinimal()
    pcm.loop()