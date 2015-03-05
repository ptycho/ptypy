"""
A client wrapper for plotting.
"""

import time
import numpy as np
from threading import Thread, Lock

if __name__ == "__main__":
    from ptypy.utils.verbose import logger, report
    from ptypy.utils.parameters import Param
else:
    from .verbose import logger, report
    from .parameters import Param

class PlotClient(object):
    """
    A client that connects and continually gets the data required for plotting.
    Note: all data is transferred as soon as the server provides it. This might
    be a waste of bandwidth if all that is required is a client that plots "on demand"...

    This PlotClient doesn't actually plot.
    """

    def __init__(self, client_pars=None):
        """
        Create a client and attempt to connect to a running reconstruction server.
        """
        # This avoids circular imports.
        from ptypy.io.interaction import Client

        # If client_pars is None, connect with the defaults defined in ptypy.io.interaction
        self.client = Client(client_pars)

        # Data requests are stored in this dictionary:
        # self.cmd_dct['cmd'] = [ticket, buffer, key]
        # When the data associated with a ticket arrives it is places in buffer[key].
        self.cmd_dct = {}

        # Initialize data containers. Here we use our own "Param" class, which adds attribute access
        # On top of dictionary.
        self.pr = Param()  # Probe
        self.ob = Param()  # Object
        self.runtime = Param()  # Runtime information (contains reconstruction metrics)
        self._new_data = False

        # The thread that will manage incoming data in the background
        self._thread = None
        self._stopping = False
        self._lock = Lock()

        # Here you should initialize plotting capabilities.

    @property
    def new_data(self):
        """
        True only if new data were acquired since last time checked.
        """
        if self._new_data:
            self._new_data = False
            return True
        else:
            return False

    def start(self):
        """
        This needs to be run for the thread to initialize.
        """
        self._stopping = False
        self._thread = Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def get_data(self):
        """
        Thread-safe way to copy data buffer.
        :return:
        """
        with self._lock:
            pr = self.pr.copy(depth=10)
            ob = self.ob.copy(depth=10)
            runtime = self.runtime.copy(depth=10)
        return pr, ob, runtime

    def _connect(self):
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

    def _initialize(self):
        """
        Wait for reconstruction to start, then grab synchronously basic information
        for initialization.
        """
        logger.info('Requesting configuration parameters')
        self.config = self.client.get_now("Ptycho.p.plotclient")
        logger.info('I have received the following configuration:')
        logger.info(report(self.config))
        
        
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
            S = Param()
            self.ob[ID] = S
            self.cmd_dct["Ptycho.obj.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.obj.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.obj.S['%s'].center" % str(ID)] = [None, S, 'center']

        # Get the list of probe IDs
        pr_IDs = self.client.get_now("Ptycho.probe.S.keys()")
        logger.info('1 probe to plot.' if len(pr_IDs) == 1 else '%d probes to plot.' % len(pr_IDs))

        # Prepare the data requests
        for ID in pr_IDs:
            S = Param()
            self.pr[ID] = S
            self.cmd_dct["Ptycho.probe.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.probe.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.probe.S['%s'].center" % str(ID)] = [None, S, 'center']

        # Data request for the error.
        self.cmd_dct["Ptycho.runtime['iter_info']"] = [None, self.runtime, 'iter_info']

    def _request_data(self):
        """
        Request all data to the server (asynchronous).
        """
        for cmd, item in self.cmd_dct.iteritems():
            item[0] = self.client.get(cmd)
            
    def _store_data(self):
        """
        Transfer all data from the client to local attributes.
        """
        with self._lock:
            for cmd, item in self.cmd_dct.iteritems():
                item[1][item[2]] = self.client.data[item[0]]
            # An extra step for the error. This should be handled differently at some point.
            # self.error = np.array([info['error'].sum(0) for info in self.runtime.iter_info])
            self._new_data = True
        self.client.flush()

    def _loop(self):
        # Activate the client thread.
        self._connect()

        self._initialize()
        while not self._stopping:
            self._request_data()
            self.client.wait()
            self._store_data()
            logger.info('New data arrived.')


if __name__ == "__main__":
    pc = PlotClient()
    pc.start()
    print 'Client running in the background.'
    print 'I will sleep for a while...'
    time.sleep(10)
    if pc.new_data:
        print 'New data have arrived!'
    else:
        print '...still no new data...'
    time.sleep(20)
