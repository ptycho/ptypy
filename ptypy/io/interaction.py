# -*- coding: utf-8 -*-
"""
Interaction module

Provides the server and a basic client to interact with it.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import zmq
import time
import string
import random
import sys
from threading import Thread, Event
import queue
import numpy as np
import re
import json

from ..utils.verbose import logger
from .. import defaults_tree

__all__ = ['Server', 'Client']

DEBUG = lambda x: None


# DEBUG = print

def ID_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Generate a random ID string made of capital letters and digits.
    size [default=6] is the length of the string.
    """
    return ''.join(random.choice(chars) for _ in range(size))


def is_str(s):
    """
    Test if s behaves like a string.
    """
    try:
        s + ''
        return True
    except:
        pass
    return False


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder class that take out numpy arrays from a structure
    and replace them with a code string.
    """

    def encode(self, obj):
        # Prepare the array list
        self.npy_arrays = []

        # Encode as usual
        s = json.JSONEncoder.encode(self, obj)

        # Return the list along with the encoded object
        npy_arrays = self.npy_arrays
        del self.npy_arrays
        return s, npy_arrays

    def default(self, obj):
        if hasattr(obj, '__array_interface__'):
            # obj is "array-like". Add it to the list
            self.npy_arrays.append(obj)

            # Replace obj by a key string giving the index of obj in the list
            return u'NPYARRAY[%03d]' % (len(self.npy_arrays)-1)

        return json.JSONEncoder.default(self, obj)


NE = NumpyEncoder()

# This is the string to match against when decoding
NPYARRAYmatch = re.compile(r"NPYARRAY\[([0-9]{3})\]")


def numpy_replace(obj, arraylist):
    """
    Takes an object decoded by JSON and replaces the arrays where
    they should be. (this function is recursive).
    """
    if is_str(obj):
        match = NPYARRAYmatch.findall(obj)
        if match:
            return arraylist[int(match[0])]
        return obj
    elif isinstance(obj, dict):
        newobj = {}
        for k, v in obj.items():
            newobj[k] = numpy_replace(v, arraylist)
        return newobj
    elif isinstance(obj, list):
        return [numpy_replace(x, arraylist) for x in obj]
    else:
        return obj


def numpy_zmq_send(out_socket, obj):
    """
    Send the given object using JSON, taking care of numpy arrays.
    """

    # Encode the object
    s, npy_arrays = NE.encode(obj)

    # (re-decode the string that does not contain numpy arrays anymore)
    s = json.loads(s)

    if not npy_arrays:
        # Simple case, just send the object as-is
        out = {'hasarray': False, 'message': obj}
        out_socket.send_json(out)
        return

    # Make sure arrays to be sent have contiguous buffers
    npy_arrays = [a if a.flags.contiguous else a.copy() for a in npy_arrays]

    # Prepare the shape and datatype information for each array
    arrayprops = [{'dtype': a.dtype.str, 'shape': a.shape} for a in npy_arrays]

    # Send the header
    out_socket.send_json({'hasarray': True, 'message': s, 'arraylist': arrayprops}, flags=zmq.SNDMORE)

    # Send the numpy arrays as raw binary data
    for a in npy_arrays[:-1]:
        out_socket.send(a, copy=False, track=True, flags=zmq.SNDMORE)
    out_socket.send(npy_arrays[-1], copy=False, track=True)
    return


def numpy_zmq_recv(in_socket):
    """
    Receive a JSON object, taking care of numpy arrays
    """
    numpy_container = in_socket.recv_json()
    message = numpy_container['message']
    if numpy_container['hasarray']:
        # Arrays are transmitted after the header
        arraylist = []
        for arrayinfo in numpy_container['arraylist']:
            msg = in_socket.recv()
            buf = memoryview(msg)
            arraylist.append(np.frombuffer(buf, dtype=arrayinfo['dtype']).reshape(arrayinfo['shape']))
        return numpy_replace(message, arraylist)
    else:
        # No array to process.
        return message


@defaults_tree.parse_doc('io.interaction.server')
class Server(object):
    """
    Main ZMQ server class.

    Defaults:

    [active]
    default = True
    help = Activation switch
    doc = Set to ``False`` for no  ZeroMQ interaction server
    type = bool

    [address]
    default = 'tcp://127.0.0.1'
    help = The address the server is listening to.
    doc = Wenn running ptypy on a remote server, it is the servers network address.
    type = str
    userlevel = 2

    [port]
    default = 5560
    help = The port the server is listening to.
    doc = Make sure to pick an unused port with a few unused ports close to it.
    type = int
    userlevel = 2

    [connections]
    default = 10
    help = Number of concurrent connections on the server
    doc = A range ``[port : port+connections]`` of ports adjacent :py:data:`~.io.interaction.port`
      will be opened on demand for connecting clients.
    type = int
    userlevel = 2

    [poll_timeout]
    default = 10.0
    type = float
    help = Network polling interval
    doc = Network polling interval, in milliseconds.

    [pinginterval]
    default = 2
    type = float
    help = Interval to check pings
    doc = Interval with which to check pings, in seconds.

    [pingtimeout]
    default = 10
    type = float
    help = Ping time out
    doc = Ping time out: client disconnected after this period, in seconds.

    """

    def __init__(self, pars=None, **kwargs):
        """
        Interaction server, meant to run asynchronously with process 0 to manage client requests.

        Parameters
        ----------
        pars : dict or Param
            Parameter set for the server.

        Keyword Arguments
        -----------------
        address : str
            Primary address

        port : int
            Primary port

        connections : int
            number of ports to open on demand *behind* the primary port.

        poll_timeout : float
            Network polling interval (in milliseconds!).

        pinginterval : float
            Interval to check pings (in seconds).

        pingtimeout : float
            Ping time out: client disconnected after this period (in seconds).

        """
        #################################
        # Initialize all parameters
        #################################

        p = self.DEFAULT
        p.update(pars)
        p.update(kwargs)
        self.p = p

        # sanity check for port range:
        # if str(p.port_range)==p.port_range:
        #    from ptypy.utils import str2range
        #    p.port_range = str2range(p.port_range)

        self.address = p.address
        self.port = p.port

        self.poll_timeout = p.poll_timeout
        self.pinginterval = p.pinginterval
        self.pingtimeout = p.pingtimeout

        # Object list, from which data can be transferred
        self.objects = dict()

        # Client names (might not be unique, but can be informative)
        self.names = {}

        # Ping times for all connected Clients
        self.pings = {}

        # Last time a ping check was done
        self.pingtime = time.time()

        # Command queue
        self.queue = queue.Queue()

        # Initialize flags to communicate state between threads.
        self._need_process = False
        self._can_process = False

        self._thread = None
        self._stopping = False
        self._activated = False

        # Bind command names to methods
        self.cmds = {'CONNECT': self._cmd_connect,        # Initial connection from client
                     'DISCONNECT': self._cmd_disconnect,  # Disconnect from client
                     'DO': self._cmd_queue_do,            # Execute a command (synchronous)
                     'GET': self._cmd_queue_get,          # Send an object to the client (synchronous)
                     'GETNOW': self._cmd_get_now,         # Send an object to the client (asynchronous)
                     'SET': self._cmd_queue_set,          # Set an object sent by the client (synchronous)
                     'PING': self._cmd_ping,              # Regular ping from client
                     'AVAIL': self._cmd_avail,            # Send list of available objects
                     'SHUTDOWN': self._cmd_shutdown}      # Shut down the server

        self.make_ID_pool()

    def make_ID_pool(self):

        port_range = list(range(self.port+1,self.port+self.p.connections+1))
        # Initial ID pool
        IDlist = []
        # This loop ensures all IDs are unique
        while len(IDlist) < len(port_range):
            newID = ID_generator()
            if newID not in IDlist:
                IDlist.append(newID)
        self.ID_pool = list(zip(IDlist, port_range))

    def activate(self):
        """
        This needs to be run for the thread to initialize.
        Returns port where the server listens or None if Server failed
        """
        self._stopping = False
        self._thread = Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        time.sleep(0.02)
        port = None
        for i in range(10):
            if self._activated:
                port = self.port
                break
            print("Waiting for Server to awake")
            time.sleep(0.05)

        return port

    def send_warning(self, warning_message):
        """
        Queue a warning message for all connected clients.
        """
        DEBUG('Queuing a WARN command')
        for ID in self.names.keys():
            self.queue.put({'ID': ID, 'cmd': 'WARN', 'ticket': 'WARN', 'str': warning_message})
        self._need_process = True
        return {'status': 'ok'}

    def send_error(self, error_message):
        """
        Queue an ERROR message for all connected clients.
        """
        DEBUG('Queuing a ERROR command')
        for ID in self.names.keys():
            self.queue.put({'ID': ID, 'cmd': 'ERROR', 'ticket': 'ERROR', 'str': error_message})
        self._need_process = True
        return {'status': 'ok'}

    def _run(self):
        """
        Prepare the server and start listening for connections.
        (runs on the separate thread)
        """

        # Initialize socket for entry point
        self.context = zmq.Context()
        self.in_socket = self.context.socket(zmq.REP)
        success = False
        for pp in range(0, 200, 20):
            port = self.port + pp
            fulladdress = self.address + ':' + str(port)
            try:
                self.in_socket.bind(fulladdress)
                success = True
            except zmq.ZMQError:
                print("Port %d used, increase port by 20" % self.port)
                continue
            if success:
                break

        if self.port != port:
            # not thread safe
            self.port = port
            self.make_ID_pool()

        print("Server listens on %s, port %s" % (str(self.address), str(self.port)))

        # Initialize list of requests
        self.out_sockets = {}

        # Initialize poller
        self.poller = zmq.Poller()
        self.poller.register(self.in_socket, zmq.POLLIN)

        self._activated = True
        # Start the main loop with a little bit naive protection for
        # rogue threads that block ports. Only works if thread raised
        # an exception.
        try:
            self._listen()
        finally:
            print("stop listening on %s, port %s" % (str(self.address), str(self.port)))
            self.in_socket.unbind(fulladdress)
            self.in_socket.close()

    def _listen(self):
        """
        Listen for connections and process requests when given permission
        to do so.
        (run on the separate thread)
        """
        while not self._stopping:
            # Process queued requests
            # This happens only if the main thread has given its OK to proceed.
            if self._can_process:
                self._need_process = False
                self._can_process = False
                self._process()

            # Check for new requests
            if self.poller.poll(self.poll_timeout):

                # Get new command
                message = self._recv(self.in_socket)

                # Parse it
                reply = self._parse_message(message)

                # Send reply to client
                self._send(self.in_socket, reply)

            # Regular client ping
            self._checkping()

    def _parse_message(self, message):
        """\
        Parse the message sent by the bound client and queue the corresponding
        command if needed.

        message is {'ID':ID, 'cmd':command, 'args':kwargs}
        """

        reply = self.cmds[message['cmd']](message['ID'], args=message['args'])
        return reply

    def _checkping(self):
        """\
        Check if all clients are still alive.
        """
        now = time.time()
        if now - self.pingtime > self.pinginterval:
            # Time to check
            todisconnect = []
            for ID, lastping in self.pings.items():
                if now - lastping > self.pingtimeout:
                    # Timeout! Force disconnection
                    todisconnect.append(ID)
            # Disconnection is done after the check because even self.ping is modified.
            for ID in todisconnect: self._cmd_disconnect(ID, None)
            self.pingtime = now

    def _cmd_connect(self, ID, args):
        """\
        Process a CONNECT command.
        Connect a new client and give it a new ID.
        """
        DEBUG('Processing a CONNECT command')
        # create new ID
        try:
            newID, newport = self.ID_pool.pop()
        except IndexError:
            # No more connections are allowed
            return {'ID': None, 'status': 'Error: no more connection allowed'}

        # New connection socket for the client (PUB means one way)
        out_socket = self.context.socket(zmq.PUB)
        fulladdress = self.address + ':' + str(newport)
        out_socket.bind(fulladdress)
        self.out_sockets[newID] = (out_socket, newport)
        self.names[newID] = args['name']
        logger.debug('Connected new client "%s" (ID=%s) on port %s' % (args['name'], newID, str(newport)))

        return {'status': 'ok', 'ID': newID, 'port': newport}

    def _cmd_disconnect(self, ID, args):
        """\
        Process a DISCONNECT command.
        Disconnect the client ID.
        """
        DEBUG('Processing a DISCONNECT command')
        socktuple = self.out_sockets.pop(ID, None)
        if socktuple is None:
            logger.debug('Warning: attempt at disconnecting non-existant socket ID=%s' % ID)
            return {'status': 'Error: no such socket ID.'}
        sock, port = socktuple
        sock.close()
        self.pings.pop(ID)
        self.names.pop(ID)
        logger.debug('Client ID=%s disconnected.' % ID)

        # Recycle port, generate new ID
        IDtuple = (ID_generator(), port)

        # Place this back in the pool
        self.ID_pool.append(IDtuple)

        return {'status': 'ok'}

    def _cmd_shutdown(self, ID, args):
        """
        Shutdown the server. Should a Client be allowed to do this?
        """
        self._stopping = True
        self._can_process = True
        return {'status': 'ok'}

    def _cmd_avail(self, ID, args):
        """\
        Process and AVAIL command
        Send available objects.
        """
        DEBUG('Processing an AVAIL command')
        return {'status': 'ok', 'avail': list(self.objects.keys())}

    def _cmd_ping(self, ID, args):
        """\
        Process a PING command
        send back a PONG.
        """
        DEBUG('Processing a PING command')
        self.pings[ID] = time.time()
        return {'status': 'pong'}

    def _cmd_queue_do(self, ID, args):
        """\
        Process a DO command (put it in the queue).
        """
        DEBUG('Queuing a DO command')
        self.queue.put({'ID': ID, 'cmd': 'DO', 'ticket': args['ticket'], 'str': args['str']})
        self._need_process = True
        return {'status': 'ok'}

    def _cmd_queue_get(self, ID, args):
        """\
        Process a GET command (put it in the queue).
        """
        DEBUG('Queuing a GET command')
        self.queue.put({'ID': ID, 'cmd': 'GET', 'ticket': args['ticket'], 'str': args['str']})
        self._need_process = True
        return {'status': 'ok'}

    def _cmd_queue_set(self, ID, args):
        """\
        Process a SET command (put it in the queue).
        """
        DEBUG('Queuing a SET command')
        self.queue.put({'ID': ID, 'cmd': 'SET', 'ticket': args['ticket'], 'str': args['str'], 'val': args['val']})
        self._need_process = True
        return {'status': 'ok'}

    def _cmd_get_now(self, ID, args):
        """\
        Return the requested object to be sent immediately as a reply.
        This may be dangerous (nothing is done to ensure thread safety).
        """
        DEBUG('Executing GETNOW command')
        status = 'ok'
        try:
            out = eval(args['str'], {}, self.objects)
        except:
            status = sys.exc_info()[0]
            out = None

        return {'status': status, 'out': out}

    def _send(self, out_socket, obj):
        """\
        Send the given object using JSON, taking care of numpy arrays.
        """
        try:
            numpy_zmq_send(out_socket, obj)
        except:
            print('Problem sending object %s' % repr(obj))
            raise

    def _recv(self, in_socket):
        """\
        Receive a JSON object, taking care of numpy arrays
        """
        return numpy_zmq_recv(in_socket)

    def _process(self):
        """\
        Loop through the queued commands and execute them. Normally access to any object
        is safe since the other thread is waiting for this function to complete.
        """
        DEBUG('Executing queued commands')
        t0 = None

        # Loop until the queue is empty
        while True:
            try:
                q = self.queue.get_nowait()
            except queue.Empty:
                break

            # Keep track of ticket number
            ticket = q['ticket']
            logger.debug('Processing ticket %s from client %s' % (str(ticket), str(q['ID'])))

            # Nothing to do if the client is not connected anymore
            if q['ID'] not in self.names.keys():
                self.queue.task_done()
                logger.debug('Client %s disconnected. Skipping.' % q['ID'])
                continue

            # Measure how long a transfer takes
            if t0 is None:
                t0 = time.time()

            status = 'ok'

            # Process the command
            if q['cmd'] == 'GET':
                try:
                    out = eval(q['str'], {}, self.objects)
                except:
                    status = sys.exc_info()[0]
                    out = None
            elif q['cmd'] == 'SET':
                try:
                    self.objects[q['str']] = q['val']
                    out = None
                except:
                    status = sys.exc_info()[0]
                    out = None
            elif q['cmd'] == 'DO':
                try:
                    exec (q['str'], {}, self.objects)
                    out = None
                except:
                    status = sys.exc_info()[0]
                    out = None
            elif q['cmd'] in ['WARN', 'ERROR']:
                out = q['str']

            # This is the socket to send data to
            out_socket = self.out_sockets[q['ID']][0]

            # Send the data
            try:
                self._send(out_socket, {'ticket': ticket, 'status': status, 'out': out})
            except TypeError:
                # We have tried to send something that JSON doesn't support.
                self._send(out_socket, {'ticket': ticket, 'status': 'TypeError', 'out': None})

            # Task completed!
            self.queue.task_done()

        # We get here only once the queue is empty.
        if t0 is not None:
            logger.debug('Time spent : %f' % (time.time() - t0))
        return

    def register(self, obj, name):
        """\
        Exposes the content of an object for transmission and interaction.
        For now this is equivalent to Interactor.object[name] = obj, but maybe
        use weakref in the future?
        """
        if name in self.objects:
            logger.debug('Warning an object called %s already there.' % name)
        self.objects[name] = obj

    def process_requests(self, tinterval=.001):
        """\
        Give permission to the serving thread to send objects and
        wait for it to complete (safer!)
        """
        if self._need_process:
            # Set flag to allow sending objects
            self._can_process = True

            # Wait until the command queue is empty
            self.queue.join()

    def stop(self):
        if not self._stopping:
            logger.debug("Stopping Interaction Server.")
            self._stopping = True
            self._thread.join(3)


@defaults_tree.parse_doc('io.interaction.client')
class Client(object):
    """
    Basic but complete client to interact with the server.

    Defaults:

    [address]
    default = 'tcp://127.0.0.1'
    help = The address the server is listening to.
    doc = Wenn running ptypy on a remote server, it is the servers network address.
    type = str
    userlevel = 2

    [port]
    default = 5560
    help = The port the server is listening to.
    doc = Make sure to pick an unused port with a few unused ports close to it.
    type = int
    userlevel = 2

    [poll_timeout]
    default = 100.0
    type = float
    help = Network polling interval
    doc = Network polling interval, in milliseconds.

    [pinginterval]
    default = 1
    type = float
    help = Interval to check pings
    doc = Interval with which to check pings, in seconds.

    [connection_timeout]
    default = 3600000.0
    type = float
    help = Timeout for dead server
    doc = Timeout for dead server, in milliseconds.

    """

    def __init__(self, pars=None, **kwargs):

        p = self.DEFAULT.copy()
        p.update(pars)
        p.update(kwargs)
        self.p = p

        self.req_address = p.address
        self.req_port = p.port
        self.poll_timeout = p.poll_timeout
        self.pinginterval = p.pinginterval
        self.connection_timeout = p.connection_timeout

        # Initially not connected
        self.connected = False

        # A name for us.
        self.name = self.__class__.__name__

        # Command queue
        self.cmds = []

        # Data container
        self.data = {}

        # Status container
        self.status = {}

        # ticket counter
        self.masterticket = 0

        # ticket status
        self.tickets = {}

        # list of pending transactions
        self.pending = []

        # list of completed transactions
        self.completed = []

        # Access to data with user-defined tags
        self.datatag = {}
        self.tickets_to_tags = {}
        self.tags_to_tickets = {}

        self.lastping = 0

        self._thread = None
        self._stopping = False

    def activate(self):
        """
        Create the thread.
        """
        self._stopping = False
        self._thread = Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        """
        Thread initialization: Connect to the server and create transmission sockets.
        """

        # Initialize socket for entry point
        self.context = zmq.Context()
        self.req_socket = self.context.socket(zmq.REQ)
        fulladdress = self.req_address + ':' + str(self.req_port)
        self.req_socket.connect(fulladdress)

        # Establish connection with the interactor by sending a "CONNECT" command
        self._send(self.req_socket, {'ID': None, 'cmd': 'CONNECT', 'args': {'name': self.name}})
        reply = self._recv(self.req_socket)
        if reply['status'] != 'ok':
            raise RuntimeError('Connection failed! (answer: %s)' % reply['status'])

        # Connection was successful. Prepare the data pipe
        self.ID = reply['ID']
        self.bind_port = reply['port']
        logger.debug('Connected to server as ID=%s on port %s' % (self.ID, str(self.bind_port)))
        self.bind_address = self.req_address
        fulladdress = self.bind_address + ':' + str(self.bind_port)
        self.bind_socket = self.context.socket(zmq.SUB)
        self.bind_socket.connect(fulladdress)
        self.bind_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Initialize poller
        self.poller = zmq.Poller()
        self.poller.register(self.bind_socket, zmq.POLLIN)
        self.req_poller = zmq.Poller()
        self.req_poller.register(self.req_socket, zmq.POLLIN)
        self.connected = True

        # Create "Event" flags for calls that require synchronization.
        self.getnow_flag = Event()
        self.getnow_flag.clear()
        self.avail_flag = Event()
        self.avail_flag.clear()
        self.shutdown_flag = Event()
        self.shutdown_flag.clear()

        # Start main listening loop
        self._listen()

    def _listen(self):
        """
        Main event loop (running on a thread).
        """
        while not self._stopping:
            # Process messages to send
            while self.cmds:
                # Pop the command to process (FIFO)
                cmd = self.cmds.pop(0)

                # Send the command
                self._send(self.req_socket, cmd)

                # Wait for the reply and store it
                if self.req_poller.poll(self.connection_timeout):
                    reply = self._recv(self.req_socket)
                else:
                    # Server has timed out on its reply!
                    logger.warning('Server did not reply. Shutting down the connection.')
                    self._stopping = True

                # Ignore pongs
                if reply['status'] == 'pong':
                    continue

                self.last_reply = reply

                # Flip the flag in case other threads were waiting for a synchronous call.
                if cmd['cmd'] == 'GETNOW':
                    self.getnow_flag.set()
                elif cmd['cmd'] == 'AVAIL':
                    self.avail_flag.set()
                elif cmd['cmd'] == 'SHUTDOWN':
                    self.shutdown_flag.set()

            # Check for data
            if self.poller.poll(self.poll_timeout):
                self._read_message()
            self._ping()

        self.connected = False

    def _ping(self):
        """
        Send a ping
        """
        now = time.time()
        if now - self.lastping > self.pinginterval:
            self.cmds.append({'ID': self.ID, 'cmd': 'PING', 'args': None})
            self.lastping = now
        return

    def _send(self, out_socket, obj):
        """
        Send the given object using JSON, taking care of numpy arrays.
        """
        numpy_zmq_send(out_socket, obj)

    def _recv(self, in_socket):
        """
        Receive a JSON object, taking care of numpy arrays
        """
        return numpy_zmq_recv(in_socket)

    def _read_message(self):
        """
        Read the message sent by the server and store the accompanying data
        if needed.
        """

        message = self._recv(self.bind_socket)
        ticket = message['ticket']

        # Store data
        self.data[ticket] = message['out']
        self.status[ticket] = message['status']

        # Assign data to tag if asked by the user.
        if ticket in self.tickets_to_tags:
            tag = self.tickets_to_tags[ticket]
            self.datatag[tag] = self.data[ticket]
            del self.tickets_to_tags[ticket]
            del self.tags_to_tickets[tag]

        # Done with this ticket!
        if ticket in self.pending:
            self.tickets[ticket] = 'completed'
            self.pending.remove(ticket)
            self.completed.append(ticket)
            self.newdata(ticket)
        else:
            # We are being sent something we didn't ask for (warning or error)
            self.unexpected_ticket(ticket)

    def flush(self):
        """
        Delete all stored data (and accompanying status).
        """
        self.data = {}
        self.status = {}
        self.datatag = {}

    def poll(self, ticket=None, tag=None):
        """
        Returns true if the transaction for a given ticket is completed.
        If ticket and tag are None, returns true only if no transaction is pending
        """
        if ticket is None and tag is None:
            return not self.pending
        if ticket is not None:
            return ticket in self.completed
        if tag is not None:
            ticket = self.tags_to_tickets.get(tag, None)
            return ticket in self.completed

    def wait(self, ticket=None, tag=None, timeout=None):
        """
        Blocks and return True only when the transaction for a given ticket is completed.
        If ticket is None, returns only when no more transaction are pending.
        If timeout is a positive number, wait will return False after timeout seconds if the ticket(s)
        had not been processed yet.
        """

        # Prepare timeout
        if timeout is None: timeout = 1e10
        t0 = time.time()

        # Deal with the case where a tag was given instead of a ticket
        if ticket is None and tag is not None:
            ticket = self.tags_to_tickets.get(tag, None)

        # Wait for completed ticket.
        while True:
            if (ticket is None and not self.pending) or (ticket in self.completed): return True
            time.sleep(0.01)
            if (time.time() - t0) > timeout:
                return False

    def newdata(self, ticket):
        """
        Meant to be replaced, e.g. to send signals to a GUI.
        """
        pass

    def unexpected_ticket(self, ticket):
        """
        Used to deal with warnings sent by the server.
        """
        logger.debug(str(ticket) + ': ' + str(self.data[ticket]))

    def stop_server(self):
        """
        Send a SHUTDOWN command - is this a good idea?
        """
        self.cmds.append({'ID': self.ID, 'cmd': 'SHUTDOWN', 'args': None})
        self.shutdown_flag.clear()
        self.shutdown_flag.wait()
        return self.last_reply

    def stop(self):
        if not self._stopping:
            logger.debug("Stopping.")
            self._stopping = True
            self._thread.join(3)

    def avail(self):
        """
        Queries the server for the name of objects available.
        ! Synchronous call !
        """
        self.cmds.append({'ID': self.ID, 'cmd': 'AVAIL', 'args': None})
        self.avail_flag.clear()
        self.avail_flag.wait()
        return self.last_reply

    def do(self, execstr, timeout=0, tag=None):
        """
        Modify and object using an exec string.
        This function returns the "ticket number" which identifies the object once
        it will have been transmitted. If timeout > 0 and the requested object has
        been transmitted within timeout seconds, return a tuple (ticket, data).
        """
        ticket = self.masterticket + 1
        self.masterticket += 1
        self.cmds.append({'ID': self.ID, 'cmd': 'DO', 'args': {'ticket': ticket, 'str': execstr}})
        self.tickets[ticket] = 'pending'
        self.pending.append(ticket)
        if tag is not None:
            self.tags_to_tickets[tag] = ticket
            self.tickets_to_tags[ticket] = tag
        if timeout > 0:
            if self.wait(ticket, timeout):
                return ticket, self.data[ticket]
        return ticket

    def get(self, evalstr, timeout=0, tag=None):
        """
        Requests an object (or part of it) using an eval string.
        This function returns the "ticket number" which identifies the object once
        it will have been transmitted. If timeout > 0 and the requested object has
        been transmitted within timeout seconds, return a tuple (ticket, data).
        """
        ticket = self.masterticket + 1
        self.masterticket += 1
        self.cmds.append({'ID': self.ID, 'cmd': 'GET', 'args': {'ticket': ticket, 'str': evalstr}})
        self.tickets[ticket] = 'pending'
        self.pending.append(ticket)
        if tag is not None:
            self.tags_to_tickets[tag] = ticket
            self.tickets_to_tags[ticket] = tag
        if timeout > 0:
            if self.wait(ticket, timeout):
                return ticket, self.data[ticket]
        return ticket

    def set(self, varname, varvalue, timeout=0, tag=None):
        """
        Sets an object named varname to the value varvalue.
        """
        ticket = self.masterticket + 1
        self.masterticket += 1
        self.cmds.append({'ID': self.ID, 'cmd': 'SET', 'args': {'ticket': ticket, 'str': varname, 'val': varvalue}})
        self.tickets[ticket] = 'pending'
        self.pending.append(ticket)
        if tag is not None:
            self.tags_to_tickets[tag] = ticket
            self.tickets_to_tags[ticket] = tag
        if timeout > 0:
            if self.wait(ticket, timeout):
                return ticket, self.data[ticket]
        return ticket

    def get_now(self, evalstr):
        """
        Synchronous get. May be dangerous, but should be safe for small objects like parameters.
        """
        self.cmds.append({'ID': self.ID, 'cmd': 'GETNOW', 'args': {'str': evalstr}})
        self.getnow_flag.clear()
        self.getnow_flag.wait()

        return self.last_reply['out']
