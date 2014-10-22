# -*- coding: utf-8 -*-
"""\
Verbose package, based on the standard logging library.

Use as:
from verbose import logger
logger.warn('This is a warning')
logger.info('This is an information')
...

TODO:
- Handlers for file and for interaction server
- option for logging.disable to reduce call overhead

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import time
import sys
import inspect
import logging

from . import parallel

__all__ = ['logger', 'set_level', '_']

CONSOLE_FORMAT = {logging.ERROR : 'ERROR %(name)s - %(message)s',
                  logging.WARNING : 'WARNING %(name)s - %(message)s',
                  logging.INFO : '%(message)s',
                  logging.DEBUG : 'DEBUG %(pathname)s [%(lineno)d] - %(message)s'}

FILE_FORMAT = {logging.ERROR : '%(asctime)s ERROR %(name)s - %(message)s',
                  logging.WARNING : '%(asctime)s WARNING %(name)s - %(message)s',
                  logging.INFO : '%(asctime)s %(message)s',
                  logging.DEBUG : '%(asctime)s DEBUG %(pathname)s [%(lineno)d] - %(message)s'}

# Monkey patching logging.Logger - is this a good idea?

#def _MPIlog(self, *args, **kwargs):
#    MPIswitch = kwargs.pop('MPI', False)
#    self._factory_log(*args, extra={'MPI':MPIswitch}, **kwargs)
#
#logging.Logger._factory_log = logging.Logger._log
#logging.Logger._log = _MPIlog 

# Logging filter
class MPIFilter(object):
    """
    A filter that ensures that logging is done only by the master process.
    """
    def __init__(self, allprocesses=False):
        """
        A filter that stops logging for all processes except the master node.
        This behavior can be altered with allprocesses=True.
        """
        self.allprocesses = allprocesses
        
    def filter(self, record):
        """
        Filter method, as expected by the logging API.
        """
        # look for an extra attribute 'allprocesses' to enable logging from all processes
        # usage: logger.info('some message',extra={'allprocesses':True})
        try:
            return record.allprocesses
        except:
            return self.allprocesses or parallel.master

# Logging formatter
class CustomFormatter(logging.Formatter):
    """
    Flexible formatting, depending on the logging level.

    Adapted from http://stackoverflow.com/questions/1343227
    Will have to be updated for python > 3.2.
    """
    DEFAULT = '%(levelname)s: %(message)s'

    def __init__(self, FORMATS=None):
        logging.Formatter.__init__(self)
        self.FORMATS = {} if FORMATS is None else FORMATS

    def format(self, record):
        self._fmt = self.FORMATS.get(record.levelno, self.DEFAULT)
        return logging.Formatter.format(self, record)

# Create logger
logger = logging.getLogger()

# Default level - should be changed as soon as possible
logger.setLevel(logging.WARNING)

# Create console handler
consolehandler = logging.StreamHandler()
logger.addHandler(consolehandler)

# Add formatter
consoleformatter = CustomFormatter(CONSOLE_FORMAT)
consolehandler.setFormatter(consoleformatter)

# Add filter
consolefilter = MPIFilter()
logger.addFilter(consolefilter)

level_from_verbosity = {0:logging.CRITICAL, 1:logging.ERROR, 2:logging.WARN, 3:logging.INFO, 4:logging.DEBUG}
level_from_string = {'CRITICAL':logging.CRITICAL, 'ERROR':logging.ERROR, 'WARN':logging.WARN, 'WARNING':logging.WARN, 'INFO':logging.INFO, 'DEBUG':logging.DEBUG}

def set_level(level):
    """
    Set verbosity level. Kept here for backward compatibility
    """
    if str(level) == level:
        logger.setLevel(level_from_string[level.upper()])
    else:
        logger.setLevel(level_from_verbosity[level])
    logger.info('Verbosity set to %s' % str(level))

# Formatting helper
def _(label, value):
    return '%-25s%s' % (label + ':', str(value))

