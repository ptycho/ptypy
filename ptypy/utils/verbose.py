# -*- coding: utf-8 -*-
"""\
Verbose package, based on the standard logging library.

Use as:
from verbose import logger
logger.warning('This is a warning')
logger.info('This is an information')
...

TODO:
- Handlers for file and for interaction server
- option for logging.disable to reduce call overhead

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import time
import sys
import inspect
import logging
from time import perf_counter

from . import parallel

__all__ = ['logger', 'set_level', 'report', 'log']

# custom logging levels
INSPECT = 15
INTERACTIVE = 35
CITATION = 45

CONSOLE_FORMAT = {CITATION: '%(message)s',
                  logging.ERROR : 'ERROR %(name)s - %(message)s',
                  INTERACTIVE : '%(message)s',
                  logging.WARNING : 'WARNING %(name)s - %(message)s',
                  logging.INFO : '%(message)s',
                  INSPECT : 'INSPECT %(message)s',
                  logging.DEBUG : 'DEBUG %(pathname)s [%(lineno)d] - %(message)s'}

FILE_FORMAT = {CITATION: '%(message)s',
               logging.ERROR : '%(asctime)s ERROR %(name)s - %(message)s',
               INTERACTIVE : '%(asctime)s %(message)s',
               logging.WARNING : '%(asctime)s WARNING %(name)s - %(message)s',
               logging.INFO : '%(asctime)s %(message)s',
               INSPECT : '%(asctime)s INSPECT %(message)s',
               logging.DEBUG : '%(asctime)s DEBUG %(pathname)s [%(lineno)d] - %(message)s'}

# How many characters per line in console
LINEMAX = 80


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

    Adapted from https://stackoverflow.com/questions/14844970
    """
    DEFAULT = '%(levelname)s: %(message)s'

    def __init__(self, FORMATS=None):
        logging.Formatter.__init__(self)
        self.FORMATS = {} if FORMATS is None else FORMATS

    def format(self, record):
        self._style._fmt = self.FORMATS.get(record.levelno, self.DEFAULT)
        return logging.Formatter.format(self, record)

# Create logger
logger = logging.getLogger("ptypy")

# Default level - should be changed as soon as possible
logger.setLevel(logging.WARNING)

# Create console handler
consolehandler = logging.StreamHandler(stream = sys.stdout)
logger.addHandler(consolehandler)

# Add formatter
consoleformatter = CustomFormatter(CONSOLE_FORMAT)
consolehandler.setFormatter(consoleformatter)

# Add filter
consolefilter = MPIFilter()
logger.addFilter(consolefilter)

# Capture warnings and log them
logging.captureWarnings(True)

level_from_verbosity = {0:logging.CRITICAL, 1:logging.ERROR, 2:logging.WARN, 3:logging.INFO, 4: INSPECT, 5:logging.DEBUG}
level_from_string = {'CITATION':CITATION, 'CRITICAL':logging.CRITICAL, 'ERROR':logging.ERROR, 'WARN':logging.WARN, 'WARNING':logging.WARN, 
                     'INTERACTIVE':INTERACTIVE, 'INFO':logging.INFO, 'INSPECT': INSPECT, 'DEBUG':logging.DEBUG}
vlevel_from_logging = dict([(v,k) for k,v in level_from_verbosity.items()])
slevel_from_logging = dict([(v,k) for k,v in level_from_string.items()])

def ilog_message(msg):
    """
    Interactive logging for jupyter notebooks, prints a normal message.
    """
    if not slevel_from_logging[logger.level] == "INTERACTIVE":
        return
    logger.log(level_from_string["INTERACTIVE"], msg)

def ilog_streamer(msg):
    """
    Interactive logging for jupyter notebooks, 
    streams a message by overwriting the same line.
    """
    if not slevel_from_logging[logger.level] == "INTERACTIVE":
        return
    consolehandler.terminator = ""
    logger.log(level_from_string["INTERACTIVE"], "\r"+msg)
    consolehandler.terminator = "\n"

def ilog_newline():
    """
    Interactive logging for jupyter notebooks, 
    moves cursor to next line. Call this after 
    ilog_streamer() to escape the streaming.
    """
    if not slevel_from_logging[logger.level] == "INTERACTIVE":
        return
    consolehandler.terminator = ""
    logger.log(level_from_string["INTERACTIVE"], "\n")
    consolehandler.terminator = "\n"

def log(level,msg,parallel=False):
    if isinstance(level, int):
        _level = level_from_verbosity[level]
    elif isinstance(level, str):
        _level = level_from_string[level.upper()]
    else:
        raise TypeError("Verbosity level should be an integer or a string")
    if not parallel:
        logger.log(_level, msg)
    else:
        logger.log(_level, msg,extra={'allprocesses':True})

def set_level(level):
    """
    Set verbosity level. Kept here for backward compatibility
    """
    logger.info('Verbosity set to %s' % str(level))
    if isinstance(level, str):
        if level.upper() not in level_from_string:
            raise KeyError("Verbosity level %s does not exist" %level)
        logger.setLevel(level_from_string[level.upper()])
    elif isinstance(level, int):
        logger.setLevel(level_from_verbosity[level])
    else:
        raise TypeError("Verbosity level should be an integer or a string")
    logger.info('Verbosity set to %s' % str(level))

def get_level():
    """
    inverse to set level
    """
    try:
        return vlevel_from_logging[logger.level]
    except:
        return slevel_from_logging[logger.level]

# Formatting helper
def _(label, value):
    return '%-25s%s' % (label + ':', str(value))

def headerline(info='',align = 'c',fill='-'):
    li = len(info)
    if li>=60:
        return headerline(info[li//2:],align,fill)+'\n'+headerline(info[:li//2],align,fill)
    else:
        if li != 0:
            li+=2
            info = ' '+info+' '
        empty = LINEMAX-li
        if align=='c':
            left = empty // 2
            right = empty-left
        elif align=='l':
            left = 4
            right = empty-left
        else:
            right = 4
            left = empty-right
    return fill*left+info+fill*right

def report(thing,depth=4,noheader=False):
    """
    no protection for circular references
    """
    import time
    import numpy as np

    indent = report.indent
    level = report.level
    maxchar = report.maxchar
    hn = report.headernewline
    star = report.asterisk

    header = '\n---- Process #%d ---- ' % parallel.rank + time.asctime() + ' -----\n'

    def _(label,level,obj):
        pre = " "*indent*level + star + ' '
        extra = str(type(obj)).split("'")[1]
        pre+=str(label) if label is not None else "id"+np.base_repr(id(obj),base=32)
        if len(pre)>=25:
            pre = pre[:21] + '... '
        return "%-25s:" % pre, extra

    def _format_dict(label, level, obj):
        header,extra = _(label, level, obj)
        header+= ' %s(%d)' % (extra,len(obj)) + hn
        if level <= depth:
            #level +=1
            for k,v in obj.items():
                header += _format(k,level+1,v)
        return header

    def _format_iterable(label, level, lst):
        l = len(lst)
        header,extra = _(label, level, lst)
        header+= ' %s(%d)' % (extra,l)
        string = str(lst)
        if len(string) <= maxchar-25 or level>=depth:
            return header +'= '+ string + hn
        elif l >0:
            header += hn
            for v in lst[:5]:
                header += _format(None,level+1,v)
            header += _('...',level+1,' ')[0] + ' ....\n'
            return header
        else:
            pass

    def _format_other(label, level, obj):
        header,extra = _(label, level, obj)
        if np.isscalar(obj):
            header += ' '+str(obj)
        else:
            header += ' ' +extra+' = '+str(obj)
        return header[:maxchar]+'\n'

    def _format_numpy(key, level,a):
        header,extra = _(key, level, a)
        if len(a) < 5 and a.ndim == 1:
            stringout = header + ' [array = ' + str(a.ravel()) + ']\n'
        else:
            stringout = header + ' [' + (('%dx'*(a.ndim-1) + '%d') % a.shape) + ' ' + str(a.dtype) + ' array]\n'
        return stringout

    def _format_None(key,level, obj):
        return _(key, level, obj)[0] + ' None\n'

    def _format(key,level, obj):
        if hasattr(obj,'items'):
            stringout = _format_dict(key,level, obj)
        elif type(obj) is np.ndarray:
            stringout = _format_numpy(key,level, obj)
        elif str(obj)==obj:
            stringout = _format_other(key,level, obj)
        elif obj is None:
            stringout = _format_None(key,level, obj)
        elif np.iterable(obj):
            stringout = _format_iterable(key,level, obj)
        else:
            stringout = _format_other(key,level, obj)
        return stringout

    if noheader:
        return _format(None,0,thing)
    else:
        return header +_format(None,0,thing)

report.indent = 2
report.level = 0
report.maxchar = LINEMAX
report.headernewline='\n'
report.asterisk='*'


class LogTime:
    def __init__(self, active=False):
        self.active = active
        self.duration = 0
    def __enter__(self):
        if not self.active:
            return
        self.time = perf_counter()
        return self
    def __exit__(self, type, value, traceback):
        if self.active:
            self.duration = perf_counter() - self.time
            self.duration = parallel.MPImax([self.duration])
            self.readout = f'{self.duration:.3f} seconds'
