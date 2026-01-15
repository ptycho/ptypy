# -*- coding: utf-8 -*-
"""
Tool to embed an ipython kernel for debugging.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os, threading

__all__ = ['IPythonKernelThread', 'start_ipython_kernel']

try:
    from IPython import embed_kernel
    import mock

    class IPythonKernelThread(threading.Thread):
        def __init__(self, ns):
            threading.Thread.__init__(self)
            self.ns = ns
            self.daemon = True

        def run(self):
            with mock.patch('signal.signal'):
                embed_kernel(local_ns=self.ns)

    def start_ipython_kernel(ns):
        print("ipython console --existing kernel-%d.json" % os.getpid())
        IPythonKernelThread(ns).start()

except:
    class IPythonKernelThread(object):
        def __init__(self, ns):
            return

        def start(self):
            return

    def start_ipython_kernel(ns):
        return


