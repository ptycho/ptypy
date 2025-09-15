"""
Reporting functions

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
from .. import abs2
import numpy as np

def calculate_metrics(ptycho):
    print('metrics')
    ptycho.print_stats()
    return {}


def calculate_maps(ptycho):
    print('maps')
    ptycho.print_stats()
    return {}


def map(ptycho, ID='fluence'):
    """
    Compute fluence or transmission map(s) for all object storages and modes

    Parameters
    ----------
    ptycho : The Ptycho object instance to draw information from.
    ID : one of 'fluence' (default), 'transmission' or 'coverage'
    """
    if ID not in ['fluence', 'transmission', 'coverage']:
        raise NotImplementedError(f'Unknown map {ID}')

    # Delete existing copy if it exists
    if f'C{ID}' in [c.ID for c in ptycho.obj.copies]:
        del ptycho.obj.owner._pool['C'][f'C{ID}']

    # Create copy of object container
    fmap = ptycho.obj.copy(ID=ID, fill=0., dtype='real')

    # Loop through storages and add the probe intensities
    for sname, sobj in fmap.S.items():
        for v in sobj.views:
            if not v.active:
                continue
            for pid, pod in v.pods.items():
                if ID == 'transmission':
                    sobj[v] += abs2(pod.probe*pod.obj)
                elif ID == 'fluence':
                    sobj[v] += abs2(pod.probe)
                elif ID == 'coverage':
                    sobj[v] += np.ones_like(pod.probe.real)
    return fmap