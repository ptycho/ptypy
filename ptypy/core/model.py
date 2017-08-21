# -*- coding: utf-8 -*-
"""
Data sharing models.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from .. import utils as u
from ..utils.verbose import logger
from classes import STORAGE_PREFIX

__all__ = ['parse_model']

DEFAULT = u.Param(
        model_type='basic',
        scan_per_probe=1,
        scan_per_object=1,
        npts=None
)

MAX_SCAN_COUNT = 100


def parse_model(pars, sharing_dct):
    """
    This factory function takes a model description in the input parameters
    and returns an object that can be called with a scan_label (or index) and
    a diffraction pattern index, and returns probe and object ids.
    """
    p = u.Param(DEFAULT)
    p.update(pars)
    if p.model_type.lower() == 'basic':
        return BasicSharingModel(sharing_dct,
                                 p.scan_per_probe,
                                 p.scan_per_object,
                                 p.npts)
    else:
        raise RuntimeError('model type %s not supported.' % p.model_type)


class BasicSharingModel(object):
    """
    BasicSharingModel: implements the most common scan-sharing patterns.
    """

    def __init__(self, sharing_dct, scan_per_probe, scan_per_object, npts=None):
        """
        BasicSharingModel: implements the most common scan-sharing patterns.

        Parameters:
        -----------
        scan_per_probe: float < 1 or int
            number of contiguous scans using the same probe. If a int, the
            number of scans. If a float < 0, split the scans into
            1/scan_per_probe independent probes. For instance,
            scan_per_probe = .5 will split all scans in two and assign a
            different probe to each.
        scan_per_object: int
            number of contiguous scans using the same object.
        npts: int
            number of diffraction patterns in a given scan. Needed only if
            scan_per_probe < 1.
        """
        # Prepare probe sharing
        if scan_per_probe == 0:
            self.shared_probe = True
            self.single_probe = True
            logger.info('Sharing a single probe for ALL scans.')
        elif scan_per_probe >= 1:
            self.shared_probe = True
            self.single_probe = False
            self.scan_per_probe = int(scan_per_probe)
            logger.info(
                'Model: sharing probe between scans '
                '(one new probe every %d scan)' % self.scan_per_probe)
        else:
            self.shared_probe = False
            self.single_probe = False
            # The following will fail if npts wasn't provided.
            self.diff_per_probe = int(npts * scan_per_probe)
            self.npts = npts
            logger.info(
                'Model: splitting scans (every %d diffraction patter)'
                % self.diff_per_probe)

        # Prepare object sharing
        if scan_per_object == 0:
            self.single_object = True
            self.shared_object = True
            logger.info('Sharing a single object for ALL scans.')
        elif scan_per_object >= 1:
            self.single_object = False
            self.shared_object = True
            self.scan_per_object = int(scan_per_object)
            logger.info(
                'Model: sharing object between scans '
                '(one new object every %d scan)' % self.scan_per_object)
        else:
            raise RuntimeError(
                'scan_per_object < 1. not supported. What does it mean anyway?')

        self.scan_labels = []
        self.probe_ids = sharing_dct['probe_ids']
        self.object_ids = sharing_dct['object_ids']

    def __call__(self, scan_label, diff_index):
        """
        Return probe and object ids given a scan and diffraction pattern index.

        Parameters:
        -----------
        scan_label: str or int
                    An identifier for a scan (label or index)
        diff_index: int
                    The index of the diffraction pattern in the given scan.
        """
        # Get an index for the scan_label
        if str(scan_label) == scan_label:
            # If it is a string, look it up
            if scan_label in self.scan_labels:
                scan_index = self.scan_labels.index(scan_label)
            else:
                scan_index = len(self.scan_labels)
                self.scan_labels.append(scan_label)
        else:
            # Nothing to do if it is an index
            scan_index = scan_label

        # Apply the rules for probe sharing
        if self.single_probe:
            probe_id = 0
        elif self.shared_probe:
            probe_id = scan_index // self.scan_per_probe
        else:
            probe_id = (scan_index * (self.npts // self.diff_per_probe)
                        + diff_index // self.diff_per_probe)

        # Follow the format specified in Viewmanager
        probe_id = STORAGE_PREFIX + '%02d' % probe_id

        # ... and the rules for object sharing
        if self.single_object:
            object_id = 0
        else:
            object_id = scan_index // self.scan_per_object

        # Follow the format specified in Viewmanager
        object_id = STORAGE_PREFIX + '%02d' % object_id

        logger.debug(
            "Model assigned frame %d of scan %s to probe %s & object %s"
            % (diff_index, str(scan_label), probe_id, object_id))

        # Store sharing info
        pl = self.probe_ids.get(probe_id, [])
        if scan_label not in pl:
            pl.append(scan_label)
        self.probe_ids[probe_id] = pl

        ol = self.object_ids.get(object_id, [])
        if scan_label not in ol:
            ol.append(scan_label)
        self.object_ids[object_id] = ol

        return probe_id, object_id
