# -*- coding: utf-8 -*-
"""\
Parameter validation

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from .parameters import Param
import pkg_resources
import csv
from .verbose import logger
import logging
# Message codes
codes = Param(
    PASS=1,
    FAIL=0,
    UNKNOWN=2,
    MISSING=3,
    INVALID=4)

code_label = dict((v, k) for k, v in codes.items())

# Logging levels
logging_levels = Param(
    PASS=logging.INFO,
    FAIL=logging.CRITICAL,
    UNKNOWN=logging.WARN,
    MISSING=logging.WARN,
    INVALID=logging.ERROR)

typemap = {'int': 'int',
           'float': 'float',
           'str': 'str',
           'bool': 'bool',
           'tuple': 'tuple',
           'list': 'list',
           'array': 'ndarray',
           'Param': 'Param',
           'None': 'NoneType',
           '': 'NoneType'}


class PDesc(object):
    """

    """

    def __init__(self, description, parent=None):
        """
        Store description list for validation and documentation.
        """
        # Name is a string
        self.name = description.get('name', '')
        
        self.parent = parent
            
        if parent is not None:
            parent.children[self.name] = self
            
        # Type can be a comma-separated list of types
        self.type = description.get('type', None)

        if 'param' in self.type.lower() or 'dict' in self.type.lower():
            self.children = {}
            entry_points_Param[self.entry_point] = self
            
        if self.type is not None:
            self.type = [typemap[x.strip()] if x.strip() in typemap else x.strip() for x in self.type.split(',')]

        # Default value can be any type. None if unknown
        dflt = description.get('default', '')
        self.default = dflt if dflt else None

        # Static is 'TRUE' or 'FALSE'
        self.static = (description.get('static', 'TRUE') == 'TRUE')

        # lowlim/uplim are floats, None if unknown
        ll = description.get('lowlim', None)
        ul = description.get('uplim', None)
        if 'int' in self.type:
            self.lowlim = int(ll) if ll else None
            self.uplim = int(ul) if ul else None
        else:
            self.lowlim = float(ll) if ll else None
            self.uplim = float(ul) if ul else None

        # Doc is a string
        self.shortdoc = description.get('shortdoc', '')
        self.longdoc = description.get('longdoc', '')

        # User level (for gui stuff) is an int
        ul = description.get('userlevel', 1)
        self.userlevel = int(ul) if ul else None

        # Validity is a string (the name of another parameter)
        # FIXME: this is not used currently
        self.validity = description.get('validity', '')

        parameter_descriptions[self.entry_point] = self

    @property
    def entry_point(self):
        if self.parent is None:
            return ''
        else:
            return '.'.join([self.parent.entry_point, self.name])

    def check(self, pars, walk):
        """
        Check that input parameter pars is consistent with parameter description.
        If walk is True and pars is a Param object, checks are also conducted for all
        sub-parameters.
        """
        ep = self.entry_point
        out = {}
        val = {}

        # 1. Data type
        if self.type is None:
            # Unconclusive
            val['type'] = codes.UNKNOWN
        else:
            val['type'] = codes.PASS if (type(pars).__name__ in self.type) else codes.FAIL

        # 2. limits
        if self.lowlim is None:
            val['lowlim'] = codes.UNKNOWN
        else:
            val['lowlim'] = codes.PASS if (pars >= self.lowlim) else codes.FAIL
        if self.uplim is None:
            val['uplim'] = codes.UNKNOWN
        else:
            val['uplim'] = codes.PASS if (pars <= self.uplim) else codes.FAIL

        # 3. Extra work for parameter entries
        if 'Param' in self.type:
            # Check for missing entries
            for k, v in self.children.items():
                if k not in pars:
                    val[k] = codes.MISSING

            # Check for excess entries
            for k, v in pars.items():
                if k not in self.children:
                    val[k] = codes.INVALID
                elif walk:
                    # Validate child
                    out.update(self.children[k].check(v, walk))

        out[ep] = val
        return out

    def __str__(self):
        return ''

    def make_doc(self):
        """
        Create documentation.
        """
        return '{self.entry_point}\n\n{self.shortdoc}\n{self.longdoc}'.format(self=self)

# Load all documentation on import
csvfile = pkg_resources.resource_filename('ptypy', 'resources/parameters_descriptions.csv')
desc_list = list(csv.DictReader(file(csvfile, 'r')))

# Populate the dictionary of all entry points.
from collections import OrderedDict
parameter_descriptions = OrderedDict()
entry_points_Param = {}

# Create the root
pdroot = PDesc(description={'name': '', 'type': 'Param'}, parent=None)
entry_pts = ['']
entry_dcts = [pdroot]  # [{}]
entry_level = 0
for num, desc in enumerate(desc_list):
    # Get parameter name and level in the hierarchy
    level = int(desc.pop('level'))
    name = desc['name']

    # Manage end of branches
    if level < entry_level:
        # End of a branch
        entry_pts = entry_pts[:(level + 1)]
        entry_dcts = entry_dcts[:(level + 1)]
        entry_level = level
    elif level > entry_level:
        raise RuntimeError('Problem parsing csv file %s, entry %d, name %s' % (csvfile, num,name))

    # Create Parameter description object
    pd = PDesc(desc, parent=entry_dcts[level])

    # save a number
    pd.ID = num
    
    # Manage new branches
    if 'param' in desc['type'].lower() or 'dict' in desc['type'].lower():
        # A new node
        entry_pt = pd.entry_point
        entry_dcts.append(pd)  # entry_dcts.append(new_desc)
        entry_level = level + 1


def validate(pars, entry_point, walk=True, raisecode=[codes.FAIL, codes.INVALID]):
    """
    Check that the parameter structure "pars" matches the documented constraints at
    the given entry_point.

    The function raises a RuntimeError if one of the code in the list 'raisecode' has
    been found. If raisecode is empty, the function will always return successfully
    but problems will be logged using logger.

    :param pars: A Param object
    :param entry_point: the Node in the structure to match to.
    :param walk: if True (default), navigate sub-parameters.
    :param raisecode: list of codes that will raise a RuntimeError.
    """
    pdesc = parameter_descriptions[entry_point]
    d = pdesc.check(pars, walk=walk)
    do_raise = False
    for ep, v in d.items():
        for tocheck, outcome in v.items():
            logger.log(logging_levels[code_label[outcome]], '%-50s %-20s %7s' % (ep, tocheck, code_label[outcome]))
            do_raise |= (outcome in raisecode)
    if do_raise:
        raise RuntimeError('Parameter validation failed.')

