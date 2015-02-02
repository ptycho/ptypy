# -*- coding: utf-8 -*-
"""\
Parameter validation

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import pkg_resources
import csv


class PDesc(object):
    """

    """
    def __init__(self, description, parent=None):
        """
        Store description list for validation and documentation.
        """

        # Type can be a comma-separated list of types
        self.type = description.get('type', None)
        if self.type is not None:
            self.type = [x.strip() for x in self.type.split(',')]

        # Name is a string
        self.name = description.get('name', '')

        # Default value is a string
        self.default = description.get('default', '')

        # Static is 'TRUE' or 'FALSE'
        self.static = (description.get('static', 'TRUE') == 'TRUE')

        # lowlim/uplim are floats, None if unknown
        ll = description.get('lowlim', None)
        ul = description.get('uplim', None)
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

        self.parent = parent

        if self.type == ['Param']:
            self.children = {}

        if parent is not None:
            parent.children[self.name] = self

        parameter_descriptions[self.entry_point] = self

    def validate(self, pars, walk=True):
        """
        Verify that the parameter structure pars is consistent with internal description.
        If walk is True (default), validate also all sub-parameters.

        Returns a list of failures and a list of warnings.
        """
        d = self._do_check(pars, walk)
        failed = []
        warning = []
        for ep, v in d.items():
            for tocheck, outcome in v.items():
                if outcome is None:
                    warning.append(ep + ':' + tocheck)
                if outcome is False:
                    failed.append(ep + ':' + tocheck)
        return failed, warning

    def report(self, pars, walk=True, level=1):
        """
        Create a report on the validity of the passed parameters in addition to the
        validity structure.
        If walk is True (default), validate also all chidren entry points.

        Return validity_dict, report
        """
        d = self._do_check(pars, walk)
        report = ''
        for ep, v in d.items():
            report += '%s\n' % ep
            report += '-'*len(ep) + '\n'
            for tocheck, outcome in v.items():
                if (outcome is True) and (level == 2):
                    report += ' * %10s%7s\n' % (tocheck, 'PASS')
                elif (outcome is None) and (level >= 1):
                    report += ' * %10s%7s\n' % (tocheck, '????')
                elif (outcome is False) and (level >= 0):
                    report += ' * %10s%7s\n' % (tocheck, 'FAIL')
            report += '\n'
        return report

    @property
    def entry_point(self):
        if self.parent is None:
            return ''
        else:
            return '.'.join([self.parent.entry_point, self.name])

    def _do_check(self, pars, walk):
        """
        """
        ep = self.entry_point
        out = {}
        val = {}

        # 1. Data type
        if self.type is None:
            # Unconclusive
            val['type'] = None
        else:
            val['type'] = (type(pars).__name__ in self.type)

        # 2. limits
        if self.lowlim is None:
            val['lowlim'] = None
        else:
            val['lowlim'] = (pars >= self.lowlim)
        if self.uplim is None:
            val['uplim'] = None
        else:
            val['uplim'] = (pars <= self.uplim)

        # 3. Extra work for parameter entries
        if self.type == ['Param']:
            # Check for missing entries
            for k, v in self.children.items():
                if k not in pars:
                    val[k] = None

            # Check for excess entries
            for k, v in pars.items():
                if k not in self.children:
                    val[k] = False
                elif walk:
                    # Validate child
                    d, newrep = self.children[k]._do_check(v, walk)
                    # Append dict
                    out[self.children[k].entry_point] = d

        out[ep] = val
        return out

    def __str__(self):
        return ''

    def make_doc(self, format=None):
        """
        Create documentation.
        """
        pass

# Load all documentation on import
csvfile = pkg_resources.resource_filename('ptypy', 'resources/parameters_descriptions.csv')
desc_list = list(csv.DictReader(file(csvfile, 'r')))

# Populate the dictionary of all entry points.
parameter_descriptions = {}

# Create the root
pdroot = PDesc(description={'name': '', 'type': 'Param'}, parent=None)
entry_pts = ['']
entry_dcts = [pdroot] # [{}]
entry_level = 0
for num, desc in enumerate(desc_list):
    # Get parameter name and level in the hierarchy
    level = int(desc.pop('level'))
    name = desc['name']

    # Manage end of branches
    if level < entry_level:
        # End of a branch
        entry_pts = entry_pts[:(level+1)]
        entry_dcts = entry_dcts[:(level+1)]
        entry_level = level
    elif level > entry_level:
        raise RuntimeError('Problem parsing csv file %s, entry %d' % (csvfile, num))

    # Create Parameter description object
    pd = PDesc(desc, parent=entry_dcts[level])

    # Manage new branches
    if desc['type'] == 'Param':
        # A new node
        entry_pt = pd.entry_point
        entry_dcts.append(pd)  # entry_dcts.append(new_desc)
        entry_level = level+1


def validate(pars, entry_point=None, level=0):
    """
    Verify that the parameter structure "pars" matches the documented constraints.

    :param pars: A Param object
    :param entry_point: the Node in the structure to match to.
    """
    pdesc = parameter_descriptions[entry_point]
    failed, warning = pdesc.validate(pars)
    if (level == 0) and failed:
        return failed
    elif (level == 1) and (failed or warning):
        return failed, warning

