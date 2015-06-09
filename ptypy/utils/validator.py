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
from .verbose import logger, headerline
import logging

__all__= ['create_default_template','make_sub_default','validate']
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
           'complex': 'complex',
           'str': 'str',
           'bool': 'bool',
           'tuple': 'tuple',
           'list': 'list',
           'array': 'ndarray',
           'Param': 'Param',
           'None': 'NoneType',
           '': 'NoneType'}

evaltypes = ['int','float','tuple','list','complex']
copytypes = ['str','file']

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
    
    @property
    def is_evaluable(self):
        for t in self.type:
            if t in evaltypes:
                return True
                break
        return False
        
    @property
    def value(self):
        """
        Returns python type / object for default string if possible
        """
        if self.default is None:
            out = None
        # should be only strings now
        elif self.default.lower()=='none':
            out = None
        elif self.default.lower()=='true':
            out = True
        elif self.default.lower()=='false':
            out = False
        elif self.is_evaluable:
            out = eval(self.default)
        else:
            out = self.default
            
        return out
        
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

def make_sub_default(entry_point, depth=1):
    """
    Creates a default parameter structure, from the loaded parameter
    descriptions in this module
    
    Parameters
    ----------
    entry_point : str
        The node in the default parameter file
        
    depth : int
        The depth in the structure to which all sub nodes are expanded
        All nodes beyond depth will be returned as empty :any:`Param` 
        structure.
        
    Returns
    -------
    pars : Param
        A parameter branch.
    
    Examples
    --------
    >>> from ptypy.utils import validator
    >>> print validator.make_sub_default('.io')
    """
    pd = entry_points_Param[entry_point]
    out = Param()
    if depth<=0:
        return out
    for name,child in pd.children.iteritems():
        if hasattr(child,'children'):
            out[name] = make_sub_default(child.entry_point, depth=depth-1)
        else:
            out[name] = child.value
    return out
    
def validate(pars, entry_point, walk=True, raisecodes=[codes.FAIL, codes.INVALID]):
    """
    Check that the parameter structure `pars` matches the documented 
    constraints at the given entry_point.

    The function raises a RuntimeError if one of the code in the list 
    `raisecodes` has been found. If raisecode is empty, the function will 
    always return successfully but problems will be logged using logger.

    Parameters
    ----------
    pars : Param
        A parameter set to validate
        
    entry_point : str
        The node in the parameter structure to match to.
    
    walk : bool
        If ``True`` (*default*), navigate sub-parameters.
    
    raisecodes: list
        List of codes that will raise a RuntimeError.
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

def create_default_template(filename=None,user_level=0,doc_level=2):
    """
    Creates a (descriptive) template for ptypy.
    
    Parameters
    ----------
    filename : str
        python file (.py) to generate, will be overriden if it exists
    
    user_level : int
        Filter parameters to display on those with less/equal user level
    
    doc_level : int
        - if ``0``, no comments. 
        - if ``1``, *short_doc* as comment in script
        - if ``>2``, *long_doc* and *short_doc* as comment in script
    """
    def _format_longdoc(doc):
        ld = doc.strip().split('\n')
        out = []
        for line in ld:
            if len(line)==0:
                continue
            if len(line)>75:
                words = line.split(' ')
                nline = ''
                count = 0
                for word in words:
                    nline+=word+' '
                    count+=len(word)
                    if count > 70:
                        count = 0
                        out.append(nline[:-1])
                        nline=""
                out.append(nline[:-1])
            else:
                out.append(line)
        if out:
            return '# '+'\n# '.join(out)+'\n' 
        else:
            return ''
            
    if filename is None:
        f = open('ptypy_template.py','w')
    else:
        f = open(filename,'w')
    h = '"""\nThis Script was autogenerated using\n'
    h+= '``u.create_default_template("%s",%d,%d)``\n' %(str(filename),user_level,doc_level)
    h+= 'It is only a TEMPLATE and not a working reconstruction script.\n"""\n\n'
    h+= "import numpy as np\n"
    h+= "import ptypy\n"
    h+= "from ptypy.core import Ptycho\n"
    h+= "from ptypy import utils as u\n\n"
    h+= headerline('Ptypy Parameter Tree','l','#')+'\n'
    #h+= "p = u.Param()\n"
    f.write(h)
    for entry, pd in parameter_descriptions.iteritems():
        if user_level < pd.userlevel:
            continue
        if hasattr(pd,'children'):
            value = "u.Param()"
        else:
            val = pd.value
            if str(val)== val :
                value = '"%s"' % str(val)
            else:
                value = str(val)
        ID ="%02d" % pd.ID if hasattr(pd,'ID') else 'NA'
        if doc_level > 0:
            f.write('\n'+"## (%s) " % ID +pd.shortdoc.strip()+'\n')
        if doc_level > 1:
            f.write(_format_longdoc(pd.longdoc))
        f.write('p'+entry+ ' = ' + value+'\n')
        
    f.write('\n\nPtycho(p,level=5)\n')
    f.close()
