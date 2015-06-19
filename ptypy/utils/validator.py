# -*- coding: utf-8 -*-
"""\
Parameter validation. This module parses the file
``resources/parameters_descriptions.csv`` to extract the parameter
defaults for |ptypy|. It saves all parameters in the form of 
a :py:class:`PDesc` object, which are flat listed in 
`parameter_descriptions` or in `entry_points_dct`, which only contains
parameters with subparameters (children).

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import pkg_resources
import csv
# Load all documentation on import
_csvfile = pkg_resources.resource_filename('ptypy', 'resources/parameters_descriptions.csv')
_desc_list = list(csv.DictReader(file(_csvfile, 'r')))
del csv
del pkg_resources

if __name__=='__main__':
    from ptypy.utils.parameters import Param
else:
    from .parameters import Param

__all__= ['create_default_template','make_sub_default','validate',\
          'entry_points_dct', 'parameter_descriptions', 'PDesc']

#! Validator message codes
CODES = Param(
    PASS=1,
    FAIL=0,
    UNKNOWN=2,
    MISSING=3,
    INVALID=4)

#! Inverse message codes
CODE_LABEL = dict((v, k) for k, v in CODES.items())

# Populate the dictionary of all entry points.
from collections import OrderedDict

# All ptypy parameters in an ordered dictionary as (entry_point, PDesc) pairs.
parameter_descriptions = OrderedDict()
del OrderedDict

# All parameter containers in a dictionary as (entry_point, PDesc) pairs.
# Subset of :py:data:`parameter_descriptions`
entry_points_dct = {}

# Logging levels
import logging

_logging_levels = Param(
    PASS=logging.INFO,
    FAIL=logging.CRITICAL,
    UNKNOWN=logging.WARN,
    MISSING=logging.WARN,
    INVALID=logging.ERROR)

del logging

_typemap = {'int': 'int',
           'float': 'float',
           'complex': 'complex',
           'str': 'str',
           'bool': 'bool',
           'tuple': 'tuple',
           'list': 'list',
           'array': 'ndarray',
           'Param': 'Param',
           'None': 'NoneType',
           'file': 'str',
           '': 'NoneType'}

_evaltypes = ['int','float','tuple','list','complex']
_copytypes = ['str','file']



class PDesc(object):
    """
    Small class to store all attributes of a ptypy parameter
    
    """

    def __init__(self, description, parent=None):
        """
        Stores description list for validation and documentation.
        """
        #: Name of parameter
        self.name = description.get('name', '')
        
        #: Parent parameter (:py:class:`PDesc` type) if it has one.
        self.parent = parent
            
        if parent is not None:
            parent.children[self.name] = self
            
        #: Type can be a comma-separated list of types
        self.type = description.get('type', None)
        
        if 'param' in self.type.lower() or 'dict' in self.type.lower():
            self.children = {}
            entry_points_dct[self.entry_point] = self
        else:
            self.children = None
            
        if self.type is not None:
            self.type = [_typemap[x.strip()] if x.strip() in _typemap else x.strip() for x in self.type.split(',')]
        
        
        self.default = None
        """ Default value can be any type. None if unknown. """
        
        self.set_default(description.get('default', ''))
        
        # Static is 'TRUE' or 'FALSE'
        self.static = (description.get('static', 'TRUE') == 'TRUE')
        
        #: Lower limit of parameter, None if unknown
        self.lowlim = None
        #: Upper limit of parameter, None if unknown
        self.uplim = None
        
        ll = description.get('lowlim', None)
        ul = description.get('uplim', None)
        if 'int' in self.type:
            self.lowlim = int(ll) if ll else None
            self.uplim = int(ul) if ul else None
        else:
            self.lowlim = float(ll) if ll else None
            self.uplim = float(ul) if ul else None
            
        # choices is an evaluable list
        c =  description.get('choices', '')
        #print c, self.name
        if str(c)=='':
            c=None
        else:
            try:
                c = eval(c.strip())
            except SyntaxError('Evaluating `choices` %s for parameter %s failed' %(str(c),self.name)):
                c = None
        
        #: If parameter is a list of choices, these are listed here.
        self.choices = c

        
        # Docs are strings
        
        #: Short descriptive string of parameter
        self.shortdoc = description.get('shortdoc', '')
        
        #: Longer documentation, may contain *sphinx* inline markup.
        self.longdoc = description.get('longdoc', '')

        # User level (for gui stuff) is an int
        ul = description.get('userlevel', 1)
        
        self.userlevel = int(ul) if ul else None
        """User level, a higher level means a parameter that is less 
        likely to vary or harder to understand.
        """
        
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
            if t in _evaltypes:
                return True
                break
        return False
        
    def set_default(self,default='', check=False):
        """
        Sets default (str) and derives value (python type)
        """
        default = str(default)
        
        # this destroys empty strings
        self.default = default if default else None
        
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
        
        self.value = out
        return out
    """
    @property
    def value(self):

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
    """
    
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
            val['type'] = CODES.UNKNOWN
        else:
            val['type'] = CODES.PASS if (type(pars).__name__ in self.type) else CODES.FAIL

        # 2. limits
        if self.lowlim is None:
            val['lowlim'] = CODES.UNKNOWN
        else:
            val['lowlim'] = CODES.PASS if (pars >= self.lowlim) else CODES.FAIL
        if self.uplim is None:
            val['uplim'] = CODES.UNKNOWN
        else:
            val['uplim'] = CODES.PASS if (pars <= self.uplim) else CODES.FAIL

        # 3. Extra work for parameter entries
        if 'Param' in self.type:
            # Check for missing entries
            for k, v in self.children.items():
                if k not in pars:
                    val[k] = CODES.MISSING

            # Check for excess entries
            for k, v in pars.items():
                if k not in self.children:
                    val[k] = CODES.INVALID
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



## maybe this should be a function in the end.
# Create the root
pdroot = PDesc(description={'name': '', 'type': 'Param'}, parent=None)
entry_pts = ['']
entry_dcts = [pdroot]  # [{}]
entry_level = 0

for num, desc in enumerate(_desc_list):
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
        raise RuntimeError('Problem parsing csv file %s, entry %d, name %s' % (_csvfile, num,name))

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

#cleanup
del level, name, entry_level, pd, entry_pt, entry_dcts

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
    pd = entry_points_dct[entry_point]
    out = Param()
    if depth<=0:
        return out
    for name,child in pd.children.iteritems():
        if child.children is not None:
            out[name] = make_sub_default(child.entry_point, depth=depth-1)
        else:
            out[name] = child.value
    return out
    
def validate(pars, entry_point, walk=True, raisecodes=[CODES.FAIL, CODES.INVALID]):
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
    from ptypy.utils.verbose import logger
    
    pdesc = parameter_descriptions[entry_point]
    d = pdesc.check(pars, walk=walk)
    do_raise = False
    for ep, v in d.items():
        for tocheck, outcome in v.items():
            logger.log(_logging_levels[CODE_LABEL[outcome]], '%-50s %-20s %7s' % (ep, tocheck, CODE_LABEL[outcome]))
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
    try:
        from ptypy.utils.verbose import headerline
        h+= headerline('Ptypy Parameter Tree','l','#')+'\n'
    except ImportError:
        h+= '### Ptypy Parameter Tree ###\n\n'
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

def _add2argparser(parser=None, entry_point='',root=None,\
                        excludes=('scans','engines'), mode = 'add',group=None):
    
    sep = '.'
    
    pd = parameter_descriptions[entry_point]
    
    has_children = hasattr(pd,'children') and pd.children is not None
    
    if root is None:
        root = pd.entry_point if has_children else pd.parent.entry_point 
        
    assert root in entry_points_dct.keys()
    
    if excludes is not None:
        # remove leading '.'
        lexcludes = [e.strip()[1:] for e in excludes if e.strip().startswith(sep) ]
        loc_exclude = [e.split(sep)[0] for e in lexcludes if len(e.split())==1 ]
    else:
        excludes =['baguette'] #:)
    
    if pd.name in excludes: return
    
    if parser is None:
        from argparse import ArgumentParser
        description = """
        Parser for PDesc %s
        Doc: %s
        """ % (pd.name, pd.shortdoc)
        parser = ArgumentParser(description=description)
    
    # overload the parser
    if not hasattr(parser,'_ptypy_translator'): 
        parser._ptypy_translator={}
    
    
    # convert to parser variable
    name = pd.entry_point.replace(root,'',1).partition(sep)[2].replace('.','-')
    
    if has_children:
        entry = pd.entry_point
        if name !='':
            group=parser.add_argument_group(title=name, description=None)
        else:
            group=None
        # recursive behavior here
        new_excludes = [e[len(entry):] for e in excludes if e.startswith(entry)]
        for key,child in pd.children.iteritems():
            _add2argparser(parser, entry_point=child.entry_point,root=root,\
             excludes=excludes,mode = 'add',group=group)
    else:
        parse = parser if group is None else group
        try:
            typ= eval(pd.type[0])
        except BaseException:
            return
        if type(typ) is not type:
            u.verbose.logger.debug('Cannot parse type %s of parameter %s' % (str(typ),name))
            return
        elif typ is bool:
            flag = '--no-'+name if pd.value else '--'+name
            action='store_false' if pd.value else 'store_true'
            parse.add_argument(flag, dest=name, action=action, 
                             help=pd.shortdoc )
        else:
            parse.add_argument('--'+name, dest=name, type=typ, default = pd.value, choices=pd.choices, 
                             help=pd.shortdoc +' (default=%s)' % pd.default.replace('%(','(') )            
        
        parser._ptypy_translator[name] = pd
        
    return parser
    
if __name__ =='__main__':
    from ptypy import utils as u
    
    
    
    parser = _add2argparser(entry_point='.scan.illumination')
    parser.parse_args()
    
