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

import ast
import weakref
from collections import OrderedDict

if __name__ == '__main__':
    from ptypy.utils.parameters import Param
else:
    from .parameters import Param

__all__ = ['create_default_template', 'make_sub_default', 'validate',
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

"""
# Populate the dictionary of all entry points.

# All ptypy parameters in an ordered dictionary as (entry_point, PDesc) pairs.
parameter_descriptions = OrderedDict()
del OrderedDict

# All parameter containers in a dictionary as (entry_point, PDesc) pairs.
# Subset of :py:data:`parameter_descriptions`
entry_points_dct = {}
"""

# Logging levels
import logging

_logging_levels = Param(
    PASS=logging.INFO,
    FAIL=logging.CRITICAL,
    UNKNOWN=logging.WARN,
    MISSING=logging.WARN,
    INVALID=logging.ERROR)

del logging


class Parameter(object):
    """
    Base class for parameter descriptions and validation. This class is used to hold both command line arguments
    and Param-type parameter descriptions.
    """

    def __init__(self, name, parent=None, separator='.', options_def=None):
        """

        :param name: The name of the parameter represented by this instance
        :param parent: Parent parameter or None
        :param separator: defaults to '.'
        :param options_def: a dictionary whose keys are attribute names and values are description
                     of the attribute. It this description contains the text "required" or
                     "mandatory", the attribute is registered as required.
        """
                
        #: Name of parameter
        self.name = name
        
        #: Parent parameter (:py:class:`Parameter` type) if it has one.
        self.parent = parent

        #: Hierarchical tree of sub-Parameters.
        self.children = OrderedDict()
        
        self.separator = separator
        
        # Required and optional attributes
        self.required = []
        self.optional = []
        self.options_def = OrderedDict()
        self._parse_options_def(options_def)

        self.num_id = 0

        #: Attributes to the parameters.
        self.options = dict.fromkeys(self.required, '')

        self._all_options = {}

        self.implicit = False
        
    @property
    def option_keys(self):
        return self._all_options.keys()
        
    @property
    def is_child(self):
        """
        Type check
        """
        return type(self.parent) is self.__class__

    def _parse_options_def(self, options_def=None):
        """
        Parse and store options definitions.
        :param options_def: a dictionary whose keys are the options names and values are a description of the options.
                            If these descriptions contain "required" or "mandatory", the options is registered as
                            such.
        """
        if options_def is not None:
            self.options_def.update(options_def)
            
            r = []
            o = []
        
            for option, text in self.options_def.items():
                if 'required' in text or 'mandatory' in text:
                    r += [option]
                else:
                    o += [option]
            self.required = r
            self.optional = o

    def new_child(self, name, options=None):
        """
        Create a new descendant and pass new options.

        If name contains separators, intermediate children are created.

        If name already exists, update options and return existing child.

        If name already exists and had been created implicitly to create a child further down,
        the order in self.children is corrected to honor the order of explicitly created children.
        """
        if options is None:
            options = self.options

        if self.separator in name:
            # Creating a sub-level
            name, next_name = name.split(self.separator, 1)
            subparent = self.children.get(name, None)
            if subparent is None:
                # Create subparent
                subparent = self.__class__(name=name, parent=self, separator=self.separator, options_def=self.options_def)

                # Remember that creation was implicit
                subparent.implicit = True

                # Insert in children dict
                self.children[name] = subparent
            child = subparent.new_child(next_name, options)
            self._all_options.update(subparent.options)
        else:
            if name in self.children.keys():
                # The child already exists
                child = self.children[name]
                if child.implicit:
                    # Tricky bit: this child had already been created implicitly, but now is being created
                    # explicitly. We use this to enforce a proper order in self.children
                    self.children.pop(name)
                    child.implicit = False

                    explicit = [(k, v) for k, v in self.children.items() if not v.implicit]
                    implicit = [(k, v) for k, v in self.children.items() if v.implicit]
                    self.children = OrderedDict(explicit + [(name, child)] + implicit)
            else:
                child = self.__class__(name=name, parent=self, separator=self.separator, options_def=self.options_def)
                self.children[name] = child
            child._store_options(options)
            self._all_options.update(child.options)

        return child

    def _store_options(self, dct):
        """
        Read and store options and check that the minimum selections
        of options is present.
        """

        if self.required is not None and type(self.required) is list:
            missing = [r for r in self.required if r not in dct.keys()]
            if missing:
                raise ValueError('Missing required option(s) <%s> for parameter %s.' % (', '.join(missing), self.name))

        self.options = dict.fromkeys(self.required)
        self.options.update(dct)

    def _find(self, name):
        """
        Walk the tree and return the first encountered element called "name", None if none is found.
        :param name:
        :return:
        """
        root = None
        for k, d in self.descendants:
            if k.endswith(name):
                root = d.parent
                break
        return root

    def __getitem__(self, name):
        """
        Get a descendant
        """
        if self.separator in name:
            root, name = name.split(self.separator, 1)
            if not root:
                parent = self._find(name.split(self.separator)[0])
            else:
                parent = self.children[root]
            return parent[name]
        else:
            return self.children[name]

    def __setitem__(self, name, desc):
        """
        Insert a descendant
        """
        if self.separator not in name:
            if name != desc.name:
                raise RuntimeError("Descendant '%s' being inserted in '%s' as '%s'." % (desc.name, self.path, name))
            self.children[name] = desc
            self._all_options.update(desc.options)
        else:
            root, name = name.split(self.separator, 1)
            if not root:
                subparent = self._find(name.split(self.separator)[0])
                if subparent is None:
                    raise RuntimeError('No attachment point for .%s found.' % name)
            else:
                subparent = self.children.get(root, None)
                if not subparent:
                    subparent = self.new_child(root)
            subparent[name] = desc

    def add_child(self, desc):
        self[desc.name] = desc

    def _get_root(self):
        """
        Return root of parameter tree.
        """
        if self.parent is None:
            return self
        else:
            return self.parent.root

    @property    
    def root(self):
        return self._get_root()
            
    @property
    def path(self):
        if self.parent is None:
            return self.name
        else:
            return self.parent.path + self.separator + self.name

    @property
    def descendants(self):
        for k, v in self.children.items():
            yield (k, v)
            for d, v1 in v.descendants:
                yield (k + self.separator + d, v1)

    def load_csv(self, fbuffer, **kwargs):
        """
        Load from csv as a fielded array. Keyword arguments are passed
        on to csv.DictReader
        """
        from csv import DictReader
        CD = DictReader(fbuffer, **kwargs)
        
        if 'level' in CD.fieldnames:
            chain = []
            
            # old style CSV, name + level sets the path
            for num, dct in enumerate(list(CD)):
            
                # Get parameter name and level in the hierarchy
                level = int(dct.pop('level'))
                name = dct.pop('name')
            
                # translations
                dct['help'] = dct.pop('shortdoc')
                dct['doc'] = dct.pop('longdoc')
                if dct.pop('static').lower() != 'yes':
                    continue

                if level == 0:  
                    chain = [name]
                else:
                    chain = chain[:level]+[name]

                name = self.separator.join(chain)
                desc = self.new_child(name, options=dct)
        else:
            # new style csv, name and path are synonymous
            for dct in list(CD):
                name = dct['path']
                desc = self.new_child(name, options=dct)

    def save_csv(self, fbuffer, **kwargs):
        """
        Save to fbuffer. Keyword arguments are passed
        on to csv.DictWriter
        """
        from csv import DictWriter
        
        fieldnames = self.required + self.optional
        fieldnames += [k for k in self._all_options.keys() if k not in fieldnames]
        
        DW = DictWriter(fbuffer, ['path'] + fieldnames)
        DW.writeheader()
        for key, desc in self.descendants:
            dct = {'path': key}
            dct.update(desc.options)
            DW.writerow(dct)
        
    def load_json(self, fbuffer):
        
        raise NotImplementedError
    
    def save_json(self, fbuffer):
        
        raise NotImplementedError

    def load_conf_parser(self, fbuffer, **kwargs):
        """
        Load Parameter defaults using Python's ConfigParser
        
        Each parameter occupies its own section.
        Separator characters in sections names map to a tree-hierarchy.
        
        Keyword arguments are forwarded to `ConfigParser.RawConfigParser`
        """
        from ConfigParser import RawConfigParser as Parser
        parser = Parser(**kwargs)
        parser.readfp(fbuffer)
        for num, sec in enumerate(parser.sections()):
            desc = self.new_child(name=sec, options=dict(parser.items(sec)))

        return parser
            
    def save_conf_parser(self, fbuffer, print_optional=True):
        """
        Save Parameter defaults using Pythons ConfigParser
        
        Each parameter occupies its own section.
        Separator characters in sections names map to a tree-hierarchy.
        """
        from ConfigParser import RawConfigParser as Parser
        parser = Parser()
        for name, desc in self.descendants:
            parser.add_section(name)
            for k, v in desc.options.items():
                if (v or print_optional) or (k in self.required):
                    parser.set(name, k, v)
        
        parser.write(fbuffer)
        return parser

    def __str__(self):
        """
        Pretty-print the Parameter options in ConfigParser format.
        """
        from ConfigParser import RawConfigParser as Parser
        import StringIO
        parser = Parser()
        parser.add_section(self.name)
        for k, v in self.options.items():
            parser.set(self.name, k, v)
        s = StringIO.StringIO()
        parser.write(s)
        return s.getvalue().strip()


class ArgParseParameter(Parameter):
    DEFAULTS = OrderedDict([
        ('default', 'Default value for parameter.'),
        ('help', 'A small docstring for command line parsing (required).'),
        ('choices', 'If parameter is list of choices, these are listed here.')
    ])

    def __init__(self, *args, **kwargs):
        
        options_def = self.DEFAULTS.copy()
        extra_def = kwargs.get('options_def')
        if extra_def is not None:
            options_def.update(extra_def)
            
        kwargs['options_def'] = options_def
        
        super(ArgParseParameter, self).__init__(*args, **kwargs)

    @property
    def help(self):
        """
        Short descriptive explanation of parameter
        """
        return self.options.get('help', '')

    @property
    def default(self):
        """
        Returns default as a Python type
        """
        default = str(self.options.get('default', ''))
        
        if not default:
            return None
        else:
            return self.eval(default)

    def eval(self, val):
        """
        A more verbose wrapper around `ast.literal_eval`
        """
        try:
            return ast.literal_eval(val)
        except ValueError as e:
            msg = e.message+". could not read %s for parameter %s" % (val, self.name)
            raise ValueError(msg)
            
    @property
    def choices(self):
        """
        If parameter is a list of choices, these are listed here.
        """
        # choices is an evaluable list
        c = self.options.get('choices', '')
        if str(c) == '':
            c = None
        else:
            try:
                c = ast.literal_eval(c.strip())
            except SyntaxError('Evaluating `choices` %s for parameter %s failed' % (str(c), self.name)):
                c = None
        
        return c

    def make_default(self, depth=1):
        """
        Creates a default parameter structure from the loaded parameter
        descriptions in this module
        
        Parameters
        ----------            
        depth : int
            The depth in the structure to which all sub nodes are expanded
            All nodes beyond depth will be returned as empty dictionaries
            
        Returns
        -------
        pars : dict
            A parameter branch as nested dicts.
        
        Examples
        --------
        >>> from ptypy import parameter
        >>> print parameter.children['io'].make_default()
        """
        out = {}
        if depth <= 0:
            return out
        for name, child in self.children.iteritems():
            if child.children: # and child.default is None:
                out[name] = child.make_default(depth=depth-1)
            else:
                out[name] = child.default
        return out
        
    def _get_type_argparse(self):
        """
        Returns type or callable that the argparser uses for 
        reading in cmd line argements.
        """
        return type(self.default)
        
    def add2argparser(self, parser=None, prefix='', excludes=('scans', 'engines'), mode='add'):
        """
        Add parameter to an argparse.ArgumentParser instance (or create and return one if parser is None)
        prefix is
        """

        sep = self.separator
        pd = self
        argsep = '-'

        if parser is None:
            from argparse import ArgumentParser
            description = """
            Parser for %s
            Doc: %s
            """ % (pd.name, pd.help)
            parser = ArgumentParser(description=description)
        
        # overload the parser
        if not hasattr(parser, '_aux_translator'):
            parser._aux_translator = {}

        # get list of descendants and remove separator
        ndesc = dict((k.replace(sep, argsep), self[k]) for k, _ in self.descendants)

        groups = {}
        
        for name, pd in ndesc.items():
            if pd.name in excludes:
                continue
            if pd.children:
                groups[name] = parser.add_argument_group(title=prefix+name, description=pd.help)

        for name, pd in ndesc.iteritems():
            
            if pd.name in excludes:
                continue
            up = argsep.join(name.split(argsep)[:-1])
            # recursive part
            parse = groups.get(up, parser)

            """
            # this should be part of PDesc I guess.
            typ = type(pd.default)
            
            for t in pd.type:
                try:
                    typ= eval(t)
                except BaseException:
                    continue
                if typ is not None:
                    break

            if typ is None:
                u.verbose.logger.debug('Failed evaluate type strings %s of parameter %s in python' % (str(pd.type),name))
                return parser
                
            if type(typ) is not type:
                u.verbose.logger.debug('Type %s of parameter %s is not python type' % (str(typ),name))
                return parser
            """
            typ = pd._get_type_argparse()
            
            if typ is bool:
                # Command line switches have no arguments, so treated differently
                flag = '--no-'+name if pd.value else '--'+name
                action = 'store_false' if pd.value else 'store_true'
                parse.add_argument(flag, dest=name, action=action, help=pd.help)
            else:
                d = pd.default
                defstr = d.replace('%(', '%%(') if str(d) == d else str(d)
                parse.add_argument('--'+name, dest=name, type=typ, default=pd.default, choices=pd.choices,
                                   help=pd.help + ' (default=%s)' % defstr)
        
            parser._aux_translator[name] = pd
            
        return parser

        
class EvalParameter(ArgParseParameter):
    """
    Parameter class to store metadata for all ptypy parameters (default, limits, documentation, etc.)
    """
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

    _evaltypes = ['int', 'float', 'tuple', 'list', 'complex']
    _copytypes = ['str', 'file']
    
    DEFAULTS = OrderedDict([
        ('default', 'Default value for parameter (required).'),
        ('help', 'A small docstring for command line parsing (required).'),
        ('doc', 'A longer explanation for the online docs.'),
        ('type', 'Comma separated list of acceptable types.'),
        ('userlevel', """User level, a higher level means a parameter that is 
                     less likely to vary or harder to understand."""),
        ('choices', 'If parameter is list of choices, these are listed here.'),
        ('uplim', 'Upper limit for scalar / integer values'),
        ('lowlim', 'Lower limit for scalar / integer values'),
    ])
     
    def __init__(self, *args, **kwargs):
        # self.DEFAULT the only valid options definition to provide to the superclass.
        kwargs['options_def'] = self.DEFAULTS.copy()
        super(EvalParameter, self).__init__(*args, **kwargs)
        
    @property
    def default(self):
        """
        Default value as a Python type
        """
        default = str(self.options.get('default', ''))
        
        # this destroys empty strings
        default = default if default else None
        
        if default is None:
            out = None
        # should be only strings now
        elif default.lower() == 'none':
            out = None
        elif default.lower() == 'true':
            out = True
        elif default.lower() == 'false':
            out = False
        elif self.is_evaluable:
            out = ast.literal_eval(default)
        else:
            out = default
        
        return out 
        
    @property
    def type(self):
        """
        List of possible data types.
        """
        types = self.options.get('type', None)
        tm = self._typemap
        if types is not None:
            types = [tm[x.strip()] if x.strip() in tm else x.strip() for x in types.split(',')]
        
        return types        
       
    @property
    def limits(self):
        """
        (lower, upper) limits if applicable. (None, None) otherwise
        """
        if self.type is None:
            return None, None
            
        ll = self.options.get('lowlim', None)
        ul = self.options.get('uplim', None)
        if 'int' in self.type:
            lowlim = int(ll) if ll else None
            uplim = int(ul) if ul else None
        else:
            lowlim = float(ll) if ll else None
            uplim = float(ul) if ul else None
            
        return lowlim, uplim
        
    @property
    def doc(self):
        """
        Long documentation, may contain *sphinx* inline markup.
        """
        return self.options.get('doc', '')

    @property
    def userlevel(self):
        """
        User level, a higher level means a parameter that is less 
        likely to vary or harder to understand.
        """
        # User level (for gui stuff) is an int
        ul = self.options.get('userlevel', 1)
        if ul == 'None':
            ul = None
        return int(ul) if ul else None
     
    @property
    def is_evaluable(self):
        for t in self.type:
            if t in self._evaltypes:
                return True
                break
        return False

    def check(self, pars, walk):
        """
        Check that input parameter pars is consistent with parameter description.
        If walk is True and pars is a Param object, checks are also conducted for all
        sub-parameters.

        Returns a dictionary report using CODES values.
        """
        ep = self.path
        out = {}
        val = {}

        # 1. Data type
        if self.type is None:
            # Inconclusive
            val['type'] = CODES.UNKNOWN
            val['lowlim'] = CODES.UNKNOWN
            val['uplim'] = CODES.UNKNOWN
            return {ep: val}
        else:
            val['type'] = CODES.PASS if (type(pars).__name__ in self.type) else CODES.FAIL

        # 2. limits
        lowlim, uplim = self.limits
        
        if lowlim is None:
            val['lowlim'] = CODES.UNKNOWN
        else:
            val['lowlim'] = CODES.PASS if (pars >= self.lowlim) else CODES.FAIL
        if uplim is None:
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
        
    def validate(self, pars, walk=True, raisecodes=(CODES.FAIL, CODES.INVALID)):
        """
        Check that the parameter structure `pars` matches the documented 
        constraints for this node / parameter.
    
        The function raises a RuntimeError if one of the code in the list 
        `raisecodes` has been found. If raisecode is empty, the function will 
        always return successfully but problems will be logged using logger.
    
        Parameters
        ----------
        pars : Param
            A parameter set to validate
        
        walk : bool
            If ``True`` (*default*), navigate sub-parameters.
        
        raisecodes: list
            List of codes that will raise a RuntimeError.
        """
        from ptypy.utils.verbose import logger
        
        d = self.check(pars, walk=walk)
        do_raise = False
        for ep, v in d.items():
            for tocheck, outcome in v.items():
                logger.log(_logging_levels[CODE_LABEL[outcome]], '%-50s %-20s %7s' % (ep, tocheck, CODE_LABEL[outcome]))
                do_raise |= (outcome in raisecodes)
        if do_raise:
            raise RuntimeError('Parameter validation failed.')
            
    def sanity_check(self, depth=10):
        """
        Checks if default parameters from configuration are 
        self-constistent with limits and choices.
        """
        self.validate(self.make_default(depth=depth))

    def make_doc_rst(self, prst, use_root=True):
        """
        Pretty-print in RST format the whole structure.
        """
        Header = '.. _parameters:\n\n'
        Header += '************************\n'
        Header += 'Parameter tree structure\n'
        Header += '************************\n\n'
        prst.write(Header)

        root = self.root
        shortdoc = 'help'
        longdoc = 'doc'
        default = 'default'
        lowlim = 'lowlim'
        uplim = 'uplim'

        for name, desc in root.descendants:
            if name == '':
                continue
            if hasattr(desc, 'children') and desc.parent is root:
                prst.write('\n' + name + '\n')
                prst.write('=' * len(name) + '\n\n')
            if hasattr(desc, 'children') and desc.parent.parent is root:
                prst.write('\n' + name + '\n')
                prst.write('-' * len(name) + '\n\n')

            prst.write('.. py:data:: ' + name)
            # prst.write('('+', '.join([t for t in opt['type']])+')')
            prst.write('(' + ', '.join(desc.type) + ')')
            prst.write('\n\n')
            # num = str(desc.num_id)
            # prst.write('   *(' + num + ')* ' + desc.help + '\n\n')
            prst.write('   ' + desc.doc.replace('\n', '\n   ') + '\n\n')
            prst.write('   *default* = ``' + str(desc.default))
            lowlim, uplim = desc.limits
            if lowlim is not None and uplim is not None:
                prst.write(' (>' + str(lowlim) + ', <' + str(uplim) + ')``\n')
            elif lowlim is not None and uplim is None:
                prst.write(' (>' + str(lowlim) + ')``\n')
            elif lowlim is None and uplim is not None:
                prst.write(' (<' + str(uplim) + ')``\n')
            else:
                prst.write('``\n')

            prst.write('\n')
        prst.close()


class parse_parameters(object):
    """
    Decorator that parses the doc string of a function or class and extracts metainformation
    on input parameters.
    """

    entries = []

    def __init__(self, path=None):
        """
        I path is none, look for attachment points somewhere else,
        """
        self.path = path

    def __call__(self, cls):
        # Extract and truncate doc string

        # Parse parameter section

        # Add to entries

        # Populate cls.DEFAULT

        return cls


# Load all documentation on import
import pkg_resources
_file = pkg_resources.resource_filename('ptypy', 'resources/parameter_descriptions.configparser')
parameter_descriptions = EvalParameter(name='')
parameter_descriptions.load_conf_parser(open(_file, 'r'))
del pkg_resources


def create_default_template(filename=None, user_level=0, doc_level=2):
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
    import textwrap

    def wrapdoc(x):
        if not x.strip():
            return ''
        x = ' '.join(x.strip().split('\n'))
        return '# ' + '\n# '.join(textwrap.wrap(x, 75, break_long_words=False, replace_whitespace=False)).strip() + '\n'

    if filename is None:
        f = open('ptypy_template.py', 'w')
    else:
        f = open(filename, 'w')
    h = '"""\nThis Script was autogenerated using\n'
    h += '``u.create_default_template("%s",%d,%d)``\n' % (str(filename), user_level, doc_level)
    h += 'It is only a TEMPLATE and not a working reconstruction script.\n"""\n\n'
    h += "import numpy as np\n"
    h += "import ptypy\n"
    h += "from ptypy.core import Ptycho\n"
    h += "from ptypy import utils as u\n\n"
    try:
        from ptypy.utils.verbose import headerline
        h += headerline('Ptypy Parameter Tree', 'l', '#')+'\n'
    except ImportError:
        h += '### Ptypy Parameter Tree ###\n\n'
    f.write(h)
    for entry, pd in parameter_descriptions.descendants:
        if user_level < pd.userlevel:
            continue
        if pd.children:
            value = "u.Param()"
        else:
            val = pd.default
            if str(val) == val:
                value = '"%s"' % str(val)
            else:
                value = str(val)
        #ID ="%02d" % pd.ID if hasattr(pd,'ID') else 'NA'
        if doc_level > 0:
            # f.write('\n'+"## (%s) " % ID +pd.shortdoc.strip()+'\n')
            f.write('\n## ' + pd.help.strip() + '\n')
        if doc_level > 1:
            #f.write(_format_longdoc(pd.doc))
            f.write(wrapdoc(pd.doc))
        f.write('p.'+entry + ' = ' + value+'\n')
        
    f.write('\n\nPtycho(p,level=5)\n')
    f.close()


if __name__ =='__main__':
    from ptypy import utils as u
    
    
    
    parser = _add2argparser(entry_point='.scan.illumination')
    parser.parse_args()
    
