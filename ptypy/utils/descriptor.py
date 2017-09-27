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
from collections import OrderedDict
import textwrap


__all__ = ['Descriptor', 'ArgParseDescriptor', 'EvalDescriptor']


class _Adict(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# ! Validator message codes
CODES = _Adict(
    PASS=1,
    FAIL=0,
    UNKNOWN=2,
    MISSING=3,
    INVALID=4)

# ! Inverse message codes
CODE_LABEL = dict((v, k) for k, v in CODES.__dict__.items())


class Descriptor(object):
    """
    Base class for parameter descriptions and validation. This class is used to hold both command line arguments
    and Param-type parameter descriptions.
    """

    # Options definitions as a class variable:
    # a dictionary whose keys are attribute names and values are description
    # of the attribute. It this description contains the text "required" or
    # "mandatory", the attribute is registered as required.
    OPTIONS_DEF = None

    def __init__(self, name, parent=None, separator='.'):
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

        #: Parent parameter (:py:class:`Descriptor` type) if it has one.
        self.parent = parent

        #: Hierarchical tree of sub-Parameters.
        self.children = OrderedDict()

        self.separator = separator

        # Required and optional attributes
        self.required = []
        self.optional = []
        self.options_def = OrderedDict()
        self._parse_options_def()

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

    def _parse_options_def(self):
        """
        Parse and store options definitions.
        """
        if self.OPTIONS_DEF is not None:
            r = []
            o = []

            for option, text in self.OPTIONS_DEF.items():
                if 'required' in text or 'mandatory' in text:
                    r += [option]
                else:
                    o += [option]
            self.required = r
            self.optional = o

    def new_child(self, name, options=None, implicit=False):
        """
        Create a new descendant and pass new options.

        If name contains separators, intermediate children are created.

        If name already exists, update options and return existing child.

        If name already exists and had been created implicitly to create
        a child further down, the order in self.children is corrected to
        honor the order of explicitly created children.
        This behaviour can be deactivated by setting implicit=True.
        """
        if options is None:
            options = self.options

        if self.separator in name:
            # Creating a sub-level
            name, next_name = name.split(self.separator, 1)
            subparent = self.children.get(name, None)
            if subparent is None:
                # Create subparent
                subparent = self.__class__(name=name, parent=self, separator=self.separator)

                # Remember that creation was implicit
                subparent.implicit = True

                # Insert in children dict
                self.children[name] = subparent
            child = subparent.new_child(next_name, options, implicit)
            self._all_options.update(subparent.options)
        else:
            if name in self.children.keys():
                # The child already exists
                child = self.children[name]
                if child.implicit and not implicit:
                    # Tricky bit: this child had already been created implicitly, but now is being created
                    # explicitly. We use this to enforce a proper order in self.children
                    self.children.pop(name)
                    child.implicit = False

                    explicit = [(k, v) for k, v in self.children.items() if not v.implicit]
                    implicit = [(k, v) for k, v in self.children.items() if v.implicit]
                    self.children = OrderedDict(explicit + [(name, child)] + implicit)
            else:
                child = self.__class__(name=name, parent=self, separator=self.separator)
                self.children[name] = child
                child.implicit = implicit
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
            # dot-prefix as a way to search through the tree deactivated for now.
            # if not root:
            #    parent = self._find(name.split(self.separator)[0])
            # else:
            #    parent = self.children[root]
            # return parent[name]
            return self.children[root][name]
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
            desc.parent = self
            self._all_options.update(desc.options)
        else:
            root, name = name.split(self.separator, 1)
            # dot-prefix as a way to search the tree is deactivated for now.
            # if not root:
            #    subparent = self._find(name.split(self.separator)[0])
            #    if subparent is None:
            #        raise RuntimeError('No attachment point for .%s found.' % name)
            # else:
            #    subparent = self.children.get(root, None)
            #    if not subparent:
            #        subparent = self.new_child(root)
            # subparent[name] = desc
            subparent = self.children.get(root, None)
            if not subparent:
                subparent = self.new_child(root)
            subparent[name] = desc

    def get(self, path):
        """
        return self.root[path] if it exists, None otherwise.
        """
        try:
            link = self.root[path]
            return link
        except (KeyError, TypeError) as e:
            return None

    def add_child(self, desc):
        self[desc.name] = desc

    def prune_child(self, name):
        """
        Remove and return the parameter "name" and all its children.
        :param name: The descendant name
        :return: The Parameter object.
        """
        # Use __getitem__ to take care of all naming syntaxes
        desc = self[name]

        # Pop out from parent
        desc.parent.children.pop(desc.name)

        # Make standalone
        desc.parent = None

        return desc

    @property
    def root(self):
        """
        Return root of parameter tree.
        """
        if self.parent is None:
            return self
        else:
            return self.parent.root

    @property
    def path(self):
        """
        Return complete path from root of parameter tree.
        (self.root[self.path] == self should always be True unless self.root is root)
        """
        if self.parent is None:
            # The base node has no path
            return None
        elif self.parent.parent is None:
            return self.name
        else:
            return self.parent.path + self.separator + self.name

    @property
    def descendants(self):
        """
        Iterator over all descendants as a pair (path name, object).
        """
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
                    chain = chain[:level] + [name]

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

    def from_string(self, s, **kwargs):
        """
        Load Parameter from string using Python's ConfigParser

        Each parameter occupies its own section.
        Separator characters in sections names map to a tree-hierarchy.

        Keyword arguments are forwarded to `ConfigParser.RawConfigParser`
        """
        from StringIO import StringIO
        return self.load_conf_parser(StringIO(s), **kwargs)

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


class ArgParseDescriptor(Descriptor):
    OPTIONS_DEF = OrderedDict([
        ('default', 'Default value for parameter.'),
        ('help', 'A small docstring for command line parsing (required).'),
        ('choices', 'If parameter is list of choices, these are listed here.')
    ])

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

    @default.setter
    def default(self, val):
        """
        Set default, ensuring that is it stored as a string.
        """
        if val is None:
            self.options['default'] = ''
        elif str(val) == val:
            self.options['default'] = "'%s'" % val
        else:
            self.options['default'] = str(val)

    def eval(self, val):
        """
        A more verbose wrapper around `ast.literal_eval`
        """
        try:
            return ast.literal_eval(val)
        except ValueError or SyntaxError as e:
            msg = e.args[0] + ". could not read %s for parameter %s" % (val, self.path)
            raise ValueError(msg)
        except SyntaxError as e:
            msg = e.args[0] + ". could not read %s for parameter %s" % (val, self.path)
            raise SyntaxError(msg)

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

    def make_default(self, depth=0, flat=False):
        """
        Creates a default parameter structure from the loaded parameter
        descriptions in this module
        
        Parameters
        ----------            
        depth : int
            The depth in the structure to which all sub nodes are expanded
            All nodes beyond depth will be ignored.

        flat : bool
            If `True` returns flat dict with long keys, otherwise nested
            dicts with short keys. default=`False`
            
        Returns
        -------
        pars : dict
            A parameter branch as nested dicts.
        
        Examples
        --------
        >>> from ptypy import descriptor
        >>> print descriptor.children['io'].make_default()
        """
        if flat:
            return dict([(k, v.default) for k, v in self.descendants])

        out = {}
        # Interpret a string default as a link to another part
        # of the structure.
        if str(self.default) == self.default:
            link = self.get(self.default)
            if link and depth >= 0:
                return link.make_default(depth=depth-1)

        if not self.children:
            return self.default

        for name, child in self.children.iteritems():
            if depth >= 0:
                out[name] = child.make_default(depth=depth-1)

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
                groups[name] = parser.add_argument_group(title=prefix + name, description=pd.help)

        for name, pd in ndesc.iteritems():

            if pd.name in excludes:
                continue
            up = argsep.join(name.split(argsep)[:-1])
            # recursive part
            parse = groups.get(up, parser)

            typ = pd._get_type_argparse()

            if typ is bool:
                # Command line switches have no arguments, so treated differently
                flag = '--no-' + name if pd.value else '--' + name
                action = 'store_false' if pd.value else 'store_true'
                parse.add_argument(flag, dest=name, action=action, help=pd.help)
            else:
                d = pd.default
                defstr = d.replace('%(', '%%(') if str(d) == d else str(d)
                parse.add_argument('--' + name, dest=name, type=typ, default=pd.default, choices=pd.choices,
                                   help=pd.help + ' (default=%s)' % defstr)

            parser._aux_translator[name] = pd

        return parser


class EvalDescriptor(ArgParseDescriptor):
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
    _limtypes = ['int', 'float']

    OPTIONS_DEF = OrderedDict([
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

        # strip any extra quotation marks
        if type(out) == str:
            out = out.strip('"').strip("'")

        return out

    @property
    def is_evaluable(self):
        for t in self.type:
            if t in self._evaltypes:
                return True
                break
        return False

    @property
    def type(self):
        """
        List of possible data types.
        """
        types = self.options.get('type', None)
        tm = self._typemap
        if types is not None:
            types = [tm[x.strip()] if x.strip() in tm else x.strip() for x in types.split(',')]
        elif self.default is not None:
            types = [type(self.default).__name__, ]
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

    def check(self, pars, walk):
        """
        Check that input parameter pars is consistent with parameter description.
        If walk is True and pars is a Param object, checks are also conducted for all
        sub-parameters.

        Returns a dictionary report using CODES values.
        """
        # FIXME: this needs a lot of testing and verbose.debug lines.
        ep = self.path
        val = {}
        out = {ep: val}

        symlinks = None

        # Data type
        if self.type is None:
            # No type: inconclusive
            val['type'] = CODES.UNKNOWN
            val['lowlim'] = CODES.UNKNOWN
            val['uplim'] = CODES.UNKNOWN
            return out
        elif type(pars).__name__ in self.type:
            # Standard type: pass
            val['type'] = CODES.PASS
        else:
            # Type could be a symlink to another part of the tree
            symlinks = {}
            for tp in self.type:
                link = self.get(tp)
                if not link:
                    # Abort because one of the types was not recognised as a link.
                    val['type'] = CODES.INVALID
                    symlinks = None
                    break
                symlinks[link.name] = link

        # Manage symlinks
        if symlinks:
            # Look for name
            name = pars.get('name', None)
            if not name:
                # The entry does not have a name, that's not good.
                val['symlink'] = CODES.INVALID
                val['name'] = CODES.MISSING
                return out
            if name not in symlinks:
                # The entry name is not found, that's not good.
                val['symlink'] = CODES.INVALID
                val['name'] = CODES.UNKNOWN
                return out
            if walk:
                # Follow symlink
                symlink = symlinks[name]
                sym_out = symlink.check(pars, walk)
                # Rehash names
                for k, v in sym_out.iteritems():
                    k1 = k.replace(symlink.path, ep)
                    out[k1] = v
            return out

        # Check limits
        if any([i in self._limtypes for i in self.type]):
            lowlim, uplim = self.limits
            if lowlim is None:
                val['lowlim'] = CODES.UNKNOWN
            else:
                val['lowlim'] = CODES.PASS if (pars >= lowlim) else CODES.FAIL
            if uplim is None:
                val['uplim'] = CODES.UNKNOWN
            else:
                val['uplim'] = CODES.PASS if (pars <= uplim) else CODES.FAIL

        # Nothing left to check except for Param or dict.
        if not hasattr(pars, 'items'):
            return out

        # Detect wildcard
        wildcard = (self.children.keys() == ['*'])
        if wildcard:
            if not walk:
                return out
        else:
            # Check for missing entries
            for k, v in self.children.items():
                if k not in pars:
                    val[k] = CODES.MISSING

        # Check for invalid entries
        if wildcard and not pars:
            # At least one child is required.
            out[ep + '.*'] = {'*': CODES.MISSING}
            return out
        for k, v in pars.items():
            if wildcard:
                if walk:
                    w_out = self.children['*'].check(v, walk)
                    for kk, vv in w_out.iteritems():
                        k1 = kk.replace('*', k, 1)
                        out[k1] = vv
            elif k not in self.children:
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
        pars : Param, dict
            A parameter set to validate
        
        walk : bool
            If ``True`` (*default*), navigate sub-parameters.
        
        raisecodes: list
            List of codes that will raise a RuntimeError.
        """
        from ptypy.utils.verbose import logger
        import logging

        _logging_levels = dict(
            PASS=logging.INFO,
            FAIL=logging.CRITICAL,
            UNKNOWN=logging.WARN,
            MISSING=logging.WARN,
            INVALID=logging.ERROR
        )

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

    def parse_doc(self, name=None, recursive=True):
        """
        Decorator to parse docstring and automatically attach new parameters.
        The parameter section is identified by a line starting with the word "Parameters"

        :param name: The descendant name under which all parameters will be held. If None, use self
        :return: The decorator function
        """
        return lambda cls: self._parse_doc_decorator(name, cls, recursive)

    def _parse_doc_decorator(self, name, cls, recursive):
        """
        Actual decorator returned by parse_doc.
        """
        # Find or create insertion point
        if name is None:
            desc = self
        else:
            try:
                desc = self[name]
            except KeyError:
                desc = self.new_child(name, implicit=True)

        # Get the parameter section, including from base class(es) if recursive.
        parameter_string = self._extract_doc_from_class(cls, recursive)

        # Maybe check here if a non-Param descendant is being overwritten?
        desc.options['type'] = 'Param'

        if not recursive and cls.__base__ != object:
            desc_base = getattr(cls.__base__, '_descriptor')
            typ = desc_base().path if desc_base is not None else None
            desc.default = typ

        # Parse parameter section and store in desc
        desc.from_string(parameter_string)

        # Attach the Parameter group to cls
        from weakref import ref
        cls._descriptor = ref(desc)

        # FIXME: This should be solved more elegantly
        from ptypy.utils import Param
        cls.DEFAULTS = Param()
        cls.DEFAULTS.update(desc.make_default(depth=99), Convert=True)

        return cls

    def _extract_doc_from_class(self, cls, recursive=True):
        """
        Utility method used recursively by _parse_doc_decorator to extract doc strings
        from all base classes and cobble the "Parameters" section.
        """
        if cls == object:
            # Reached "object" base class. No doc string here.
            return ''

        # Get doc from base
        if recursive:
            base_parameters = self._extract_doc_from_class(cls.__base__)
        else:
            base_parameters = ''

        # Append doc from class
        docstring = cls.__doc__ if cls.__doc__ is not None else ' '

        # Because of indentation it is safer to work line by line
        doclines = docstring.splitlines()
        for n, line in enumerate(doclines):
            if line.strip().startswith('Defaults:'):
                break
        parameter_string = textwrap.dedent('\n'.join(doclines[n + 1:]))

        return base_parameters + parameter_string


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
        h += headerline('Ptypy Parameter Tree', 'l', '#') + '\n'
    except ImportError:
        h += '### Ptypy Parameter Tree ###\n\n'
    f.write(h)

    # Write data structure
    for entry, pd in defaults_tree.descendants:

        # Skip entries above user level
        if user_level < pd.userlevel:
            continue

        # Manage wildcards
        if pd.name == '*':
            pname = pd.parent.name
            if pname[-1] != 's':
                raise RuntimeError('Wildcards are supposed to appear only in plural container.'
                                   '%s does not end with an "s"' % pname)
            entry = entry.replace(pd.name, pname[:-1] + '_00')

        if pd.children:
            value = "u.Param()"
        else:
            val = pd.default
            link = pd.get(val)
            if link:
                value = 'p.' + pd.default
            elif str(val) == val:
                value = '"%s"' % str(val)
            else:
                value = str(val)
        # ID ="%02d" % pd.ID if hasattr(pd,'ID') else 'NA'
        if doc_level > 0:
            # f.write('\n'+"## (%s) " % ID +pd.shortdoc.strip()+'\n')
            f.write('\n## ' + pd.help.strip() + '\n')
        if doc_level > 1:
            # f.write(_format_longdoc(pd.doc))
            f.write(wrapdoc(pd.doc))
        f.write('p.' + entry + ' = ' + value + '\n')

    f.write('\n\nPtycho(p,level=5)\n')
    f.close()


defaults_tree = EvalDescriptor('root')