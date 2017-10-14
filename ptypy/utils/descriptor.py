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
from parameters import Param


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

    def __init__(self, name, parent=None, separator='.'):
        """
        Parameter class to store metadata for all ptypy parameters (default, limits, documentation, etc.)
        """
        super(EvalDescriptor, self).__init__(name, parent=parent, separator=separator)
        self.options['type'] = 'Param'

    @property
    def default(self):
        """
        Default value as a Python type
        """
        default = str(self.options.get('default', ''))
        types = str(self.options.get('type', []))

        # this destroys empty strings
        default = default if default else None

        if 'Param' in types or 'dict' in types:
            out = Param()
        elif default is None:
            out = None
        # should be only strings now
        elif default.lower() == 'none':
            out = None
        elif default.lower() == 'true':
            out = True
        elif default.lower() == 'false':
            out = False
        elif default.startswith('@'):
            out = self.get(default[1:])
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
    def is_symlink(self):
        """
        True if type/default are symlinks.
        """
        types = self.options.get('type', '')
        return '@' in types

    @property
    def is_target(self):
        """
        True if parent of symlink targets.
        """
        if self.parent is not self.root:
            return False
        for n, d in self.root.descendants:
            if d.is_symlink and d.type[0].path.startswith(self.name):
                return True
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
            # symlinks
            if types[0].startswith('@'):
                # wildcard in symlink: needed to grab dynamically added entries
                if types[0].endswith('.*'):
                    parent = self.get(types[0][1:-2])
                    types = [c for n, c in parent.children.items()]
                else:
                    types = [self.get(t[1:]) for t in types]
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

    def _walk(self, depth=0, pars=None, ignore_symlinks=False, ignore_wildcards=False, path=None):
        """
        Generator that traverses the complete tree up to given depth, either on its own,
        or following parameter structure in pars, following symlinks and honouring wildcards.

        Parameters
        ----------
        depth: How many levels to traverse in the tree.
        pars: optional parameter tree to match with descriptor tree
        ignore_symlinks: If True, do not follow symlinks. Default to False.
        ignore_wildcards: If True, do not interpret wildcards. Default to False.
        path: Used internally for recursion.

        Returns
        -------
        A generator. Yields a dict with structure
        {'d': <Descriptor instance>,
         'path': <relative path in structure>,
         'status': <status message>,
         'info': <additional information depending on status>}
        """

        if path is None:
            # This happens only at top level: ensure proper construction of relative paths.
            path = ''

        # Resolve symlinks
        if self.is_symlink and not ignore_symlinks:
            if len(self.type) == 1:
                # No name needed
                s = self.type[0]
            else:
                if pars is not None:
                    # Look for name in pars
                    name = pars.get('name', None)
                    # Is this the intended behaviour? Instead maybe s = self.default
                    if name is None:
                        s = None
                        yield {'d': self, 'path': path, 'status': 'noname', 'info': ''}
                    else:
                        s = dict((link.name, link) for link in self.type).get(name, None)
                        if not s:
                            yield {'d': self, 'path': path, 'status': 'nolink', 'info': name}
                else:
                    # No pars, resolve default
                    s = self.default
            # Follow links
            if s:
                for x in s._walk(depth=depth, pars=pars, ignore_symlinks=ignore_symlinks,
                                 ignore_wildcards=ignore_wildcards, path=path):
                    yield x
            return

        # Detect wildcard
        wildcard = (self.children.keys() == ['*'])

        # Grab or check children
        if wildcard:
            if ignore_wildcards:
                yield {'d': self, 'path': path, 'status': 'wildcard', 'info': ''}
                return
            else:
                if pars is None:
                    # Generate default name for single entry
                    children = {self.name[:-1] + '_00': self.children['*']}
                else:
                    # Grab all names from pars
                    children = {k: self.children['*'] for k in pars.keys()}
        else:
            children = self.children

        # Main yield: check type here.
        if not pars or (type(pars).__name__ in self.type) or (hasattr(pars, 'items') and 'Param' in self.type):
            yield {'d': self, 'path': path, 'status': 'ok', 'info': ''}
        else:
            yield {'d': self, 'path': path, 'status': 'wrongtype', 'info': type(pars).__name__}
            return

        if not children or depth == 0:
            # Nothing else to do
            return

        # Look for unrecognised entries in pars
        if pars:
            for k, v in pars.items():
                if k not in children:
                    yield {'d': self, 'path': path, 'status': 'nochild', 'info': k}

        # Loop through children
        for cname, c in children.items():
            new_path = '.'.join([path, cname]) if path else cname
            if pars:
                if cname not in pars:
                    yield {'d': c, 'path': path, 'status': 'nopar', 'info': cname}
                else:
                    for x in c._walk(depth=depth-1, pars=pars[cname], ignore_symlinks=ignore_symlinks,
                                     ignore_wildcards=ignore_wildcards, path=new_path):
                        yield x
            else:
                for x in c._walk(depth=depth-1, ignore_symlinks=ignore_symlinks,
                                 ignore_wildcards=ignore_wildcards, path=new_path):
                    yield x
        return

    def check(self, pars, depth=99):
        """
        Check that input parameter pars is consistent with parameter description, up to given depth.

        Parameters
        ----------
        pars: The input parameter or parameter tree
        depth: The level at wich verification is done.

        Returns
        -------
        A dictionary report using CODES values.

        """
        out = {}
        for res in self._walk(depth=depth, pars=pars):
            # Switch through all possible statuses
            if res['status'] == 'ok':
                # Check limits
                d = res['d']
                val = {'type': CODES.PASS}
                if any([i in d._limtypes for i in d.type]):
                    lowlim, uplim = d.limits
                    if lowlim is None:
                        val['lowlim'] = CODES.UNKNOWN
                    else:
                        val['lowlim'] = CODES.PASS if (pars[res['path']] >= lowlim) else CODES.FAIL
                    if uplim is None:
                        val['uplim'] = CODES.UNKNOWN
                    else:
                        val['uplim'] = CODES.PASS if (pars[res['path']] <= uplim) else CODES.FAIL
                out[res['path']] = val
            elif res['status'] == 'wrongtype':
                # Wrong type
                out[res['path']] = {'type': CODES.INVALID}
            elif res['status'] == 'noname':
                # Symlink name could not be found
                out[res['path']] = {'symlink': CODES.INVALID, 'name': CODES.MISSING}
            elif res['status'] == 'nolink':
                # Link was not resolved
                out[res['path']] = {'symlink': CODES.INVALID, 'name': CODES.UNKNOWN}
            elif res['status'] == 'nochild':
                # Parameter entry without corresponding Descriptor
                out[res['path']] = {res['info']: CODES.INVALID}
            elif res['status'] == 'nopar':
                # Missing parameter entry
                out[res['path']] = {res['info']: CODES.MISSING}

        return out

    def validate(self, pars, raisecodes=(CODES.FAIL, CODES.INVALID)):
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

        d = self.check(pars)
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

    def make_default(self, depth=0):
        """
        Creates a default parameter structure.

        Parameters
        ----------
        depth : int
            The depth in the structure to which all sub nodes are expanded
            All nodes beyond depth will be ignored.

        Returns
        -------
        pars : Param
            A parameter branch as Param.

        Examples
        --------
        >>> from ptypy.utils.descriptor import defaults_tree
        >>> print(defaults_tree['io'].make_default(depth=5))
        """
        out = Param()
        for ret in self._walk(depth=depth, ignore_symlinks=True, ignore_wildcards=True):
            path = ret['path']
            if path == '': continue
            out[path] = ret['d'].default
        return out

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

        Parameters
        ----------
        name: str
              The descendant name under which all parameters will be held. If None, use self.
        recursive: bool
              Whether or not to traverse the docstring of base classes. *Is there are use case for this?*

        Returns
        -------
        The decorator function.
        """
        return lambda cls: self._parse_doc_decorator(name, cls, recursive)

    def _parse_doc_decorator(self, name, cls, recursive):
        """
        Actual decorator returned by parse_doc.

        Parameters
        ----------
        name: str
             Descendant name.
        cls:
             Class to decorate.
        recursive:
             If false do not parse base class doc.

        Returns
        -------
        Decorated class.
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

        # PT: I don't understand this.
        if not recursive and cls.__base__ != object:
            desc_base = getattr(cls.__base__, '_descriptor')
            typ = desc_base().path if desc_base is not None else None
            desc.default = typ

        # Parse parameter section and store in desc
        desc.from_string(parameter_string)

        # Attach the Parameter group to cls
        from weakref import ref
        cls._descriptor = ref(desc)

        # Render the defaults
        cls.DEFAULT = desc.make_default(depth=99)

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

    def create_template(self, filename=None, start_at_root=True, user_level=0, doc_level=2):
        """ 
        Creates templates for ptypy scripts from an EvalDescriptor instance.
        """
        desc = self
        if start_at_root:
            desc = self.root

        base = 'p'

        # open file
        filename = 'ptypy_template.py' if filename is None else filename
        with open(filename, 'w') as fp:

            # write header
            h = '"""\nThis template was autogenerated using an EvalDescriptor instance.\n'
            h += 'It is only a template and not a working reconstruction script.\n"""\n\n'
            h += "import numpy as np\n"
            h += "import ptypy\n"
            h += "from ptypy.core import Ptycho\n"
            h += "from ptypy import utils as u\n\n"
            h += '### Ptypy parameter tree ###' + '\n\n'
            fp.write(h)

            # write the parameter defaults
            fp.write(base + ' = Param()\n\n')
            for ret in self._walk(depth=99, ignore_wildcards=False):
                d = ret['d']
                # user level
                if d.userlevel > user_level: continue
                # skip the root, handled above
                if d.root is d: continue
                # handle line breaks already in the help/doc strings
                hlp = '# ' + d.help.replace('\n', '\n# ')
                doc = '# ' + d.doc.replace('\n', '\n# ')
                # doclevel 2: help and doc before parameter
                if doc_level == 2:
                    fp.write('\n')
                    fp.write(hlp + '\n')
                    fp.write(doc + '\n')
                # Container defaults come as Params. It would be more elegant 
                # to check 'if d.children' here but not sure that is safe
                if isinstance(d.default, Param):
                    if doc_level < 2:
                        fp.write('\n')
                    line = base + '.' + ret['path'] + ' = u.Param()'
                    fp.write(line)
                # not Param: actual default value
                else:
                    val = str(d.default)
                    if 'str' in d.type and not d.default is None:
                        val = "'" + val + "'"
                    line = base + '.' + ret['path'] + ' = ' + val
                    fp.write(line)
                # doclevel 1: inline help comments
                if doc_level == 1:
                    fp.write(' ' * max(1, 50 - len(line)) + hlp + '\n')
                else:
                    fp.write('\n')

            # write the Ptycho instantiation
            fp.write('\n\n### Reconstruction ###\n\n')
            fp.write('Ptycho(%s,level=5)\n'%base)

defaults_tree = EvalDescriptor('root')
