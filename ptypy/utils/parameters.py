# -*- coding: utf-8 -*-
"""\
Parameter definition.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os

__all__ = ['Param', 'asParam', 'param_from_json', 'param_from_yaml']  # 'load',]

PARAM_PREFIX = 'pars'


class Param(dict):
    """
    Convenience class: a dictionary that gives access to its keys
    through attributes.

    Note: dictionaries stored in this class are also automatically converted
    to Param objects:
    >>> p = Param()
    >>> p.x = {}
    >>> p
    Param({'x': {}})
    
    While dict(p) returns a dictionary, it is not recursive, so it is better in this case
    to use p.todict(). However, p.todict does not check for infinite recursion. So please
    don't store a dictionary (or a Param) inside itself.

    BE: Please note also that the recursive behavior of the update function will create
    new references. This will lead inconsistency if other objects refer to dicts or Params
    in the updated Param instance.
    """
    _display_items_as_attributes = True
    _PREFIX = PARAM_PREFIX

    def __init__(self, __d__=None, **kwargs):
        """
        A Dictionary that enables access to its keys as attributes.
        Same constructor as dict.
        """
        dict.__init__(self)
        if __d__ is not None:
            self.update(__d__)
        self.update(kwargs)

    def __getstate__(self):
        return list(self.__dict__.items())

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __str__(self):
        from .verbose import report
        return report(self, depth=7, noheader=True)

    def __setitem__(self, key, value):
        # BE: original behavior modified as implicit conversion may destroy references
        # Use update(value,Convert=True) instead
        # return super(Param, self).__setitem__(key, Param(value) if type(value) == dict else value)

        s = self
        # Parse dots in key
        if type(key) == str and '.' in key:
            keys = key.split('.')
            # Extract or generate subtree
            for k in keys[:-1]:
                try:
                    s = s.__getitem__(k)
                except KeyError:
                    super(Param, s).__setitem__(k, Param())
                    s = super(Param, s).__getitem__(k)
            key = keys[-1]
        return super(Param, s).__setitem__(key, value)

    def __getitem__(self, name):
        # Parse dots in key
        if type(name) == str and '.' in name:
            name_, name__ = name.split('.', 1)
            s = self[name_]
            if type(s) is not type(self):
                raise KeyError(name)
            return s[name__]
        return super(Param, self).__getitem__(name)

    def __delitem__(self, name):
        return super(Param, self).__delitem__(name)

    def __delattr__(self, name):
        return super(Param, self).__delitem__(name)

    # __getattr__ = __getitem__
    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError as ke:
            raise AttributeError(ke)

    __setattr__ = __setitem__

    def copy(self, depth=0):
        """
        :returns Param: A (recursive) copy of P with depth `depth`
        """
        d = Param(self)
        if depth > 0:
            for k, v in d.items():
                if isinstance(v, self.__class__): d[k] = v.copy(depth - 1)
        return d

    def __dir__(self):
        """
        Defined to include the keys when using dir(). Useful for
        tab completion in e.g. ipython.
        If you do not wish the dict key's be displayed as attributes
        (although they are still accessible as such) set the class
        attribute `_display_items_as_attributes` to False. Default is
        True.
        """
        if self._display_items_as_attributes:
            return list(self.keys())
        else:
            return []

    def __contains__(self, key):
        """
        Redefined to support dot keys.
        """
        if '.' in key:
            name_, name__ = key.split('.', 1)
            if name_ in self:
                s = self[name_]
                if type(s) is type(self):
                    return name__ in self[name_]
                else:
                    return False
            else:
                return False
        return super(Param, self).__contains__(key)

    def update(self, __d__=None, in_place_depth=0, Convert=False, **kwargs):
        """
        Update Param - almost same behavior as dict.update, except
        that all dictionaries are converted to Param if `Convert` is set
        to True, and update may occur in-place recursively for other Param
        instances that self refers to.

        Parameters
        ----------
        Convert : bool
                  If True, convert all dict-like values in self also to Param.
                  *WARNING*
                  This mey result in misdirected references in your environment
        in_place_depth : int
                  Counter for recursive in-place updates
                  If the counter reaches zero, the Param to a key is
                  replaced instead of updated
        """

        def _k_v_update(k, v):
            # If an element is itself a dict, convert it to Param
            if Convert and hasattr(v, 'keys') and not isinstance(v, self.__class__):
                v = Param(v)
                v.update(list(v.items()), in_place_depth - 1, Convert)

            # new key
            if k not in self:
                self[k] = v

            # If this key already exists and is already dict-like, update it
            elif in_place_depth > 0 and hasattr(v, 'keys') and isinstance(self[k], self.__class__):
                self[k].update(v, in_place_depth - 1, Convert)

                """
                if isinstance(self[k],self.__class__):
                    # Param gets recursive in_place updates
                    self[k].update(v, in_place_depth - 1)
                else:
                    # dicts are only updated in-place once
                    self[k].update(v)
                """

            # Otherwise just replace it
            else:
                self[k] = v

        if __d__ is not None:
            if hasattr(__d__, 'keys'):
                # Iterate through dict-like argument
                for k, v in __d__.items():
                    _k_v_update(k, v)

            else:
                # here we assume a (key,value) list.
                for (k, v) in __d__:
                    _k_v_update(k, v)

        for k, v in kwargs.items():
            _k_v_update(k, v)

        return None

    def _to_dict(self, Recursive=False):
        """
        Convert to dictionary (recursively if needed).
        """
        if not Recursive:
            return dict(self)
        else:
            d = dict(self)
            for k, v in d.items():
                if isinstance(v, self.__class__): d[k] = v._to_dict(Recursive)
        return d

    @classmethod
    def _from_dict(cls, dct):
        """
        Make Param from dict. This is similar to the __init__ call
        but kept here for coherent usage among ptypy.core.Base children
        """
        # p=Param()
        # p.update(dct.copy())
        return Param(dct.copy())


def validate_standard_param(sp, p=None, prefix=None):
    """\
    validate_standard_param(sp) checks if sp follows the standard parameter convention.
    validate_standard_param(sp, p) attemps to check if p is a valid implementation of sp.

    NOT VERY SOPHISTICATED FOR NOW!
    """
    if p is None:
        good = True
        for k, v in sp.items():
            if k.startswith('_'): continue
            if type(v) == type(sp):
                pref = k if prefix is None else '.'.join([prefix, k])
                good &= validate_standard_param(v, prefix=pref)
                continue
            else:
                try:
                    a, b, c = v
                    if prefix is not None:
                        print('    %s.%s = %s' % (prefix, k, str(v)))
                    else:
                        print('    %s = %s' % (k, str(v)))
                except:
                    good = False
                    if prefix is not None:
                        print('!!! %s.%s = %s <--- Incorrect' % (prefix, k, str(v)))
                    else:
                        print('!!! %s = %s <--- Incorrect' % (k, str(v)))

        return good
    else:
        raise NotImplementedError('Checking if a param fits with a standard is not yet implemented')


def format_standard_param(p):
    """\
    Pretty-print a Standard Param class.
    """
    lines = []
    if not validate_standard_param(p):
        print('Standard parameter does not')
    for k, v in p.items():
        if k.startswith('_'): continue
        if type(v) == type(p):
            sublines = format_standard_param(v)
            lines += [k + '.' + s for s in sublines]
        else:
            lines += ['%s = %s #[%s] %s' % (k, str(v[1]), v[0], v[2])]
    return lines


def asParam(obj):
    """
    Convert the input to a Param.

    Parameters
    ----------
    a : dict_like
        Input structure, in any format that can be converted to a Param.

    Returns:
    out : Param
        The Param structure built from a. No copy is done if the input
        is already a Param.
    """
    return obj if isinstance(obj, Param) else Param(obj)


def load(filename):
    """\
    Helper function to load a parameter file
    """
    # This avoid circular import
    from .. import io

    filename = os.path.abspath(os.path.expanduser(filename))
    param_dict = io.h5read(filename)
    param = Param(param_dict)
    param.autoviv = False
    return param


def make_default(default_dict_or_file):
    """
    convert description dict to a module dict using a possibly verbose Q & A game
    """
    pass

def param_from_json(filename):
    """
    Convert param tree from JSON file into Param().
    """
    import json
    p = Param()
    with open(filename) as f:
        p.update(json.load(f), Convert=True)
    return p

def param_from_yaml(filename):
    """
    Convert param tree from YAML file into Param().
    """
    import yaml
    p = Param()
    with open(filename) as f:
        p.update(yaml.load(f, Loader=yaml.FullLoader), Convert=True)
    return p