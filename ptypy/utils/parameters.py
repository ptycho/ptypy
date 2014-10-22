# -*- coding: utf-8 -*-
"""\
Parameter definition.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import os
from .. import io

__all__ = ['Param', 'asParam'] # 'load',]

PARAM_PREFIX = 'pars'

class Param_legacy(object):
    """Parameter container
    
    In normal use case, this class is not meant to be instantiated by the user.
    See FSParamFactory and NetworkParamFactory.
    
    By default the class has "autovivification" disabled. When enabled, access to any
    unknown attribute results in its automatic creation (an instance of Param). As a result,
    this code is legal:
        
    >>> p = Param(autoviv=True)
    >>> p.x = 1
    >>> p.y.a = 1  # This creates p.y as a Param object
    
    To turn it off, one only needs to do
    >>> p.autoviv = False
    This will be propagated to all Param children as well.
    """

    def __init__(self, old_param=None, autoviv=False, none_if_absent=False, **kwargs):
        """Parameter structure initialization
        
        (old_param can be a Param object or a nested dict).
        """
        self._autoviv = autoviv
        self._none_if_absent = none_if_absent
        if isinstance(old_param, dict):
            self._from_dict(old_param)
        elif isinstance(old_param, type(self)):
            self._from_dict(old_param._to_dict())
        if kwargs:
            self._from_dict(kwargs)
            
    def __getattr__(self, item):
        if self._autoviv:
            if '(' in item: return
            value = type(self)(autoviv=True)
            setattr(self, item, value)
            return value
        elif self._none_if_absent:
            setattr(self, item, None)
            return None
        else:
            raise AttributeError
            
    def _to_dict(self):
        return dict( (k,v._to_dict()) if isinstance(v, type(self)) else (k,v) for k,v in self.__dict__.iteritems() if not k.startswith('_'))

    def _from_dict(self, d):
        for k,v in d.iteritems():
            if isinstance(v, dict):
                setattr(self, k, type(self)(v))
            else:
                setattr(self, k, v)

    def _copy(self):
        return Param(self._to_dict())
        
    # These are properties to make sure that an error is raised if an attempt is made
    # at overwriting them.
    @property
    def keys(self):
        """Access the key method"""
        return self.__dict__.keys

    @property
    def iteritems(self):
        """Access the key method"""
        return self.__dict__.iteritems


    def update(self, *args, **kwargs):
        """Append parameters or dict"""
        if args:
            self._from_dict(args[0])
        if kwargs:
            self._from_dict(kwargs)
        
    @property
    def copy(self):
        "Deep copy"
        return self._copy

    @property
    def none_if_absent(self):
        "Switch for returning None if absent"
        return self._none_if_absent
        
    @none_if_absent.setter
    def none_if_absent(self, value):
        # propagate to children
        self._none_if_absent = value
        self._autoviv = False
        for k,v in self.__dict__.iteritems():
            if isinstance(v, type(self)) and k[0] != '_':
                v.none_if_absent = value
                v._autoviv = False

    @property
    def autoviv(self):
        "Autovivication switch"
        return self._autoviv

    @autoviv.setter
    def autoviv(self, value):
        # propagate to children
        self._autoviv = value
        self._none_if_absent = False
        for k,v in self.__dict__.iteritems():
            if isinstance(v, type(self)) and k[0] != '_':
                v.autoviv = value
                v._none_if_absent = False
        
    def __repr__(self):
        return 'Param(%s)' % repr(self._to_dict())      

#class STDParam(Param):
#    """\
#    Standard parameters.
#    """
#    def __init__(self,**kwargs):
#        Param.__init__(self, **kwargs)
#        self.autoviv = False
#
#    def _validate(self, p=None):
#        """\
#        STDParam._validate(): internal check if all is ok.
#        STDParam._validate(p): checks if the Param object p follows the convention.
#        """
#        if p is None:
#            # Internal check
#            return True
#
#        p = asParam(p)
#        for k,v in p.iteritems():
#            if k.startswith('_'): continue
#            if k not in self.keys():
#                raise ValueError('Parameter %s missing (%s)' % (k, self._format(k)))
#
#
#    def _format(self, key=None):
#        """\
#        Pretty format the parameter for the provided key (all keys if None).
#        """
#        lines = []
#        if key is None:
#            for k in self.keys():
#                lines += [self._format(k)]
#            
#        
#        v = self.__dict__[k]
#        if isinstance(v, Param):

class Param(dict):
    """
    Convenience class: a dictionary that gives access to its keys
    through attributes.
    
    Note: dictionaries stored in this class are also automatically converted
    to Param objects:
    >>> p = Param()
    >>> p.x = {}
    >>> p
    Param({})
    
    While dict(p) returns a dictionary, it is not recursive, so it is better in this case
    to use p.todict(). However, p.todict does not check for infinite recursion. So please
    don't store a dictionary (or a Param) inside itself.
    
    BE: Please note also that the recursive behavior of the update function will create
    new references. This will lead inconsistency if other objects refer to dicts or Params
    in the updated Param instance. 
    """
    _display_items_as_attributes=True
    _PREFIX = PARAM_PREFIX
    
    def __init__(self,__d__=None,**kwargs):
        """
        A Dictionary that enables access to its keys as attributes.
        Same constructor as dict.
        """
        dict.__init__(self)
        if __d__ is not None: self.update(__d__)
        self.update(kwargs)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        # BE: original behavior modified as implicit conversion may destroy references
        # Use update(value,Convert=True) instead
        #return super(Param, self).__setitem__(key, Param(value) if type(value) == dict else value)
        return super(Param, self).__setitem__(key, value) 

    def __getitem__(self, name):
        #item = super(Param, self).__getitem__(name)
        #return Param(item) if type(item) == dict else item
        return super(Param, self).__getitem__(name)

    def __delitem__(self, name):
        return super(Param, self).__delitem__(name)

    def __delattr__(self, name):
        return super(Param, self).__delitem__(name)
        
    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def copy(self,depth=0):
        """
        P.copy() -> A (recursive) copy of P with depth `depth` 
        """
        d = Param(self)
        if depth>0:
            for k,v in d.iteritems():
                if isinstance(v,self.__class__): d[k] = v.copy(depth-1)
        return d     
     

    def __dir__(self):
        """
        Defined to include the keys when using dir(). Useful for
        tab completion in e.g. ipython.
        If you do not wish the dict key's be displayed as attributes
        (although they are still accessible as such) set the class 
        attribute '_display_items_as_attributes' to False. Default is
        True.
        """
        if self._display_items_as_attributes:
            return self.keys()
            #return [item.__dict__.get('name',str(key)) for key,item in self.iteritems()]
        else:
            return []

    def update(self, __d__=None,Convert=False,Replace=True, **kwargs):
        """
        Update Param - almost same behavior as dict.update, except
        that all dictionaries are converted to Param, and update
        is done recursively, in such a way that as little info is lost.
        
        additional Parameters:
        ----------------------
        Convert : bool (False)
                  If True, convert all dict-like values in self also to Param
                  *WARNING* 
                  This mey result in misdirected references in your environment
        Replace : bool (True)
                  If False, values in self are not replaced by but 
                  updated with the new values.
        """
        def _k_v_update(k,v):
            # If an element is itself a dict, convert it to Param
            if Convert and hasattr(v, 'keys'):
                #print 'converting'
                v = Param(v)
            # If this key already exists and is already dict-like, update it
            if not Replace and hasattr(self.get(k, None), 'keys'):
                self[k].update(v)
            # Otherwise just replace it
            else:
                self[k] = v
            
        if __d__ is not None:
            if hasattr(__d__, 'keys'):
                # Iterate through dict-like argument
                for k,v in __d__.iteritems():
                    _k_v_update(k,v)
                    
            else:
                for (k,v) in __d__:
                    _k_v_update(k,v)
                    
        for k,v in kwargs.iteritems():
            _k_v_update(k,v)
            
        return None
            
    def _to_dict(self,Recursive=False):
        """
        Convert to dictionary (recursively if needed).
        """
        if not Recursive:
            return dict(self)
        else:
            d = dict(self)
            for k,v in d.iteritems():
                if isinstance(v,self.__class__): d[k] = v._to_dict(Recursive)
        return d
        
    @classmethod
    def _from_dict(cls,dct):
        """
        Make Param from dict. This is similar to the __init__ call
        but kept here for coherent usage among ptypy.core.Base children
        """
        #p=Param()
        #p.update(dct.copy())
        return Param(dct.copy())
        
def validate_standard_param(sp, p=None, prefix=None):
    """\
    validate_standard_param(sp) checks if sp follows the standard parameter convention.
    validate_standard_param(sp, p) attemps to check if p is a valid implementation of sp.

    NOT VERY SOPHISTICATED FOR NOW!
    """
    if p is None:
        good = True
        for k,v in sp.iteritems():
            if k.startswith('_'): continue
            if type(v) == type(sp):
                pref = k if prefix is None else '.'.join([prefix, k])
                good &= validate_standard_param(v, prefix=pref)
                continue
            else:
                try:
                    a,b,c = v
                    if prefix is not None:
                        print '    %s.%s = %s' % (prefix, k, str(v))
                    else:   
                        print '    %s = %s' % (k, str(v))
                except:
                    good = False
                    if prefix is not None:
                        print '!!! %s.%s = %s <--- Incorrect' % (prefix, k, str(v))
                    else:
                        print '!!! %s = %s <--- Incorrect' % (k, str(v))

        return good
    else:
        raise NotimplementedError('Checking if a param fits with a standard is not yet implemented')

def format_standard_param(p):
    """\
    Pretty-print a Standard Param class.
    """
    lines = []
    if not validate_standard_param(p):
        print 'Standard parameter does not'
    for k,v in p.iteritems():
        if k.startswith('_'): continue
        if type(v) == type(p):
            sublines = format_standard_param(v)
            lines += [k + '.' + s for s in sublines]
        else:
            lines += ['%s = %s #[%s] %s' % (k, str(v[1]),v[0],v[2])] 
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
