

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

