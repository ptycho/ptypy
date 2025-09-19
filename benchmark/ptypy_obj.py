import numpy as np
import time

try:
    from pympler.asizeof import asizeof
    use_asizeof = True
except ImportError:
    use_asizeof = False

class Base(object):

    _CHILD_PREFIX = 'ID'
    _PREFIX = 'B'
    
    __slots__ = ['ID','numID','owner','_pool','_recs','_record']
    _fields = [('ID','<S16'),
               ('active', 'b1'),
                # ('dlayer', '<i8'),
                # ('layer', '<i8'), 
                # ('dhigh', '(5,)i8'),
                # ('dlow', '(5,)i8'),
                # ('shape', '(5,)i8'),
                # ('dcoord', '(5,)i8'),
                # ('psize', '(5,)f8'),
                # ('coord', '(5,)f8'),
                ('sp', '(5,)f8')]
    
    def __init__(self, owner=None, ID=None, BeOwner=True):
        """
        Ptypy Base class to support some kind of hierarchy,
        conversion to and from dictionaries as well as a 'cross-node' ID
        management of python objects

        Parameters
        ----------
        owner : Other subclass of Base or Base
            Owner gives IDs to other ptypy objects that refer to it
            as owner. Owner also keeps a reference to these objects in
            its internal _pool where objects are key-sorted according
            to their ID prefix

        ID : None, str or int

        BeOwner : bool
            Set to `False` if this instance is not intended to own other
            ptypy objects.
        """
        self.owner = owner
        self.ID = ID
        self.numID = None
        
        # Try register yourself to your owner if it exists
        if isinstance(owner,Base):
            owner._new_ptypy_object(obj=self)
        else:
            self._record = None
            print(
                'Failed registering instance of %s with ID %s to object %s'
                % (type(self), self.ID, owner))
        
        # Make a pool for your own ptypy objects
        self._pool = {} if BeOwner else None
        self._recs = {} if BeOwner else None

    def _new_ptypy_object(self, obj):
        """
        Registers a new ptypy object into this object's pool and records.

        Parameters:
        -----------
        obj : [any object] or None
            The object to register.
        """
        try:
            prefix = obj._PREFIX
        except:
            prefix = self._CHILD_PREFIX

        if self._pool.get(prefix) is None:
            self._pool[prefix] = {}
            self._recs[prefix] = np.zeros((8,),dtype=obj.__class__._fields)
            
        d = self._pool[prefix]
        # Check if ID is already taken and assign a new one
        ID = obj.ID
        try:
            if str(ID) == ID:
                nID = prefix + ID
            else:
                nID = prefix + self._num_to_id(ID)
            if nID in d:
                print('Overwriting ID %s in pool of %s'
                                % (nID, self.ID))
        except:
            idx = len(d)
            nID = prefix + self._num_to_id(idx)
            while nID in d:
                idx += 1
                nID = prefix + self._num_to_id(idx)
            
        d[nID] = obj
        obj.ID = nID
        idx = len(d)
        obj.numID = idx
        recs = self._recs[prefix]
        l = len(recs)
        if idx >= l:
            nl = l + 8192 if idx > 10000 else 2*l
            recs = np.resize(recs,(nl,))
            self._recs[prefix] = recs
        rec = recs[idx] 
        obj._record = rec
        rec['ID'] = nID
        
        return
        
    @staticmethod
    def _num_to_id(num):
        """
        maybe more sophisticated in future
        """
        return '%04d' % num

    def calc_mem_usage(self):
        space = 64   # that is for the class itself
        pool_space = 0
        npy_space = 0
        if hasattr(self, '_pool') and self._pool is not None:
            if use_asizeof:
                space += asizeof(self._pool, limit=0)
            for k, v in self._pool.items():
                if use_asizeof:
                    space += asizeof(v, limit=0)
                for kk, vv in v.items():
                    pool_space += vv.calc_mem_usage()[0]
        
        if hasattr(self, '__dict__'):
            for k, v in self.__dict__.items():
                if issubclass(type(v), Base):
                    continue
                elif str(k) == '_pool' or str(k) == 'pods':
                    continue
                else:
                    if use_asizeof:
                        s = asizeof(v)
                        space += s
                    if type(v) is np.ndarray:
                        npy_space += v.nbytes

        return space + pool_space + npy_space, pool_space, npy_space

chunk=8192
N = 100
B=Base()
for i in range(N):
    t0=time.time()
    for b in range(chunk):
        h = Base(B, BeOwner=False)
    dt=time.time()-t0
    print(h.ID, B.calc_mem_usage(), dt)
