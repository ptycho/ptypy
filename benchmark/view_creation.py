from ptypy.core.classes import View, Container, Storage, Base, POD, VIEW_PREFIX
import time
import profile, cProfile
C1 = Container(data_type='real')
S1 = C1.new_storage(shape=(1, 7, 7))
nviews = 10000
def make_views(view_cls):
    for k in range(int(nviews)):
        view_cls(C1, 
                 ID=None,#"V%04d" %k, 
                 storageID = S1.ID, 
                 psize = None, 
                 shape = (4, 4), 
                 coord = 0.,
                 active=False)
#profile.run("make_views()")
# t0 = time.time()
# make_views(View)
# dt = time.time() - t0
# print(f"Total: {int(dt)} seconds for {nviews} views, {dt/nviews} seconds per view")

class Test:
    def __init__(self):
        pass
t0 = time.time()
for k in range(int(nviews)):
    Test()
dt = time.time() - t0
print(f"Total: {int(dt)} seconds for {nviews} dummy instances, {dt/nviews} seconds per instance")
B = Base()
t0 = time.time()
for k in range(int(nviews)):
    Base(B, None, False)
dt = time.time() - t0
print(f"Total: {int(dt)} seconds for {nviews} Base instances, {dt/nviews} seconds per instance")

class FastView(View):
    def __init__(self, container, 
                 ID=None, active=True, storageID=None, coord=None,
                 psize=1.0,shape=None, layer=0):
                # Prepare a dictionary for PODs (volatile!)
        
        super(View, self).__init__(container, ID, False)

        self._pods = None 
        r""" Potential volatile dictionary for all :any:`POD`\ s that 
            connect to this view. Set by :any:`POD` """

        # A single pod lookup (weak reference), set by POD instance.
        self._pod = None

        self.active = True
        """ Active state. If False this view will be ignored when
            resizing the data buffer of the associated :any:`Storage`."""

        #: The :py:class:`Storage` instance that this view applies to by default.
        self.storage = None

        self.storageID = None
        """ The storage ID that this view will be forward to if applied
            to a :any:`Container`."""

        #: The "layer" i.e. first axis index in Storage data buffer
        self.dlayer = 0

        self.active = active

        self.storageID = storageID

        # shape == None means "full frame"
        self.shape = shape

        # Look for storage, create one if necessary
        s = self.owner.storages.get(self.storageID, None)
        if s is None:
            sh = (1,) + tuple(self.shape) if self.shape is not None else None
            s = self.owner.new_storage(ID=self.storageID,
                                       psize=psize,
                                       origin=coord,
                                       shape=sh)
        self.storage = s


        if self.shape is None:
            self._set_full_frame(s)

        # Information to access the slice within the storage buffer
        self.psize = psize
        self.coord = coord
        self.layer = layer

        # This ensures self-consistency (sets pixel coordinate and ROI)
        if self.active:
            self.storage.update_views(self)


t0 = time.time()
make_views(FastView)
dt = time.time() - t0
print(f"Total: {int(dt)} seconds for {nviews} views, {dt/nviews} seconds per view")