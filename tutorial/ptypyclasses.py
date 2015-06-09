# .. _ptypyclasses:

# Tutorial: Data access abstraction - Storage, View, Container
# ============================================================

# First a hort reminder from the classes Module about the classes
# this tutorial covers
 
"""
This module defines flexible containers for the various quantities needed
for ptychographic reconstructions.

**Container class**
    A high-level container that keeps track of sub-containers (Storage)
    and Views onto them. A container can copy itself to produce a buffer
    needed for calculations. Mathematical operations are not implemented at
    this level. Operations on arrays should be done using the Views, which
    simply return numpyarrays.


**Storage class**
    The sub-container, wrapping a numpy array buffer. A Storage defines a
    system of coordinate (for now only a scaled translation of the pixel
    coordinates, but more complicated affine transformation could be
    implemented if needed). The sub-class DynamicStorage can adapt the size
    of its buffer (cropping and/or padding) depending on the Views.


**View class**
    A low-weight class that contains all information to access a 2D piece
    of a Storage within a Container.

"""
# Import some modules
import matplotlib as mpl
import numpy as np
import ptypy
from ptypy import utils as u
from ptypy.core import View,Container,Storage,Base
plt = mpl.pyplot

# A single Storage in one Container
# ---------------------------------

# As master class we create a :any:`Container` instance
C1=Container(data_type='real')

# This class itself holds nothing at first. In order to store data in 
# ``C1`` we have to add a :any:`Storage` to that container.
S1=C1.new_storage(shape=(1,7,7))

# Since we haven't specified an ID the Container class picks one for ``S1``
# In this case that will be ``S0000`` where the *S* refers to the class type.
print S1.ID

# We can have a look now what kind of data Storage holds. 
print S1.formatted_report()[0]

# Apart from the ID on the left we discover a few other entries, for
# example the quantity ``psize`` which refers to the physical pixelsize 
# for the last two dimensions of the stored data.
print S1.psize

# Many attributes of a :any:`Storage` are in fact *properties*. Changing
# their value may have an impact on other methods or attributes of the
# class. For example. One convenient method is Storage.\ :py:meth:`~ptypy.core.classes.Storage.grids`
# that creates grids for the last two dimensions (see also
# :py:func:`ptypy.utils.array_utils.grids`)
y,x = S1.grids()
print y
print x

# These are cooridinate grids for vertical and horizontal axes respectively
# We also see that these coordinates have their center at::
print S1.center

# So now we change a few properties. For example,
S1.center = (2,2)
S1.psize = 0.1
g = S1.grids()
print g[0]
print g[1]

# We see that the center has moved one pixel up and one down. If we want 
# to use a physical quantity for the center, we may also set the top left
# pixel to a new value, which shifts the center to a new position.
S1.origin -= 0.12
y,x = S1.grids()
print y
print x
print S1.center

# Up until now our actual *data* numpy array located at ``S1.data`` is 
# still filled with the not so exciting ones. We can use 
# :any:`Storage.fill` to fill that container
S1.fill(x+y)
print S1.data

# We can also plot the data using 
# :py:func:`~ptypy.utils.plot_utils.plot_storage` 
fig = u.plot_storage(S1,0)
fig.savefig('ptypyclasses_%d.png' % fig.number, dpi=300)
# This is a test of a figure plot

# Adding Views as a way to access data
# ------------------------------------

# Besides being able to access the data directly through its attribute
# and the corresponding *numpy* syntax, ptypy offers acces through a
# :any:`View` instance. The View invocation is a bit more complex.
from ptypy.core.classes import DEFAULT_ACCESSRULE
ar = DEFAULT_ACCESSRULE.copy()
print ar

# Now let's say we want a 4x4 view on Storage ``S1`` around the origin.
# We set
ar.shape = (4,4)  # ar.shape = 4 would have been also valid
ar.coord = 0.      # ar.coord = (0.,0.)
ar.storageID = S1.ID
ar.psize = None

# Now we can construct the View. The last step in this process is an 
# update of the View by the Storage ``S1`` which transfers data
# data ranges/coordinates to the View.
V1 = View(C1, ID=None, accessrule = ar)

# We see that a number of the accessrule items appear in the View now.
print V1.shape
print V1.coord
print V1.storageID

# A few other were set by the automatic update of Storage
print V1.psize
print V1.storage

# The update also set new attributes of the View that start with 
# a lower 'd' and are locally stored information about data access. 
print V1.dlayer, V1.dlow, V1.dhigh

# Finally, we can retrieve the data subset by applying the View to the storage.
data = S1[V1]
print data

# It does not matter if we apply the View to Storage ``S1`` or the 
# container ``C1``, or use the View internal 
# View.\ :py:meth:`~ptypy.core.classes.View.data` property.
print np.allclose(data,C1[V1])
print np.allclose(data,V1.data)

# The first access yielded a similar result because the 
# :py:attr:`~ptypy.core.classes.View.storageID` is in ``C1`` and the
# second acces method worked because it uses the View's 
# :py:attr:`~ptypy.core.classes.View.storage` attribute
print V1.storage is S1
print V1.storageID in C1.S.keys()

# We observe that the coordinate [0.0,0.0] is not part of the grid
# in S1 anymore. Consequently, the View was put as close to [0.0,0.0]
# as possible. The coordinate in data space, that the View would have as
# center is the attribute :py:meth:`~ptypy.core.classes.View.pcoord` while
# :py:meth:`~ptypy.core.classes.View.dcoord` is the closest data coordinate
# The difference is held by :py:meth:`~ptypy.core.classes.View.sp` such 
# that a subpixel correction may be applied if needed (future release)
print V1.dcoord, V1.pcoord, V1.sp

# .. note::
#    Please note that we cannot guarantee any API stability for other 
#    attributes / properties besides *.data*, *.shape* and *.coord*

# If we set the coordinate to some other value in the grid, we can eliminate
# the subpixel misfit. By changing the *.coord* property, we need to
# update the View manually, as the View-Storage interaction is non-automatic
# apart from the View construction - a measure of caution.
V1.coord = (0.08,0.08)
S1.update_views(V1)
print V1.dcoord, V1.pcoord, V1.sp 

# Oh we see that the high range limit of the View is close to the border 
# of the data buffer... so what happens if we push the coordinate further?
print V1.dhigh
V1.coord = (0.28,0.28)
S1.update_views(V1)
print V1.dhigh

# Now the higher range limit of the View is certianly off bounds.
# Applying this View to the Storage can lead to undesired behavior, i.e.
# concatenation or data access errors.
print S1[V1]
print S1[V1].shape , V1.shape

# One important feature of the :any:`Storage` class is that it can detect
# all out-of-bounds accesses and reformat the data buffer accordingly.
# A simple call to 
# *Storage*.\ :py:meth:`~ptypy.core.classes.Storage.reformat` should do. 
print S1.shape
mn = S1[V1].mean()
S1.fill_value = mn
S1.reformat()
print S1.shape

# Oh no, the Storage data buffer has shrunk! .. Don't worry. That is
# intended behavior. A call to *.reformat()* crops and pads the data 
# buffer around all **active** Views. 
# You need to set
S1.padonly = True

# if you want to avoid that the data buffer is cropped. We leave this
# as an exercise to the user. Instead, we add a new View at different 
# location to verify that the buffer will try to reach both Views.
ar2 = ar.copy()
ar2.coord = (-0.82,-0.82)
V2 = View(C1, ID=None, accessrule = ar2)
S1.fill_value = 0.
S1.reformat()
print S1.shape

# Ok we see that the the buffer has grown in size. Now we give the new
# View a copied values of the other view for a nice figure
V2.data = V1.data.copy()
fig = u.plot_storage(S1,2)
fig.savefig('ptypyclasses_%d.png' % fig.number, dpi=300)
# Storage with 4x4 views of the same content.

# We observe that the data buffer spans both views.
# Now let us add more....
for i in range(1,11):
    ar2 = ar.copy()
    ar2.coord = (-0.82+i*0.1,-0.82+i*0.1)
    View(C1, ID=None, accessrule = ar2)

# A handy method of the :any:`Storage` class is that it can determine
# its own coverage by views.
S1.data[:] = S1.get_view_coverage()
fig = u.plot_storage(S1,3)
fig.savefig('ptypyclasses_%d.png' % fig.number, dpi=300)
# View coverage in data buffer of ``S1``.

# Another handy feature of the :any:`View` class is that it automatically
# create a Storage instance to the ``storageID`` if it does not already
# exist.
ar = DEFAULT_ACCESSRULE.copy()
ar.shape = 200
ar.coord = 0.      
ar.storageID = 'S100'
ar.psize = 1.0
V3=View(C1,ID=None,accessrule = ar)

# Finally we have a look at the mischief we managed so far.
print C1.formatted_report()


