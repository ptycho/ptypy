#Tutorial on ptypy classes
#=========================

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
import ptypy
from ptypy import utils as u
from ptypy.core import View,Container,Storage,Base
import matplotlib as mpl
plt = mpl.pyplot

#A single Storage in one Container
#---------------------------------

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
# class. For example. One convenient property is :py:meth:`Storage.grids`
# that creates grids for the last two dimensions (see also) 
# :py:func:`~ptypy.utils.array_utils.grids`
print S1.grids()[0]
print S1.grids()[1]

# These are cooridinate grids for vertical and horizontal axes respectively
# We also see that these coordinates have their center at::
print S1.center

# So now we change a few properties. For example,
S1.center = (2,2)
S1.psize = 0.1
print S1.grids()[0]
print S1.grids()[1]

# We see that the center has moved one pixel up and one down. If we want 
# to use a physical quantity for the center, we may also set the top left
# pixel to a new value, which shifts the center to a new position.
S1.origin -= 0.12
print S1.grids()[0]
print S1.grids()[1]
print S1.center

# Up until now our actual *data* numpy array located at ``S1.data`` is 
# still filled with the not so exciting ones. We can use 
# :any:`Storage.fill` to fill that container
S1.fill(S1.grids()[0]+S1.grids()[1])
print S1.data

# Test
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(S1.data[0])
fig.tight_layout()
plt.show(block=False)

# make fig ;Test figure; This is a test of a figure plot

# Adding View as a way to access data
# -----------------------------------

# Besides being able to access the data directly through its attribute
# and the corresponding *numpy* syntax, ptypy offers acces through a
# :any:`View` instance. The View invocation is a bit more complex.
from ptypy.core.classes import DEFAULT_ACCESSRULE
ar = DEFAULT_ACCESSRULE.copy()
print ar

# Hallo lieber Marco
 

