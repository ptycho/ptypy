Writing your own engine
=======================

This tutorial aims at providing you with the information you need to build 
your own reconstruction engine from scratch. First, Lets have a look 
how the :any:`Ptycho` class expects an *Engine* to work. 

In :any:`Ptycho.init_engines` we find that an Engine is constructed
with the Ptycho instance and a Parameter set.

.. literalinclude:: ../../ptypy/core/ptycho.py
   :language: python
   :linenos:
   :lineno-start: 243
   :pyobject: Ptycho.init_engines
   :emphasize-lines: 26
   :dedent: 4

In :any:`Ptycho.run` we find that three methods are called:

.. literalinclude:: ../../ptypy/core/ptycho.py
   :language: python
   :linenos:
   :lineno-start: 307
   :lines: 305-330
   :emphasize-lines: 8,22,25
   :dedent: 4
   
Hence an *Engine* class should consist of at least the following methods

- Engine. *__init__(self,ptycho, parameters)*
- Engine. *initialize()*
- Engine. *prepare()*
- Engine. *iterate()*

Of course the :any:`BaseEngine` class of ptypy has these methods, but we 
save the art of subclassing BaseEngine for a later time. Let us start with
a clean slate.


.. literalinclude:: ../../ptypy/engines/DM_minimal.py
   :language: python
   :lines: 10-144
