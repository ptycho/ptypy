
=====================
Contributing to PTYPY
=====================

1. Introduction
---------------
     
If you are reading this, it might mean that you are considering improving |ptypy| with your contributions. Awesome! We welcome all improvement to the code, and we are always happy to discuss scientific collaborations.

Because we are not professional maintainers, we need your help for code additions and modifictaions to be done in a proper way. So here's a quick description of how we would like you to proceed.

2. Code organisation
--------------------

Folder structure
^^^^^^^^^^^^^^^^

|ptypy| is a python package. Its source code is completely contained in the ``/ptypy/`` directory. Other directories contain additional resources (``/resources/``), executable scripts (``/scripts/``), tutorials (``/tutorials/``) and some example reconstruction scripts that should be used as a starting point for any reconstruction (``/templates/``).

Coding standards
^^^^^^^^^^^^^^^^

Please ensure you satisfy most of PEP8_ recommendations. We are not dogmatic about this, but following the same standards as others makes reading code much easier. For scientific code we consider it OK to break a few of the rules, like line length if what is written maps to a well-defined mathematical operation, or the use of single character variables for a more direct mapping to equations.

Branches
^^^^^^^^

The important permanent branches are:
 - ``master``: the current cutting-edge but functional package.
 - ``stable``: the latest release, recommended for production use.
 - ``target``: target for a next release. This branch should stay up-to-date with ``master``, and contain planned updates that will break compatibility with the current version.
 - other thematic and temporary branches will appear and disappear as new ideas are tried out and merged in.


Development cycle
^^^^^^^^^^^^^^^^^

There has been only two releases of the code up to now, so what we can tell about the *normal development cycle* for |ptypy| is rather limited. However the plan is as follows:
 - Normal development usually happens on thematic branches. These branches are merged back to master when it is clear that (1) the feature is sufficiently debugged and tested and (2) no current functionality will break.
 - At regular interval admins will decide to freeze the development for a new stable release. During this period, development will be allowed only on feature branches but master will accept only bug fixes. Once the stable release is done, development will continue.


3. Pull requests
----------------

Most likely you are a member of the |ptypy| team, which give you access to the full repository, but no right to commit changes. The proper way of doing this is *pull requests*. You can read about how this is done on github's `pull requests tutorial`_.

Pull requests can be made against one of the feature branches, or against ``target`` or ``master``. In the latter cases, if your changes are deemed a bit too substantial, the first thing we will do is create a feature branch for your commits, and we will let it live for a little while, making sure that it is all fine. We will then merge it onto ``master`` (or ``target``).

In principle bug fixes can be requested on the ``stable`` branch. 

3. Direct commits
-----------------

If you are one of our power-users (or power-developers), you can be given rights to commit directly to |ptypy|. This makes things much simpler of course, but with great power comes great responsibility.

To make sure that things are done cleanly, we encourage all the core developers to create thematic remote branches instead of committing always onto master. Merging these thematic branches will be done as a collective decision during one of the regular admin meetings.


.. |ptypy| replace:: PtyPy


.. _PEP8: https://www.python.org/dev/peps/pep-0008/

.. _`pull requests tutorial`: https://help.github.com/articles/using-pull-requests/