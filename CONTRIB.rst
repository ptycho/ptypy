
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

Testing
^^^^^^^

All tests are in the (``/test/``) folder and our CI pipeline runs these test for every commit (?). Please note that tests that require GPUs are disabled for the CI pipeline. Make sure to supply tests for new code or drastic changes to the existing code base. Smaller commits or bug fixes don't require an extra test.

Branches
^^^^^^^^

We are following the Gitflow https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow development model where a development branch (``dev``) is merged into the master branch for every release. Individual features are developed on topic branches from the development branch and squash-merged back into it when the feature is mature

The important permanent branches are:
 - ``master``: (protected) the current release plus bugfixes / hotpatches.
 - ``dev``: (protected) current branch for all developments. Features are branched this branch and merged back into it upon completion.


Development cycle
^^^^^^^^^^^^^^^^^

|ptypy| does not follow a rigid release schedule. Releases are prepared for major event or when a set of features have reached maturity.

 - Normal development usually happens on thematic branches from the ``dev`` branch. These branches are merged back to ``dev`` when it is clear that (1) the feature is sufficiently debugged and tested and (2) no current functionality will break.
 - For a release the dev branch will be merged back into master and that merge tagged as a release.


3. Pull requests
----------------

Most likely you are a member of the |ptypy| team, which give you access to the full repository, but no right to commit changes. The proper way of doing this is *pull requests*. You can read about how this is done on github's `pull requests tutorial`_.

Pull requests shall be made against one of the feature branches, or against ``dev`` or ``master``. For PRs against master we will only accept bugifxes or smaller changes. Every other PR should be made against ``dev``. Your PR will be reviewed and discussed anmongst the core developer team. The more you touch core libraries, the more scrutiny your PR will face. However, we created two folders in the main source folder where you have mmore freedom to try out things. For example, if you want to provide a new reconstruction engine, place it into the ``custom/`` folder. A new ``PtyScan`` subclass that prepares data from your experiment is best placed in the ``experiment/`` folder.

If you develop a new feature on a topic branch, it is your responsibility to keep it current with dev branch to avoid merge conflicts. 


.. |ptypy| replace:: PtyPy


.. _PEP8: https://www.python.org/dev/peps/pep-0008/

.. _`pull requests tutorial`: https://help.github.com/articles/using-pull-requests/
