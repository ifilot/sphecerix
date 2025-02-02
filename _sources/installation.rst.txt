.. _installation:
.. index:: Installation

Installation
============

.. tip::
    For Windows users with relatively little experience with Python, we warmly
    recommend to use the `Anaconda distribution <https://www.anaconda.com/products/distribution>`_.
    Anaconda is an all-in-one package containing the Python compiler,
    an integrated desktop environment (Spyder) and plenty of useful Python
    packages such as numpy and matplotlib.

:program:`Sphecerix` is distributed via both Anaconda package as well as PyPI. For
Windows, it is recommended to install :program:`Sphecerix` via Anaconda, while
for Linux, we recommend to use PyPI.

Windows / Anaconda
------------------

To install :program:`Sphecerix` under Windows, open an Anaconda Prompt window
and run::

    conda install -c ifilot sphecerix

.. note::
    Sometimes Anaconda is unable to resolve the package dependencies. This can
    be caused by a broken environment. An easy solution is to create a new
    environment. See the "Troubleshooting" section at the end of this page
    for more information.

Linux / PyPI
------------

To install :program:`Sphecerix` systemwide, run::

    sudo pip install sphecerix

or to install :program:`Sphecerix` only for the current user, run::

    pip install sphecerix
