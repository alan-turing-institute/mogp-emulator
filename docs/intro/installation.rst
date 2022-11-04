.. _installation:

Installation
============

In most cases, the easiest way to install ``mogp_emulator`` is via ``pip``, which should install the
library and all of its dependencies: ::

   pip install mogp-emulator

You can also use ``pip`` to install directly from the github repository using ::

   pip install git+https://github.com/alan-turing-institute/mogp-emulator

This will accomplish the same thing as the manual installation instructions
below.
   
Manual Installation
-------------------

To install the package manually, for instance to have access to a development version or to take part in
active development of the package, the following instructions can be used to install the package.

Download
~~~~~~~~

You can download the code as a zipped archive from the Github repository. This will download all files
on the `master` branch, which can be unpacked and then used to install following the instructions
below.

If you prefer to check out the Github repository, you can download the code using: ::

	git clone https://github.com/alan-turing-institute/mogp-emulator/

This will clone the entire git history of the software and check out the ``master`` branch by default.
The ``master`` branch is the most stable version of the code, but will not have all features as the
code is under active development. The ``devel`` branch is the more actively developed branch, and all new
features will be available here as they are completed. All code in the ``devel`` branch is well tested and
documented to the point that results can be trusted, but may still have some minor bugs and issues. Any
other branch is used to develop new features and should be considered untested and experimental. Please
get in touch with one of the team members if you are unsure if a particular feature is
available.

Requirements
~~~~~~~~~~~~

The code requires Python 3.6 or later, and working Numpy, Scipy, and Patsy installations are required. You should
be able to install these packages using ``pip`` if you do not have them already available on your system.
From the base ``mogp_emulator`` directory, you can install all required packages using: ::

   pip install -r requirements.txt

This will install the minimum requirements needed to use ``mogp_emulator``. There are a few addditional
packages that are not required but can be useful. Installation of the optional dependencies can be done via: ::

   pip install -r requirements-optional.txt


Installation
~~~~~~~~~~~~

Then to install the main code, run the following command: ::

   python setup.py install

This will install the main code in the system Python installation. You may need adminstrative priveleges
to install the software itself, depending on your system configuration. However, any updates to the code
cloned through the github repository (particularly if you are using the devel branch, which is under more
active development) will not be reflected in the system installation using this method. If you would like
to always have the most active development version, install using: ::

   python setup.py develop

This will insert symlinks to the repository files into the system Python installation so that files
are updated whenever there are changes to the code files.

Documentation
-------------

The code documentation is available on `readthedocs <https://mogp-emulator.readthedocs.io>`_. A current
build of the ``master`` and ``devel`` branches should always be available in HTML or PDF format.

To build the documentation yourself requires Sphinx, which can be installed using ``pip``. This can also
be done in the ``docs`` directory using ``pip install -r requirements.txt``. To build the documentatation,
change to the ``docs`` directory. There is a Makefile in the `docs` directory to facilitate building the
documentation for you. To build the HTML version, enter the
following into the shell from the ``docs`` directory: ::

   make html

This will build the HTML version of the documentation. A standalone PDF version can be built, which
requires a standard LaTeX installation, via: ::

   make latexpdf

In both cases, the documentation files can be found in the corresponding directories in the ``docs/_build``
directory. Note that if these directories are not found on your system, you may need to create them in
order for the build to finish correctly. A version of the documentation can also be found at the link
above on Read the Docs.

Testing the Installation
------------------------

Unit Tests
~~~~~~~~~~

``mogp_emulator`` includes a full set of unit tests. To run the test suite, you will need to install the
development dependencies, which include ``pytest`` and ``pytest-cov`` to give coverage reports,
which can be done in the main ``mogp_emulator`` directory via ``pip install -r requirements-dev.txt``.
The ``pytest-cov`` package is not required to run the test suite, but is useful if you are developing
the code to determine test coverage.

The tests can be run from the base ``mogp_emulator`` directory or the ``mogp_emulator/tests`` directory
by entering ``pytest``, which will run all tests and print out the results to the console. In the
``mogp_emulator/tests`` directory, there is also a ``Makefile`` that will run the tests for you.
You can simply enter ``make tests`` into the shell.
