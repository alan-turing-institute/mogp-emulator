.. _MetaNotation:

Meta-pages: Notation
====================

This page sets out the notational conventions used in the
:ref:`MUCM<DefMUCM>` toolkit.

General Conventions
-------------------

Vectors and matrices
~~~~~~~~~~~~~~~~~~~~

Vectors in the toolkit can be either row (\(1\times n`) or column
(\(n\times 1`) vectors. When not specified explicitly, a vector is a
column vector by default.

We do not identify vectors or matrices by bold face.

Functions
~~~~~~~~~

Functions are denoted with a dot argument, e.g. :math:`f(\cdot)`, while
their value at a point :math:`x` is :math:`f(x)`.

The following compact notation is used for arrays of function values.
Suppose that :math:`g(\cdot)` is a function taking values :math:`x` in some
space, and let :math:`D` be a vector with :math:`n` elements :math:`D =
\\{x_1,x_2, \\ldots, x_n\}` in that space; then :math:`g(D)` is made up of
the function values :math:`g(x_1),g(x_2),\ldots,g(x_n)`. More specifically,
we have the following cases:

-  If :math:`g(x)` is a scalar, then :math:`g(D)` is a :math:`(n\times 1)`
   (column) vector by default, unless explicitly defined as a
   :math:`(1\times n)` (row) vector.
-  If :math:`g(x)` is a :math:`(t\times 1)` column vector, then :math:`g(D)` is a
   matrix of dimension :math:`t\times n`.
-  If :math:`g(x)` is a :math:`(1\times t)` row vector, then :math:`g(D)` is a
   matrix of dimension :math:`(n\times t)`.
-  If :math:`g(x,x')` is a scalar, then :math:`g(D,x')` is a (\(n\times 1`)
   column vector, :math:`g(x,D')` is a (\(1\times n`) row vector and
   :math:`g(D,D')` is a :math:`(n\times n)` matrix.

Other notation
~~~~~~~~~~~~~~

The :math:`\,^*` superscript denotes a posterior value or function.

The matrix/vector transpose is denoted with a roman superscript
:math:`\,^{\textrm{T}}`.

Expectations, Variances and Covariances are denoted as
:math:`\textrm{E}[\cdot], \\textrm{Var}[\cdot],
\\textrm{Cov}[\cdot,\cdot]`.

The trace of a matrix is denoted as :math:`\textrm{tr}`.

Reserved symbols
----------------

The following is a list of symbols that represent fundamental quantities
across the toolkit. These *reserved* symbols will always represent
quantities of the same generic type in the toolkit. For instance, the
symbol :math::ref:`n` will always denote a number of `design<DefDesign>`
points. Where two or more different designs are considered in a toolkit
page, their sizes will be distinguished by subscripts or superscripts,
e.g. :math::ref:`n_{t}` might be the size of a `training
sample<DefTrainingSample>` design, while :math::ref:`n_v` is the size of
a `validation<DefValidation>` design. Notation should always be
defined properly in toolkit pages, but the use of reserved symbols has
mnemonic value to assist the reader in remembering the meanings of
different symbols.

The reserved symbols comprise a relatively small set of symbols (and
note that if only a lower-case symbol is reserved the corresponding
upper-case symbol is not). Non-reserved symbols have no special meanings
in the toolkit.

`Symbol </foswiki/bin/rest/PublishPlugin/publish?validation_key=84284d726120ad351dbbb99d93422a85;googlefile=;defaultpage=;relativeurl=/;destinationftpserver=;destinationftppath=;destinationftpusername=;destinationftppassword=;fastupload=on;extras=;sortcol=0;table=61;up=0#sorted_table>`__

`Meaning </foswiki/bin/rest/PublishPlugin/publish?validation_key=84284d726120ad351dbbb99d93422a85;googlefile=;defaultpage=;relativeurl=/;destinationftpserver=;destinationftppath=;destinationftpusername=;destinationftppassword=;fastupload=on;extras=;sortcol=1;table=61;up=0#sorted_table>`__

Dimensions

:math:`n`

Number of :ref:`design<DefDesign>` points

:math:`p`

Number of :ref:`active inputs<DefActiveInput>`

:math:`q`

Number of :ref:`basis functions<DefBasisFunctions>`

:math:`r`

Number of outputs

:math:`s`

Number of :ref:`hyperparameter<DefHyperparameter>` sets in an
:ref:`emulator<DefEmulator>`

Input - Output

:math:`x`

Point in the :ref:`simulator<DefSimulator>`'s input space

:math:`y`

Reality - the actual system value

:math:`z`

Observation of reality :math:`y`

:math:`D`

Design, comprising an ordered set of points in an input space

:math:`d(\cdot)`

Model discrepancy function

:math:`f(\cdot)`

The output(s) of a simulator

:math:`h(\cdot)`

Vector of basis functions

Hyperparameters

:math:`\beta`

:ref:`Hyperparameters<DefHyperparameter>` of a
:ref:`mean<AltMeanFunction>` function

:math:`\delta`

Hyperparameters of a :ref:`correlation<AltCorrelationFunction>`
function

:math:`\sigma^2`

Scale hyperparameter for a :ref:`covariance<DiscCovarianceFunction>`
function

:math:`\theta`

Collection of hyperparameters on which the emulator is conditioned

:math:`\nu`

:ref:`Nugget<DefNugget>`

:math:`\pi`

:ref:`Distribution of hyperparameters<AltGPPriors>`

Statistics

:math:`m(\cdot)`

Mean function

:math:`v(\cdot,\cdot)`

Covariance function

:math:`m^*(\cdot)`

Emulator's posterior mean, conditioned on the hyperparameters and design
points

:math:`v^*(\cdot,\cdot)`

Emulator's posterior covariance, conditioned on the hyperparameters and
design points

:math:`c(\cdot,\cdot)`

Correlation function
