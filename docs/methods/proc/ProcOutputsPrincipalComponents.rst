.. _ProcOutputsPrincipalComponents:

Procedure: Principal components and related transformations of simulator outputs
================================================================================

Description and Background
--------------------------

One of the methods for :ref:`emulating<DefEmulator>` several outputs
from a :ref:`simulator<DefSimulator>` considered in the alternatives
page on approaches to emulating multiple outputs
(:ref:`AltMultipleOutputsApproach<AltMultipleOutputsApproach>`) is to
transform the outputs into a set of 'latent outputs' that can be
considered uncorrelated, and then to emulate the latent outputs
separately. We consider here various linear transformations for this
purpose, although probably the most useful is the principal components
transformation.

There are two steps in the procedure.

#. Obtain an :math:`r\times r` variance matrix :math:`V` for the :math:`r`
   outputs.
#. Derive an :math:`r\times r` transformation matrix :math:`P` and apply the
   transformation.

We consider each of these steps in outline before describing the
procedure itself.

The variance matrix :math:`V` for the outputs is unknown and must be
estimated. The basic source for such estimation is the :ref:`training
sample<DefTrainingSample>`, comprising output vectors from the
simulator at :math:`n` points in the space of possible input
configurations.

It is simple to compute the sample variance matrix of these vectors, but
sample covariances and correlations measure correlation between the
deviations of pairs of outputs from their sample means, whereas for
:math:`V` we require correlation in deviations from the mean *functions*
:math:`m_u(\cdot) (u=1,2,\ldots,r)` of the :math:`r` outputs. The procedure
below therefore involves first estimating these mean functions.

For the second step, there are various ways to derive a linear
transformation for a row vector of random quantities :math:`w` with
variance matrix :math:`V` so that the elements of the transformed row
vector :math:`w^\star = wP^{\textrm T}` are uncorrelated. In general, the
covariance matrix of :math:`w^\star` is :math:`B=PVP^{\textrm T}`, and we seek
a matrix :math:`P` for which this becomes diagonal. Then the elements of
:math:`w^\star` are uncorrelated with variances equal to the diagonal
elements of :math:`B`.

Probably the most useful for transforming simulator outputs is the
principal components transformation. Here we let :math:`P` be the matrix of
eigenvectors of :math:`V`, whereupon :math:`B` is the diagonal matrix with the
corresponding eigenvalues of :math:`V` down the diagonal.

Another class of transformations arise by letting :math:`P=S^{-1}`, where
:math:`S` is a square root of :math:`V` in the sense that :math:`V=SS^{\textrm
T}`. Then :math:`B=I`, the :math:`r\times r` identity matrix. One square root
matrix that is easy to calculate (and to invert to get :math:`P`) is the
Cholesky square root; see also the procedure page on Pivoted Cholesky
decomposition (:ref:`ProcPivotedCholesky<ProcPivotedCholesky>`).
However, it is easy to see that if :math:`S` is a square root of :math:`V`
then so is :math:`RS` for any orthogonal matrix :math:`R`.

Inputs
------

-  The (\(r\times 1`) row vectors :math:`f(x_i)` (for :math:`i=1,2,\ldots n`)
   of outputs from the simulator at :math:`n` design points
   :math:`D=\{x_1,x_2,\ldots,x_n\}`.

Outputs
-------

-  Transformation matrix :math:`P`.
-  Transformed 'latent' output vectors :math:`f^\star(x_i)` (for
   :math:`i=1,2,\ldots n`) at the :math:`n` design points.

Procedure
---------

Derive :math:`V`
~~~~~~~~~~~~~

#. Choose a set of :ref:`basis functions<DefBasisFunctions>` in the
   form of a :math:`q\times 1` vector function :math:`h(\cdot)`. The same set
   of basis functions should be used for all the outputs, so if a
   certain basis function is thought *a priori* to be appropriate to
   model the expected behaviour of any one of the outputs (as a function
   of the inputs) then this should be included in :math:`h(\cdot)`.
#. For each :math:`u=1,2,\ldots,r`, construct the :math:`n\times 1` vector
   :math:`F_u` of values of output :math:`u`. Thus, :math:`F_u` is the :math:`u`-th
   column of the matrix :math::ref:`f(D)`. (See the toolkit notation page
   (`MetaNotation<MetaNotation>`) for the use of :math:`D` as a
   function argument, as in :math:`f(D)`.)
#. For each :math:`u`, fit a conventional linear regression model to
   vector\(F_u`, with :math:`h(D)` as the usual :math:`n\times q` X-matrix.
   Let the :math:`q\times 1` vector :math:`\hat\beta_u` be the fitted (i.e.
   estimated) regression coefficients.
#. For each :math:`u`, form the :math:`n\times 1` vector of residuals
   :math:`E_u=F_u-h(D)^{\textrm T}\hat\beta_u` and form these columns into
   the :math:`n\times r` residual matrix :math:`E`.
#. Set :math:`V=n^{-1}E^{\textrm T}E`.

The fitting of linear regression models is available in all the major
statistical software packages and in many programming languages. An
alternative to fitting the conventional linear regression model would be
to build an emulator for each output using the core thread that treats
the core problem using Gaussian methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) and to define :math:`\hat\beta` as
in the procedure page for building a Gaussian process emulator
(:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`); however, although this
might in principle be better it is unlikely in practice to make an
appreciable difference or to be worth the extra effort.

Derive transformation
~~~~~~~~~~~~~~~~~~~~~

Having obtained :math:`V`, the transformation matrix :math:`P` and the
diagonal variance matrix :math:`B` can be obtained using standard software
to apply the relevant decomposition method. For instance, software for
eigenvalue decomposition is available in numerous computing packages
(e.g. Matlab), the Cholesky decomposition is almost as widely available
and the pivoted Cholesky decomposition is detailed in
:ref:`ProcPivotedCholesky<ProcPivotedCholesky>`.

The latent outputs are then characterised by the (row) vectors
:math:`f^\star(x_i)=f(x_i)P^{\textrm T}`, for :math:`i=1,2,\ldots,n`.
Equivalently, :math:`f^\star(D)=f(D)P^{\textrm T}`.

Additional Comments
-------------------

Transformations are a very general idea, and there are related uses in
the toolkit. The procedure page on transforming outputs
(:ref:`ProcOutputTransformation<ProcOutputTransformation>`) concerns
transforming individual outputs (generally in a nonlinear way),
motivated principally by making the assumption of a Gaussian process
more appropriate.

Principal components and related transformations, particularly the
pivoted Cholesky decomposition, are used in
:ref:`validation<DefValidation>`; see the procedure page on
validating a Gaussian process emulator
(:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`). The variance matrix
:math:`V` is in that case obtained directly from the emulator as the
predictive variance matrix of the validation sample.

If the outputs are strongly correlated, then it might be possible to
reduce the number of outputs, for example by retaining only the first
:math:`r^\star < r` eigenvectors of the :math:`\strut P` matrix with the
largest eigenvalues, although in this case it is necessary to include an
additional nugget effect as discussed in the alternatives page on
emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`).
