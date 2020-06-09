.. _AltMultivariateCovarianceStructures:

Alternatives: Choice of covariance function in the multivariate Gaussian process emulator
=========================================================================================

Overview
--------

The covariance function in the multivariate Gaussian process emulator
for a simulator with :math:`r` outputs is the :math:`r \times r` matrix
function :math:`v(.,.)`. The :math:`i,j`th element of :math:`v(x,x^\prime)`
represents the covariance between :math:`f_i(x)` and :math:`f_j(x^\prime)`.
That means that :math:`v(.,.)` must encode two kinds of correlation:
correlation between the outputs, and correlation over the input space.
In order to build these two kinds of correlation into a valid covariance
function (that is, one for which the covariance matrix :math:`v(D,D)` is
positive semi-definite for any design :math:`D`), we must impose some
structure on :math:`v(.,.)`. Here we describe some alternatives for this
structure.

Do covariances matter?
~~~~~~~~~~~~~~~~~~~~~~

Suppose that we are able to emulate all the outputs of interest
accurately using single-output emulators, so that the uncertainty in
each case is very small. Then we will know that the true values of any
two simulator outputs will be very close to the emulator mean values.
Knowing also the covariance will allow us to assess just how probable it
is that the true values for both outputs lie above their respective
emulator means, but when the uncertainty is very small this extra
information is of little importance.

So it may be argued that if we have enough training data to emulate all
outputs of interest very accurately, in the sense that the emulation
uncertainty is small enough for us to be able to ignore it, then we can
also ignore covariances between outputs. It is generally the case that
the more accurately we can emulate the outputs the less importance
attaches to our choice of emulation strategy, and so with large training
samples all alternatives considered here will give essentially the same
results. However, the reality with complex simulators is that we are
rarely in the happy position of being able to make enough training runs
to achieve this kind of accuracy, particularly if the emulator input and
output spaces are high dimensional. When emulator uncertainty is
appreciable, covariances between outputs need to be considered and the
choice between alternative ways of emulating multiple outputs does
matter.

The Alternatives
----------------

Independent outputs
~~~~~~~~~~~~~~~~~~~

If we are willing to treat the outputs as being independent then the
covariance function :math:`v(.,.)` is diagonal, with diagonal elements
:math:`\sigma^2_1c_1(.,.),\ldots,\sigma^2_rc_r(.,.)`, where each output :math:`i`
has its own input space correlation function :math:`c_i(.,.)`. The
hyperparameters in this case are :math:`\omega=(\sigma^2,\delta)`, where
:math:`\sigma^2=(\sigma^2_1,\ldots,\sigma^2_r)` and
:math:`\delta=(\delta_1,\ldots,\delta_r)`, with :math:`\delta_i` being the
correlation length hyperparameters for :math:`c_i(.,.)`. A typical prior
for :math:`\omega=(\sigma^2,\delta)` is of the form

.. math::
   \pi(\sigma^2,\delta) \propto \pi_\delta(\delta) \prod_{i=1}^{r}\sigma^{-2}_i\,,

which expresses weak knowledge about :math:`\sigma^2`. Alternatives for the
correlation length prior, :math:`\pi_\delta(.)`, are discussed in
:ref:`AltGPPriors<AltGPPriors>`.

Separable covariance
~~~~~~~~~~~~~~~~~~~~

A :ref:`separable<DefSeparable>` covariance has the form

.. math::
   v(.,.) = \Sigma c(\cdot,\cdot)\, ,

where :math:`\Sigma` is a :math:`r \times r` covariance matrix between
outputs and :math:`c(.,.)` is a correlation function between input points.
The hyperparameters for the separable covariance are
:math:`\omega=(\Sigma,\delta)`, where :math:`\delta` are the hyperparameters
for :math:`c(.,.)`. A typical prior for :math:`\omega=(\Sigma,\delta)` is of
the form

.. math::
   \pi(\Sigma,\delta) \propto \pi(\delta) |\Sigma|^\frac{k+1}{2}\,,

which expresses weak knowledge about :math:`\Sigma`. This is also discussed
in :ref:`AltMultivariateGPPriors<AltMultivariateGPPriors>` where the
variance parameters are considered together with those for the mean
function and :math:`\delta` is assumed known or fixed. The choice of
:math:`\pi(\delta)`, the arbitrary prior on the correlations lengths, is
discussed in :ref:`AltGPPriors<AltGPPriors>`.

Nonseparable covariance based on convolution methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One way of constructing a Gaussian process is by taking the convolution
of a Gaussian white noise process with an arbitrary :ref:`smoothing
kernel<DefSmoothingKernel>`. This method lends itself to
defining a nonseparable multivariate covariance function, which is often
called a 'convolution covariance function'.

To construct a convolution covariance function :math:`v(.,.)`, we choose a
smoothing kernel :math:`\kappa_i` corresponding to each output :math:`i`. We
also have a positive semidefinite :math:`r \times r` matrix
:math:`\tilde{\Sigma}` with elements :math:`\tilde{\sigma}_{ij}`. Then for
each :math:`i,j \in \{1,...,r\}`, the :math:`i,j`th element of :math:`v(.,.)`
is

.. math::
   v_{ij}(x,x^\prime) = \tilde{\sigma}_{ij} \int_{R^p}
   \kappa_i(u-x)\kappa_j(u-x^\prime)\,du \,.

The smoothing kernels determine the input space correlation function for
each output, and :math:`\tilde{\Sigma}` controls the between-output
covariances. The hyperparameters in the convolution covariance function
are :math:`\omega = (\tilde{\Sigma},\delta)`, where
:math:`\delta=\{\delta_1,...,\delta_r\}` are the parameters in the
smoothing kernels, which will typically correspond to something like
correlation lengths.

A convenient choice of smoothing kernel is

.. math::
   \kappa_i(x)=\left[ \left(\frac{4}{\pi}\right)^{p}\prod_{\ell=1}^p
   \phi_i^{(\ell)} \right] ^{\frac{1}{4}}\exp\{-2x^T\Phi_ix\} \, ,

where each :math:`\Phi_i` is a :math:`p \times p` diagonal matrix with
diagonal elements
:math:`(\phi_i^{(1)},...,\phi_i^{(p)}=((\delta_i^{(1)})^{-2},...,(\delta_i^{(p)})^{-2})`,
for then the covariance function has elements

.. math::
   v_{ij}(x,x^\prime) = \tilde{\Sigma}_{ij} \rho_{ij}
   \exp\{-2(x-x^\prime)^T\Phi_i(\Phi_i+\Phi_j)^{-1}\Phi_j(x-x^\prime)\}\,,

where

.. math::
   \rho_{ij}=2^\frac{p}{2}\prod_{\ell=1}^p
   \left[(\phi_i^{(\ell)}\phi_j^{(\ell)})^\frac{1}{4}
   (\phi_i^{(\ell)}+\phi_j^{(\ell)})^{-\frac{1}{2}}\right]\,.

This means that the covariance function for an individual output :math:`i`
is :math:`v_{ii}(x,x^\prime)
=\sigma_i^2\exp\{-(x-x^\prime)^T\Phi_i(x-x^\prime)\}`. Thus each output
:math:`i` has an input space correlation function of the Gaussian (squared
exponential) form, as described in
:ref:`AltCorrelationFunction<AltCorrelationFunction>`, with
correlation lengths :math:`\delta_i=(\delta_i^{(1)},\ldots,\delta_i^{(p)}) =
((\phi_i^{(1)})^{-1/2},\ldots,(\phi_i^{(p)})^{-1/2})`. A possible choice
of prior for the hyperparameters is

.. math::
   \pi(\tilde{\Sigma},\delta) \propto
   \pi(\delta)|\tilde{\Sigma}|^\frac{k+1}{2}\,

where :math:`\pi(\delta)` is an arbitrary prior on the correlations
lengths, as discussed in :ref:`AltGPPriors<AltGPPriors>`. Note that
the hyperparameter :math:`\tilde{\Sigma}` is not itself the between-outputs
covariance matrix, so this prior is not directly equivalent to the weak
prior for the separable covariance function given above.

Nonseparable covariance based on the linear model of coregionalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear model of coregionalization (LMC) was developed in the field
of geostatistics as a tool to model multivariate spatial processes, and
provides an alternative way of constructing a nonseparable multivariate
covariance function, which we call the 'LMC covariance function'.

To construct a LMC covariance function :math:`v(.,.)`, we choose a set of
:math:`r` 'basis correlation functions'
:math:`\kappa_1(.,.),\ldots,\kappa_r(.,.)`. These can be any correlation
function, as discussed in
:ref:`AltCorrelationFunction<AltCorrelationFunction>`. We also have
the :math:`r \times r` between-outputs covariance matrix :math:`\Sigma`, for
which we must choose a square-root decomposition :math:`\Sigma=RR^T`. We
usually choose :math:`R` to be the symmetric eigendecomposition of
:math:`\Sigma`, that is :math:`R=Q\mathrm{diag}(\sqrt{d_1},\ldots,\sqrt{d_r})Q^T`
where :math:`d_1,\ldots,d_r` are the eigenvalues of :math:`\Sigma` and :math:`Q` is
a matrix with the eigenvectors of :math:`\Sigma` as its columns (arranged
from left to right in the order corresponding to :math:`d_1,\ldots,d_r`). An
advantage of using the symmetric eigendecompostion of :math:`\Sigma` is
that the resulting emulator is not dependent on the ordering of the
outputs, but note that other square-root decompositions, such as the
Cholesky decomposition, may be used.

For any given choice of decomposition :math:`\Sigma=RR^T`, the LMC
covariance function is

.. math::
   v(.,.) = \sum_{i=1}^{r} \Sigma_i \kappa_i(.,.)\,,

where, for each :math:`i=1,...,r`, :math:`\Sigma_i = r_ir_i^T`, with :math:`r_i`
the :math:`i`th column of the matrix :math:`R`.

We can see that the covariance function for an individual output is a
weighted sum of the basis correlation functions. The weights are
determined by the elements of the 'coregionalization matrices'
:math:`\{\Sigma_i:i=1,...,k\}`. With the LMC covariance function, the
emulator covariance matrices defined in
:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>` become sums of
Kronecker products of the coregionalization matrices and basis
correlation matrices:

.. math::
   V &= \sum_{i=1}^r \Sigma_i \otimes \kappa_i(D,D)\,, \\
   u(x) &= \sum_{i=1}^r \Sigma_i \otimes \kappa_i(D,x)\,.

The hyperparameters in the LMC covariance function are
:math:`\omega=(\Sigma,\tilde{\delta})`, where :math:`\tilde{\delta}` denotes
the collection of hyperparameters in the basis correlation functions. A
possible choice of prior which expresses weak knowledge about
:math:`\Sigma` is

.. math::
   \pi(\Sigma,\tilde{\delta}) \propto
   \pi(\tilde{\delta})|\Sigma|^\frac{k+1}{2}\,

where :math:`\pi(\tilde{\delta})` is an arbitrary prior on the correlations
length of the basis correlation functions, as discussed in
:ref:`AltGPPriors<AltGPPriors>`. Note that the hyperparameters
:math:`\tilde{\delta}` do not correspond directly to the correlation
lengths of individual outputs, so this prior is not directly equivalent
to the weak prior for the separable covariance function given above.

Comparison of the alternatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The independent outputs approach is perhaps the simplest case, for then
the procedure :ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`
becomes equivalent to building :math:`r` separate, independent emulators,
one for each output, using the methods set out in the core thread
:ref:`ThreadCoreGP<ThreadCoreGP>`. This allows different mean and
correlation functions to be fitted for each output, and in particular
different outputs can have different correlation length parameters in
their hyperparameter vectors. Using the independent emulators for tasks
such as prediction, uncertainty and sensitivity analyses is set out in
:ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`,
and is also relatively simple.

However, if the number of outputs :math:`r` is large, building :math:`r`
emulators may be time consuming. Also, the independent outputs approach
fails to capture any correlation between outputs, which may be a
particular problem if we are interested in a function that combines
outputs, as discussed in
:ref:`ProcPredictMultiOutputFunction<ProcPredictMultiOutputFunction>`.
In that case the easiest non-independent approach is to use a separable
covariance function, which leads to several simplifications to the
general multivariate emulator, resulting in a simple emulation
methodology. This is described in
:ref:`ProcBuildMultiOutputGPSep<ProcBuildMultiOutputGPSep>`. A
drawback of the separable covariance is that it imposes a restriction
that all the outputs have the same correlation function :math:`c(.,.)`
across the input space. This means that all outputs have the same
correlation length, so the resulting emulator may not perform well if
the outputs represent several different types of physical quantity,
since it assumes that all outputs have the same smoothness properties.

It is worth noting that outputs which are very highly correlated will
necessarily have very similar correlation functions. So in the situation
where the independent outputs approach is most inappropriate because we
expect the outputs to be strongly correlated, the separable covariance
function is more likely to be reasonable.

The situation in which the separable covariance function is most likely
to be inappropriate is when multiple outputs represent a variety of
types of physical quantities. In that case we may expect there to be
some correlation between the outputs due to shared dependence on some
underlying processes, but the outputs are likely to have different
smoothness properties, so require different correlation functions across
the input space. The most flexible approach for such situations is to
use a non-independent, nonseparable covariance function.

The downside of nonseparable covariance functions are their complexity:
they require a large number of hyperparameters to be estimated before
they can be used. Described above are two types of nonseparable
covariance function. In practical terms, the main difference between
them is the way in which they are parameterised. The convolution
covariance function is controlled by hyperparameters
:math:`\omega=(\tilde{\Sigma},\delta)`, where the :math:`\delta` directly
control the input space correlation functions but :math:`\tilde{\Sigma}`
gives only limited control over the between-outputs covariance matrix.
Conversely, in the LMC covariance function we have parameters
:math:`\omega=(\tilde{\delta},\Sigma)`, where :math:`\Sigma` is the
between-outputs covariance matrix, but :math:`\tilde{\delta}` gives only
limited control over the input space correlation functions. Therefore,
if we have more meaningful prior information about the correlation
lengths than the between-outputs covariances, we may favour the
convolution covariance function, whereas if we have more meaningful
prior information about the between-outputs covariances than the
correlation lengths, we may favour the LMC covariance function. In cases
where we have weak information about both correlation lengths and the
between-outputs covariances, it may be wise to try both and see which
performs the best.

We conclude that, when building a multivariate emulator, it would be
desirable to try a variety of the above covariance functions, perform
diagnostics on the resulting emulators (as described in
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`), and select that which
is most fit for purpose. If the outputs have different correlation
lengths, and there is interest in joint predictions of multiple outputs,
then the nonseparable covariance functions may be best. However,
nonseparable covariance functions may not be practicable in large
dimension simulators, so then we must choose between the independent
outputs approach and the separable covariance function. The former is
likely to be best if interest is confined to marginal output
predictions, while the latter may be necessary if joint predictions are
required. An alternative approach to treating high dimensional output
spaces is to attempt dimension reduction, for example using PCA, and
then construct a separable or nonseparable Gaussian process in the
reduced dimension space as discussed in
:ref:`ProcOutputsPrincipalComponents<ProcOutputsPrincipalComponents>`.
