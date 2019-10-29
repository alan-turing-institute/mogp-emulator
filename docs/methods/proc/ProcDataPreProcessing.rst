.. _ProcDataPreProcessing:

Procedure: Data Pre-Processing and Standardisation
====================================================

In this page we describe the process of pre-processing data, which might
often be undertaken prior to for example
:ref:`screening<DefScreening>` or more general
:ref:`emulation<DefEmulator>`. This can take several forms. A very
common pre-processing step is centring, which produces data with zero
mean. If the range of variation is known a priori a simple linear
transformation to the range [0,1] is often used. It might also be useful
to standardise (sometimes called normalise) data to produce zero mean
and unit variance. For multivariate data it can be useful to whiten (or
sphere) the data to have zero mean and identity covariance, which for
one variable is the same as standardisation. The linear transformation
and normalisation processes are not equivalent since the latter is a
probabilistic transformation using the first two moments of the observed
data.

Centring
--------

It is often useful to remove the mean from a data set. In general the
mean, :math:`\textrm{E}[x]`, will not be known and thus must be estimated
and the centered data is given by: :math:` x' =x-\textrm{E}[x]`. Centring
will often be used if a zero mean Gaussian process is being used to
build the emulator, although in general it would be better to include an
explicit mean function in the emulator.

Linear transformations
----------------------

To linearly transform the data region :math:`x \\in [c,d]` to another
domain :math:`x' \\in [a,b]`:

:math:`x' = \\frac{x-c}{d-c} (b-a) + a`

In experimental design the convention is for :math:`[a,b]=[0,1]`.

Standardising
-------------

If the domain of the design region is not known, samples from the design
space can be used to rescale the data to have 0 mean, unit variance by
using the process of standardisation. If on the other hand the design
domain is known we can employ a linear rescaling.

The process involves estimating the mean :math:`\mu = \\textrm{E}[x]` and
standard deviation of the data :math:`\sigma` and applying the
transformation :math:` x' = \\frac{x-\mu} { \\sigma } \`. It is possible to
standardise each input / output separately which rescales the data, but
does not render the outputs uncorrelated. This might be useful in
situations where correlations or covariances are difficult to estimate,
or where these relationships want to be preserved, so that individual
inputs can still be distinguished.

Sphering / Whitening
--------------------

For multivariate inputs and outputs it is possible to whiten the data,
that is convert the data to zero mean, identity variance. This process
is a linear transformation of the data and is described in more detail,
including a discussion of how to treat a mean function in the procedure
page
:ref:`ProcOutputsPrincipalComponents<ProcOutputsPrincipalComponents>`,
and would typically be applied to outputs rather than inputs.

The data sphering process involves estimating the mean
:math:`\textrm{E}[x]` and variance matrix of the data :math:`\textrm{Var}[x]`,
computing the eigen decomposition :math:`\strut P {\Delta} P^{T}` of
:math:`\textrm{Var}[x]` and applying the transformation :math:`\strut x' = P
{\Delta}^{-1/2} P^T (x-\textrm{E}[x])`. Alternative approaches are
possible and are discussed in
:ref:`ProcOutputsPrincipalComponents<ProcOutputsPrincipalComponents>`.
