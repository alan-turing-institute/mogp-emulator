.. _AltMeanFunctionMultivariate:

Alternatives: Multivariate emulator prior mean function
=======================================================

Overview
--------

In the process of building a :ref:`multivariate
GP<DefMultivariateGP>` emulator it is necessary to specify a
mean function for the GP model. The mean function for a set of outputs
is a vector function. Its elements are the mean functions for each
output, defining a prior expectation of the form of that output's
response to the simulator's inputs. Alternative choices on the emulator
prior mean function for an individual output are discussed in
:ref:`AltMeanFunction<AltMeanFunction>`. Here we discuss how these
may be adapted for use in the multivariate emulator.

Choosing the alternatives
-------------------------

If the linear form of mean function is chosen with the same :math::ref:`q\times
1` vector :math:`h(\cdot)` of `basis functions<DefBasisFunctions>`
for each output, then the vector mean function can be expressed simply
as :math:`\beta^T h(\cdot)`, where :math:`\beta` is a :math:`q\times r` matrix of
regression coefficients and :math:`r` is the number of outputs. The choice
of regressors for the two simple cases shown in
:ref:`AltMeanFunction<AltMeanFunction>` can then be adapted as
follows:

\* For :math:`q=1` and :math:`h(x)=1`, the mean function becomes :math:`m(x) =
\\beta`, where :math:`\beta` is a :math:`r`-dimension vector of
hyperparameters representing an unknown multivariate overall mean (i.e.
the mean vector) for the simulator output.

\* For :math:`h(x)^T=(1,x)`, so that :math:`q=1+p`, where :math:`p` is the number
of inputs and the mean function is an :math:`r`-dimensional vector whose
:math:`j`th element is the scalar: :math:`\{m(x)\}_j=\beta_{1,j} +
\\beta_{2,j}\,x_1 + \\ldots + \\beta_{1+p,j}\,x_p`.

This linear form of mean function in which all outputs have the same set
of basis functions is generally used in the multivariate emulator
described in
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`,
where it is the natural choice for the :ref:`multivariate Gaussian
process<DefMultivariateGP>` and has mathematical and
computational advantages. However, the linear form does not have to be
used, and there may be situations in which we believe the outputs behave
like a function that is non-linear in the unknown coefficients, such as
:math:`\{m(x)\}_j = x / (\beta_{1,j} + \\beta_{2,j} x)`. The general theory
still holds for a non-linear form, albeit without the mathematical
simplification which allow the analytic integration of the regression
parameters :math:`\beta`.

It would also be possible to use different mean functions for different
outputs, but this is not widely done in practice and renders the
notation more verbose.
