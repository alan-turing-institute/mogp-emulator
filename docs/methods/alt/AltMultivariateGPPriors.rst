.. _AltMultivariateGPPriors:

Alternatives: Prior distributions for multivariate GP hyperparameters
=====================================================================

Overview
--------

The prior specification for the :ref:`ThreadCoreGP<ThreadCoreGP>`
priors is discussed and alternatives are described in
:ref:`AltGPPriors<AltGPPriors>`. :ref:`ThreadCoreGP<ThreadCoreGP>`
deals with the :ref:`core problem<DiscCore>` and in particular with
emulating a single output - the univariate case. The multivariate case
of emulating several outputs is considered in the variant thread
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`,
and there are a range of possible parameterisations (see the
alternatives page on approaches to emulating multiple outputs
(:ref:`AltMultipleOutputsApproach<AltMultipleOutputsApproach>`))
which require different prior specifications. Some of the alternatives
reduce to building independent univariate emulators, for which the
discussion in :ref:`AltGPPriors<AltGPPriors>` is appropriate. We
focus here on the case of a input-output
:ref:`separable<DefSeparable>` multi output emulator and we assume
that the mean function has the linear form. For further discussion of
nonseparable covariances several alternative parameterisations are
discussed in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.

In the case of a input-output :ref:`separable<DefSeparable>` multi
output emulator, the prior specification will be over a matrix
:math::ref:`\beta` of `hyperparameters<DefHyperparameter>` for the mean
function, a between-outputs variance matrix :math:`\Sigma` and a vector
:math:`\delta` of hyperparameters for the correlation function over the
input space.

A fully Bayesian analysis requires hyperparameters to be given prior
distributions. We consider here alternative ways to specify prior
distributions for the hyperparameters of the multi output problem.

Choosing the Alternatives
-------------------------

The prior distributions should be chosen to represent whatever prior
knowledge the analyst has about the hyperparameters. However, the prior
distributions will be updated with the information from a set of
training runs, and if there is substantial information in the training
data about one or more of the hyperparameters then the prior information
about those hyperparameters may be essentially irrelevant. In the multi
output case it can often make sense to choose prior distributions which
are more simple to work with and can accommodate prior beliefs
reasonably well.

In general we require a *joint* distribution
:math:`\pi(\beta,\Sigma,\delta)` for all the hyperparameters. Where
required, we will denote the marginal distribution of :math:`\delta` by
:math:`\pi_\delta(\cdot)`, and similarly for marginal distributions of
other groups of hyperparameters.

The Nature of the Alternatives
------------------------------

Priors for :math:`\Sigma`
~~~~~~~~~~~~~~~~~~~~~~

In most applications, there will be information about :math:`\Sigma` in the
training data, although we note that :math:`\Sigma` contains :math:`r(r-1)/2`
unique values in the between outputs variance matrix (where :math:`r` is
the number of outputs), so more training runs might be required to
ensure this is well identified from the data compared to the single
output case. Unless there is strong prior information available
regarding this hyperparameter, it would be acceptable to use the
conventional :ref:`weak prior<DefWeakPrior>` specification

:math:`\pi_{\Sigma}(\Sigma) \\propto \| \\Sigma \|^{-\frac{r+1}{2}} \`

independently of the other hyperparameters.

In situations where the training data are more sparse, which may arise
for instance when the simulator is computationally demanding, prior
information about :math:`\Sigma` may make an important contribution to the
analysis.

Genuine prior information about :math::ref:`\Sigma` in the form of a
`proper<DefProper>` prior distribution should be specified by a
process of :ref:`elicitation<DefElicitation>` - see comments at the
end of this page. See also the discussion of conjugate prior
distributions below.

Priors for :math:`\beta`
~~~~~~~~~~~~~~~~~~~~~

As with the univariate case, we would expect to find that in most
applications there is enough evidence in the training data to identify
:math:`\beta` well, particularly when the mean function is specified in the
linear form, so that the elements of :math:`\beta` are a matrix of
regression parameters. Then it is acceptable to use the conventional
weak prior specification

:math:`\pi_{\beta}(\beta) \\propto 1 \`

independently of the other hyperparameters.

If there is a wish to express genuine prior information about :math:`\beta`
in the form of a proper prior distribution, then this should be
specified by a process of elicitation - see comments at the end of this
page. See also the discussion of conjugate prior distributions below.

Conjugate priors for :math:`\beta` and :math:`\Sigma`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When substantive prior information exists and is to be specified for
:math::ref:`\beta` and/or :math:`\Sigma`, then it is convenient to use
`conjugate<DefConjugate>` prior distributions if feasible.

If prior information is to be specified for :math:`\Sigma` alone (with the
weak prior specification adopted for :math:`\beta`), the conjugate prior
family is the inverse Wishart family. Elicitation of such distributions
is not a trivial matter and will be developed further at a later date.

:math:`\beta` is a matrix of regression parameters in a linear form of mean
function, and if prior information is to be specified about both
:math:`\beta` and :math:`\Sigma`, then the conjugate prior family is the
matrix normal inverse Wishart family. Specifying such a distribution is
a complex business and knowledge is still developing in this area.

Although these conjugate prior specifications make subsequent updating
using the training data as simple as in the case of weak priors, the
details are not given in the :ref:`MUCM<DefMUCM>` toolkit because it
is expected that weak priors for :math:`\beta` and :math:`\Sigma` will
generally be used. In the multivariate setting it becomes increasingly
difficult to capture beliefs about covariances in a simple manner. Also
the number of judgements that have to be made increases quadratically
with the dimension of the output space, which makes expert elicitation a
real challenge.

If prior information is to be specified for :math:`\beta` alone, the
conjugate prior family is the matrix normal family, but for full
conjugacy the between-columns variance matrix of :math:`\beta` should be
equal to :math:`\Sigma` in the same way as is found in the matrix normal
inverse Wishart family. This seems unrealistic when weak prior
information is to be specified for :math:`\Sigma`, and so we do not discuss
this conjugate option further.

Priors for :math:`\delta`
~~~~~~~~~~~~~~~~~~~~~~

This case is very similar to that discussed in the core
:ref:`AltGPPriors<AltGPPriors>` thread and we do not repeat it here.
We note that in the case of an input - output separable emulator there
are no more correlation function parameters to estimate than in the
univariate case - the extra complexity is all in :math:`\beta` and
:math:`\Sigma`. For more complex representations, such as the Linear Model
of Coregionalisation covariances and convolution covariances, discussed
in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`,
there are more parameters and prior specification is discussed on that
page.

Additional Comments, References, and Links
------------------------------------------

References to the literature on elicitation of prior distributions may
be found at the end of :ref:`AltGPPriors<AltGPPriors>`. However, the
elicitation of distributions for matrices such as :math:`\beta` or
:math:`\Sigma` is very challenging and there is very little literature to
provide guidance. Accordingly, there is relatively little experience of
elicitation for multivariate GPs.
