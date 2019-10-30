.. _AltGPPriors:

Alternatives: Prior distributions for GP hyperparameters
========================================================

Overview
--------

In the fully :ref:`Bayesian<DefBayesian>` approach to
:ref:`emulating<DefEmulator>` a complex
:ref:`simulator<DefSimulator>`, a :ref:`Gaussian process<DefGP>`
(GP) is formulated to represent prior knowledge of the simulator. The GP
specification is conditional on some
:ref:`hyperparameters<DefHyperparameter>`, as discussed in the
alternatives pages for emulator prior mean function
(:ref:`AltMeanFunction<AltMeanFunction>`), and correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`) and the
discussion page on the GP covariance function
(:ref:`DiscCovarianceFunction<DiscCovarianceFunction>`).
Specifically, in the :ref:`core problem<DiscCore>` that is the
subject of the core threads using Gaussian process methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) or Bayes linear
(:ref:`ThreadCoreBL<ThreadCoreBL>`) a vector :math:`\beta` defines the
detailed form of the mean function, a scalar :math:`\sigma^2` quantifies
the uncertainty or variability of the simulator around the prior mean
function, while :math:`\delta` is a vector of hyperparameters defining
details of the correlation function. Threads that deal with variations
on the basic core problem may introduce further hyperparameters.

In particular in the multi output variant (see the thread on the
analysis of a simulator with multiple outputs using Gaussian process
methods
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`))
there are a range of possible parameterisations which require different
(more general) prior specifications. The univariate case is typically
the simplification to scalar settings. For example if a input-output
:ref:`separable<DefSeparable>` multi output emulator is used, then
the prior specification will be over a matrix :math:`\beta`, and a matrix
:math:`\Sigma` and a set of correlation function hyperparameters
:math:`\delta`. Priors for this case are discussed in the alternatives page
on prior distributions for multivariate GP hyperparameters
(:ref:`AltMultivariateGPPriors<AltMultivariateGPPriors>`).

A fully Bayesian analysis requires hyperparameters to be given prior
distributions. We consider here alternative ways to specify prior
distributions for the hyperparameters of the core problem. Prior
distributions for other hyperparameters are addressed in the relevant
variant thread. Hyperparameters may be handled differently in the :ref:`Bayes
Linear<DefBayesLinear>` approach - see
:ref:`ThreadCoreBL<ThreadCoreBL>`.

Choosing the Alternatives
-------------------------

The prior distributions should be chosen to represent whatever prior
knowledge the analyst has about the hyperparameters. However, the prior
distributions will be updated with the information from a set of
training runs, and if there is substantial information in the training
data about one or more of the hyperparameters then the prior information
about those hyperparameters may be essentially irrelevant.

In general we require a *joint* distribution
:math:`\pi(\beta,\sigma^2,\delta)` for all the hyperparamters. Where
required, we will denote the marginal distribution of :math:`\delta` by
:math:`\pi_{\delta}(\delta)`, and similarly for marginal distributions of
other groups of hyperparameters.

The Nature of the Alternatives
------------------------------

Priors for :math:`\sigma^2`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most applications, there will be plenty of information about
:math:`\sigma^2` in the training data. We typically have at least 100
training runs, and 100 observations would usually be considered adequate
to estimate a variance. Unless there is strong prior information
available regarding this hyperparameter, it would be acceptable to use
the conventional :ref:`weak prior<DefWeakPrior>` specification

.. math::
   \pi_{\sigma^2}(\sigma^2) \propto \sigma^{-2}

independently of the other hyperparameters.

In situations where the training data are more sparse, which may arise
for instance when the simulator is computationally demanding, prior
information about :math:`\sigma^2` may make an important contribution to
the analysis. Genuine prior information about :math:`\sigma^2` in the form
of a :ref:`proper<DefProper>` prior distribution should be specified
by a process of :ref:`elicitation<DefElicitation>` - see references
at the end of this page. See also the discussion of conjugate prior
distributions below.

Priors for :math:`\beta`
~~~~~~~~~~~~~~~~~~~~~

Again, we would expect to find that in most applications there is enough
evidence in the training data to identify :math:`\beta` well, particularly
when the mean function is specified in the linear form, so that the
elements of :math:`\beta` are regression parameters. Then it is acceptable
to use the conventional weak prior specification

.. math::
   \pi_{\beta}(\beta) \propto 1

independently of the other hyperparameters.

If there is a wish to express genuine prior information about :math:`\beta`
in the form of a proper prior distribution, then this should be
specified by a process of elicitation - see references at the end of
this page. See also the discussion of conjugate prior distributions
below.

Conjugate priors for :math:`\beta` and :math:`\sigma^2`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When substantive prior information exists and is to be specified for
either :math:`\beta` or :math:`\sigma^2`, then it is convenient to use
:ref:`conjugate<DefConjugate>` prior distributions if feasible.

If prior information is to be specified for :math:`\sigma^2` alone (with
the weak prior specification adopted for :math:`\beta`), the conjugate
prior family is the inverse gamma family. This can be elicited using the
SHELF package referred to at the end of this page.

If :math:`\beta` is a vector of regression parameters in a linear form of
mean function, and prior information is to be specified about both
:math:`\beta` and :math:`\sigma^2`, then the conjugate prior family is the
normal inverse gamma family. Specifying such a distribution is a complex
business - see the reference to Oakley (2002) at the end of this page.

Although these conjugate prior specifications make subsequent updating
using the training data as simple as in the case of weak priors, the
details are not given in the :ref:`MUCM<DefMUCM>` toolkit because it
is expected that weak priors for :math:`\beta` and :math:`\sigma^2` will
generally be used. If needed, information on using conjugate priors in
building an emulator can be found in the Oakley (2002) reference at the
end of this page. (The case of an inverse gamma prior for :math:`\sigma^2`
combined with a weak prior for :math:`\beta` is a special case of the
general normal inverse gamma prior.)

If prior information is to be specified for :math:`\beta` alone, the
conjugate prior family is the normal family, but for full conjugacy the
variance of :math:`\beta` should be proportional to :math:`\sigma^2` in the
same way as is found in the normal inverse gamma family. This seems
unrealistic when weak prior information is to be specified for
:math:`\sigma^2`, and so we do not discuss this conjugate option further.

Priors for :math:`\delta`
~~~~~~~~~~~~~~~~~~~~~~~~~

The situation with :math:`\delta` is quite different, in that the
information in the training data will often fail to identify these
hyperparameters to sufficient accuracy. It can therefore be difficult to
estimate their values effectively, and inappropriate estimates can lead
to poor emulation. Also, it is known that in the case of a Gaussian or
an exponential power correlation function (see the alternatives page on
emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`)) conventional
weak prior distributions may lead to improper posterior distributions.
See the alternatives page on estimators of correlation hyperparameters
(:ref:`AltEstimateDelta<AltEstimateDelta>`).

Accordingly, prior information about :math:`\delta` can be very useful, or
even essential, in obtaining a valid emulator. Fortunately, genuine
prior information about :math:`\delta` often exists.

Consider the case of a Gaussian correlation function, where the elements
of :math:`\delta` are correlation lengths which define the
:ref:`smoothness<DefSmoothness>` of the simulator output as each
input is varied. The experience of the users and developers of the
simulator may suggest how stable the output should be in response to
varying individual inputs. In particular, it may be possible to specify
a range for each input such that it is not expected the output will
"wiggle" (relative to the mean function). For other forms of correlation
function, there may again be genuine prior information about the
smoothness and/or :ref:`regularity<DefRegularity>` that is controlled
by various elements of :math:`\delta`. We suggest that even quite crudely
elicited distributions for these parameters will be better than adopting
any default priors.

Additional Comments, References, and Links
------------------------------------------

The following resources on elicitation of prior distributions are
mentioned in the text above. The first is a thorough review of the field
of elicitation, and provides a wealth of general background information
on ideas and methods. The second (SHELF) is a package of documents and
simple software that is designed to help those with less experience of
elicitation to elicit expert knowledge effectively. SHELF is based on
the authors' own experiences and represents current best practice in the
field. Finally, the third reference deals specifically with eliciting a
conjugate prior for :math:`\beta` and :math:`\sigma^2` in a GP model with a
linear mean function.

O'Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P.
H., Jenkinson, D. J., Oakley, J. E. and Rakow, T. (2006). Uncertain
Judgements: Eliciting Expert Probabilities. John Wiley and Sons,
Chichester. 328pp. ISBN 0-470-02999-4.

SHELF - the Sheffield Elicitation Framework - can be downloaded from
`http://tonyohagan.co.uk/shelf <http://tonyohagan.co.uk/shelf>`__
(:ref:`Disclaimer<MetaSoftwareDisclaimer>`)

Oakley, J. (2002). Eliciting Gaussian process priors for complex
computer codes. *The Statistician* 51, 81-97.

The following paper discusses issues of propriety in GP posterior
distributions related to the choice of prior when it is desired to
express weak prior information (about :math:`\delta` as well as other
hyperparameters).

Paulo, R. (2005). Default priors for Gaussian
processes. *Annals of Statistics*, 33, 556-582.
