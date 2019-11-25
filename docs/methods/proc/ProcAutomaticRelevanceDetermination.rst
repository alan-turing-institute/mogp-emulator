.. _ProcAutomaticRelevanceDetermination:

Procedure: Automatic Relevance Determination (ARD)
==================================================

We describe here the method of Automatic Relevance Determination (ARD)
where the :ref:`correlation length scales<DefCorrelationLength>`
:math:`\delta_i` in a covariance function can be used to determine the
input relevance. This is also known as the application of independent
priors over the length scales in the covariance models.

The purpose of the procedure is to perform
:ref:`screening<DefScreening>` on the simulator inputs, identifying
the :ref:`active<DefActiveInput>` inputs.

ARD is typically applied using a zero mean :ref:`Gaussian
Process<DefGP>` :ref:`emulator<DefEmulator>`. Provided the
inputs have been standardised (see the procedure page on data
preprocessing (:ref:`ProcDataPreProcessing<ProcDataPreProcessing>`)),
the correlation length scales may be directly used as importance
measures. Another case where ARD may be used is with non-zero mean
function Gaussian Process where we wish to identify factor effects in
the residual process. For example with a linear mean, correlation length
scales indicate non-linear and interaction effects. If the effect of a
factor is strictly linear with no interaction with other factors, it can
still be screened out by subtracting from the simulator output prior to
emulation.

Choice of Covariance Function
-----------------------------

To implement the ARD method, a range of covariance functions can be
used. In fact any covariance function that has a length scale vector
included can be used for ARD, for example the squared exponential
covariance used in most of the toolkit. Such a covariance function is
the Rational Quadratic (RQ) :

:math:`v(x_p,x_q) =\sigma^2 [1 + (x_p - x_q)^{\tau} P^{-1} (x_p - x_q)/(2
\alpha)]^{-\alpha}` where :math:`\sigma` is the scale parameter and
:math:`P=\mathrm{diag}(\delta_i)^{2}` a diagonal matrix of correlation
length scale parameters. Taking the limit :math:`a\rightarrow\infty`
parameter, we obtain the squared exponential kernel.

Assuming :math:`p` input variables, each hyperparameter :math:`\delta_i` is
associated with a single input factor. The :math:`\delta_i` hyperparameters
are referred to as characteristic length scales and can be interpreted
as the distance required to move along a particular axis for the
function values to become uncorrelated. If the length-scale has a very
large value the covariance becomes almost independent of that input,
effectively removing that input from the model. Thus length scales can
be viewed as a total effect measure and used to determine the relevance
of a particular input.

Lastly, if the simulator produces random outputs the emulator should no
longer exactly interpolate the observations. In this case, a nugget term
:math:`\nu` should be added to the covariance function to capture the
response uncertainty.

Implementation
--------------

Given a set of simulator runs, the ARD procedure can be implemented in
the following order:

#. **Standardisation**. It is important to first
   :ref:`standardise<ProcDataPreProcessing>` the input data so all
   input factors operate on the same scale. If rescaling is not done
   prior to the inference stage, length scale parameters will generally
   have larger values for input factors operating on larger scales.
#. **Inference**. The Maximum-A-Posteriori (MAP) values of the length
   scale hyper-parameters are typically obtained by iterative non-linear
   optimisation using standard algorithms such as scaled conjugate
   gradients, although in a fully Bayesian treatment posterior
   distributions could be approximated using Monte Carlo methods.
   Maximum-A-Posteriori (MAP) is the process of identifying the mode of
   the posterior distribution of the hyperparameter (see the discussion
   page :ref:`DiscPostModeDelta<DiscPostModeDelta>`). One difficulty
   using ARD stems from the use of an optimisation process since the
   optimisation is not guaranteed to converge to a global minimum and
   thus ensure robustness. The algorithm can be run multiple times from
   different starting points to assess robustness at the cost of
   increasing the computational resources required. In case of a very
   high dimensional input space, maximum likelihood may be too costly or
   intractable due to the high number of free parameters (one length
   scale for each dimension). In this case Welch at al (1992) propose a
   constrained version of maximum likelihood where initially all inputs
   are assumed to have the same length scale and iteratively, some
   inputs are assigned separate length scales based on the improvement
   in the likelihood score.
#. **Validation**. To ensure robustness of the screening results, prior
   to utilising the length scales as importance measures the emulator
   should be validated as described in procedure page
   :ref:`ProcValidateCoreGP<ProcValidateCoreGP>`.

An example of applying the ARD process is provided in
:ref:`ExamScreeningAutomaticRelevanceDetermination<ExamScreeningAutomaticRelevanceDetermination>`.

References
----------

Williams, C. K. I. and C. E. Rasmussen (2006). `Gaussian Processes for
Machine Learning <http://www.gaussianprocess.org/gpml/>`_. MIT Press.

William J. Welch, Robert. J. Buck, Jerome Sacks, Henry P. Wynn, Toby J.
Mitchell and Max D. Morris. " Screening, Predicting, and Computer
Experiments", *Technometrics*, Vol. 34, No. 1 (Feb., 1992), pp. 15-25.
Available at `http://www.jstor.org/stable/1269548 <http://www.jstor.org/stable/1269548>`_.
