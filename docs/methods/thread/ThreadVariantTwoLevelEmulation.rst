.. _ThreadVariantTwoLevelEmulation:

Thread Variant: Two-level emulation of the core model using a fast approximation
================================================================================

Overview
--------

Often, a :ref:`computer model<DefSimulator>` can be evaluated at
different levels of accuracy resulting in different versions of the
computer model of the same system. For example, this can arise from
simplifying the underlying mathematics, by adjusting the model gridding,
or by changing the accuracy of the model's numerical solver. Lower
accuracy models can often be evaluated for a fraction of the cost of the
full computer model, and may share many qualitative features with the
original. Often, the coarsened computer model is informative for the
accurate computer model, and hence for the physical system itself. By
using evaluations of the coarse model in addition to those of the full
model, we can construct a single multiscale
:ref:`emulator<DefEmulator>` of the computer simulation.

When an approximate version of the simulator is available, we refer to
it as the *coarse simulator*, :math:`f^c(\cdot)`, and the original as the
*accurate simulator*, :math:`f^a(\cdot)`, to reflect these differences in
precision. We consider evaluations of the coarse model to be relatively
inexpensive when compared to :math:`f^a(x)`. In such a setting, we can
obtain many evaluations of the coarse simulator and use these to
construct an informed emulator for the coarse model. This provides a
basis for constructing an informed prior specification for the accurate
emulator. We then select and evaluate a small number of accurate model
runs to update our emulator for the accurate model. This transfer of
beliefs from coarse to accurate emulator is the basis of :ref:`multilevel
emulation<DefMultilevelEmulation>` and is the focus of this
thread.

This thread considers the problem of constructing an emulator for a
single-output deterministic computer model :math:`f^a(x)` when we have
access to a single approximate version of the same simulator,
:math:`f^c(x)`.

Requirements
------------

The requirements for the methods and techniques described in this page
differ from the :ref:`core problem<DiscCore>` by relaxing the
requirement that we are only concerned with a single simulator. We now
generalise this problem to the case where

-  We have two computer models for the same complex system - one of the
   models (the *coarse simulator*) is comparatively less expensive to
   evaluate, though consequently less accurate, than the second
   simulator (the *accurate simulator*)

General process
---------------

The general process of two-level emulation follows three distinct
stages:

#. **Emulate the coarse model** - :ref:`design<DefDesign>` for and
   perform :ref:`evaluations<DefTrainingSample>` of :math:`f^c(x)`, use
   these evaluations to construct an emulator for the coarse model
#. **Link the coarse and accurate models** - by parametrising the
   relationship between the two simulators, we use information from the
   coarse emulator to build a prior emulator for :math:`f^a(x)`
#. **Emulate the accurate model** - design for and evaluate a small
   number of runs of :math:`f^a(x)`, use these evaluations to update the
   prior emulator

Stages 1 and 3 in this process are applications of standard emulation
methodology discussed extensively in the Toolkit. Stage 2 is unique to
multiscale emulation and the combination of information obtained from
the emulator :math:`f^c(x)` with beliefs about the relationship between the
two models is at the heart of this approach.

Emulating the coarse model
--------------------------

To represent our uncertainty about the high-dimensional coarse computer
model :math:`f^c(x)`, we build a coarse emulator for the coarse simulator.
This gives an emulator of the form:

.. math::
   f^c(x) = m^c(x) + w^c(x)

where :math:`m^c(x)` represents the emulator :ref:`mean
function<AltMeanFunction>`, and :math:`w^c(x)` is a stochastic
residual process with a specified :ref:`covariance
function<DiscCovarianceFunction>`.

The choice of the methods of emulator construction depends on the
problem. In the case where the coarse simulator is very fast and very
large amounts of model evaluations can be obtained for little expense,
we may consider a purely empirical method of emulator construction as
described in the procedure for the empirical construction of a Bayes
linear emulator
(:ref:`ProcBuildCoreBLEmpirical<ProcBuildCoreBLEmpirical>`). When the
coarse simulator requires a moderate amount of resource to evaluate
(albeit far less than :math:`f^a(x)`) and when appropriate prior beliefs
about the model are available, then we can apply the
:ref:`fully-Bayesian<DefBayesian>` methods of the thread for the
analysis of the core model using Gaussian process methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) or the :ref:`Bayes
linear<DefBayesLinear>` methods of
:ref:`ThreadCoreBL<ThreadCoreBL>`.

The manner in which we construct the emulator is not important, merely
that we obtain an emulator as described in the form of either numerical
estimates, :ref:`adjusted beliefs<DefBLAdjust>`, or posterior
distributions for the emulator mean function parameters :math:`\beta^c`,
the variance/correlation hyperparameters :math:`\{(\sigma^c)^2,\delta^c\}`,
and an updated residual process. If the coarse emulator is built using
Bayes linear methods, the necessary mean, variance and covariance
specifications are provided within the relevant thread. Details of how
to obtain the corresponding quantities for a fully-Bayesian emulator
will be provided in a later release of the toollkit.

Linking the coarse and accurate emulators
-----------------------------------------

Given that the coarse simulator is informative for the accurate
simulator, we can use our coarse emulator as a basis for constructing
our prior beliefs for the emulator of :math:`f^a(x)`. To construct such a
prior, we model the relationship between the two simulators and then
combine information from :math:`f^c(x)` with appropriate belief
specifications about this relationship. We express our emulator for the
accurate model in a similar form as the coarse emulator

.. math::
   f^a(x) = m^a(x) + w^a(x),

In general, we express the accurate emulator in terms of either the
coarse simulator itself or elements of the coarse simulator in
conjunction with some additional parameters which capture how we believe
the two simulators are related.

There are many ways to parametrise the relationship between the two
computer models. Common approaches include:

**Single multiplier:** A simple approach to linking the computer models
is to consider the accurate simulator to be a re-scaled version of the
coarse simulator plus additional residual variation. This yields an
accurate emulator of the form:

.. math::
   f^a(x)=\rho f^c(x) + {w^a}'(x),

where :math:`\rho` is an unknown scaling parameter, and :math:`{w^a}'(x)` is a
new stochastic residual process unique to the accurate computer model.
We may consider the single multiplier method when we believe that the
difference in behaviour between the two models is mainly a matter of
scale, rather than changes in the shape or location of the output.

In this case, we can consider the mean function of the accurate emulator
to be :math:`m^a(x)=\rho m^c(x)`, and the residual process can be
expressed as :math:`w^a(x) = \rho w^c(x) +{w^a}'(x)`.

**Regression multipliers:** When the coarse emulator mean function takes
a :ref:`linear form<AltMeanFunction>`, :math:`m^c(x)=\sum_j \beta^c_j(x)
h_j(x)`, the single multiplier method can be generalised. Instead of
re-scaling the value of the coarse simulator itself, we can consider
re-scaling the contributions from each of the regression :ref:`basis
functions<DefBasisFunctions>` to the emulator's mean function.
This gives an accurate emulator of identical structure to the coarse
emulator though with modified values of the regression coefficients,

.. math::
   f^a(x)=\sum_j \rho_j \beta^c_j h_j(x) + \rho_w w^c(x) + {w^a}'(x)

where :math:`\rho_j` is an unknown scaling parameter for basis function
:math:`h_j(x)`, and :math:`\rho_w` scales the contribution of the coarse
residual process to the accurate emulator. We might choose to use this
regression form, for example, when we consider that each term in the
regression represents a physical process and the effects represented by
:math:`h_j(x)` change as we move between the the two simulators.

In this case, we can consider the mean function of the accurate emulator
to be :math:`m^a(x)=\sum_j \beta^a_j h_j(x)` where
:math:`\beta^a_j=\rho_j\beta^c_j`, and the residual process can be
expressed as :math:`w^a(x) = \rho_w w^c(x) +{w^a}'(x)`. In some cases
it can be appropriate to express this relationship in the alternative
form :math:`\beta^a_j=\rho_j\beta^c_j +\gamma_j`, where :math:`\gamma_j` is an
additional unknown parameter. This alternative form can better
accommodate models which have mean function effects which "switch on" as
we move onto the accurate model.

When the mean function of the emulators has a linear form, the single
multiplier method is a special case of the regression multipliers method
obtained by setting :math:`\rho_i=\rho^w=\rho`.

**Spatial multiplier:** Similar to the single multiplier method, we
still consider the accurate simulator to be a re-scaling of the coarse
simulator. However, the scaling factor is no longer a single unknown
value but a stochastic process, :math:`\rho(x)`, over the input space.

.. math::
   f^a(x)=\rho(x) f^c(x) + w^a(x).

This spatial multiplier approach is applicable when we expect the nature
of the relationship between the two models to change as we move
throughout the input space. Similarly to (1), we can write the mean
function of the accurate emulator to be :math:`m^a(x)=\rho(x) m^c(x)`,
and the residual process can be expressed as :math:`w^a(x) = \rho(x)
w^c(x) +{w^a}'(x)`.

In general, we obtain a form for the accurate emulator given by the
appropriate expressions for :math:`m^a(x)` and :math:`w^a(x)`. Each of these
components is expressed in terms of (elements of) the emulator for
:math:`f^c(x)` and an additional residual process :math:`w^a(x)`, and is
parametrised by a collection of unknown linkage hyperparameters
:math:`\rho`.

Specifying beliefs about :math:`\rho_j` and :math:`w^a(x)`
----------------------------------------------------------

Given the coarse emulator :math:`f^c(x)` and a model linking :math:`f^c(x)` to
:math:`f^a(x)`, then a prior specification for :math:`\rho` and :math:`w^a(x)`
are sufficient to develop a prior for the emulator for :math:`f^a(x)`. In
general, our uncertainty judgements about :math:`\rho` and :math:`w^a(x)` will
be problem-specific. For now, we describe a simple structure that these
beliefs may take and offer general advice for making such statements.

We begin by considering that :math:`\rho_j` and :math:`{w^a}'(x)` are
independent of :math:`\beta^c` and :math:`w^c(x)`. The simplest general
specification of prior beliefs for the multipliers :math:`\rho` corresponds
to considering that there exists no known systematic biases between the
two models. This equates to the belief that the expected value of
:math:`m^a(x)` is the same as :math:`m^c(x)`, which implies

.. math::
   \textrm{E}[\rho_j]=1

The simplest specification for the variance and covariance of the
:math:`\rho_j` is to parametrise the variance matrix by two constants
:math:`\sigma^2_\rho \` and :math:`\alpha` such that

.. math::
   \textrm{Var}[\rho_j]&=&\sigma^2_\rho \\
   \textrm{Corr}[\rho_j,\rho_k]&=&\alpha, i\neq j

where :math:`\sigma^2_\rho\geq 0` and :math:`\alpha\in[-1,1]`. This belief
specification is relatively simple. However by adjusting the value of
:math:`\sigma^2_\rho` we can tighten or relax the strength of the
relationship between the two simulators. By varying the value of
:math:`\alpha` we can move from beliefs that the accurate simulator is a
direct re-scaling of the coarse simulator (method (1) above) when
:math:`\alpha=1`, to a model where the contribution from each of the
regression basis functions varies independently when :math:`\alpha=0`.
Specification of these values will typically come from expert judgement.
However, performing a small number of paired evaluations on the two
simulators and assessing the degree of association can prove informative
when specifying values of :math:`\sigma^2_\rho` and :math:`\alpha`.
Additionally, considering heuristic statements can be insightful - for
example, the belief that it is highly unlikely that :math:`\beta^a_j` has a
different sign to :math:`\beta^c_j` might suggest the belief that
:math:`3\textrm{sd}[\rho_j]=1`.

For method (3), the corresponding beliefs would be that the prior mean
of the stochastic process :math:`\rho(x)` was the constant 1, and that the
prior variance was :math:`\sigma^2_\rho` with a given correlation function
(likely of the same form as :math:`w^c(x)`).

Beliefs about :math:`{w^a}'(x)` are more challenging to structure. In
general, we often consider that the :math:`{w^a}'(x)` behaves similarly to
:math:`w^c(x)` and so has a zero mean, variance :math:`(\sigma^a)^2`, and the
same correlation function and hyperparameter values as :math:`w^c(x)`. More
complex belief specifications can be used when we have appropriate prior
information relevant to those judgements. For example, we may wish to
have higher values of :math:`\textrm{Corr}[\rho_j,\rho_k]` when :math:`h_j(x)`
and :math:`h_k(x)` are functions of the same input parameter or are of
similar functional forms.

Constructing the prior emulator for :math:`f^a(x)`
---------------------------------------------------

In the :ref:`Bayes linear<DefBayesLinear>` approach to emulation, our
prior beliefs about the emulator :math:`f^a(x)` are defined entirely by the
:ref:`expectation and variance<DefSecondOrderSpec>`. Using the
regression multiplier method (2) of linking the simulators, these
beliefs are as follows:

.. math::
   \textrm{E}[f^a(x)] &=& \sum_j \textrm{E}[\rho_j \beta^c_j]
   h_j(x) + \textrm{E}[\rho_w w^c(x)] + \textrm{E}[{w^a}'(x)] \\
   \textrm{Var}[f^a(x)] &=& \sum_j\sum_k h_j(x)h_k(x)
   \textrm{Cov}[\rho_j \beta^c_j,\rho_k \beta^c_k] +
   \textrm{Var}[\rho_w w^c(x)] + \textrm{Var}[{w^a}'(x)] +
   2\sum_j h_j(x)\textrm{Cov}[\rho_j \beta^c_j,\rho_w w^c(x)]

where the constituent elements are either expressed directly in terms of
our beliefs about :math:`\rho_j` and :math:`{w^a}'(x)`, or are obtained from
the expressions below:

.. math::
   \textrm{E}[\rho_j \beta^c_j] &=& \textrm{E}[\rho_j] \textrm{E}[\beta^c_j] \\
   \textrm{E}[\rho_w w^c(x)]    &=& \textrm{E}[\rho_w] \textrm{E}[w^c(x)] \\
   \textrm{Var}[\rho_w w^c(x)]  &=& \textrm{Var}[\rho_w]\textrm{Var}[w^c(x)] +
                                    \textrm{Var}[\rho_w]\textrm{E}[w^c(x)]^2 +
                                    \textrm{E}[\rho_w]^2\textrm{Var}[w^c(x)] \\
   \textrm{Cov}[\rho_j \beta^c_j,\rho_k \beta^c_k] &=& \textrm{Cov}[\rho_j,\rho_k] \textrm{Cov}[\beta^c_j,\beta^c_k] +
                                                       \textrm{Cov}[\rho_j,\rho_k]\textrm{E}[\beta^c_j]\textrm{E}[\beta^c_k] +
                                                       \textrm{Cov}[\beta^c_j,\beta^c_k]\textrm{E}[\rho_j]\textrm{E}[\rho_k] \\
   \textrm{Cov}[\rho_j \beta^c_j,\rho_w w^c(x)]    &=& \textrm{Cov}[\rho_j,\rho_w] \textrm{Cov}[\beta^c_j,w^c(x)] +
                                                       \textrm{Cov}[\rho_j,\rho_w]\textrm{E}[\beta^c_j]\textrm{E}[w^c(x)] +
                                                       \textrm{Cov}[\beta^c_j,w^c(x)]\textrm{E}[\rho_j]\textrm{E}[\rho_w]

Expressions for the single multiplier approach are obtained by replacing
all occurrences of :math:`\rho_j` and :math:`\rho_w` with the single parameter
:math:`\rho` and beliefs about that parameter are substituted into the
above expressions with :math:`\textrm{Corr}[\rho,\rho] =1`. Similarly
for the spatial multiplier method (3), :math:`\rho_j` and :math:`\rho_w` are
replaced by the process :math:`\rho(x)`.

In the case where the coarse simulator is well-understood, much of the
uncertainties surrounding the coarse coefficients :math:`\beta^c_j` and the
coarse residuals :math:`w^c(x)` will be eliminated. Any unresolved
variation on these quantities is often negligible in comparison to the
other uncertainties associated with :math:`f^a(x)`. In such cases, we may
make the assumption that the :math:`\beta^c_j` (and hence the :math:`w^c(x)`)
are known and thus substantially simplify the expressions for
:math:`\textrm{E}[f^a(x)]` and :math:`\textrm{Var}[f^a(x)]` as follows:

.. math::
   \textrm{E}[f^a(x)] &=& \sum_j \textrm{E}[\rho_j] g_j(x) +
                          \textrm{E}[\rho_w] w^c(x) +
                          \textrm{E}[{w^a}'(x)] \\
   \textrm{Var}[f^a(x)] &=& \sum_j\sum_k g_j(x)g_k(x)\textrm{Cov}[\rho_j ,\rho_k] +
                            w^c(x)^2 \textrm{Var}[\rho_w] +
                            \textrm{Var}[{w^a}'(x)] + 2\sum_j g_j(x)w^c(x)\textrm{Cov}[\rho_j,\rho_w ]

where we define :math:`g_j(x)=\beta^c_jh_j(x)`. These simplifications
substantially reduce the complexity of the emulation calculations as now
the only uncertain quantities in :math:`f^a(x)` are :math:`\rho_j`,
:math:`\rho_w` and :math:`{w^a}'(x)`.

These quantities are sufficient to describe the Bayes linear emulator of
:math:`f^a(x)`. The :ref:`Gaussian process<DefGP>` approach requires a
probability distribution for :math:`f^a(x)`, which will be taken to be
Gaussian with the above specified mean and variance.

Design for the accurate simulator
---------------------------------

We are now ready to make a small number of evaluations of the accurate
computer model and update our emulator for the :math:`f^a(x)`. Since the
accurate computer model is comparatively very expensive to evaluate,
this design will be small -- typically far fewer runs than those
available for the coarse simulator.

Due to the small number of design points, the choice of design is
particularly important. If the cost of evaluating :math:`f^a(x)` permits,
then a space-filling design for the accurate simulator would still be
effective. However as the number of evaluations will be typically
limited, we may consider seeking an optimal design which has the
greatest effect in reducing uncertainty about :math:`f^a(x)`. The general
procedure for generating such a design is described in
:ref:`ProcOptimalLHC<ProcOptimalLHC>`, where the design criterion is
given by the adjusted (or posterior) variance of the accurate emulator
given the simulator evaluations (see the procedure for building a Bayes
linear emulator for the core problem
(:ref:`ProcBuildCoreBL<ProcBuildCoreBL>`) and the expression given
above).

Building the accurate emulator
------------------------------

Our prior beliefs about the accurate emulator and the design and
evaluations of the accurate simulator provide sufficient information to
directly apply the Bayesian emulation methods described in
:ref:`ThreadCoreBL<ThreadCoreBL>` or
:ref:`ThreadCoreGP<ThreadCoreGP>`. Once we have constructed the
accurate emulator we can then perform appropriate diagnostics and
validation, and use the emulator as detailed for suitable post-emulation
tasks.
