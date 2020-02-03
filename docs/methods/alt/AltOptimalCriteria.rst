.. _AltOptimalCriteria:

Alternatives: Optimal design criteria
=====================================

Overview
--------

Several toolkit operations call for a set of points to be specified in
the :ref:`simulator<DefSimulator>` space of inputs, and the choice of
such set is called a design. For instance, to create an
:ref:`emulator<DefEmulator>` of the simulator we need a set of
points, the :ref:`training sample<DefTrainingSample>`, at which the
simulator is to be run to provide data in order to build the emulator.

:ref:`Optimal design<DefModelBasedDesign>` seeks a design which
maximises or minimises a suitable criterion relating to the efficiency
of the experiment in terms of estimating parameters, prediction and so.

In mainstream optimal design this is based on some function of the
variance-covariance matrix :math:`\sigma^2(X^TX)^{-1}` from a regression
model :math:`Y = X \beta + \varepsilon`. It is heavily model based
since :math:`X` depends on the model. Thus, recall that in
regression will be built on a vector of functions :math:`f(x)`: so that
the mean at a particular :math:`x` is :math:`\eta(x) = E(Y_x) = \sum
\beta_j f_j(x)= \beta^T f(x)`. Then :math:`X = f_j(x_i)` and
:math:`x_1, \cdots, x_n` is the design.

Below, a term like :math:`\min_{design}` means minimum over some class of
designs, e.g. all designs of sample size :math:`n` with design
points :math:`x_1, \cdots, x_n` in a design space :math:`\mathcal X`.
We may standardise the design space to be, for example
:math:`[-1, 1]^d`, where :math:`\strut{d}` is the number of scalar
inputs, but we can have very non standard design spaces.

For details about optimal design algorithms see,
:ref:`AltOptimalDesignAlgorithms<AltOptimalDesignAlgorithms>`.

Choosing the Alternatives
-------------------------

Each of the criteria listed below serves a specific use, and some
comments are made after each.

The Nature of the Alternatives
------------------------------

Classical Optimality Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D-optimality
^^^^^^^^^^^^

D-optimality is used to choose a design that minimises the generalised
variance of the estimators of the model parameters namely
:math:`\min_{design} det((X^TX)^{-1}) \Leftrightarrow \max_{design} det
(X^TX)`. This is perhaps the most famous of optimal design criteria,
and has some elegant theory associated with it. Because the determinant
is the product of the eigenvalues it is sensitive to :math:`X`,
or equivalently :math:`X^TX`, being close to not being full rank.
Conversely it guards well against this possibility: a small eigenvalue
implies that the model is close to being "collinear", i.e. :math:`\strut{X}`
is not of full rank. The criterion has the nice property
that a linear reparametrisation (such as shifting and rescaling the
parameters) has no effect on the criterion

G-optimality
^^^^^^^^^^^^

This criterion used to find the design that minimises (over the design
space) the variance of the prediction of the mean :math:`\eta(x)` at a
point :math:`x`. This variance is equal to :math:`\sigma^2 f(x)^T
(X^TX)^{-1} f(x)` and therefore :math:`G`-optimality means:
minimise the maximum value of this variance: :math:`\min_{design} \max_{x
\in \mathcal X} f(x)^T(X^TX)^{-1}f(x)`. The General Equivalence
Theorem of Kiefer and Wolfowitz says that :math:`D`- and :math:`G`-optimality
are equivalent (under a general definition of
a design as a mass function and over the same design space).

E-optimality
^^^^^^^^^^^^

The E-optimality criterion is used when we are interested in estimating
any (normalized) linear function of the parameters. It guards against
the worst such value: :math:`\min_{design} \max_j
\lambda_j((X^TX)^{-1})`. As such it is even stronger than D-optimality
in guarding against collinearity. In other words large eigen-values
occur when :math:`X` is close to singularity.

Trace optimality
^^^^^^^^^^^^^^^^

This is used when the aim of the experiment is to minimize the sum of
the variances of the least squares estimators of the parameters:
:math:`min_{design} {\rm Tr} (X^TX)^{-1}`. It is one of the easiest criteria
in conception, but suffers from not being invariant under linear
reparametrisations (unlike D-optimality). It is more appropriate when
parameter have clear meanings, eg as the effect of a particular factor.

A-optimality:
^^^^^^^^^^^^^

It is used when the aim of the experiment is to estimate more than one
special linear functions of the parameters, e.g. :math:`K\beta`. The
least squares estimate has :math:`{\rm Var}[K^T \hat{\beta}] = \sigma^2
K^T (X^TX)^{-1}K`. Using the trace criterion, this has trace:
:math:`\sigma^2 {\rm Tr} (X^TX)^{-1}A` with :math:`A=KK^T`. Care has to
be taken because the criterion is not invariant with respect to
reparametrisation, including rescaling, of parameters and because :math:`A`
affects the importance given to different parameters.

c-optimality
^^^^^^^^^^^^

This is to minimise the variance of a particular linear function of the
parameters: :math:`\phi = \sum c_j \beta_j =c^T\beta` and
:math:`\min_{design} c^T (X^TX)^{-1}c`. It is a special case of
A-optimality and has a nice geometric theory of its own. It was one of
the earliest studied criteria.

Bayesian optimal experimental design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is an important general approach to design in the Bayesian
setting. The choice of experimental design is a decision which may be
made after the specification of any prior distribution on the parameters
in a model but before the observations are made (outputs are measured).
In this sense it is an pre-posterior decision. Suppose that :math:`\beta`
refers to the unknown parameters of interest; not just the labeled
parameters but may include predicted outputs at special points. Let
:math:`\phi(\pi(\beta|Y))` be a functional on the posterior distribution
:math:`\pi(\beta|Y)` (after the experiment is conducted) which measures
some feature of the posterior such as its peakedness. In the case that
:math:`\phi = E(L(\hat{\beta}, \beta))`, for some estimator
:math:`\hat{\beta}` and loss function :math:`L` then :math:`\phi` is the
posterior Bayes risk of :math:`\hat{\beta}`.

If :math:`\phi = \min_{\hat{\beta}} E(L(\hat{\beta}, \beta))` then
:math:`\hat{\beta}` is the Bayes estimator with respect to :math:`L`
and we have the (posterior) Bayes risk. This is ideal from the Bayes
standpoint although it may be computationally easier to use a simpler
non-Bayes estimator but still compute the Bayes risk.

The full Bayesian optimal design criterion with respect to :math:`\phi` is
:math:`\min_{design} E_Y \phi(\pi(\beta|Y)),` where :math:`Y` is
the output generated by the experiment. Here, of course, the
distribution of :math:`Y` is affected by the design. In areas such as
non-linear regression one make be able to compute a local optimal design
using a classical estimator such as the maximum likelihood estimator. In
such a case the :math:`\phi` value may depend on this unknown :math:`\beta`:
:math:`\phi(\beta)`. An approximate Bayesian criteria is then
:math:`\min_{design} E_{\beta}(\phi(\beta)),` where the expectation is with
respect to the prior distribution on :math:`\beta`, :math:`\pi(\beta)`. The
approximate full Bayes criteria (which is computationally harder) and
approximate Bayes criteria can give similar solutions.

There are Bayesian analogues of all the classical optimal design
criteria listed above. The idea is to replace the variance matrix of the
least squares estimates of the regression parameter :math:`\beta`,
by the posterior variance matrix :math:`{\rm Var}[\beta |Y ]`. Thus,
if we take the standard regression model: :math:`Y = X \beta + \varepsilon`
and let :math:`\beta \sim N(\mu, \sigma^2 I )` and :math:`\mu \sim N(0,
\Gamma)`, then :math:`{\rm Var}[\beta |Y ] = (\sigma^{-2}X^TX +
\Gamma^{-1})^{-1}`.

Important note: in this simple case the posterior covariance does not
depend on the observed data, so that the prior expectation :math:`E_Y` in
the Bayes rule for design, is not needed.

Bayesian Information criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The information-theoretical approach to experimental design goes some
way towards being an all purpose philosophy. It is easiest to explain by
:math:`\phi(\pi)` in the Bayes formulation to be Shannon entropy. For a
general random variable :math:`X` with pdf :math:`p(x)` this is :math:`{\rm
Ent}(X)= - E_X(\log(p(X)) = - \int \log(p(x)) p(x) dx`.

Shannon information is :math:`Inf(X) =- {\rm Ent}(X)`.

The information approach is to minimise the expected posterior entropy.

.. math::
   \min_{design} E_Y {\rm Ent}(\beta|Y)

This is often expressed as: maximise the expected information gain
:math:`E_Y(G)`, where: :math:`G=Inf(\beta|Y) - Inf(\beta)`, where the second
term on the right hand side is the prior information. An important
result says that :math:`E_Y(G)` is always non-negative, although in actual
experiment cases :math:`G` may decrease. This result can be generalised to
a wider class of information criteria which include Tsallis entropy
:math:`E_X \left(\frac{p(X)^{\alpha}-1}{\alpha}\right),\;\;\alpha > -1`.

Maximum Entropy Sampling (MES)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a special way of using the Bayesian information criterion for
prediction. We exploit an importat formula for Shannon entropy which
applies to two random variables: :math:`( U,V )`:

.. math::
   \mbox{Ent}(U,V) = \mbox{Ent}(U) + \mbox{E}_U (\mbox{Ent}(V|U).

Let :math:`S` represent a set of candidate points, which covers the whole
design space well. This could for example be a large factorial design,
or a large space-filling design. Let and let :math:`s` be a
possible subset of :math:`s`, to be use as a design. Then the
complement of :math:`s` namely :math:`s` can be
thought of as the "unsampled" points.

Then partition :math:`Y`: :math:`(Y_s,Y_{S \setminus s})`. Then for
prediction we could consider: :math:`\mbox{Ent}(Y_s,Y_{S \setminus s}) =
\mbox{Ent}(Y_s) + \mbox{E}_{Y_s} [\mbox{Ent}(Y_{S \setminus
s}|Y_s)]`. The joint entropy on the left hand side is fixed, that is
does not depend on the choics of design. Therefore, since the Bayes
criterion is to minimise, over the choice of design, the second term on
the right, it is optimal to maximise the first term on the right. This
is simply the entropy of the sample, and typically requires no
conditioning computation. For the simple Bayesian Gaussian case we have

.. math::
   \max_{design} |R + X \Gamma X^T|.

Where :math:`R` is the variance matrix of the process part of
the model, :math:`X` is the design matrix for the regression part
of the model and :math:`\Gamma` is prior variance of the
regression parameters (as above).

We see that :math:`Y_{S \setminus s}` plays the role of the unknown
parameter in the general Bayes formulation. This is familiar in the
Bayesian context under the heading *predictive distributions*. To
summarise: select the design to achieve mimumum entropy (= maximum
information) of the joint predictive distribution for all the unsampled
points, by maximising the entropy of the sampled points.

Integrated Mean Squared Error (IMSE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This criterion aims at minimising the mean squared error of prediction
over the unsampled points. As for MES, above, this is based on the
predictive distribution for the unsampled points.

The mean squared prediction error (MSPE), under the standard model at a
single point is given by

.. math::
   \mbox{MSPE}(\hat{Y}(x))=\sigma^2\left[ 1-(f(x)^T \quad r(x)^T)\left[
   \begin{array}{cc}
   0 & F^T \\
   F & R
   \end{array}\right]\left(\begin{array}{c}
   f(x)\\
   r(x)
   \end{array}\right)\right]

Several criteria could be based on this quantity as the point
:math:`x` ranges over the design space :math:`\mathcal{X}`.

The integrated mean squared prediction error, which is the predictive
error averaged over, the design space, :math:`\mathcal{X}`,

.. math::
   J(\mathcal{D})=\int_{\mathcal{X}}\frac{\mbox{MSPE}[\hat{Y}(x)]}{\sigma^2}\omega(x)dx

where :math:`\omega(\cdot)` is a specified non-negative weight function
satisfying :math:`\int_{\mathcal{X}}\omega(x) dx`.

This criterion has been favoured by several authors in computer
experiments, (Sacks et al,1989).

Other criteria are (i) minimising over the design the maximum MSPE over
the unsampled points (ii) minimising the biggest eigenvalues of the
predictive (posterior) covariance matrix. Note that each of the above
criteria is the analogue of a classical criterion in which the
parameters are replaced by the values of the process at the unsampled
points in the candidate set: thus Maximum Entropy Sampling (in the
Gaussian case) is the analogue of D-optimality, IMSE is a type of
trace-optimality and (i) and (ii) above are types of types of G- and
E-optimality respectively.

Bayesian sequential optimum design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Bayesian paradigm is very useful in understanding sequential design.

After we have selected a criterion :math:`\phi`, see above, the
first stage design is to :math:`\min_{design_1} E_{Y_1}
\phi(\pi(\beta|Y_1)),` where :math:`Y_1` is the output generated in the
first stage experiment.

The (naive/myopic) one-step-ahead method is to take the prior
distribution (process) at stage 2 to be the posterior process having
observed :math:`Y_1` and procead to choose the stage 2 design to minimise
:math:`\min_{design_2} E_{Y_2} \phi(\pi(\beta|Y_1,Y_2)),`, and so on
through further stages, if necessary.

The full sequential rule, which is very difficult to implement, uses the
knowledge that one will use an optimal design at future stages to adjust
the "risk" at the first stage. For two stages this would give, working
backwards: :math:`R_2= \min_{design_2} E_{Y_2} \phi(\pi(\beta|Y_1,Y_2)),`
and then at the first stage choose the first design to achieve
:math:`\min_{design_1} E_{Y_1}( R_2)`. The general case is essentially
dynamic programming (Bellman rule) and with a finite horizon :math:`N` the
scheme would look like:

.. math::
   \min_{design_1} E_{Y_1} \min_{design_2} E_{Y_2} \ldots
   \min_{design_{N-1} } E_{Y_{N-1}} \phi(\pi(\beta|Y_1, \ldots, Y_N))

This full version of sequential design is very "expensive" because the
internal expectations require much numerical integration.

One can perform global optimisation to obtain an optimal design over a
design space :math:`\mathcal{X}`, but it is convenient to use a large
candidate set. As mentioned, this candidate set is typically taken to be
a *full factorial design* or a large *space-filling design*. In block
sequential design one may add a block optimally which may itself be a
smaller space-filling design.

It is useful to stress again notation for the design used in the MES
method, above. Thus let :math:`S` be the candidate set and
:math:`s` the design points and :math:`\bar{s} = S \setminus s` the
unsampled points. Then the optimal design problem is an optimal subset
problem: choose :math:`s \subset S` in an optimal way. This notation helps
to describe the important class of :ref:`exchange
algorithms<ProcExchangeAlgorithm>` in where points are exchanged
between :math:`s` and :math:`\bar{s}`.

A one-point-at-a-time myopic sequential design based on the MES
principle places each new design point at the point with maximum
posterior variance from the design to date. See also some discussion in
:ref:`DiscCoreDesign<DiscCoreDesign>`.

Additional Comments, References, and Links
------------------------------------------

There is a large literature on algorithms for optimal design and
algorithms are incorporated into commercial software, with the most
prevalent being algorithms for *D*-optimality. See, also
:ref:`AltOptimalDesignAlgorithms<AltOptimalDesignAlgorithms>`.

A. C. Atkinson and A. N. Donev. Optimum Experimental Designs. Oxford
Statistical Science Series, vol. 8. Oxford: Clarendon Press, 1992.

A. C. Atkinson, A. N. Donev and R. D. Tobias. Optimum Experimental
Designs, with SAS. 2007 Oxford, Oxford University Press. 207.

F. Pukelsheim. Optimal Design of Experiments Friedrich Pukelsheim,Siam:
Classics in Applied Mathematics 50. 2006 Original publication: Wiley,
1993.

J.Sacks, W. J. Welch, T. J. Mitchell, & H. P. Wynn, Design and analysis
of computer experiments, Statistical Science, 4(4):409-423, 1989.

JMP (SAS product): `http://www.jmp.com/software/ <http://www.jmp.com/software/>`_

K. Chaloner, and I. Verdinelli Bayesian experimental design: a review.
Statistical Science, 10, 3, 273--304, 1995.

M. C Shewry and H. P. Wynn. Maximum entropy sampling. Journal of Applied
Statistics, 14: 165-170, 1987.

MATLAB: Design of Experiments (Statistics Toolbox\ :sup:`TM`)
