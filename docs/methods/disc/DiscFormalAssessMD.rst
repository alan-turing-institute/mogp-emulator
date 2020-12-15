.. _DiscFormalAssessMD:

Discussion: Formal Assessment of Model Discrepancy
==================================================

Description and background
--------------------------

We consider formal :ref:`assessment<DefAssessment>` of the :ref:`model
discrepancy<DefModelDiscrepancy>` term :math:`d`
(:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`),
formulated to account for the mismatch between the
:ref:`simulator<DefSimulator>` :math:`f` evaluated at the :ref:`best
input<DefBestInput>` :math:`x^+`
(:ref:`DiscBestInput<DiscBestInput>`) and system value :math:`y`. Unlike
the informal methods described in
:ref:`DiscInformalAssessMD<DiscInformalAssessMD>`, the key ingredient
here is a specification of :math:`\textrm{Var}[d]` in terms of relatively
few unknown parameters :math:`\varphi`. The aim is to learn about
:math:`\varphi` from system observations :math:`z`
(:ref:`DiscObservations<DiscObservations>`) and simulator runs
(:ref:`ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`). Likelihood and Bayesian methods
are described. Assessing the distribution of :math:`d`, or just its
first and second order structure, necessary for a Bayes linear analysis,
is crucial for the important topics of history matching,
:ref:`calibration<DefCalibration>` and prediction for the real
system, which will be discussed in future Toolkit releases.

Discussion
----------

Our aim in this page is to describe formal methods for assessing the
distribution of the model discrepancy term :math:`d` using system
observations :math:`z` and simulator output :math:`F` from
:math:`n` simulator runs at design inputs :math:`D`. The
distribution of :math:`d` is often assumed to be Gaussian with
:math:`\textrm{E}[d]=0` and variance matrix :math:`\textrm{Var}[d]` depending
on a few parameters. The low-dimensional specification of
:math:`\textrm{Var}[d]` mostly arises from a subject matter expert, such as
the cosmologist in the galaxy formation study described in the first of
the two examples given at the end of this page. Initial assessment of
:math:`\textrm{E}[d]` is usually zero, but subsequent analysis may suggest
there is a 'trend' which is attributable to variation over and above
that accounted for by :math:`\textrm{Var}[d]`.

Notation
~~~~~~~~

We use the notation described in the discussion page on structured forms
for the model discrepancy
(:ref:`DiscStructuredMD<DiscStructuredMD>`), where there is a
'location' input :math:`u` (such as space-time) which indexes
simulator outputs as :math:`f(u,x)` and corresponding system values
:math:`y(u)` measured as observations :math:`z(u)` with model discrepancies
:math:`d(u).`

Suppose the system is observed at :math:`k` locations
:math:`u_1,\ldots, u_k`. Denote the corresponding observations by
:math:`z=(z_1,\ldots, z_k)` and the measurement errors by
:math:`\epsilon=(\epsilon_1,\ldots, \epsilon_k)`. Denote by
:math:`\Sigma_{ij}(\varphi)` the :math:`r \times r` covariance matrix
:math:`\textrm{Cov}[d(u_i),d(u_j)]` for any pair of locations :math:`u_i`
and :math:`u_j` in the set :math:`U` of allowable values of
:math:`u`. The number of unknown parameters in :math:`\varphi` is
assumed to be small compared to the number :math:`rk` of scalar
observations :math:`z_{ij}`. Denote by :math:`\Sigma_d(\varphi)` the
:math:`rk \times rk` variance matrix of the :math:`rk`-vector
:math:`d`, where
:math:`[\Sigma_d(\varphi)]_{ij}=\Sigma_{ij}(\varphi)`.

In the galaxy formation example, described below, :math:`\strut{r=1,k=11}`
and :math:`\varphi=(a,b,c)`, so that :math:`\Sigma_d(\varphi)` is an
:math:`11 \times 11` matrix. Initially, the cosmologist gave
specific values for :math:`a`, :math:`b` and :math:`c`, but
subsequently gave ranges. There are 11 observations in this
example, so inference about :math:`a`, :math:`b` and :math:`c` is
likely to be imprecise, unless prior information about them is precise
and reliable.

In the spot welding example, :math:`\strut{r=1}` and there are
10 replicate observations at each of the :math:`k=12= 2
\times 3 \times 2` locations. A simple form of separable covariance
is

.. math::
   \textrm{Cov}[d(l,c,g),d(l',c',g')]=\sigma^2 \exp
   (-(\theta_l(l-l')^2+\theta_c(c-c')^2+\theta_g(g-g')^2))

Here, :math:`\varphi=(\sigma^2, \theta_l,\theta_c,\theta_g)` and there are
48 observations to estimate these four parameters. In fact,
the replication level of 10 observations per location
suggests that simultaneous estimates of both :math:`\varphi` and the
measurement error variance :math:`\sigma_\epsilon^2` should be quite
precise.

Inference for :math:`\varphi`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now consider how to estimate the model discrepancy variance matrix
:math:`\Sigma_d(\varphi)` using an :ref:`emulator<DefEmulator>` and
system observations :math:`z`.

We start by choosing likelihood as a basis for inference about
:math:`\varphi`. The likelihood :math:`l(\varphi)` for :math:`\varphi` can be
computed as

.. math::
   l(\varphi) \propto \int p(z|D,F,\varphi, x^+) p(x^+)dx^+

where :math:`p(x^+)` is the prior distribution for :math:`x^+`. The
form of the integrand follows because of the separation between :math:`F`
and :math:`z` given :math:`f(x^+)` due to the strong independence
property between discrepancy :math:`d` and :math:`(f,x^+)`. The first
distribution in the integrand may also be interpreted as a joint
likelihood function for :math:`\varphi` and :math:`x^+`. Integration
over :math:`x^+` hides the potential for this joint likelihood
surface to be multimodal, as there will often be fits to the
observations for some choices of :math:`x^+` with small variance
and low correlation across outputs and other fits with large variance
and high correlation across outputs. However, we assume there is a prior
distribution for :math:`x^+`, so that :math:`l(\varphi)` is the
likelihood for :math:`\varphi`.

The expectation and variance of the first distribution in the integrand
can be computed as

.. math::
   \textrm{E}[z|D,F,\varphi, x^+] = \textrm{E}[z|D,F, x^+] =
   \mu(x^+)

and

.. math::
   \textrm{Var}[z|D,F,\varphi, x^+]= \Sigma (x^+) + \Sigma_d(\varphi)+
   \Sigma_\epsilon

where :math:`\mu(x)` and :math:`\Sigma (x)` are the emulator mean and emulator
variance matrix at input :math:`x` and :math:`\Sigma_\epsilon` is the
measurement error variance matrix. For simplicity, we typically assume a
Gaussian distribution for :math:`p(z|D,F,\varphi, x^+)`, but robustness of
inference to other distributions may be considered.

The integral in the expression for :math:` l(\varphi)`, which gives the
likelihood for any particular value of :math:`\varphi`, is computed using
numerical integration or by simulating from the prior distribution
:math:`p(x^+)` for :math:`x^+`. We can then proceed to compute the
maximum likelihood estimate :math:`\hat{\varphi}` and confidence regions
for :math:`\varphi` using the Hessian of the log-likelihood function at
:math:`\hat{\varphi}`. The maximum likelihood estimate of
:math:`\Sigma_d(\varphi)` is :math:`\Sigma_d(\hat{\varphi})`. Edwards, A. W.
F. (1972) gives an interesting account of likelihood.

If we are prepared to quantify our prior information about :math:`\varphi`
(for example, using the considerations of the discussion page on expert
assessment (:ref:`DiscExpertAssessMD<DiscExpertAssessMD>`)) in terms
of a prior distribution, then we may base inferences on its posterior
distribution, computed using Bayes theorem in the usual way.

Bayes linear inference for :math:`\varphi` proceeds as follows. (i)
Simulation to derive mean and covariance structures between
:math:`z` and :math:`x^+`, which are used to identify the
Bayes linear assessment :math:`\hat{x}` for :math:`x^+` adjusted by
:math:`z`; (ii) evaluation of the hat run :math:`\hat{f} =
f(\hat{x})`, as in Goldstein, M. and Rougier, J. C. (2006); (iii)
simulation to assess the mean, variance and covariance structures across
the squared components of the difference :math:`z - \hat{f}` and the
components of :math:`\varphi`, to carry out the corresponding Bayes linear
update for :math:`\varphi`.

Additional comments and examples
--------------------------------

It should be noted that when prediction for the real system and
calibration are considered in future releases of the toolkit, it will be
necessary to account for uncertainty in :math:`\varphi` in the overall
uncertainty of these procedures.

Galaxy formation
~~~~~~~~~~~~~~~~

Goldstein, M. and Vernon, I. (2009) consider the galaxy formation model
'Galform' which simulates two outputs, the :math:`b_j` and :math:`K`
band luminosity functions. The :math:`b_j` band gives numbers of young
galaxies :math:`s` per unit volume of different luminosities, while
the :math:`K` band describes the number of old galaxies
:math:`l`. The authors consider 11 representative
outputs, 6 from the :math:`b_j` band and 5 from
the :math:`K` band. Here, :math:`u=(A,\lambda)` is age :math:`A`
and luminosity :math:`\lambda` and :math:`y(A,\lambda)` is count of
age :math:`A` galaxies of luminosity :math:`\lambda` per unit
volume of space. The authors carried out a careful elicitation process
with the cosmologists for :math:`\log y` and specified a covariance
:math:`\textrm{Cov}[d(A_i,\lambda_l),d(A_j,\lambda_k)]` between :math:`d(A_i,
\lambda_l)` and :math:`d(A_j, \lambda_k)` of the form

.. math::
   a \left[ \begin{array}{cccccc} 1 & b & .. & c & .. & c \\ b & 1
   & .. & c & . & c \\ : & : & : & : & : & : \\ c & .. & c & 1 & b & ..
   \\ c & .. & c & b & 1 & .. \\ : & : & : & : & : & : \end{array}
   \right]

for specified values of the overall variance :math:`a`, the
correlation within bands :math:`b` and the correlation between
bands :math:`c`. The input vector :math:`x` has eight
components.

Spot welding
~~~~~~~~~~~~

Higdon, D., Kennedy, M., Cavendish, J. C., Cafeo, J. A., and Ryne, R. D.
(2004) consider a model for spot welding which simulates spot weld
nugget diameter for different combinations of load and current applied
to two metal sheets, and gauge is the thickness of the two sheets. Here,
:math:`u=(l,c,g)` represents load :math:`l`, current :math:`c`
and gauge :math:`g` and :math:`y(l, c, g)` is the weld diameter when
load :math:`l` and current :math:`c` are applied to sheets of
gauge :math:`g` at the :math:`12=2 \times 3 \times 2` combinations.
Moreover, there is system replication of 10 observations
for each of the 12 system combinations. The authors specify
a Gaussian process for :math:`d` over :math:`(l, c, g)` combinations,
using a separable covariance structure. There is one scalar input.

References
----------

Edwards, A. W. F. (1972), "Likelihood", Cambridge (expanded edition,
1992, Johns Hopkins University Press, Baltimore): Cambridge University
Press.

Goldstein, M. and Rougier, J. C. (2006), "Bayes linear calibrated
prediction for complex systems", *Journal of the American Statistical
Association*, 101, 1132-1143.

Goldstein, M. and Vernon, I. (2009), "Bayes linear analysis of
imprecision in computer models, with application to understanding the
Universe", in 6th International Symposium on Imprecise Probability:
Theories and Applications.

Higdon, D., Kennedy, M., Cavendish, J. C., Cafeo, J. A., and Ryne, R. D.
(2004), "Combining field data and computer simulations for calibration
and prediction", *SIAM Journal on Scientific Computing*, 26, 448–466.
