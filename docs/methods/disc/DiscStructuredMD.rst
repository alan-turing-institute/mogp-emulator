.. _DiscStructuredMD:

Discussion: Structured Forms for Model Discrepancy
==================================================

Description and background
--------------------------

The basic ideas of :ref:`model discrepancy<DefModelDiscrepancy>` are
presented in the variant thread on linking models to reality
(:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`).
There, the key equations are :math:`y=f(x^+) + d` and
:math:`z=y+\epsilon`, where :math:`y` represents system
values, :math:`f` is the :ref:`simultaor<DefSimulator>`,
:math:`x^+` is the :ref:`best input<DefBestInput>` and
:math:`z` are observations on :math:`y` with measurement error
:math:`\epsilon` (:ref:`DiscObservations<DiscObservations>`). We
expand that notation here to allow explicit indexing of these quantities
by other inputs; for example, space-time co-ordinates. Possible
situations where :math:`d` has a highly structured form due to
physical structures present in the model are discussed.

Discussion
----------

In accordance with the standard toolkit notation, we denote the model
simulator by :math:`f` and its inputs by :math:`x`. We extend
the notation used in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
to include generalised 'location' inputs :math:`u`; for example,
:math:`u` might index space-time, such as latitude, longitude and
date, and :math:`y(u)` might denote the real average temperature of
the earth's surface at that latitude and longitude over a particular
day. We write :math:`f(u, x)` and :math:`y(u)` to denote the
simulator output for inputs :math:`x` at location :math:`u`
and the system value at location :math:`u`, respectively. The
fundamental relationship linking model to reality given in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
now becomes

.. math::
   y(u)= f(u, x^+) + d(u)

where :math:`d(u)` is the model discrepancy between the real system
value :math:`y(u)` at :math:`u` and the simulator value
:math:`f(u, x^+)` for the best input :math:`x^+` at
:math:`u`: for detailed discussions about :math:`x^+` and
:math:`d(u)` see :ref:`DiscBestInput<DiscBestInput>` and
:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>`. As there may
be replicate observations of the real system at a location
:math:`u`, we introduce notation to cover this case. Denote by
:math:`z_r(u)` the :math:`r`-th replicate observation of the
system value :math:`y(u)` at location :math:`u`. As in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
and :ref:`DiscObservations<DiscObservations>`, we assume that the
:math:`z_r(u)` and the real system value :math:`y(u)` are
additively related through the observation equation

.. math::
   z_r(u) = y(u) + \epsilon_r(u)

where :math:`\epsilon_r(u)` is the measurement error of the
:math:`r`-th replicate observation on :math:`y(u)`. We write
:math:`\epsilon(u)` when there is no replication. Combining these
equations, we obtain the following relationship between system
observations and simulator output

.. math::
   z_r(u)=f(u, x^+) + d(u) + \epsilon_r(u)

Statistical assumptions
~~~~~~~~~~~~~~~~~~~~~~~

#. The components of the collection :math:`f(u, \cdot )`,
   :math:`x^+`, :math:`d(u)`, :math:`\epsilon_r(u)` are
   assumed to be independent random quantities for each :math:`u`
   in the collection :math:`U` of :math:`u`-values considered,
   or just uncorrelated in the Bayes linear case
   (:ref:`ThreadCoreBL<ThreadCoreBL>`).
#. The distribution of :math:`f(u, \cdot )`, :math:`x^+`,
   :math:`d(u)`, :math:`\epsilon_r(u)` over all finite
   collections of :math:`u`-values in :math:`U` needs to be
   specified in any particular application.
#. The model discrepancy term :math:`d(\cdot)` is regarded as a
   random process over the collection :math:`U`. Either the
   distribution of :math:`d(\cdot)` is specified over all finite
   subsets of :math:`U`, such as a :ref:`Gaussian
   process<DefGP>` for a full Bayes analysis
   (:ref:`ThreadCoreGP<ThreadCoreGP>`), or just the expectation and
   covariance structure of :math:`d(\cdot)` are specified over all
   finite subsets of :math:`U` as is required for a Bayes linear
   analysis (:ref:`ThreadCoreBL<ThreadCoreBL>`). A separable
   covariance structure for :math:`d(\cdot)`, similar to that
   defined in :ref:`DefSeparable<DefSeparable>`, might be a
   convenient simple choice when, for example, :math:`u` indexes
   space-time.
#. As the system is observed at only a finite number of locations
   :math:`u_1, \ldots, u_k` (say), unlike :math:`d(\cdot)`,
   we do not regard :math:`\epsilon_r(\cdot)` as a random process
   over :math:`U`. In most applications, the measurement errors
   :math:`\epsilon_{r_i}(u_i)` are assumed to be independent and
   identically distributed random quantities with zero mean and variance
   :math:`\Sigma_\epsilon`. However, measurement errors may be
   correlated across the locations :math:`u_1, \ldots, u_k`, which
   might be the case, for example, when :math:`u` indexes time. On
   the other hand, the replicate observations within each of locations
   :math:`u_1, \ldots, u_k` are almost always assumed to be
   independent and identically distributed; see
   :ref:`DiscObservations<DiscObservations>`.

Additional comments and Examples
--------------------------------

Kennedy, M. C. and O’Hagan, A. (2001) generalise the combined equation
to

.. math::
   z_r(u)=\rho f(u, x^+) + d(u) + \epsilon_r(u)

where :math:`\rho` is a parameter to be specified or estimated.

Rainfall runoff
~~~~~~~~~~~~~~~

Iorgulescu, I., Beven, K. J., and Musy, A. (2005) consider a rainfall
runoff model which simulates consecutive hourly measurements of water
discharge :math:`D` and Calcium :math:`C` and Silica
:math:`S` concentrations for a particular catchment area. Here,
:math:`u=t` is hour :math:`t` and :math:`y(t)=
(D(t),C(t),S(t))` is the amount of water discharged into streams and
the Calcium and Silica concentrations at hour :math:`t`. There were
839 hourly values of :math:`t`. While the authors did
not consider model discrepancy explicitly, a simple choice would be to
specify the covariance :math:`\textrm{Cov}[d(t_k),d(t_{\ell})]`
between :math:`d(t_k)` and :math:`d(t_{\ell})` to be of the
form

.. math::
   \sigma^2 \exp\left(-\theta (t_k-t_{\ell})^2\right)

A more realistic simple choice might be the covariance structure derived
from the autoregressive scheme :math:`d(t_{k+1})=\rho d(t_k) +
\eta_k`, where :math:`\eta_k` is a white noise process with
variance :math:`\sigma^2`; the parameters :math:`\rho` and
:math:`\sigma^2` are to be specified or estimated. Such a
covariance structure would be particularly appropriate when forecasting
future runoff.

The input vector :math:`x` has eighteen components. An informal
:ref:`assessment<DefAssessment>` of model discrepancy for this runoff
model is given in Goldstein, M., Seheult, A. and Vernon, I. (2010): see
also :ref:`DiscInformalAssessMD<DiscInformalAssessMD>`.

Galaxy formation
~~~~~~~~~~~~~~~~

Goldstein, M. and Vernon, I. (2009) consider the galaxy formation model
'Galform' which simulates two outputs, the :math:`b_j` and
:math:`K` band luminosity functions. The :math:`b_j` band
gives numbers of young galaxies :math:`s` per unit volume of
different luminosities, while the :math:`K` band describes the
number of old galaxies :math:`l`. The authors consider
11 representative outputs, 6 from the
:math:`b_j` band and 5 from the :math:`K` band.
Here, :math:`u=(A,\lambda)` is age :math:`A` and luminosity
:math:`\lambda` and :math:`y(A,\lambda)` is count of age
:math:`A` galaxies of luminosity :math:`\lambda` per unit
volume of space. The authors carried out a careful elicitation process
with the cosmologists for :math:`\log y` and specified a covariance
:math:`\textrm{Cov}[d(A_i,\lambda_l),d(A_j,\lambda_k)]` between
:math:`d(A_i, \lambda_l)` and :math:`d(A_j, \lambda_k)` of
the form

.. math::
   a \left[ \begin{array}{cccccc} 1 & b & .. & c & .. & c \\
   b & 1 & .. & c & . & c \\ : & : & : & : & : & : \\ c & .. & c & 1 &
   b & .. \\ c & .. & c & b & 1 & .. \\ : & : & : & : & : & :
   \end{array} \right]

for specified values of the overall variance :math:`a`, the
correlation within bands :math:`b` and the correlation between
bands :math:`c`. The input vector :math:`x` has eight
components.

Hydrocarbon reservoir
~~~~~~~~~~~~~~~~~~~~~

Craig, P. S., Goldstein, M., Rougier, J. C., and Seheult, A. H. (2001)
consider a hydrocarbon reservoir model which simulates bottom-hole
pressures of different wells through time. Here, :math:`u=(w,t)` is
the pair well :math:`w` and time :math:`t` and
:math:`y(w,t)` is the bottom-hole pressure for well :math:`w`
at time :math:`t`. There were 34 combinations of
:math:`w` and :math:`t` considered. The authors specify a
non-separable covariance
:math:`\textrm{Cov}[d(w_i,t_k),d(w_j,t_{\ell})]` between
:math:`d(w_i,t_k)` and :math:`d(w_j,t_{\ell})` of the form

.. math::
   \sigma_1^2 \exp\left(-\theta_1(t_k-t_{\ell})^2\right) +
   \sigma_2^2 \exp\left(-\theta_2 (t_k-t_{\ell})^2\right) I_{w_i=w_j}

where :math:`I_P` denotes the indicator function of the proposition
:math:`P`. The input vector :math:`x` has four active
components: see :ref:`DefActiveInput<DefActiveInput>`.

Spot welding
~~~~~~~~~~~~

Higdon, D., Kennedy, M., Cavendish, J. C., Cafeo, J. A., and Ryne, R. D.
(2004) consider a model for spot welding which simulates spot weld
nugget diameters for different combinations of load, and current applied
to two metal sheets, and gauge is the thickness of the two sheets. Here,
:math:`u=(l,c,g)` is the triple load :math:`l`, current
:math:`c` and gauge :math:`g` and :math:`y(l, c, g)` is
the weld diameter when load :math:`l` and current :math:`c`
are applied to sheets of gauge :math:`g` at the :math:`12=2
\times 3 \times 2` combinations. Moreover, there is system
replication of 10 observations for each of the
12 system combinations. The authors specify a Gaussian
process for the model discrepancy :math:`d` over :math:`(l, c,
g)` with a separable covariance structure. The input vector
:math:`x` has one component.

References
----------

Craig, P. S., Goldstein, M., Rougier, J. C., and Seheult, A. H. (2001),
"Bayesian forecasting for complex systems using computer simulators",
*Journal of the American Statistical Association*, 96, 717-729.

Goldstein, M. and Vernon, I. (2009), "Bayes linear analysis of
imprecision in computer models, with application to understanding the
Universe", in 6th International Symposium on Imprecise Probability:
Theories and Applications.

Goldstein, M., Seheult, A. and Vernon, I. (2010), "Assessing Model
Adequacy", MUCM Technical Report 10/04.

Higdon, D., Kennedy, M., Cavendish, J. C., Cafeo, J. A., and Ryne, R. D.
(2004), "Combining field data and computer simulations for calibration
and prediction", *SIAM Journal on Scientific Computing*, 26, 448-466.

Iorgulescu, I., Beven, K. J., and Musy, A. (2005), "Data-based modelling
of runoff and chemical tracer concentrations in the Haute-Mentue
research catchment (Switzerland)", *Hydrological Processes*, 19,
2557-2573.

Kennedy, M. C. and O’Hagan, A. (2001), "Bayesian calibration of computer
models", *Journal of the Royal Statistical Society, Series B*, 63,
425-464.
