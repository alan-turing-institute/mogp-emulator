.. _DiscStructuredMD:

Discussion: Structured Forms for Model Discrepancy
==================================================

Description and background
--------------------------

The basic ideas of :ref:`model discrepancy<DefModelDiscrepancy>` are
presented in the variant thread on linking models to reality
(:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`).
There, the key equations are :math:`\strut{y=f(x^+) + d}` and
:math::ref:`\strut{z=y+\epsilon}`, where :math:`\strut{y}` represents system
values, :math:`\strut{f}` is the `simultaor<DefSimulator>`,
:math::ref:`\strut{x^+}` is the `best input<DefBestInput>` and
:math::ref:`\strut{z}` are observations on :math:`\strut{y}` with measurement error
:math:`\strut{\epsilon}` (`DiscObservations<DiscObservations>`). We
expand that notation here to allow explicit indexing of these quantities
by other inputs; for example, space-time co-ordinates. Possible
situations where :math:`\strut{d}` has a highly structured form due to
physical structures present in the model are discussed.

Discussion
----------

In accordance with the standard toolkit notation, we denote the model
simulator by :math:`\strut{f}` and its inputs by :math:`\strut{x}`. We extend
the notation used in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
to include generalised \`location' inputs :math:`\strut{u}`; for example,
:math:`\strut{u}` might index space-time, such as latitude, longitude and
date, and :math:`\strut{y(u)}` might denote the real average temperature of
the earth's surface at that latitude and longitude over a particular
day. We write :math:`\strut{ f(u, x)}` and :math:`\strut{y(u)}` to denote the
simulator output for inputs :math:`\strut{x}` at location :math:`\strut{u}`
and the system value at location :math:`\strut{u}`, respectively. The
fundamental relationship linking model to reality given in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
now becomes

:math:`\strut{ y(u)= f(u, x^+) + d(u) }`

where :math:`\strut{d(u)}` is the model discrepancy between the real system
value :math:`\strut{y(u)}` at :math:`\strut{u}` and the simulator value
:math:`\strut{ f(u, x^+)}` for the best input :math:`\strut{x^+}` at
:math::ref:`\strut{u}`: for detailed discussions about :math:`\strut{x^+}` and
:math:`\strut{d(u)}` see `DiscBestInput<DiscBestInput>` and
:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>`. As there may
be replicate observations of the real system at a location
:math:`\strut{u}`, we introduce notation to cover this case. Denote by
:math:`\strut{z_r(u)}` the :math:`\strut{r}`-th replicate observation of the
system value :math::ref:`\strut{y(u)}` at location :math:`\strut{u}`. As in
`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
and :ref:`DiscObservations<DiscObservations>`, we assume that the
:math:`\strut{z_r(u)}` and the real system value :math:`\strut{y(u)}` are
additively related through the observation equation

:math:`\strut{ z_r(u) = y(u) + \\epsilon_r(u) }`

where :math:`\strut{\epsilon_r(u)}` is the measurement error of the
:math:`\strut{r}`-th replicate observation on :math:`\strut{y(u)}`. We write
:math:`\strut{\epsilon(u)}` when there is no replication. Combining these
equations, we obtain the following relationship between system
observations and simulator output

:math:`\strut{ z_r(u)=f(u, x^+) + d(u) + \\epsilon_r(u) }`

Statistical assumptions
~~~~~~~~~~~~~~~~~~~~~~~

#. The components of the collection :math:`\strut{\{f(u, \\cdot )}`,
   :math:`\strut{x^+}`, :math:`\strut{d(u)}`, :math:`\strut{\epsilon_r(u)\}}` are
   assumed to be independent random quantities for each :math:`\strut{u}`
   in the collection :math:`\strut{U}` of :math:`\strut{u}`-values considered,
   or just uncorrelated in the Bayes linear case
   (:ref:`ThreadCoreBL<ThreadCoreBL>`).
#. The distribution of :math:`\strut{\{f(u, \\cdot )}`, :math:`\strut{x^+}`,
   :math:`\strut{d(u)}`, :math:`\strut{\epsilon_r(u)\}}` over all finite
   collections of :math:`\strut{u}`-values in :math:`\strut{U}` needs to be
   specified in any particular application.
#. The model discrepancy term :math:`\strut{d(\cdot)}` is regarded as a
   random process over the collection :math:`\strut{U}`. Either the
   distribution of :math:`\strut{d(\cdot)}` is specified over all finite
   subsets of :math::ref:`\strut{U}`, such as a `Gaussian
   process<DefGP>` for a full Bayes analysis
   (:ref:`ThreadCoreGP<ThreadCoreGP>`), or just the expectation and
   covariance structure of :math:`\strut{d(\cdot)}` are specified over all
   finite subsets of :math::ref:`\strut{U}` as is required for a Bayes linear
   analysis (`ThreadCoreBL<ThreadCoreBL>`). A separable
   covariance structure for :math::ref:`\strut{d(\cdot)}`, similar to that
   defined in `DefSeparable<DefSeparable>`, might be a
   convenient simple choice when, for example, :math:`\strut{u}` indexes
   space-time.
#. As the system is observed at only a finite number of locations
   :math:`\strut{u_1, \\ldots, u_k}` (say), unlike :math:`\strut{d(\cdot)}`,
   we do not regard :math:`\strut{\epsilon_r(\cdot)}` as a random process
   over :math:`\strut{U}`. In most applications, the measurement errors
   :math:`\strut{\epsilon_{r_i}(u_i)}` are assumed to be independent and
   identically distributed random quantities with zero mean and variance
   :math:`\strut{\Sigma_\epsilon}`. However, measurement errors may be
   correlated across the locations :math:`\strut{u_1, \\ldots, u_k}`, which
   might be the case, for example, when :math:`\strut{u}` indexes time. On
   the other hand, the replicate observations within each of locations
   :math:`\strut{u_1, \\ldots, u_k}` are almost always assumed to be
   independent and identically distributed; see
   :ref:`DiscObservations<DiscObservations>`.

Additional comments and Examples
--------------------------------

Kennedy, M. C. and O’Hagan, A. (2001) generalise the combined equation
to

:math:`\strut{ z_r(u)=\rho f(u, x^+) + d(u) + \\epsilon_r(u) }`

where :math:`\strut{\rho}` is a parameter to be specified or estimated.

Rainfall runoff
~~~~~~~~~~~~~~~

Iorgulescu, I., Beven, K. J., and Musy, A. (2005) consider a rainfall
runoff model which simulates consecutive hourly measurements of water
discharge :math:`\strut{D}` and Calcium :math:`\strut{C}` and Silica
:math:`\strut{S}` concentrations for a particular catchment area. Here,
:math:`\strut{u=t}` is hour :math:`\strut{t}` and :math:`\strut{y(t)=
(D(t),C(t),S(t))}` is the amount of water discharged into streams and
the Calcium and Silica concentrations at hour :math:`\strut{t}`. There were
:math:`\strut{839}` hourly values of :math:`\strut{t}`. While the authors did
not consider model discrepancy explicitly, a simple choice would be to
specify the covariance :math:`\strut{\textrm{Cov}[d(t_k),d(t_{\ell})]}`
between :math:`\strut{d(t_k)}` and :math:`\strut{d(t_{\ell})}` to be of the
form

:math:`\strut{ \\sigma^2 \\exp\left(-\theta (t_k-t_{\ell})^2\right) }`

A more realistic simple choice might be the covariance structure derived
from the autoregressive scheme :math:`\strut{d(t_{k+1})=\rho d(t_k) +
\\eta_k}`, where :math:`\strut{\{\eta_k\}}` is a white noise process with
variance :math:`\strut{\sigma^2}`; the parameters :math:`\strut{\rho}` and
:math:`\strut{\sigma^2}` are to be specified or estimated. Such a
covariance structure would be particularly appropriate when forecasting
future runoff.

The input vector :math::ref:`\strut{x}` has eighteen components. An informal
`assessment<DefAssessment>` of model discrepancy for this runoff
model is given in Goldstein, M., Seheult, A. and Vernon, I. (2010): see
also :ref:`DiscInformalAssessMD<DiscInformalAssessMD>`.

Galaxy formation
~~~~~~~~~~~~~~~~

Goldstein, M. and Vernon, I. (2009) consider the galaxy formation model
\`Galform' which simulates two outputs, the :math:`\strut{b_j}` and
:math:`\strut{K}` band luminosity functions. The :math:`\strut{b_j}` band
gives numbers of young galaxies :math:`\strut{s}` per unit volume of
different luminosities, while the :math:`\strut{K}` band describes the
number of old galaxies :math:`\strut{l}`. The authors consider
:math:`\strut{11}` representative outputs, :math:`\strut{6}` from the
:math:`\strut{b_j}` band and :math:`\strut{5}` from the :math:`\strut{K}` band.
Here, :math:`\strut{u=(A,\lambda)}` is age :math:`\strut{A}` and luminosity
:math:`\strut{\lambda}` and :math:`\strut{y(A,\lambda)}` is count of age
:math:`\strut{A}` galaxies of luminosity :math:`\strut{\lambda}` per unit
volume of space. The authors carried out a careful elicitation process
with the cosmologists for :math:`\strut{\log y}` and specified a covariance
:math:`\strut{\textrm{Cov}[d(A_i,\lambda_l),d(A_j,\lambda_k)]}` between
:math:`\strut{d(A_i, \\lambda_l)}` and :math:`\strut{d(A_j, \\lambda_k)}` of
the form

:math:`\strut{ a \\left[ \\begin{array}{cccccc} 1 & b & .. & c & .. & c \\\\
b & 1 & .. & c & . & c \\\\ : & : & : & : & : & : \\\\ c & .. & c & 1 &
b & .. \\\\ c & .. & c & b & 1 & .. \\\: & : & : & : & : & :
\\end{array} \\right] }`

for specified values of the overall variance :math:`\strut{a}`, the
correlation within bands :math:`\strut{b}` and the correlation between
bands :math:`\strut{c}`. The input vector :math:`\strut{x}` has eight
components.

Hydrocarbon reservoir
~~~~~~~~~~~~~~~~~~~~~

Craig, P. S., Goldstein, M., Rougier, J. C., and Seheult, A. H. (2001)
consider a hydrocarbon reservoir model which simulates bottom-hole
pressures of different wells through time. Here, :math:`\strut{u=(w,t)}` is
the pair well :math:`\strut{w}` and time :math:`\strut{t}` and
:math:`\strut{y(w,t)}` is the bottom-hole pressure for well :math:`\strut{w}`
at time :math:`\strut{t}`. There were :math:`\strut{34}` combinations of
:math:`\strut{w}` and :math:`\strut{t}` considered. The authors specify a
non-separable covariance
:math:`\strut{\textrm{Cov}[d(w_i,t_k),d(w_j,t_{\ell})]}` between
:math:`\strut{d(w_i,t_k)}` and :math:`\strut{d(w_j,t_{\ell})}` of the form

:math:`\strut{ \\sigma_1^2 \\exp\left(-\theta_1(t_k-t_{\ell})^2\right) +
\\sigma_2^2 \\exp\left(-\theta_2 (t_k-t_{\ell})^2\right) I_{w_i=w_j} }`

where :math:`\strut{I_P}` denotes the indicator function of the proposition
:math::ref:`\strut{P}`. The input vector :math:`\strut{x}` has four active
components: see `DefActiveInput<DefActiveInput>`.

Spot welding
~~~~~~~~~~~~

Higdon, D., Kennedy, M., Cavendish, J. C., Cafeo, J. A., and Ryne, R. D.
(2004) consider a model for spot welding which simulates spot weld
nugget diameters for different combinations of load, and current applied
to two metal sheets, and gauge is the thickness of the two sheets. Here,
:math:`\strut{u=(l,c,g)}` is the triple load :math:`\strut{l}`, current
:math:`\strut{c}` and gauge :math:`\strut{g}` and :math:`\strut{y(l, c, g)}` is
the weld diameter when load :math:`\strut{l}` and current :math:`\strut{c}`
are applied to sheets of gauge :math:`\strut{g}` at the :math:`\strut{12=2
\\times 3 \\times 2}` combinations. Moreover, there is system
replication of :math:`\strut{10}` observations for each of the
:math:`\strut{12}` system combinations. The authors specify a Gaussian
process for the model discrepancy :math:`\strut{d}` over :math:`\strut{(l, c,
g)}` with a separable covariance structure. The input vector
:math:`\strut{x}` has one component.

References
----------

Craig, P. S., Goldstein, M., Rougier, J. C., and Seheult, A. H. (2001),
"Bayesian forecasting for complex systems using computer simulators",
Journal of the American Statistical Association, 96, 717-729.

Goldstein, M. and Vernon, I. (2009), "Bayes linear analysis of
imprecision in computer models, with application to understanding the
Universe", in 6th International Symposium on Imprecise Probability:
Theories and Applications.

Goldstein, M., Seheult, A. and Vernon, I. (2010), "Assessing Model
Adequacy", MUCM Technical Report 10/04.

Higdon, D., Kennedy, M., Cavendish, J. C., Cafeo, J. A., and Ryne, R. D.
(2004), "Combining field data and computer simulations for calibration
and prediction", SIAM Journal on Scientific Computing, 26, 448-466.

Iorgulescu, I., Beven, K. J., and Musy, A. (2005), "Data-based modelling
of runoff and chemical tracer concentrations in the Haute-Mentue
research catchment (Switzerland)", Hydrological Processes, 19,
2557-2573.

Kennedy, M. C. and O’Hagan, A. (2001), "Bayesian calibration of computer
models", Journal of the Royal Statistical Society, Series B, 63,
425-464.
