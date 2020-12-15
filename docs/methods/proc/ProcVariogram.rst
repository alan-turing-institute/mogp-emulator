.. _ProcVariogram:

Procedure: Variogram estimation of covariance function hyperparameters
======================================================================

Description and Background
--------------------------

Variogram estimation is an empirical procedure used to :ref:`estimate the
variance and correlation parameters<AltEstimateDelta>`,
:math:`(\sigma^2,\delta)` in a stochastic process. Variogram estimation has
been typically used to assess the degree of spatial dependence in
spatial random fields, such as models based on geological structures.
Since the general :ref:`emulation<DefEmulator>` methodology shares
many similarities with spatial processes, variogram estimation can be
applied to the output of complex computer models in order to assess
covariance :ref:`hyperparameters<DefHyperparameter>`.

Since variogram estimation is a numerical optimisation procedure it
typically requires a very large number of evaluations, is relatively
computationally intensive, and suffers the same problems as other
optimisation strategies such as maximum likelihood approaches.

The variogram itself is defined to be the expected squared increment of
the output values between input locations :math:`x` and :math:`x'`:

.. math::
   2\gamma(x,x')=\textrm{E}[|f(x)-f(x')|^2]

Technically, the function :math:`\gamma(x,x')` is referred to as the
"semi-variogram" though the two terms are often used interchangeably.
For full details of the properties of and the theory behind variograms,
see Cressie (1993) or Chiles and Delfiner (1999).

The basis of this procedure is the result that any two points :math:`x`
and :math:`x'` which are separated by a distance of (approximately)
:math:`t` will have (approximately) the same value for the variogram
:math:`\gamma(x,x')`. Given a large sample of observed values, we can
identify all point pairs which are approximately separated by a distance
:math:`t` in the input space, and use their difference in observed
values to estimate the variogram.

If the stochastic process :math:`f(x)` has mean zero then the variogram is
related to the variance parameter :math:`\sigma^2` and the :ref:`correlation
function<AltCorrelationFunction>` as follows

.. math::
   2\gamma(x,x') = 2\sigma^2(1-\textrm{Cor}[f(x),f(x')]).

Thus estimation of the variogram function permits the estimation of the
collection of covariance hyperparameters.

Typically, the variogram estimation is applied to the emulator residuals
which are a weakly stationary stochastic process with zero mean. For
each point at which we have evaluated the computer model, we calculate
:math:`w(x)=f(x)-\hat{\beta}^Th(x)`, where :math:`\hat{\beta}` are the
updated values for the coefficients of the linear mean function. We then
apply the procedure below to obtain estimates for :math:`(\sigma^2,\delta)`.

Inputs
------

-  A vector of :math:`n` emulator residuals :math:`(w(x_1), w(x_2), \dots,
   w(x_n))`
-  The :math:`n`-point design, :math:`X`
-  A form for the correlation function :math:`c(x,x')`
-  Starting values for the hyperparameters :math:`(\sigma^2, \delta)`

Outputs
-------

-  Variogram estimates for :math:`(\sigma^2, \delta)`

Procedure
---------

Collect empirical information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. For each pair of residuals :math:`(w(x_i), w(x_j))`, calculate the
   absolute inter-point distance :math:`h_i=||x_i-x_j||^2` and the
   absolute residual difference :math:`e_{ij}=|w(x_i)-w(x_j)|`
#. Thus we obtain the :math:`n(n-1)/2` vector of inter-point distances
   :math:`t=(t_k)`, and the vector of absolute residual differences
   :math:`e=(e_k)`
#. Divide the range of :math:`t` into :math:`N` intervals,
   :math:`\mathcal{I}_a`
#. For each of the :math:`N` intervals, calculate:

   #. The number of distances within that interval, :math:`n_a=\#\{t_{ij} :
      t_{ij}\in \mathcal{I}_a\}`,
   #. The average inter-point separation for that interval,
      :math:`\bar{t}_a=\frac{1}{n_a} \sum_{t_{ij}\in\mathcal{I}_a} t_{ij}`
   #. An empirical variogram estimate -- the classical estimator
      :math:`\hat{g}_a`, or the robust estimator, :math:`\tilde{g}_a`, as
      follows:

      .. math::
         \hat{g}_a=\frac{1}{n_a} \sum_{t_{ij}\in\mathcal{I}_a} e_{ij}^2

      .. math::
         \tilde{g}_a= \frac{(\sum_{t_{ij}\in\mathcal{I}_a} e_{ij}^{0.5} /
         n_a)^4}{(0.457+0.494/n_a)}

Either of the two variogram estimators can be used in the estimation
procedure, however the classical estimator is noted to be sensitive to
outliers whereas the robust estimator has been developed to mitigate
this.

Fit the variogram model
~~~~~~~~~~~~~~~~~~~~~~~

Given the statistics :math:`n_a`, :math:`t_a`, and an empirical variogram
estimate, :math:`g_a`, for each interval, :math:`\mathcal{I}_a`, we now fit
the variogram model by weighted least squares. This typically requires
extensive numerical optimisation over the space of possible values for
:math:`(\sigma^2, \delta)`.

For a given choice of :math:`(\sigma^2, \delta)`, we calculate the
theoretical variogram :math:`\gamma_a` for each interval :math:`\mathcal{I}_a`
at mean separation :math:`\bar{t}_a`. The theoretical variogram for a
Gaussian correlation function with correlation length parameter
:math:`\delta` and at inter-point separation :math:`\bar{t}_a` is given by

.. math::
   \gamma_a=\gamma(\bar{t}_a)=\sigma^2(1-\exp\{-\bar{t}_a^TM\bar{t}_a\}),

where :math:`M` is a diagonal matrix with elements :math:`1/\delta^2`.

Similarly, the theoretical variogram for a Gaussian correlation function
in the presence of a nugget term with variance :math:`\alpha\sigma^2` is

.. math::
   \gamma=\gamma(\bar{t}_a)=\sigma^2(1-\alpha)
   (1-\exp\{-\bar{t}_a^TM\bar{t}_a\})+\alpha\sigma^2.

Beginning with the specified starting values for :math:`(\sigma^2,\delta)`,
we then numerically minimise the following expression,

.. math::
   W = \sum_a 0.5 n_a (g_a/\gamma_a-1)^2,

for :math:`(\sigma^2,\delta)` over their feasible ranges.

References
----------

#. Cressie, N., 1993, Statistics for spatial data, Wiley Interscience
#. Chiles, J.P., P. Delfiner, 1999, Geostatististics, Modelling Spatial
   Uncertainty, Wiley-Interscience
