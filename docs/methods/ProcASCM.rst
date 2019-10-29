.. _ProcASCM:

Procedure: Adaptive Sampler for Complex Models (ASCM)
=====================================================

Description and Background
--------------------------

This procedure aims at sequentially selecting the design points and
updating the information at every stage. The framework used to implement
this procedure is the Bayesian decision theory. Natural conjugate priors
are assumed for the unknown parameters. The ASCM procedure allows for
learning about the parameters and assessing the emulator performance at
every stage. It uses the Bayesian optimal design principals in :ref:`Optimal
design<DefModelBasedDesign>`.

Inputs
------

-  The design size :math:`\strut{n}`, the subdesign size :math:`\strut{n_i}`,
   a counter :math:`\strut{nn=0}`, initial design :math:`\strut{D_0}` usually
   a space filling design, the candidate set :math:`\strut{E}` of size
   :math:`\strut{N}`, also another space filling design usually chosen to
   be a grid defined on the design space :math:`\strut{\mathcal{X}}`.
-  The output at the initial design points, :math:`Y(D_{0})`.
-  A model for the output, for simplicity, the model is chosen to be
   :math:`Y(x)=X\theta+Z(x)`.
-  A covariance function associated with the Gaussian process model. In
   the Karhunen-Loeve method the "true" covariance function is replaced
   by a truncated version of the K-L expansion
   :ref:`DiscKarhunenLoeveExpansion<DiscKarhunenLoeveExpansion>`.
-  A prior distribution for the unkonwn model parameters :math:`\theta`,
   which is taken here to be :math:`N(\mu,V)` .
-  An optimality criterion.
-  Note that :math:`\Sigma_{nn}` is within-design process covariance matrix
   and :math:`\Sigma_{nr}` the between design and non-design process
   covariance matrix, and similarly, in the case of unknown
   :math:`\strut{\sigma^2}\;`, :math:`\Sigma_{nn}= \\sigma^2 R_{nn}` and
   :math:`\Sigma_{nr}=\sigma^2 R_{nr}` where :math:`R_{nn},R_{nr}` are the
   correlation matrices.

Outputs
-------

-  A sequentially chosen optimal design :math:`\strut{D}`.
-  The value of the chosen criterion :math:`\strut{\mathcal{C}}`.
-  The posterior distribution :math:`\pi(\Theta|Y_n)` and the predictive
   distribution :math:`f(Y_r|Y_n)`, where :math:`\strut{Y_n}` is the vector on
   :math:`\strut{n}` observed outputs and :math:`\strut{Y_r}` is the vector of
   :math:`\strut{r}` unobserved outputs. The predictive distribution is
   used as an emulator.

Procedure
---------

#. Check if the candidate set :math:`\strut{E}` contains any points of the
   initial design points :math:`\strut{D_0}`. If it does then :math:`E=E
   \\setminus D_0`.
#. Compute the posterior distribution for the unknown parameters
   :math:`\pi(\Theta|Y_n)` and the predictive distribution :math:`f(Y_r|Y_n)`.
   The posterior distribution can be obtained analytically or
   numerically.
#. Choose the next design point :math:`{D_i}` or points to optimize the
   chosen criterion. The selection is done using the :ref:`exchange
   algorithm<ProcExchangeAlgorithm>`. The criterion is based on
   the posterior distribution. For example, the maximum entropy sampling
   criterion has approximately the form
   :math:`\det((X_rVX_r^T+\Sigma_{rr}-(X_{r}VX_{n}^T+\Sigma_{rn})(X_nVX_n^T+\Sigma_{nn})^{-1}(X_nVX_r^T+\Sigma_{nr})))`
   if the predictive distribution :math:`f(Y_r|Y_n)` is a Gaussian process
   or approximately the form
   :math:`a^*(X_rVX_r^T+R_{rr}-(X_{r}VX_{n}^T+R_{rn})(X_nVX_n^T+R_{nn})^{-1}(X_nVX_r^T+R_{nr}))`
   if the predictive distribution is a Student :math:`\strut{t}` process.
   They are almost the same because :math:`\strut{a^*}` is just a constant
   not dependent on the design; the unknown :math:`\strut{\sigma^2}` does
   not affect the choice of the design, in this case.
#. Observe the output at the design points :math:`D_i` selected in step 3.
   The observation itself is useful for the purposes of assessing the
   uncertainty about prediction. It is not neccesary for the computing
   the criterion but it is necessary for computing the predictive
   distribution :math:`f(Y_r|Y_n)`.
#. Update the predictive distribution :math:`f(Y_r|Y_n)`.
#. Compute the measures of accuracy in order to assess the improvement
   of prediction.
#. Update the candidate set :math:`\strut{E=E \\setminus D_i}`, the design
   :math:`\strut{S}`, :math:`\strut{D=D \\cup D_i}`, and the design size
   :math:`\strut{nn=nn+n_i}` .
#. Stop if :math:`\strut{nn = n}` or a certain value of the criterion is
   achieved stop otherwise go to step 3.

Additional Comments, References, and Links
------------------------------------------

The above algorithm is a foundation for several other versions under
development. The criterion mentioned is the entropy criterion (see step
3 of the procedure), but in principle any other optimal design criterion
can be use. The methodology used here allows for learning about the
parameters, that is to say the posterior distributions are computed at
every design stage, using a basic random regression Bayesian
formulation. It also employs the K-L expansion, but any set of basis
functions can be used.

The more fully adaptive version, which is under development will allow
the variance parameters on the regression terms to be updated, using
hyper-parameters. The aim is to allow adaptation to (i) global
smoothness, because low/high varying basis function represent less/more
smoothness and (ii) local smoothness.
