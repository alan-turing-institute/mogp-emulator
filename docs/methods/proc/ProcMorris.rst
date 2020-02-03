.. _ProcMorris:

Procedure: Morris screening method
==================================

The Morris method, also know as the Elementary Effect (EE) method,
utilises a discrete approximation of the average value of the Jacobian
(matrix of the partial derivatives of the
:ref:`simulator<DefSimulator>` output with respect to each of the
simulator inputs) of the simulator over the input space. The motivation
for the method was :ref:`screening<DefScreening>` for
:ref:`deterministic<DefDeterministic>` computer models with moderate
to large numbers of inputs. The method relies on a one-factor-at-a-time
(OAT) experimental :ref:`design<DefDesign>` where the effects of a
single factor on the output is assessed sequentially. The individual
randomised experimental designs are known as trajectories. The method's
main advantage is the lack of assumptions on the inputs and functional
dependency of the output to inputs such as monotonicity or linearity.

Algorithm
---------

The algorithm involves generating :math:`R` trajectory designs, as
described below. Each trajectory design is used to compute the expected
value of the elementary effects in the simulator function locally. By
averaging these a global approximation is obtained. The steps are:

#. Rescale all input variables to operate on the same scale using either
   standardisation or linear scaling as shown in the procedure page for
   data standardisation
   (:ref:`ProcDataPreProcessing<ProcDataPreProcessing>`). Otherwise
   different values of the step size (see below) will be needed for each
   input variable.

#. Each point in the trajectory differs from the previous point in only
   one input variable by a fixed step size, :math:`\Delta`. For
   :math:`k` variables, each trajectory has :math:`k+1` points,
   changing each variable exactly once. The start point for the
   trajectory is random, although this can be modified to improve the
   algorithm. See the discussion below.

#. Compute the elementary effect for each input variable
   :math:`1,\ldots,k`: :math:`EE_i(x)=\frac{f(x+\Delta
   e_i)-f(x)}{\Delta}`. :math:`e_i` is the unit vector in the
   direction of the :math:`i^{th}` axis for :math:`i=1,\ldots,k`.
   Each elementary effect is computed with observations
   at the pair of points :math:`x, x+\Delta e_i` that differ in the
   :math:`i^{th}` input variable by the fixed step size :math:`\Delta`.

#. Compute the moments of the elementary effects distribution for each
   input variable:

   .. math::
      \mu_i &=& \sum_{r=1} ^{R} \frac{EE_i(x_r)}{R}, \\
      \mu^{*}_i &=& \sum_{r=1} ^{R} \left|\frac{EE_i(x_r)}{R}\right|, \\
      \sigma_i &=& \sqrt{\sum_{r=1}^{R} \frac{(EE_i(x_r) - \mu_i)^2}{R}}.

The sample moment :math:`\mu_i` is an average effect measure, and a
high value suggests a dominant contribution of the :math:`i^{th}`
input factor in positive or negative response values (i.e. typically
linear, or at least monotonic). The sample moment :math:`\mu^{*}_i`
is a total effect measure; a high value indicates large influence of the
corresponding input factor. :math:`\mu_i` may prove misleading due
to cancellation effects (that is on average over the input space the
output goes up in response to the input as much as it comes down), thus
to capture main effects :math:`\mu^{*}_i` should be used.
Non-linear and interaction effects are estimated with :math:`\sigma_i`.
The total number of model runs needed in the Morris's
method is :math:`(k+1)R`.

An effects plot is constructed by plotting :math:`\mu_i` or
:math:`\mu^*_i` against :math:`\sigma_i`. This plot is a
visual tool to detecting and ranking effects.

An example on synthetic data demonstrating the Morris method is provided
at the example page :ref:`ExamScreeningMorris<ExamScreeningMorris>`.

Setting the parameters of the Morris method
-------------------------------------------

There is interest in undertaking input screening with as few simulator
runs as possible, but as the number of input factors :math:`k` is
fixed, the size of the experiment required is controlled by the number
of trajectory designs :math:`R`. Usually small values of :math:`R`
are used; for instance, in Morris (1991) the values :math:`R=3`
and :math:`R=4` were used in the examples. A value of :math:`R`
between 10 and 50 is mentioned in the more recent literature (see
References). A larger value of :math:`R` may improve the quality of
the global estimates at the price of extra runs. For a reasonably high
dimensional input space, with more than say 10 inputs, it would seem
unwise to select :math:`R` less than 10, since coverage of the
space, and thus global effects estimates require something close to
space filling. It is likely for large :math:`k` the number of
trajectory designs will need some dependency on :math:`R`.

The step size :math:`\Delta` is selected in such a way that all the
simulator runs lie in the input space and the elementary effects are
computed with reasonable precision. The usual choice of :math:`\Delta`
in the literature is determined by the input space considered
for experimentation, which is a :math:`k` dimensional grid
constructed with :math:`p` uniformly spaced values for each input.
The number :math:`p` is recommended to be even and :math:`\Delta`
to be an integer multiple of :math:`1/(p-1)`. Morris
(1991) suggests a value of :math:`\Delta = p/2(p-1)` that ensures
good coverage of the input space with few trajectories. One value for
:math:`\Delta` is generally used for all the inputs, but the method
can be generalised to instead use different values of :math:`\Delta`
and :math:`p` for every input.

Extending the Morris method
---------------------------

In Morris's original proposal, the starting points of the trajectory
designs were taken at random from the input space grid. Campolongo
(2007) proposed generating a large number of trajectories, selecting the
subset that maximise the distance between trajectories in order to cover
the design space. Another option is to use a :ref:`Latin Hypercube
design<ProcLHC>` or a :ref:`Sobol<DiscSobol>` sequence to
select the starting points of the trajectories.

A potential drawback of OAT runs in the Morris's method is that design
points fall on top of each other when projected into lower dimensions.
This disadvantage becomes more apparent when the design runs are to be
used in further modelling after discarding unimportant factors. An
alternative is to construct a randomly rotated simplex at every point
from which elementary effects are computed (Pujol, 2009). The
computation of distribution moments :math:`\mu_i,\mu^*_i,\sigma_i`
and further analysis is similar as the Morris's method, with the
advantage that projections of the resulting design do not fall on top of
existing points, and all observations can be reused in a later stage. A
potential disadvantage of this approach is the loss of efficiency in the
computation of elementary effects.

Lastly, it is possible to modify the standard Morris algorithm to
minimize the number of simulator runs required by employing a sequential
version of the algorithm. Details can be found in Boukouvalas et al
(2010).

References
----------

Morris, M. D. (1991, May). Factorial sampling plans for preliminary
computational experiments. *Technometrics*, 33 (2), 161–174.

Boukouvalas, A., Gosling, J.P. and Maruri-Aguilar, H., `An efficient
screening method for computer
experiments <http://wiki.aston.ac.uk/twiki/pub/AlexisBoukouvalas/WebHome/screenReport.pdf>`_.
NCRG Technical Report, Aston University (2010)

Saltelli, A., Chan, K. and Scott, E. M. (eds.) (2000). `Sensitivity
Analysis <http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0471998923>`_.
Wiley.

Francesca Campolongo, Jessica Cariboni, and Andrea Saltelli. An
effective screening design for sensitivity analysis of large models.
*Environ. Model. Softw.*, 22(10):1509–18, 2007.

Francesca Campolongo, Jessica Cariboni, Andrea Saltelli, and W.
Schoutens. Enhancing the Morris Method. In *Sensitivity Analysis of Model
Output*, pages 369–79, 2004.

Gilles Pujol. Simplex-based screening designs for estimating metamodels.
*Reliability Engineering & System Safety*, 94:1156–60, 2009.
