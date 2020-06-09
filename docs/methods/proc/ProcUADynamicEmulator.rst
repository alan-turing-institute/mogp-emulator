.. _ProcUADynamicEmulator:

Procedure: Uncertainty analysis for dynamic emulators
=====================================================

Description and Background
--------------------------

We describe an approximate method for quantifying uncertainty about
:ref:`dynamic simulator<DefDynamic>` outputs given uncertainty about
the simulator inputs. For a general discussion of uncertainty analysis,
see its definition page
(:ref:`DefUncertaintyAnalysis<DefUncertaintyAnalysis>`) and the
procedure page for carrying out uncertainty analysis using a GP emulator
(:ref:`ProcUAGP<ProcUAGP>`). This method is based on
:ref:`emulating<DefEmulator>` the :ref:`single step
function<DefSingleStepFunction>`, following the procedures
described in the variant thread on dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

We suppose that there is a true, but uncertain sequence of :ref:`forcing
inputs<DefForcingInput>` :math:`A_1,\ldots,A_T` and true values
of the simulator parameters :math:`\phi` and initial conditions :math:`W_0`,
corresponding to the modelling situation of interest. We denote the
true values of all these quantities by a vector :math:`X`. Uncertainty
about :math:`X` is described by a joint probability distribution
:math:`\omega(\cdot)`. The corresponding, uncertain sequence of :ref:`state
variables<DefStateVector>` that would be obtained by running the
simulator at inputs :math:`X` is denoted by :math:`W_1,\ldots,W_T`. The
procedure we describe quantifies uncertainty about :math:`W_1,\ldots,W_T`
given uncertainty about :math:`X`.

Inputs
------

-  An emulator for the single step function :math:`w_t=f(w_{t-1},a_t,\phi)`,
   formulated as a :ref:`GP<DefGP>` or
   :ref:`t-process<DefTProcess>` conditional on hyperparameters,
   training inputs :math:`D` and training outputs :math:`f(D)`.
-  A set :math:`\{\theta^{(1)},\ldots,\theta^{(s)}\}` of emulator
   hyperparameter values.
-  A joint distribution :math:`\omega(\cdot)` for the forcing variables,
   initial conditions and simulator parameters

Outputs
-------

-  Approximate mean and variance for each of :math:`W_1,\ldots,W_T`

Procedure
---------

We describe a Monte Carlo procedure with :math:`N` defined to be the number
of Monte Carlo iterations. For notational convenience, we suppose that
:math:`N\le s`. For discussion of the choice of :math:`N`, including the case
:math:`N>s`, see the discussion page on Monte Carlo estimation
(:ref:`DiscMonteCarlo<DiscMonteCarlo>`).

#. Generate a random value of :math:`X` from its distribution :math:`\omega(\cdot)`.
   Denote this random value by :math:`X_i`
#. Given :math:`X_i`, and one set of emulator hyperparameters
   :math:`\theta_i`, iterate the single step emulator using the approximation
   method described in the procedure page
   :ref:`ProcApproximateIterateSingleStepEmulator<ProcApproximateIterateSingleStepEmulator>`
   to obtain :math:`\textrm{E}[W_t \|f(D),X_i,\theta^{(i)}]` and
   :math:`\textrm{Var}[W_t |f(D),X_i,\theta^{(i)}]` for all :math:`t` of
   interest.
#. Repeat steps 2 and 3 :math:`N` times and estimate :math:`\textrm{E}[W_t
   |f(D)]` and :math:`\textrm{Var}[W_t|f(D)]` by

   .. math::
      \hat{\textrm{E}}[W_t |f(D)] &= \frac{1}{N}\sum_{i=1}^N
      \textrm{E}[W_t |f(D),X_i,\theta^{(i)}] \\
      \widehat{\textrm{Var}}[W_t \|f(D)] &= \frac{1}{N}\sum_{i=1}^N
      \textrm{Var}[W_t |f(D),X_i,\theta^{(i)}] + \frac{1}{N-1}\sum_{i=1}^N
      \left\{\textrm{E}[W_t |f(D),X_i,\theta^{(i)}] - \hat{\textrm{E}}[W_t
      |f(D)] \right\}^2

Additional Comments
-------------------

Note that this procedure does not enable us to fully consider the two
sources of uncertainty (uncertainty about inputs and uncertainty about
the simulator) separately. (See the discussion page on uncertainty
analysis (:ref:`DiscUncertaintyAnalysis<DiscUncertaintyAnalysis>`)).
However, one term that is useful to consider is

.. math::
   \frac{1}{N-1}\sum_{i=1}^N \left\{\textrm{E}[w_t
   |f(D),X_i,\theta^{(i)}] - \hat{\textrm{E}}[W_t |f(D)] \right\}^2.

This gives us the expected reduction in our variance of :math:`W_t`
obtained by learning the true inputs :math:`X`. If this term is small
relative to :math:`\textrm{Var}[W_t |f(D)]`, it suggests that
uncertainty about the simulator is large, and that more training runs of
the simulator would be beneficial for reducing uncertainty about :math:`W_t`.
