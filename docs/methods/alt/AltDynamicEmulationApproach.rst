.. _AltDynamicEmulationApproach:

Alternatives: Dynamic Emulation Approaches
==========================================

Overview
--------

This page discusses two approaches for building an
:ref:`emulator<DefEmulator>` of a :ref:`dynamic
simulator<DefDynamic>`. Dynamic simulators produce multivariate
outputs: a time series of :ref:`state variables<DefStateVector>`
:math:`w_1,\ldots,w_T`. The strategy for emulating such a simulator
presented in the variant thread for dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`) is to construct
an emulator of the :ref:`single step function<DefSingleStepFunction>`
:math:`w_t=f(w_{t-1},a_t,\phi)`. An alternative is to treat the
simulator like any other multivariate output simulator, and construct an
emulator directly as in the variant thread for the analysis of a
simulator with multiple outputs
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`).
Here, we consider the relative merits of the two approaches.

The Nature of the Alternatives
------------------------------

#. Build an emulator of the single step function
   :math:`w_t=f(w_{t-1},a_t,\phi)`. If we wish to obtain a joint
   distribution of :math:`w_1,\ldots,w_T` given :math:`a_1,\ldots,a_T`
   and :math:`\phi`, we cannot do so directly. We must either use
   simulation methods
   (:ref:`ProcExactIterateSingleStepEmulator<ProcExactIterateSingleStepEmulator>`)
   or an approximation approach
   (:ref:`ProcApproximateIterateSingleStepEmulator<ProcApproximateIterateSingleStepEmulator>`).
#. Build an emulator of the full times series simulator
   :math:`(w_1,\ldots,w_T)=f_{full}(w_0,a_1,\ldots,a_T,\phi)`. This
   simulator and corresponding emulator have a much larger input and
   output space, but the emulator will directly give us the joint
   distribution of :math:`w_1,\ldots,w_T` given :math:`w_0,a_1,\ldots,a_T`
   and :math:`\phi`. We may only be interested in some subvector or
   function of :math:`(w_1,\ldots,w_T)`, which could be emulated directly,
   reducing the dimension of the output space.

Choosing the Alternatives
-------------------------

We argue that emulating the single step function is preferable under any
of the following circumstances:

#. We wish to emulate the outputs over a range of different series of
   :ref:`forcing inputs<DefForcingInput>` :math:`a_1,\ldots,a_T`. This
   may be because we are uncertain about the value of a ‘true’ forcing
   input series, or we may just wish to investigate how the outputs vary as
   :math:`a_1,\ldots,a_T` varies. In a multivariate output emulator, the
   inputs are :math:`w_0, \phi, a_1,\ldots,a_T`. Hence the input dimension
   is large if :math:`T` is large, and building any emulator becomes
   increasingly difficult as the number of inputs increases. In the single
   step emulator, the inputs are :math:`w_{t-1}, \phi, a_t`, and so the
   number of emulator inputs is fixed regardless of the value of :math:`T`.

#. We are uncertain about the maximum :math:`T` of interest. A 'black
   box' emulator of the function :math:`(w_{1}\ldots,w_{T})=f_{full}(w_0,
   \phi, a_1,\ldots,a_T)` cannot predict :math:`w_{T+t}` for any
   :math:`t>0`. It is, however, possible to extrapolate if

   -  the forcing variables are fixed and are not treated as emulator
      inputs
   -  :math:`t` is treated as an additional simulator/emulator input
   -  The prior mean function is a good approximation of the relationship
      between :math:`w_{t}` and :math:`t` in the simulator. See the
      alternatives page on emulator prior mean function
      (:ref:`AltMeanFunction<AltMeanFunction>`)

#. We wish to increase the complexity of the simulator single step
   function. As the computational expense of the single step function
   increases, it may become increasingly impractical to obtain sufficient
   training runs of the full times series simulator, depending on the value
   of :math:`T`.

Additional Comments, References and Links
-----------------------------------------

When emulating the full time series output directly, there are
particular emulator modelling choices that can speed up computation
considerably. These are presented in

Rougier, J. C. (2008), Efficient Emulators for Multivariate
Deterministic Functions, *Journal of Computational and Graphical
Statistics*, 17(4), 827-843.
