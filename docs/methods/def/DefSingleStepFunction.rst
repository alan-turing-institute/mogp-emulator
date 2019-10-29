.. _DefSingleStepFunction:

Definition of Term: Single step function
========================================

We consider :ref:`dynamic simulators<DefDynamic>` of the form that
model the evolution of a :ref:`state vector<DefStateVector>` by
iteratively applying a **single step function**. For example, suppose
the dynamic simulator takes as inputs an initial state vector :math::ref:`w_0`,
a time series of `forcing inputs<DefForcingInput>`
:math:`a_1,\ldots,a_T`, and some constant parameters :math:`\phi`, to produce
a time series output :math:`w_1,\ldots,w_T`. If the evolution of the time
series :math:`w_1,\ldots,w_T` is defined by the equation
:math:`w_t=f(w_{t-1},a_t,\phi) \`, then :math:`f(.) \` is known as the single
step function.
