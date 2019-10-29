.. _ProcUpdateDynamicMeanAndVariance:

Procedure: recursively update the dynamic emulator mean and variance in the approximation method
================================================================================================

Description and Background
--------------------------

This page is concerned with task of :ref:`emulating<DefEmulator>` a
:ref:`dynamic simulator<DefDynamic>`, as set out in the variant
thread on dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

The approximation procedure described in page
:ref:`ProcApproximateIterateSingleStepEmulator<ProcApproximateIterateSingleStepEmulator>`
recursively defines

:math:` \\mu_{t+1}= \\mathrm{E}[ m^*(w_t,a_{t+1},\phi)|f(D),\theta] \`,

:math:` V_{t+1}= \\mathrm{E}[
v^*\{(w_t,a_{t+1},\phi),(w_t,a_{t+1},\phi)\}|f(D),\theta] +
\\mathrm{Var}[m^*(w_t,a_{t+1},\phi)|f(D),\theta] \`,

where the expectations and variances are taken with respect to :math:` w_{t}
\`, with :math:` w_{t} \\sim N_r(\mu_{t},V_{t}) \`. Here, we show how to
compute :math::ref:` \\mu_{t+1} \` and :math:` V_{t+1} \` in the case of a `single
step<DefSingleStepFunction>` emulator linear mean and a
:ref:`separable<DefSeparable>` Gaussian covariance function.

To simplify notation, we now omit the constant parameters :math:`\phi` from
the simulator inputs. (We can think of :math::ref:`\phi` as an extra `forcing
input<DefForcingInput>` that is constant over time).

Inputs
------

-  :math:` \\mu_{t} \` and :math:` V_{t} \`
-  The single step emulator training inputs :math:`D \` and outputs :math:`f(D)
   \`
-  Emulator covariance function parameters :math:` B \` and :math:` \\Sigma \`
   (defined below).

Outputs
-------

-  :math:` \\mu_{t+1} \` and :math:` V_{t+1} \`

Notation
--------

:math:`x`: a :math:` p \\times 1 \` input vector :math:`( {x^{(w)}}^T,{x^{(a)}}^T
)^T`, with :math:`x^{(w)}` the corresponding :math:` r \\times 1 \` vector
state variable input, and :math:` x^{(a)} \` the corresponding :math:` (p-r)
\\times 1 \` vector forcing variable input.

:math:`h(x^{(w)},x^{(a)})= h(x)=(1\, x^T)^T \`: the prior mean function.

:math:`\Sigma c(x,x') =\Sigma \\exp\{-(x-x')^TB(x-x')\}`: the prior
covariance function, with

:math:`\Sigma \`: the covariance matrix between outputs at the same input
:math:`x`, and

:math:` B`: a diagonal matrix with :math:`(i,j)^{\mathrm{th}}` element
:math:`1/\delta_i^2`.

:math:`B_w`: the upper-left :math:`r \\times r` submatrix of :math:` B`.

:math:`B_a`: the lower-right :math:`(p-r) \\times (p-r)` submatrix of :math:`B`.

:math:`D`: the set of training data inputs :math:`x_1,\ldots,n`.

:math:`c\{(x^{(w)},x^{(a)}),D\}=c(x,D)`: an :math:`n\times 1` vector with
:math:`i^{\mathrm{th}}` element :math:`c(x,x_i)`.

:math:`A`: an :math:`n\times n` matrix with :math:`(i,j)^{\mathrm{th}}` element
:math:`c(x_i,x_j)`.

:math:`H`: an :math:`n\times (r+1)` matrix with :math:`i^{\mathrm{th}}` row
:math:`h(x_i)^T`.

:math:`f(D)`: an :math:`n\times r` matrix of training outputs with
:math:`(i,j)^{\mathrm{th}}` element the :math:`j^{\mathrm{th}}` training
output type for the :math:`i^{\mathrm{th}}` training input.

:math:`\hat{\beta}=(H^TA^{-1}H)^{-1}H^TA^{-1}f(D)`.

:math:` 0_{a\times b} \`: an :math:` {a\times b} \` matrix of zeros.

Procedure
---------

A series of constants need to be evaluated in the following order. Terms
defined at each stage can be evaluated independently of each other, but
are expressed in terms of constants defined at preceding stages.

Stage 1: compute :math:`K_{Vh},` :math:`K_{Ec},` :math:`K_{Eh},` :math:`K_{Ewc}` and :math:`K_{Ecc}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`K_{Vh}=\mathrm{Var}[h(w_t,a_{t+1})|f(D),B] =
\\left(\begin{array}{ccc}0_{1\times 1} & 0_{1\times r} & 0_{1\times
(p-r)} \\\\ 0_{p\times 1} & V_t & 0_{p\times (p-r)} \\\\ 0_{(p-r)\times
1} & 0_{(p-r)\times r} & 0_{(p-r)\times (p-r)} \\end{array}\right),`

--------------

:math:` K_{Ec}=\mathrm{E}[c\{(w_t,a_{t+1}),D\}|f(D),B] \`, an :math:` n \\times
1 \` vector, with element :math:` i \` given by

:math:` \|2V_tB_w+I_r|^{-1/2} \\exp\{-(a_{t+1} - x_i^{(a)})^TB_a(a_{t+1} -
x_i^{(a)})\}\\\ \\times
\\exp\{-(\mu_t-x_i^{(w)})^T(2V_t+B_w^{-1})^{-1}(\mu_t - x_i^{(w)})\} \`

--------------

:math:` K_{Eh}=\mathrm{E}[h(w_t,a_{t+1})|f(D),B]=(1, \\mu_{t}^T,a_{t+1}^T)^T
\`

--------------

:math:`K_{Ewc}=\mathrm{E}[w_tc\{(w_t,a_{t+1}),D\}^T|f(D),B] \` , an :math:` r
\\times n \` matrix, with column :math:` i \` given by

:math:`\mathrm{E}[w_tc(\{w_t,a_{t+1}\},x_i)|f(D),B] = \|2V_t B_w +
I_r|^{-1/2} \\times\exp\{-(a_{t+1}-x_i^{(a)})^TB_a (a_{t+1}-x_i^{(a)})
\\}\\\ \\times
\\exp\left\{-(\mu_t-x_i^{(w)})^T\left(2V_t+B_W^{-1}\right)^{-1}
(\mu_t-x_i^{(w)}) \\right\} \\times(2B_w+V_t^{-1})^{-1}(2B_wx_i^{(w)} +
V_t^{-1}\mu_t).`

--------------

:math:`
K_{Ecc}=\mathrm{E}[c\{(w_t,a_{t+1}),D\}c\{(w_t,a_{t+1}),D\}^T|f(D),B]
\`, an :math:` n \\times n \` matrix, with element :math:` i,j \` given by

:math:`\mathrm{E}[c(\{w_t,a_{t+1}\},x_i)c(\{w_t,a_{t+1}\},x_j)|f(D),B] =
\|4V_t B_w + I_r|^{-1/2} \\exp \\left\{-\frac{1}{2}(x_i^{(w)}-
x_j^{(w)})^TB_w (x_i^{(w)}- x_j^{(w)}) \\right\}\\
\\times\exp\{-(a_{t+1}-x_i^{(a)})^TB_a (a_{t+1}-x_i^{(a)})-
(a_{t+1}-x_j^{(a)})^TB_a (a_{t+1}-x_j^{(a)}) \\}\\\ \\times
\\exp\left[-\left\{\mu_t-\frac{1}{2}(x_i^{(w)}+ x_j^{(w)})
\\right\}^T\left(2V_t+\frac{1}{2}B_W^{-1}\right)^{-1}
\\left\{\mu_t-\frac{1}{2}(x_i^{(w)}+ x_j^{(w)}) \\right\} \\right]`

Stage 2: compute :math:` K_{Cwc},` :math:` K_{Ehh},` and :math:` K_{Vc}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:` K_{Cwc}=\mathrm{Cov}[w_t,c\{(w_t,a_{t+1}),D\}|f(D),B]=K_{Ewc} -
\\mu_tK_{Ec}^T \`

--------------

:math:` K_{Ehh}=\mathrm{E}[h(w_t,a_{t+1})h(w_t,a_{t+1})^T|f(D),B]=K_{Vh} +
K_{Eh}K_{Eh}^T`

--------------

:math:` K_{Vc}=\mathrm{Var}[c\{(w_t,a_{t+1}),D\}|f(D),B]=K_{Ecc} -
K_{Ec}K_{Ec}^T`

Stage 3: compute :math:` K_{Chc}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:` K_{Chc}=\mathrm{Cov}[h(w_t,a_{t+1}),c\{(w_t,a_{t+1}),D\}|f(D),B]
=\left(\begin{array}{c}0_{1\times n}\\\ K_{Cwc} \\\\ 0_{(p-r)\times n}
\\end{array}\right) \`

Stage 4: compute :math:` K_{Ehc}` and :math:` K_{Vm}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`
K_{Ehc}=\mathrm{E}[h(w_t,a_{t+1})c\{(w_t,a_{t+1}),D\}^T|f(D),B]=K_{Chc}
+ K_{Eh}K_{Ec}^T`

--------------

:math:` K_{Vm}= \\mathrm{Var}[m^*(w_t,a_{t+1})|f(D),B] = \\hat{\beta}^T
K_{Vh}\hat{\beta} +\hat{\beta}^T K_{Chc}A^{-1}(f(D)-H\hat{\beta})\\
+(f(D)-H\hat{\beta})^TK_{Chc}^T\hat{\beta}+(f(D)-H\hat{\beta})^TA^{-1}K_{Vc}A^{-1}(f(D)-H\hat{\beta})
\`

Stage 5: compute :math:` K_{Ev}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:` K_{Ev}= \\mathrm{E}[v^*\{(w_t,a_{t+1}),(w_t,a_{t+1})\}|f(D),B] = 1
-tr[\{A^{-1}-A^{-1}H(H^TA^{-1}H)^{-1}H^TA^{-1}\}K_{Ecc}]\\
+tr[(H^TA^{-1}H)^{-1}K_{Ehh} ]-2tr[A^{-1}H(H^TA^{-1}H)^{-1}K_{Ehc}] \`

Stage 6: compute the procedure outputs :math:` \\mu_{t+1}` and :math:` V_{t+1}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:` \\mu_{t+1} = K_{Eh} \\hat{\beta}+K_{Ec}^TA^{-1}(f(D)-H\hat{\beta})
\`

:math:` V_{t+1} = K_{Vm}+K_{Ev}\Sigma`

Reference
---------

Conti, S., Gosling, J. P., Oakley, J. E. and O'Hagan, A. (2009).
Gaussian process emulation of dynamic computer codes. Biometrika 96,
663-676.
