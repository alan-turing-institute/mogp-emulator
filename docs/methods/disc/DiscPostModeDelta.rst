.. _DiscPostModeDelta:

Discussion: Finding the posterior mode of correlation lengths
=============================================================

Description and Background
--------------------------

This page discusses some details involved in the maximisation of the
posterior distribution of the correlation lengths :math:`\strut \\delta`.
The details discussed are the logarithmic transformation of the
correlation lengths and the incorporation of derivatives of the
posterior distribution in the optimisation process.

Discussion
----------

Logarithmic transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying a logarithmic transformation to the correlation lengths can
help finding the posterior mode, because it transforms a constrained
optimisation problem to an unconstrained one. The transformation that
can be applied is

:math:`\tau = 2\ln(\delta)\, .`

Additionally, it is common to maximise the logarithm of the posterior
distribution, rather than the posterior itself. Therefore, the function
that has to be maximised is

:math:`g(\tau) = \\ln(\pi^*_{\delta}(\exp(\tau/2)))`

which after substitution of :math:`\pi^*_{\delta}(\delta)` from the
procedure page on building a Gaussian Process emulator for the core
problem (:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`), we have

:math:`\displaystyle g(\tau) \\propto \\ln(\pi_{\delta}(\exp(\tau/2))) -
\\frac{n-q}{2}\ln(\hat{\sigma}^2) - 0.5\ln|A\| -0.5\ln|H^{\rm
T}A^{-1}H|\, .`

In the absence of any prior information about the correlation lengths,
the prior distribution :math:`\pi_{\delta}(\delta)` can be set to a
constant, in which case the entire term
:math:`\ln(\pi_{\delta}(\exp(\tau/2)))` can be ignored in the optimisation.

We should also note, that the Jacobian of the transformation :math:`\tau =
2\ln(\delta)` is not taken into account. This implies that
:math:`\exp(g(\tau))` cannot be strictly viewed as a probability density
function, but on the other hand, this results in the mode of :math:`\strut
g(\tau)` being located exactly at the same position as the mode of
:math:`\strut \\pi^*_{\delta}(\delta)` for :math:`\delta = \\exp(\tau/2)\, .`

Optimisation using derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the function :math:`g(\tau)` can be optimised using a
derivatives-free algorithm, the existence of closed form expressions for
the two first derivatives of :math:`g(\tau)` can help speeding up the
optimisation. In the following, we give the two first derivatives of
:math:`\strut g(\tau)`, assuming a uniform correlation length prior (i.e.
:math:`\strut \\pi_{\delta}(\delta) \\propto const.`), which implies that
its two first derivatives are zero. In case a different prior is used
for :math:`\strut \\delta`, its two first derivatives need to be calculated
and added to the following expressions.

The first derivative of :math:`g(\tau)` is a :math:`\strut p \\times 1`
vector, whose k-th element is:

:math:`\displaystyle \\frac{\partial g(\tau)}{\partial \\tau_k} =
-\frac{1-n+q}{2} {\rm tr}\left[P\frac{\partial A}{\partial
\\tau_k}\right] - \\frac{n-q}{2}{\rm tr}\left[R\frac{\partial
A}{\partial \\tau_k}\right]`

The second derivative is a :math:`p\times p \` matrix, whose (k,l)-th entry
is

:math:`\displaystyle \\frac{\partial^2 g(\tau)}{\partial \\tau_l \\partial
\\tau_k} = -\frac{1-n+q}{2}{\rm tr}\left[P\frac{\partial A}{\partial
\\tau_l \\partial \\tau_k} - P\frac{\partial A}{\partial
\\tau_l}P\frac{\partial A}{\partial \\tau_k}\right] - \\frac{n-q}{2}{\rm
tr}\left[R\frac{\partial A}{\partial \\tau_l \\partial \\tau_k}
-R\frac{\partial A}{\partial \\tau_l}R\frac{\partial A}{\partial
\\tau_k}\right]`

The matrices :math:`\strut P, R` are defined as follows

:math:`\displaystyle P \\equiv A^{-1} - A^{-1}H(H^{\rm T}A^{-1}H)^{-1}H^{\rm
T}A^{-1}`

:math:`\displaystyle R \\equiv P - Pf(D)(f(D)^{\rm T}Pf(D))^{-1}f(D)^{\rm
T}P`

The derivatives of the matrix :math:`\strut A` w.r.t :math:`\strut \\tau` are
(\(n \\times n`) matrices, whose (i,j)-th element is

:math:`\displaystyle \\left(\frac{\partial A}{\partial \\tau_k}\right)_{i,j}
= (A)_{i,j}\frac{(x_{k,i}-x_{k,j})^2}{e^{\tau_k}}`

In the above equation, :math:`\strut (\cdot)_{i,j}` denotes the (i,j)-th
element of a matrix, and the double subscripted :math:`\strut x_{k,i}`
denotes the k-th input of the i-th design point. The second cross
derivative of :math:`\strut A` is

:math:`\displaystyle \\left(\frac{\partial^2 A}{\partial \\tau_l \\partial
\\tau_k}\right)_{i,j} =
(A)_{i,j}\frac{(x_{l,i}-x_{l,j})^2}{e^{\tau_l}}\frac{(x_{k,i}-x_{k,j})^2}{e^{\tau_k}}`

and finally,

:math:`\displaystyle \\left(\frac{\partial^2 A}{\partial
\\tau_k^2}\right)_{i,j} =
(A)_{i,j}\left[\left(\frac{(x_{k,i}-x_{k,j})^2}{e^{\tau_k}}\right)^2 -
\\frac{(x_{k,i}-x_{k,j})^2}{e^{\tau_k}}\right]`

If we denote by :math:`\strut \\hat{\tau}` the value of :math:`\strut \\tau`
that maximises :math:`\strut g(\tau)`, the posterior mode of :math:`\strut
\\pi^*_{\delta}(\delta)` is found simply by the back transformation
:math:`\strut \\hat{\delta} = \\exp(\hat{\tau}/2)`.

Additional Comments
-------------------

The posterior distribution of the correlation lengths can often have
multiple modes, especially in emulators with a large number of inputs.
It is therefore recommended that the optimisation algorithm is run
several times from different starting points, to ensure that the global
maximum is found.
