.. _ProcUAGP:

Procedure: Uncertainty Analysis using a GP emulator
===================================================

Description and Background
--------------------------

One of the simpler tasks that is required by users of
:ref:`simulators<DefSimulator>` is :ref:`uncertainty
analysis<DefUncertaintyAnalysis>` (UA), which studies the
uncertainty in model outputs that is induced by uncertainty in the
inputs. Although relatively simple in concept, UA is both important and
demanding. It is important because it is the primary way to quantify the
uncertainty in predictions made by simulators. It is demanding because
in principle it requires us to evaulate the output at all possible
values of the uncertain inputs. The :ref:`MUCM<DefMUCM>` approach of
first building an :ref:`emulator<DefEmulator>` is a powerful way of
making UA feasible for complex and computer-intensive simulators.

This procedure describes how to compute some of the UA measures
discussed in the definition page of Uncertainty Analysis
(:ref:`DefUncertaintyAnalysis<DefUncertaintyAnalysis>`). In
particular, we consider the uncertainty mean and variance:

:math:` \\textrm{E}[f(X)] = \\int_{{\cal X}} f(x) \\omega(x) \\mathrm{d} x
\`

:math:` \\textrm{Var}[f(X)] = \\int_{{\cal X}} (f(x) - \\textrm{E}[f(x)])^2
\\omega(x) \\mathrm{d} x \`

Notice that it is necessary to specify the uncertainty about the inputs
through a full probability distribution for the inputs. This clearly
demands a good understanding of probability and its use to express
personal degrees of belief. However, this specification of uncertainty
often also requires interaction with relevant experts whose knowledge is
being used to specify values for individual inputs. There is a
considerable literature on the elicitation of expert knowledge and
uncertainty in probabilistic form, and some references are given at the
end of this page.

In practice, we cannot evaluate either :math:`\textrm{E}[f(X)]` or
:math:`\textrm{Var}[f(X)]` directly from the simulator because the
integrals require us to know :math:`f(x)` at every :math:`x`. Even evaluating
numerically by a Monte Carlo integration approach would require a very
large number of runs of the simulator, so this is one task for which
emulation is very powerful. We build an emulator from a limited
:ref:`training sample<DefTrainingSample>` of simulator runs and then
use the emulator to evaluate these quantities. We still cannot evaluate
them exactly because of uncertainty in the emulator. We therefore
present procedures here for calculating the emulator (i.e. posterior)
mean of each quantity as an estimate; while the emulator variance
provides a measure of accuracy of that estimate. We use
:math:`\textrm{E}^*` and :math:`\textrm{Var}^*` to denote emulator mean and
variance.

We assume here that a :ref:`Gaussian process<DefGP>` (GP) emulator
has been built in the form described in the procedure page for building
a GP emulator (:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`), and that we
are only emulating a single output. Note that
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>` gives procedures for deriving
emulators in a number of different forms, and we consider here only the
"linear mean and weak prior" case where the GP has a linear mean
function, weak prior information is specified on the hyperparameters
:math:`\beta` and :math:`\sigma^2` and the emulator is derived with a single
point estimate for the hyperparameters :math:`\delta`.

Inputs
------

-  An emulator as defined in :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`
-  A distribution :math:`\omega(.)` for the uncertain inputs

Outputs
-------

-  The expected value :math:`\textrm{E}^*[\textrm{E}[f(X)]]` and variance
   :math:`\textrm{Var}^*[\textrm{E}[f(X)]]` of the uncertainty distribution
   mean
-  The expected value :math:`\textrm{E}^*[\textrm{Var}[f(X)]]` of the
   uncertainty distribution variance

Procedure
---------

In this section we describe the calculation of the above quantities. We
first give their expressions in terms of a number of integral forms,
:math:`U_p,P_p,Q_p,S_p,U,T,R`. We then give the form of these integrals for
the *general case* when no assumptions about the form of the
distribution of the inputs :math:`\omega(X)`, correlation function
:math:`c(X,X')` and regression function :math:`h(X)` are made. Finally we give
their forms for two special cases.

Calculation of :math:`\phantom{E}\textrm{E}^*[\textrm{E}[f(X)]]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:` \\textrm{E}^*[\textrm{E}[f(X)]] = R\hat{\beta} + Te \`

where

:math:` e =A^{-1}(f(D)-H\hat{\beta}) \`

Calculation of :math:`\phantom{E}\textrm{Var}^*[\textrm{E}[f(X)]]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:` \\textrm{Var}^*[\textrm{E}[f(X)]] = \\hat{\sigma}^2[U -
TA^{-1}T^{\mathrm{T}} + (R-TA^{-1}H)W(R-TA^{-1}H)^{\mathrm{T}}] \`

where

:math:` W = (H^{\mathrm{T}}A^{-1}H)^{-1} \`

Calculation of :math:`\phantom{E}\textrm{E}^*[\textrm{Var}[f(X)]]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:` \\textrm{E}^*[\textrm{Var}[f(X)]] =
\\textrm{E}^*[\textrm{E}[f(X)^2]] - \\textrm{E}^*[\textrm{E}[f(X)]^2]
\`

The first term is

:math:` \\begin{array}{r l} \\textrm{E}^*[\textrm{E}[f(X)^2]] & =
\\hat{\sigma}^2[U_p-\mathrm{tr}(A^{-1}P_p) + \\mathrm{tr}\{W(Q_p-S_p
A^{-1} H-H^{\mathrm{T}}A^{-1}S_p^{\mathrm{T}} + H^{\mathrm{T}}A^{-1}P_p
A^{-1}H)\}]\\\ & + e^{\mathrm{T}}P_pe + 2\hat{\beta}^{\mathrm{T}}S_pe +
\\hat{\beta}^{\mathrm{T}}Q_p\hat{\beta} \\end{array} \`

The second term is

:math:` \\begin{array}{r l} \\textrm{E}^*[\textrm{E}[f(X)]^2] & =
\\hat{\sigma}^2[U-TA^{-1}T^{\mathrm{T}} +\{R - TA^{-1}H\}W\{R -
TA^{-1}H\}^\mathrm{T}]\\\ & + \\left(R\hat{\beta}+Te \\right)^2
\\end{array} \`

Dimensions
~~~~~~~~~~

Before describing the terms involved in the above expressions we first
give their dimensions. We assume that we have :math:`n` *observations*,
:math:`p` *inputs* and :math:`q` *regression functions*. The dimension of the
above quantities are given in the table below.

=================== ================ ======== ================
Symbol              Dimension        Symbol   Dimension
:math:`\hat{\sigma}^2` :math:`1 \\times 1` :math:`U_p` :math:`1 \\times 1`
:math:`\hat{\beta}`    :math:`q \\times 1` :math:`P_p` :math:`n \\times n`
:math:`e`              :math:`n \\times 1` :math:`S_p` :math:`q \\times n`
:math:`f`              :math:`n \\times 1` :math:`Q_p` :math:`q \\times q`
:math:`A`              :math:`n \\times n` :math:`U`   :math:`1 \\times 1`
:math:`H`              :math:`n \\times q` :math:`T`   :math:`1 \\times n`
:math:`W`              :math:`q \\times q` :math:`R`   :math:`1 \\times q`
=================== ================ ======== ================

The terms :math::ref:`\hat{\sigma}^2`, :math:`\hat{\beta}`, :math:`f(D)`, :math:`A` and
:math:`H` are defined in `ProcBuildCoreGP<ProcBuildCoreGP>`, while
:math:`e` and :math:`W` are defined above. The terms in the right hand column
are inherent in uncertainty analysis and are described below.

The integral forms
~~~~~~~~~~~~~~~~~~

 General case
^^^^^^^^^^^^

When no assumptions are made about the distribution of the inputs, the
correlation and the regression functions we have general expressions for
the :math:`U_p, P_p, S_p, Q_p, U, R, T` terms. These are

:math:` U_p = \\int_{{\cal X}} c(x,x)\omega(x) \\mathrm{d} x \`

:math:` P_p = \\int_{{\cal X}} t(x)t(x)^{\mathrm{T}} \\omega(x) \\mathrm{d}
x \`

:math:` S_p = \\int_{{\cal X}} h(x)t(x)^{\mathrm{T}} \\omega(x) \\mathrm{d}
x \`

:math:` Q_p = \\int_{{\cal X}} h(x)h(x)^{\mathrm{T}} \\omega(x) \\mathrm{d}
x \`

:math::ref:`h(x)` is described in the alternatives page on emulator prior mean
function (`AltMeanFunction<AltMeanFunction>`). :math:`c(.,.)` is
the correlation function discussed in the alternatives page on emulator
prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`).

Also :math::ref:`t(x) = c(D,x)`, as introduced in
`ProcBuildCoreGP<ProcBuildCoreGP>`.

Finally, :math:`\omega(x)` is the joint distribution of the :math:`x` inputs.

For the :math:`U,R,T` we have

:math:` U = \\int_{{\cal X}}\int_{{\cal X}} c(x,x')\omega(x)\omega(x')
\\mathrm{d} x \\mathrm{d} x' \`

:math:` R = \\int_{{\cal X}} h(x)^{\mathrm{T}}\omega(x) \\mathrm{d} x \`

:math:` T = \\int_{{\cal X}} t(x)^{\mathrm{T}}\omega(x) \\mathrm{d} x \`

where :math:`x` and :math:`x'` are two different realisations of :math:`x`.

 Special case 1
^^^^^^^^^^^^^^

We now show how the above integrals are transformed when we make
specific choices about :math:`\omega(.)` :math:`c(.,.)` and :math:`h(.)`. We
first assume that :math:`\omega(.)` is a normal distribution given by

:math:` \\omega(x) =
\\frac{1}{(2\pi)^{d/2}|B|^{-1/2}}\exp\left[-\frac{1}{2}(x-m)^{\rm T} B
(x-m)\right] \`

We also assume the generalised Gaussian correlation function with nugget
(see :ref:`AltCorrelationFunction<AltCorrelationFunction>`)

:math:` c(x,x') = \\nu I_{x=x'} + (1-\nu)\exp\{-(x-x')^{\rm T} C (x-x')\}
\`

where :math:`I_{x=x'}` equals 1 if :math:`x=x'` but is otherwise zero, and
:math:`\nu` represents the nugget term. The case of a generalised Gaussian
correlation function without nugget is simply obtained by setting
:math:`\nu=0`.

We let both :math:`B,C` be general positive definite matrices. Also, we do
not make any particular assumption for :math:`h(x)`.

We now give the expressions for each of the integrals

--------------

:math:` U_p = 1`

Note that this result is independent of the existence of a non-zero
nugget :math::ref:`\nu`. See the discussion page on the nugget effects in
uncertainty and sensitivity (`DiscUANugget<DiscUANugget>`) for
more on this point.

--------------

:math:`P_p` is an :math:`n \\times n` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

:math:` P_p(k,l) = (1-\nu)^2\frac{|B|^{1/2}}{|F|^{1/2}}
\\exp\left\{-\frac{1}{2}\left[ r - g^{\mathrm{T}}F^{-1}g
\\right]\right\} \`

with

:math:` F = 4C+B \`

and

:math:` g = 2C(x_k+x_k - 2m) \`

and

:math:` r = (x_k - m)^{\mathrm{T}}2C(x_k - m) + (x_l - m)^{\mathrm{T}}2C(x_l
- m) \`

The subscripts :math:`k` and :math:`l` of :math:`x` denote training points.

--------------

:math:`S_p` is a :math:`q \\times n` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

:math:` S_p(k,l) = (1-\nu)\frac{|B|^{1/2}}{|F|^{1/2}} \\exp
\\left\{-\frac{1}{2}\left[r - g^{\mathrm{T}}F^{-1}g\right]\right\}
\\textrm{E}_*[h_k(x)] \`

with

:math:` F = 2C+B \`

and

:math:` g = 2C(x_l - m) \`

and

:math:` r = (x_l-m)^{\mathrm{T}}2C(x_l-m) \`

The expectation :math:`\textrm{E}_*[.]` is w.r.t. the normal distribution
:math:` {\cal{N}}(m + F^{-1}g,F^{-1}) \`. Also :math:`h_k(x)` is the :math:`k`-th
element of :math:`h(x)`.

--------------

:math:`Q_p` is a :math:`q \\times q` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

:math:` Q_s(k,l) = \\textrm{E}_*[h_k(x)h_l(x)^{\rm T}] \`

where the expectation :math:`\textrm{E}_*[.]` is w.r.t. the normal
distribution :math:`\omega(x)`

--------------

:math:` U` is the scalar

:math:` U = (1-\nu)\frac{|B|\phantom{^{1/2}}}{|F|^{1/2}} \`

with

:math:` F = \\left[ \\begin{array}{cc} 2C + B& -2C\\\ -2C& 2C + B
\\end{array}\right] \`

--------------

:math:`R` is the :math:`1 \\times q` vector with elements the mean of the
elements of :math:`h(x)`, w.r.t. :math:`\omega(x)`, i.e.,

:math:` R = \\int_{{\cal X}} h(x)^{\mathrm{T}}\omega(x) \\mathrm{d} x \`

--------------

:math:`T` is a :math:`1 \\times n` vector, whose :math:`k^{\mathrm{th}}` entry is

:math:` T(k) = \\frac{(1-\nu)|B|^{1/2}}{|2C+B|^{1/2}}
\\exp\left\{-\frac{1}{2} \\left[r-g^{\rm T}F^{-1}g\right]\right\}
\\qquad \`

with

:math:` F = 2C+B \`

:math:` g = 2C(x_k-m) \`

:math:` r = (x_k-m)^{\rm T} 2C(x_k-m) \`

 Special case 2
^^^^^^^^^^^^^^

In this special case, we further assume that the matrices :math:`B,C` are
diagonal. We also consider a special form for the vector :math:`h(x)`,
which is the linear function described in
:ref:`AltMeanFunction<AltMeanFunction>`

:math:` h(x) = [1,x]^{\mathrm{T}} \`

Hence :math:`q=p+1`. We now present the form of the integrals under the new
assumptions.

--------------

:math:` U_p = 1 \`

--------------

:math:`P_p` is an :math:`n \\times n` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

:math:` \\begin{array}{r l} P_p(k,l) = &(1-\nu)^2\prod_{i=1}^p
\\left(\frac{B_{ii}}{4C_{ii}+B_{ii}}\right)^{1/2}
\\exp\left\{-\frac{1}{2}\frac{1}{4C_{ii}+B_{ii}}\right.\\\
&\left[4C_{ii}^2(x_{ik}-x_{il})^2 + 2C_{ii} B_{ii}\left\{(x_{ik}-m_i)^2
+ (x_{il}-m_i)^2\right\}\right]\Big\} \\end{array} \`

where the double indexed :math:`x_{ik}` denotes the :math:`i^{\mathrm{th}}`
input of the :math:`k^{\mathrm{th}}` training data.

--------------

:math:`S_p` is an :math:`q \\times n` matrix whose :math:`(k,l)^{\mathrm{th}}`
entry is

:math:` \\begin{array}{r l} S_p(k,l) = &(1-\nu)\textrm{E}_*[h_k(x)] \\\\
&\prod_{i=1}^p \\frac{B_{ii}^{1/2}}{(2C_{ii}+B_{ii})^{1/2}}
\\exp\left\{-\frac{1}{2}\frac{2C_{ii}B_{ii}}{2C_{ii}+B_{ii}}
\\left[(x_{il}-m_i)^2\right]\right\} \\end{array} \`

For the expectation we have

:math:` \\begin{array}{ll} \\textrm{E}_*[h_1(x)] = 1 & \\\\
\\textrm{E}_*[h_{j+1}(x)] = \\frac{2C_{jj}x_{jl} + m_j B_{jj}}{2C_{jj} +
B_{jj}}& \\qquad \\mathrm{for} \\quad j=1,2,\ldots,p \\end{array} \`

--------------

:math:`Q_p` is the :math:`q \\times q` matrix,

:math:` Q_p = \\left[ \\begin{array}{cc} 1 &m^{\rm T}\\\ m& mm^{\rm T} +
B^{-1} \\end{array} \\right] \`

--------------

:math:` U` is the scalar

:math:` U = (1-\nu)\prod_{i=1}^p \\left(\frac{B_{ii}}{B_{ii} +
2(2C_{ii})}\right)^{1/2} \`

--------------

:math:`R` is the :math:`1 \\times q` vector

:math:` R = [1,m^{\rm T}] \`

--------------

:math:`T` is a :math:`1 \\times n` vector, whose :math:`k^{\mathrm{th}}` entry is

:math:` T(k) = (1-\nu) \\prod_{i=1}^p
\\frac{B_{ii}^{1/2}}{(2C_{ii}+B_{ii})^{1/2}}
\\exp\left\{-\frac{1}{2}\frac{2C_{ii}B_{ii}}{2C_{ii}+B_{ii}}
\\left(x_{ik}-m_i\right)^2\right\} \\qquad \`

References
----------

The topic of eliciting expert judgements about uncertain quantities is
dealt with fully in the book

O'Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P.
H., Jenkinson, D. J., Oakley, J. E. and Rakow, T. (2006). Uncertain
Judgements: Eliciting Expert Probabilities. John Wiley and Sons,
Chichester. 328pp. ISBN 0-470-02999-4.

For those with limited knowledge of probability theory, we recommend the
:ref:`SHELF <http://tonyohagan.co.uk/shelf/>`__ package
(`disclaimer<MetaSoftwareDisclaimer>`), which provides simple
templates, software and guidance for carrying out elicitation.

Oakley, J.E., O'Hagan, A., (2002). Bayesian Inference for the
Uncertainty Distribution of Computer Model Outputs, Biometrika, 89, 4,
pp.769-784.

Ongoing work
------------

We intend to provide procedures relaxing the assumption of the "linear
mean and weak prior" case of :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`
as part of the ongoing development of the toolkit.
