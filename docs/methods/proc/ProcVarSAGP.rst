.. _ProcVarSAGP:

Procedure: Variance Based Sensitivity Analysis using a GP emulator
==================================================================

Description
-----------

This page describes the formulae needed for performing :ref:`Variance
Based<DefVarianceBasedSA>` :ref:`Sensitivity
Analysis<DefSensitivityAnalysis>` (VBSA). In VBSA we consider
the effect on the output :math:`f(X)` of a
:ref:`simulator<DefSimulator>` as we vary the inputs :math:`X`, when the
variation of those inputs is described by a (joint) probability
distribution :math:`\omega(X)`. This probability distribution can be
interpreted as describing uncertainty about the best or true values for
the inputs, or may simply represent the range of input values of
interest to us. The principal measures for quantifying the sensitivity
of :math:`f(X)` to a set of inputs :math:`X_w` are the main effect, the
sensitivity index and the total effect index. We can also define the
interaction between two inputs :math:`X_i` and :math:`X_j`. These are
described below.

Mean, main and interaction effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`w` denote a subset of the indices from 1 to the number
:math:`p` of inputs. Let :math:`X_w` denote the set of inputs with indices in
:math:`w`. The mean effect of a set of inputs :math:`X_w` is the
function

.. math::
   M_w(x_w) = \textrm{E}[f(X)|x_w].

This is a function of :math:`x_w` showing how the simulator output for
given :math:`x_w`, when averaged over the uncertain values of all the other
inputs, varies with :math:`x_w`.

The deviation of the mean effect :math:`M_{\{i\}}(x_{\{i\}})` of the single
input :math:`X_i` from the overall mean,

.. math::
   I_i(x_w) = \textrm{E}[f(X)|x_i]-\textrm{E}[f(X)].

is called the *main effect* of :math:`X_i`.

Furthermore, we define the *interaction effect* between inputs :math:`X_i`
and :math:`X_j` to be

.. math::
   I_{\{i,j\}}(x_i,x_j) = \textrm{E}[f(X)|x_i,x_j]-I_i(x_i) - I_j(x_j)
   - \textrm{E}[f(X)]

Sensitivity variance
~~~~~~~~~~~~~~~~~~~~

The sensitivity variance :math:`V_w` describes the amount by which the
uncertainty in the output is reduced when we are certain about the
values of the inputs :math:`X_w`. That is,

.. math::
   V_w = \textrm{Var}[f(X)]-\textrm{E}[\textrm{Var}[f(X)|x_w]]

It can also be written in the following equivalent forms

.. math::
   V_w = \textrm{Var}[\textrm{E}[f(X)|x_w]] =
   \textrm{E}[\textrm{E}[f(X)|x_w]^2] - \textrm{E}[f(X)]^2

Total effect variance
~~~~~~~~~~~~~~~~~~~~~

The total effect variance :math:`V_{Tw}` is the expected amount of
uncertainty in the model output that would be left if we removed the
uncertainty in all the inputs except for the inputs with indices
:math:`w`. The total effect variance is given by

.. math::
   V_{Tw} = \textrm{E}[\textrm{Var}[f(X)|x_{\bar{w}}]]

where :math:`\bar{w}` denotes the set of all indices not in :math:`w`,
and hence :math:`x_{\bar{w}}` means all the inputs except for those in
:math:`x_w`.

:math:`V_{Tw}` can also be written as

.. math::
   V_{Tw} = \textrm{Var}[f(X)] - V_{\bar{w}}.

In order to define the above quantities, it is necessary to specify a
full probability distribution for the inputs. This clearly demands a
good understanding of probability and its use to express personal
degrees of belief. However, this specification of uncertainty often also
requires interaction with relevant experts whose knowledge is being used
to specify values for individual inputs. There is a considerable
literature on the elicitation of expert knowledge and uncertainty in
probabilistic form, and some references are given at the end of the
procedure page for uncertainty analysis (:ref:`ProcUAGP<ProcUAGP>`)
page.

We assume here that a :ref:`Gaussian process<DefGP>` (GP) emulator
has been built according to the procedure page
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`, and that we are only
emulating a single output. Note that
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>` gives procedures for deriving
emulators in a number of different forms, and we consider here only the
"linear mean and weak prior" case where the GP has a linear mean
function, weak prior information is specified on the hyperparameters
:math:`\beta` and :math:`\sigma^2` and the emulator is derived with a single
point estimate for the hyperparameters :math:`\delta`.

The procedure here computes the expected values (with respect to the
emulator) of the above quantities.

Inputs
------

-  An emulator as defined in :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`

-  A distribution :math:`\omega(\cdot)` for the uncertain inputs

-  A set :math:`w` of indices for inputs whose average effect or
   sensitivity indices are to be computed, or a pair {i,j} of indices
   defining an interaction effect to be computed

-  Values :math:`x_w` for the inputs :math:`X_w`, or similarly for
   :math:`X_i,X_j`

Outputs
-------

-  :math:`\textrm{E}^*[M_w(x_w)]`

-  :math:`\textrm{E}^*[I_{\{i,j\}}(x_i,x_j)]`

-  :math:`\textrm{E}^*[V_w]`

-  :math:`\textrm{E}^*[V_{Tw}]`

where :math:`\textrm{E}^*[\cdot]` denotes an expectation taken with respect
to the emulator uncertainty, i.e. a posterior mean.

Procedure
---------

In this section we provide the formulae for the calculation of the
posterior means of :math:`M_w(x_w)`, :math:`I_{\{i,j\}}(x_i,x_j)`, :math:`V_w`
and :math:`V_{Tw}`. These are given as a function of a number of integral
forms, which are denoted as :math:`U_w,P_w,S_w,Q_w,R_w` and :math:`T_w`. The
exact expressions for these forms depend on the distribution of the
inputs :math:`\omega(\cdot)`, the correlation function :math:`c(.,.)` and the
regression function :math:`h(\cdot)`. In the following section, we give
expressions for the above integral forms for the general and two special
cases.

Calculation of :math:`\textrm{E}^*[M_w(x_w)]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   \textrm{E}^*[M_w(x_w)] = R_w\hat{\beta} + T_w e,

where :math:`e = A^{-1}(f(D)-H\hat{\beta})` and :math:`\hat{\beta}, A, f(D)`
and :math:`H` are defined in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`.

For the main effect of :math:`X_i` the posterior mean is

.. math::
   \textrm{E}^*[I_i(x_i)] = \{R_{\{i\}} - R\}\hat{\beta} +
   \{T_{\{i\}}-T\}e.

It is important to note here that both :math:`R_w` and :math:`T_w` are
functions of :math:`x_w`. The dependence on :math:`x_w` has been suppressed
here for notational simplicity.

Calculation of :math:`\textrm{E}^*[I_{\{i,j\}}(x_i,x_j)]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   \begin{array}{r l} \textrm{E}^*[I_{\{i,j\}}(x_i,x_j)] = &
   \{R_{\{i,j\}} - R_{\{i\}} - R_{\{j\}} - R\}\hat{\beta} \\ + &
   \{T_{\{i,j\}} - T_{\{i\}} - T_{\{j\}} - T\}e \end{array}

where :math:`R_{\{i,j\}}` and :math:`R_{\{i\}}`, for instance, are special
cases of :math:`R_w` when the set :math:`w` of indices comprises the two
elements :math:`w=\{i,j\}` or the single element :math:`w=\{i\}`. Remember
also that these will be functions of :math:`x_{\{i,j\}}=(x_i,x_j)` and
:math:`x_{\{i\}}=x_i` respectively.

Calculation of :math:`\textrm{E}^*[V_w]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We write the posterior mean of :math:`V_w` as

.. math::
   \textrm{E}^*[V_w] =
   \textrm{E}^*[\textrm{E}[\textrm{E}[f(X)|x_w]^2]] -
   \textrm{E}^*[\textrm{E}[f(X)]^2]

The first term is

.. math::
   \begin{array}{r l} \textrm{E}^*[\textrm{E}[\textrm{E}[f(X)|x_w]^2]] & =
   \hat{\sigma}^2[U_w-\mathrm{tr}(A^{-1}P_w) + \mathrm{tr}\{W(Q_w-S_w
   A^{-1} H \\ & \qquad \qquad - H^{\mathrm{T}}A^{-1}S_w^{\mathrm{T}} +
   H^{\mathrm{T}}A^{-1}P_w A^{-1}H)\}] \\ & \quad + e^{\mathrm{T}}P_we +
   2\hat{\beta}^{\mathrm{T}}S_we + \hat{\beta}^{\mathrm{T}}Q_w\hat{\beta}
   \end{array}

where :math:`\hat\sigma^2` is defined in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`.

The second term is

.. math::
   \begin{array}{r l} \textrm{E}^*[\textrm{E}[f(X)]^2] & =
   \hat{\sigma}^2[U-TA^{-1}T^{\mathrm{T}} +\{R - TA^{-1}H\}W\{R -
   TA^{-1}H\}^\mathrm{T}] \\ & \quad + \left(R\hat{\beta}+Te\right)^2
   \end{array}

with :math:`W = (H^{\mathrm{T}}A^{-1}H)^{-1}`.

Calculation of :math:`\textrm{E}^*[V_{Tw}]`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\textrm{E}^*[V_{Tw}]` can be calculated via the sensitivity variance
:math:`V_{\bar{w}}` using the relation

.. math::
   \textrm{E}^*[V_{Tw}] = \textrm{E}^*[\textrm{Var}[f(X)]] -
   \textrm{E}^*[V_{\bar{w}}]

with

.. math::
   \textrm{E}^*[\textrm{Var}[f(X)]] =
   \hat{\sigma}^2[U-TA^{-1}T^{\mathrm{T}} +\{R - TA^{-1}H\}W\{R -
   TA^{-1}H\}^\mathrm{T}]

Dimensions
~~~~~~~~~~

Before presenting the integral forms that appear in the above
expressions, we give the dimensions of all the involved quantities in
the table below. We assume that we have :math:`n` *observations*,
:math:`p` *inputs* and :math:`q` *regression functions*. The terms in the left
column are either described in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>` or they are shorthands :math:`(e,
W)`. The terms in the right hand side column are the integral forms,
which will be presented in the following section.

====================== ================== =========== ==================
**Symbol**             **Dimension**      **Symbol**  **Dimension**
---------------------- ------------------ ----------- ------------------
:math:`\hat{\sigma}^2` :math:`1 \times 1` :math:`U_w` :math:`1 \times 1`
:math:`\hat{\beta}`    :math:`q \times 1` :math:`P_w` :math:`n \times n`
:math:`f`              :math:`n \times 1` :math:`S_w` :math:`q \times n`
:math:`H`              :math:`n \times q` :math:`Q_w` :math:`q \times q`
:math:`A`              :math:`n \times n` :math:`R_w` :math:`1 \times q`
:math:`e`              :math:`n \times 1` :math:`T_w` :math:`1 \times n`
:math:`W`              :math:`q \times q`
====================== ================== =========== ==================

The integral forms
~~~~~~~~~~~~~~~~~~

General case
^^^^^^^^^^^^

When no assumptions are made about the distribution of the inputs, the
correlation and the regression functions we have general expressions for
the :math:`U_w, P_w, S_w, Q_w, R_w, T_w` terms. These are

.. math::
   U_w = \int_{{\cal X}_w}\int_{{\cal X}_{\bar{w}}}\int_{{\cal
   X}_{\bar{w}}}
   c(x,x^*)\omega(x_{\bar{w}}|x_w)\omega(x'_{\bar{w}}|x_w)\omega(x_w)
   \mathrm{d} x_{\bar{w}} \mathrm{d} x'_{\bar{w}} \mathrm{d} x_w

.. math::
   P_w = \int_{{\cal X}_w}\int_{{\cal X}_{\bar{w}}}\int_{{\cal
   X}_{\bar{w}}} t(x)t(x^*)^{\mathrm{T}}
   \omega(x_{\bar{w}}|x_w)\omega(x'_{\bar{w}}|x_w)\omega(x_w) \mathrm{d}
   x_{\bar{w}} \mathrm{d} x'_{\bar{w}} \mathrm{d} x_w

.. math::
   S_w = \int_{{\cal X}_w}\int_{{\cal X}_{\bar{w}}}\int_{{\cal
   X}_{\bar{w}}} h(x)t(x^*)^{\mathrm{T}}
   \omega(x_{\bar{w}}|x_w)\omega(x'_{\bar{w}}|x_w)\omega(x_w) \mathrm{d}
   x_{\bar{w}} \mathrm{d} x'_{\bar{w}} \mathrm{d} x_w

.. math::
   Q_w = \int_{{\cal X}_w}\int_{{\cal X}_{\bar{w}}}\int_{{\cal
   X}_{\bar{w}}} h(x)h(x^*)^{\mathrm{T}}
   \omega(x_{\bar{w}}|x_w)\omega(x'_{\bar{w}}|x_w)\omega(x_w) \mathrm{d}
   x_{\bar{w}} \mathrm{d} x'_{\bar{w}} \mathrm{d} x_w

.. math::
   R_w = \int_{{\cal X}_{\bar{w}}}
   h(x)^{\mathrm{T}}\omega(x_{\bar{w}}|x_w) \mathrm{d} x_{\bar{w}}

.. math::
   T_w = \int_{{\cal X}_{\bar{w}}}
   t(x)^{\mathrm{T}}\omega(x_{\bar{w}}|x_w) \mathrm{d} x_{\bar{w}}

Here, :math:`x_{\bar{w}}` and :math:`x'_{\bar{w}}` denote two different
realisations of :math:`x_{\bar{w}}`. :math:`x^*` is a vector with elements
made up of :math:`x'_{\bar{w}}` and :math:`x_w` in the same way as :math:`x` is
composed of :math:`x_{\bar{w}}` and :math:`x_w`. Remember also that :math:`R_w`
and :math:`T_w` are functions of :math:`x_w`.

:math:`h(x)` is described in the alternatives page on emulator prior mean
function (:ref:`AltMeanFunction<AltMeanFunction>`). :math:`c(.,.)` is
the correlation function discussed in the alternatives page on emulator
prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`). Also :math:`t(x)
= c(D,x)`, as introduced in :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`.

:math:`\omega(x_w)` is the joint distribution of the :math:`x_w` inputs and
:math:`\omega(x_{\bar{w}}|x_w)` is the conditional distribution of
:math:`x_{\bar{w}}` when the values of :math:`x_{w}` are known.

Finally, when one of the above integral forms appears without a
subscript (e.g. :math:`U`), it is implied that the set :math:`w` is empty.

Special case 1
^^^^^^^^^^^^^^

We now show derive closed form expressions for the above integrals when
we make specific choices about :math:`\omega(\cdot)` :math:`c(\cdot,\cdot)`
and :math:`h(\cdot)`.
We first assume that :math:`\omega(\cdot)` is a normal distribution given by

.. math::
   \omega(x) =
   \frac{1}{(2\pi)^{d/2}|B|^{-1/2}}\exp\left[-\frac{1}{2}(x-m)^T B
   (x-m)\right]

We also assume the generalised Gaussian correlation function with nugget
(see :ref:`AltCorrelationFunction<AltCorrelationFunction>`)

.. math::
   c(x,x') = \nu I_{x=x'} + (1-\nu)\exp\{-(x-x')^T C (x-x')\}

where :math:`I_{x=x'}` equals 1 if :math:`x=x'` but is otherwise zero, and
:math:`\nu` represents the nugget term. The case of a generalised Gaussian
correlation function without nugget is simply obtained by setting
:math:`\nu=0`.

We let both :math:`B,C` be general positive definite matrices, partitioned
as

.. math::
   B = \left[ \begin{array}{cc} B_{ww} & B_{w\bar{w}} \\
   B_{\bar{w}w} & B_{\bar{w}\bar{w}} \end{array} \right], \\
   C = \left[ \begin{array}{cc} C_{ww} & C_{w\bar{w}} \\
   C_{\bar{w}w} & C_{\bar{w}\bar{w}} \end{array} \right]

Finally, we do not make any particular assumption for :math:`h(x)`.

We now give the expressions for each of the integrals

--------------

:math:`U_w` is the scalar

.. math::
   U_w = (1-\nu)\frac{|B|^{1/2}|B_{\bar{w}\bar{w}}|^{1/2}}{|F|^{1/2}}

with

.. math::
   F = \left[ \begin{array}{ccc} B_{ww} +
   B_{w\bar{w}}B_{\bar{w}\bar{w}}^{-1}B_{\bar{w}w} & B_{w\bar{w}} &
   B_{w\bar{w}} \\ B_{\bar{w}w} & 2C_{\bar{w}\bar{w}} + B_{\bar{w}\bar{w}} &
   -2C_{\bar{w}\bar{w}} \\ B_{\bar{w}w} & -2C_{\bar{w}\bar{w}} &
   2C_{\bar{w}\bar{w}} + B_{\bar{w}\bar{w}} \end{array} \right]

:math:`U` is the special case when :math:`w` is the empty set. The exact
formula for :math:`U` is given in :ref:`ProcUAGP<ProcUAGP>`.

--------------

:math:`P_w` is an :math:`n \times n` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

.. math::
   P_w(k,l) =
   (1-\nu)^2\frac{|B|^{1/2}|B_{\bar{w}\bar{w}}|^{1/2}}{|F|^{1/2}}
   \exp\left\{-\frac{1}{2}\left[ r - g^{\mathrm{T}}F^{-1}g
   \right]\right\}

with

.. math::
   F = \left[ \begin{array}{ccc} 4C_{ww} + B_{ww} +
   B_{w\bar{w}}B_{\bar{w}\bar{w}}^{-1}B_{\bar{w}w} & 2C_{w\bar{w}} +
   B_{w\bar{w}}& 2C_{w\bar{w}} + B_{w\bar{w}} \\ 2C_{\bar{w}w} +
   B_{\bar{w}w} & 2C_{\bar{w}\bar{w}} + B_{\bar{w}\bar{w}}& 0 \\
   2C_{\bar{w}w} + B_{\bar{w}w} & 0 & 2C_{\bar{w}\bar{w}} +
   B_{\bar{w}\bar{w}} \end{array} \right]

and

.. math::
   g = \left[ \begin{array}{l} 2C_{ww}(x_{w,k}+x_{w,l} - 2m_w) +
   2C_{w\bar{w}}(x_{\bar{w},k}+x_{\bar{w},l} - 2m_{\bar{w}}) \\
   2C_{\bar{w}w}(x_{w,k} - m_w) + 2C_{\bar{w}\bar{w}}(x_{\bar{w},k} -
   m_{\bar{w}}) \\ 2C_{\bar{w}w}(x_{w,l} - m_w) +
   2C_{\bar{w}\bar{w}}(x_{\bar{w},l} - m_{\bar{w}}) \end{array} \right]

and

.. math::
   r = (x_k - m)^{\mathrm{T}}2C(x_k - m) + (x_l - m)^{\mathrm{T}}2C(x_l
   - m)

:math:`P` is a special case of :math:`P_w` when :math:`w` is the empty set, and
reduces to

.. math::
   P=T^T T

--------------

:math:`S_w` is an :math:`q \times n` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

.. math::
   S_w(k,l) =
   (1-\nu)\frac{|B|^{1/2}|B_{\bar{w}\bar{w}}|^{1/2}}{|F|^{1/2}} \exp
   \left\{-\frac{1}{2}\left[r - g^{\mathrm{T}}F^{-1}g\right]\right\}
   \textrm{E}_*[h_k(x)]

with

.. math::
   F = \left[ \begin{array}{ccc} 2C_{ww}+B_{ww} +
   B_{w\bar{w}}B_{\bar{w}\bar{w}}^{-1}B_{\bar{w}w} & B_{w\bar{w}} & 2C_{w\bar{w}}+B_{w\bar{w}} \\
   B_{\bar{w}w} & B_{\bar{w}\bar{w}} & 0 \\
   2C_{\bar{w}w}+B_{\bar{w}w} & 0 & 2C_{\bar{w}\bar{w}}+B_{\bar{w}\bar{w}}
   \end{array} \right]

and

.. math::
   g = \left[\begin{array}{ccc} 2C_{ww} & 0 & 2C_{w\bar{w}} \\ 0 & 0 & 0 \\
   2C_{\bar{w}w} & 0 & 2C_{\bar{w}\bar{w}} \end{array}\right] \left[
   \begin{array}{c} x_{w,l} - m_w \\ 0 \\ x_{\bar{w},l} - m_{\bar{w}}
   \end{array} \right]

and

.. math::
   r = \left[ \begin{array}{c} x_{w,l} - m_w \\ 0 \\ x_{\bar{w},l} -
   m_{\bar{w}} \end{array} \right]^{\mathrm{T}} \left[\begin{array}{ccc}
   2C_{ww} & 0 & 2C_{w\bar{w}} \\ 0 & 0 & 0 \\ 2C_{\bar{w}w} & 0 & 2C_{\bar{w}\bar{w}}
   \end{array}\right] \left[ \begin{array}{c} x_{w,l} - m_w \\ 0 \\
   x_{\bar{w},l} - m_{\bar{w}} \end{array} \right]

The expectation :math:`\textrm{E}_z[\cdot]` is w.r.t. the normal distribution
:math:`{\cal{N}}(m + F^{-1}g,F^{-1})`. Also :math:`h_k(x)` is the :math:`k`-th
element of :math:`h(x)`.

:math:`S` is a special case of :math:`S_w` when :math:`w` is the empty set, and
reduces to

:math:`S=R^T T`

--------------

:math:`Q_w` is a :math:`q \times q` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

.. math::
   Q_w(k,l) = \frac{|B|^{1/2}|B_{\bar{w}\bar{w}}|^{1/2}}{|F|^{1/2}}
   \textrm{E}_*[h_k(x)h_l^{\mathrm{T}}(x)]

where the expectation :math:`\textrm{E}_*[\cdot]` is w.r.t. the normal
distribution :math:`{\cal{N}}([m_w,m_{\bar{w}}]^{\mathrm{T}},F^{-1})`

with

.. math::
   F = \left[ \begin{array}{cc} B_{ww} +
   B_{w\bar{w}}B_{\bar{w}\bar{w}}^{-1}B_{\bar{w}w} & B_{w\bar{w}} \\
   B_{\bar{w}w} & B_{\bar{w}\bar{w}} \end{array} \right]

:math:`Q` is a special case of :math:`Q_w` when :math:`w` is the empty set, and
reduces to

.. math::
   Q=R^T R

--------------

:math:`R_w` is the :math:`1 \times q` vector with elements the mean of the
elements of :math:`h(x)`, w.r.t. :math:`\omega(x_{\bar{w}}|x_w)`, i.e.,

.. math::
   R_w = \int_{{\cal X}_{\bar{w}}}
   h(x)^{\mathrm{T}}\omega(x_{\bar{w}}|x_w) \mathrm{d} x_{\bar{w}}

and is a function of :math:`x_w`. :math:`R` is a special case of :math:`R_w`,
when :math:`w` is the empty set. The formula for :math:`R` is given in
:ref:`ProcUAGP<ProcUAGP>`.

--------------

:math:`T_w` is an :math:`1 \times n` vector, whose :math:`k^{\mathrm{th}}` entry
is

.. math::
   T_w(k) = (1-\nu)
   \frac{|B_{\bar{w}\bar{w}}|^{1/2}}{|2C_{\bar{w}\bar{w}}+B_{\bar{w}\bar{w}}|^{1/2}}
   \exp\left\{-\frac{1}{2}\left[F^{'} +r-g^TF^{-1}g\right]\right\}

with

.. math::
   F^{'} &= (x_w-m_w - (F^{-1}g)_w)^T \\
   & \big[2C_{ww}+B_{w\bar{w}}B_{\bar{w}\bar{w}}^{-1}B_{\bar{w}w} -
   (2C_{w\bar{w}} +B_{w\bar{w}})(2C_{\bar{w}\bar{w}}
   +B_{\bar{w}\bar{w}})^{-1}(2C_{\bar{w}w} +B_{\bar{w}w})\big] \\ & (x_w-m_w
   - (F^{-1}g)_w) \\
   F &= \left[ \begin{array}{cc} 2C_{ww} +
   B_{w\bar{w}}B_{\bar{w}\bar{w}}^{-1}B_{\bar{w}w}& 2C_{w\bar{w}} +
   B_{w\bar{w}} \\ 2C_{\bar{w}w} + B_{\bar{w}w}&2C_{\bar{w}\bar{w}} +
   B_{\bar{w}\bar{w}} \end{array} \right] \\
   g &= 2C(x_k-m) \\
   r &= (x_k-m)^T 2C(x_k-m)

:math:`(F^{-1}g)_w` is the part of the :math:`F^{-1}g` vector that corresponds
to the indices :math:`w`. According to the above formulation, these are the
first :math:`\#(w)` indices, where :math:`\#(w)` is the number of indices
contained in :math:`w`.

:math:`T` is a special case of :math:`T_w`, when :math:`w` is the empty set. The
formula for :math:`T` is given in :ref:`ProcUAGP<ProcUAGP>`.

Special case 2
^^^^^^^^^^^^^^

In this special case, we further assume that the matrices :math:`B,C` are
diagonal. We also consider a special form for the vector :math:`h(x)`,
which is the linear function described in
:ref:`AltMeanFunction<AltMeanFunction>`

.. math::
   h(x) = [1,x^{\mathrm{T}}]^{\mathrm{T}}

We now present the form of the integrals under the new assumptions.

--------------

:math:`U_w` is the scalar

.. math::
   U_w = (1-\nu)\prod_{i\in \bar{w}} \left(\frac{B_{ii}}{B_{ii} +
   2(2C_{ii})}\right)^{1/2}

Again, :math:`U` is the special case when :math:`w` is the empty set, and its
exact formula is given in :ref:`ProcUAGP<ProcUAGP>`.

--------------

:math:`P_w` is an :math:`n \times n` matrix, whose :math:`(k,l)^{\mathrm{th}}`
entry is

.. math::
   \begin{array}{r l} P_w(k,l) = &(1-\nu)^2\prod_{i\in {\bar{w}}}
   \frac{B_{ii}}{2C_{ii}+B_{ii}} \exp\left\{-\frac{1}{2}\frac{2C_{ii}
   B_{ii}}{2C_{ii}+B_{ii}} \left[(x_{i,k}-m_i)^2 +
   (x_{i,l}-m_i)^2\right]\right\} \\ & \prod_{i\in w}
   \left(\frac{B_{ii}}{4C_{ii}+B_{ii}}\right)^{1/2}
   \exp\left\{-\frac{1}{2}\frac{1}{4C_{ii}+B_{ii}}\right. \\
   & \left[4C_{ii}^2(x_{i,k}-x_{i,l})^2 + 2C_{ii}
   B_{ii}\left\{(x_{i,k}-m_i)^2 + (x_{i,l}-m_i)^2\right\}\right]\Big\}
   \end{array}

where the double indexed :math:`x_{i,k}` denotes the :math:`i^{\mathrm{th}}`
input of the :math:`k^{\mathrm{th}}` training data.

:math:`P` is a special case of :math:`P_w` when :math:`w` is the empty set, and
reduces to

:math:`P=T^T T`

--------------

:math:`S_w` is an :math:`q \times n` matrix whose :math:`(k,l)^{\mathrm{th}}`
entry is

.. math::
   \begin{array}{rl} S_w{(k,l)} = &(1-\nu) \textrm{E}_*[h_k(x)] \\
   &\quad\prod_{i\in \{w,\bar{w}\}}\frac{B_{ii}^{1/2}} {(2C_{ii} +
   B_{ii})^{1/2}} \exp
   \left\{-\frac{1}{2}\left[\frac{2C_{ii}B_{ii}}{2C_{ii}+B_{ii}}(x_{i,l} -
   m_i)^2\right]\right\} \end{array}

For the expectation we have

.. math::
   \begin{array}{ll} \textrm{E}_*[h_1(x)] = 1 & \\
   \textrm{E}_*[h_{j+1}(x)] = m_j & \qquad \mathrm{if} \quad j \in
   \bar{w} \\ \textrm{E}_*[h_{j+1}(x)] = \frac{2C_{jj}x_{j,l} +
   B_{jj}m_j}{2C_{jj} + B_{jj}}& \qquad \mathrm{if} \quad j \in w
   \end{array}

:math:`S` is a special case of :math:`S_w` when :math:`w` is the empty set, and
reduces to

:math:`S=R^T T`

--------------

:math:`Q_w` is a :math:`q \times q` matrix. If we assume that its :math:`q`
indices have the labels :math:`[1,\bar{w},w]`, then,

.. math::
   Q_w = \left [ \begin{array}{ccc} 1 & m_{\bar w}^T & m_w^T \\
   m_{\bar w} & m_{\bar w}m_{\bar w}^T & m_{\bar w}m_w^T \\ m_w &
   m_wm_{\bar w}^T & m_wm_w^T + B_{ww}^{-1} \end{array} \right ]

:math:`Q` is a special case of :math:`Q_w`, when :math:`w` is the empty set, and
reduces to

:math:`Q=R^T R`

--------------

:math:`R_w` is a :math:`1 \times q` vector. If we assume that its :math:`q`
indices have the labels :math:`[1,\bar{w},w]`, then,

.. math::
   R_w = [1,m_{\bar{w}}^T,x_w^T]

:math:`R` is a special case of :math:`R_w`, when :math:`w` is the empty set. The
formula for :math:`R` is given in :ref:`ProcUAGP<ProcUAGP>`.

--------------

:math:`T_w` is an :math:`1 \times n` vector, whose :math:`k^{\mathrm{th}}` entry
is

.. math::
   \begin{array}{r l} \displaystyle T_w(k) = (1-\nu) \prod_{i\in
   \{\bar{w}\}} \frac{\displaystyle B_{ii}^{1/2}}{\displaystyle
   (2C_{ii}+B_{ii})^{1/2}} & \displaystyle
   \exp\left\{-\frac{1}{2}\frac{\displaystyle 2C_{ii}B_{ii}}{\displaystyle
   2C_{ii}+B_{ii}} \left(x_{i,k}-m_i\right)^2\right\} \\ & \displaystyle
   \exp\left\{-\frac{1}{2}(x_w-x_{w,k})^T 2C_{ww} (x_w-x_{w,k})\right\}
   \end{array}

Recall that :math:`x_w` denotes the fixed values for the inputs :math:`X_w`,
upon which the measures :math:`M_w`, :math:`V_w` and :math:`V_{Tw}` are
conditioned. On the other hand, :math:`x_{w,k}` represents the :math:`w`
inputs of the :math:`k`\th design points.

:math:`T` is a special case of :math:`T_w`, when :math:`w` is the empty set. The
formula for :math:`T` is given in :ref:`ProcUAGP<ProcUAGP>`.

References
----------

The principal reference for these procedures is

Oakley, J.E., O'Hagan, A., (2004), Probabilistic Sensitivity Analysis of
Complex Models: a Bayesian Approach, *J.R. Statist. Soc. B*, 66, Part 3,
pp.751-769.

The above paper does not explicitly consider the case of a non-zero
nugget. The calculations of :math:`{\rm E}^*[V_w]` and :math:`{\rm
E}^*[V_{Tw}]` produce results that are scaled by :math:`(1-\nu)`, and in
general :math:`(1- \nu)\sigma^2` is the maximum variance reduction
achievable because the nugget :math:`\nu` represents noise that we cannot
learn about by reducing uncertainty about :math:`X`. See the discussion
page on the nugget effects in sensitivity analysis
(:ref:`DiscUANugget<DiscUANugget>`) for more details on this point.

Ongoing work
------------

We intend to provide procedures relaxing the assumption of the "linear
mean and weak prior" case of :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`
as part of the ongoing development of the toolkit. We also intend to
provide procedures for computing posterior variances of the various
measures.
