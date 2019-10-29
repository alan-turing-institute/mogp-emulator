.. _ProcBLAdjust:

Procedure: Calculation of adjusted expectation and variance
===========================================================

Description and Background
--------------------------

In the context of :ref:`Bayes linear<DefBayesLinear>` methods, the
Bayes linear :ref:`adjustment<DefBLAdjust>` is the appropriate method
for updating prior :ref:`second-order beliefs<DefSecondOrderSpec>`
given observed data. The adjustment takes the form of linear fitting of
our beliefs on the observed data quantities. Specifically, given two
random vectors, :math:`B`, :math:`D`, the *adjusted expectation* for element
:math:`B_i`, given :math:`D`, is the linear combination :math:`a_0 + a^T D`
minimising :math:`\textrm{E}[B_i - a_0 - a^T D)^2]` over choices of
:math:`\{a_0, a\}`.

Inputs
------

-  :math:`\textrm{E}[B]`, :math:`\textrm{Var}[B]` - prior expectation and
   variance for the vector :math:`B`
-  :math:`\textrm{E}[D]`, :math:`\textrm{Var}[D]` - prior expectation and
   variance for the vector :math:`D`
-  :math:`\textrm{Cov}[B,D]` - prior covariance between the vector :math:`B`
   and the vector :math:`B`
-  :math:`D_{obs}` - observed values of the vector :math:`D`

Outputs
-------

-  :math:`\textrm{E}_D[B]` - adjusted expectation for the uncertain
   quantity :math:`B` given the observations :math:`D`
-  :math:`\textrm{Var}_D[B]`- adjusted variance matrix for the uncertain
   quantity :math:`B` given the observations :math:`D`

Procedure
---------

The adjusted expectation vector, :math:`\textrm{E}_D[B]` is evaluated as

:math:` \\textrm{E}_D[B] = \\textrm{E}[B] + \\textrm{Cov}[B,D]
\\textrm{Var}[D]^{-1} (D_{obs}-\textrm{E}[D]) \`

(If :math:`\textrm{Var}[D]` is not invertible, then we use a generalised
inverse such as Moore-Penrose).

The *adjusted variance matrix* for :math:`B` given :math:`D` is

:math:` \\textrm{Var}_D[B] = \\textrm{Var}[B] -
\\textrm{Cov}[B,D]\textrm{Var}[D]^{-1}\textrm{Cov}[D,B] \`

Additional Comments
-------------------

See :ref:`DiscBayesLinearTheory<DiscBayesLinearTheory>` for a full
description of Bayes linear methods.
