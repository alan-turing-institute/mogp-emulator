.. _DiscReificationTheory:

Discussion: Reification Theory
==============================

Description and Background
--------------------------

Reification is an approach for coherently linking models to reality,
which was introduced in the discussion page
:ref:`DiscReification<DiscReification>`. Here we provide more detail
about the various techniques involved in implementing the full
Reification process. Readers should be familiar with the basic
:ref:`emulation<DefEmulator>` ideas presented in the core threads
:ref:`ThreadCoreGP<ThreadCoreGP>` and specifically
:ref:`ThreadCoreBL<ThreadCoreBL>`.

Discussion
----------

As covered in :ref:`DiscReification<DiscReification>`, the
Reification approach centres around the idea of linking the current
model :math:` \\strut{ f} \` to a Reified model :math:` \\strut{ f^+} \`. The
Reified model incorporates all possible improvements to the model that
can be currently imagined. The link to reality :math::ref:` \\strut{ y} \` is
then given by applying the `Best Input<DefBestInput>` approach
(see :ref:`DiscBestInput<DiscBestInput>`) to the Reified model only,
giving,

:math:` \\strut{ y = f^+(x^+, w^+) + d^+, \\qquad d^+ \\perp (f, f^+, x^+,
w^+) } \`

where :math:` \\strut{ w} \` are any extra model parameters that might be
introduced due to any of the considered model improvements incorporated
by :math:` \\strut{ f^+} \`. Hence, this is a more detailed method than the
Best Input approach. Here we will describe in more detail how to link a
model :math:` \\strut{ f} \` to :math:` \\strut{ f^+} \` either directly (a
process referred to as Direct Reification) or via an intermediate model
:math:` \\strut{ f'} \` (a process referred to as Structural Reification).
This involves the use of emulators to summarise our beliefs about the
behaviour of each of the functions :math:` \\strut{ f} \`, :math:` \\strut{ f'}
\` and :math:` \\strut{ f^+} \`. Here :math:`\strut{f'} \` may correspond to
any of the intermediate models introduced in
:ref:`DiscReification<DiscReification>`.

Direct Reification
~~~~~~~~~~~~~~~~~~

Direct Reification is a relatively straightforward process where we link
the current model :math:` \\strut{ f} \` to the Reified model :math:` \\strut{
f^+} \` . It should be employed when the expert does not have detailed
ideas about specific improvements to the model.

We first represent the model :math::ref:` \\strut{ f} \` using an emulator. As
described in detail in `ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`, an emulator is a probabilistic
belief specification for a deterministic function. Our emulator for the
:math:` \\strut{ i} \`th component of :math:` \\strut{ f} \` might take the
form,

:math:` \\strut{ f_i(x) = \\sum_j \\beta_{ij}\, g_{ij}(x) + u_i(x) } \`

where :math:` \\strut{ B = \\{ \\beta_{ij} \\}} \` are unknown scalars, :math:`
\\strut{ g_{ij}} \` are known deterministic functions of the inputs :math:`
\\strut{ x} \`, and :math:` \\strut{ u_i(x)} \` is a weakly stationary
stochastic process.

The Reified model might be a function, not just of the current input
parameters :math:` \\strut{ x} \` , but also of new inputs :math:` \\strut{ w}
\` which might be included in a future improvement stage. Our simplest
emulator for :math:` \\strut{ f^+} \` would then be

:math:` \\strut{ f^+_i(x, w) = \\sum_j \\beta^+_{ij}\, g_{ij}(x) + u^+_i(x)
+ u^+_i(x, w) } \`

where we might model :math:` \\strut{ B^+ = \\{ \\beta^+_{ij} \\}} \` as
:math:` \\strut{ \\beta^+_{ij} = c_{ij}\, \\beta_{ij} + \\nu_{ij}} \` for
known :math:` \\strut{ c_{ij}} \` and uncertain :math:` \\strut{ \\nu_{ij}}
\`, and correlate :math:` \\strut{ u_i(x)} \` and :math:` \\strut{ u_i^+(x)}
\`, but leave :math:` \\strut{ u_i^+(x,w)} \` uncorrelated with the other
random quantities in the emulator. Essentially this step models :math:`
\\strut{ f^+} \` by inflating each of the variances in the current
emulator for :math:` \\strut{ f} \`, and is often relatively simple. We can
represent this structure using the following Bayesian Belief Network,
where \`child' vertices that are strictly determined by their \`parents'
are indicated with dashed lines, where :math:` \\strut{ u^+} \` represents
:math:` \\strut{ \\{ u^+(x), u^+(x,w) \\}} \` and :math:` \\strut{ F} \`
represents a collection of model runs:

=========================================================
|image0|
Figure 1: Bayesian belief network for Direct Reification.
=========================================================

Structural Reification
~~~~~~~~~~~~~~~~~~~~~~

Structural Reification is a process where we link the current model :math:`
\\strut{ f} \` to an improved model :math:` \\strut{ f'} \` , and then to
the Reified model :math:` \\strut{ f^+} \`. Here :math:`\strut{f'}` may
correspond to :math:`\strut{f_{\rm theory}}` or :math:`\strut{f'_{\rm
theory}}:ref:` which were introduced in
`DiscReification<DiscReification>`.

Usually, we can think more carefully about the reasons for the model's
inadequacy. As we have advocated in the discussion page on expert
:ref:`assessment<DefAssessment>` of model discrepancy
(:ref:`DiscExpertAssessMD<DiscExpertAssessMD>`), a useful strategy is
to envisage specific improvements to the model, and to consider the
possible effects on the model outputs of such improvements. Often we can
imagine specific generalisations for :math:` \\strut{ f(x)} \` with extra
model components and new input parameters :math:` \\strut{ v} \`, resulting
in an improved model :math:` \\strut{ f'(x, v)} \` . Suppose the improved
model reduces back to the current model for some value of :math:` \\strut{
v=v_0} \` i.e. :math:` \\strut{ f'(x, v_0) = f(x)} \` . We might emulate
:math:` \\strut{ f'} \` \`on top' of :math:` \\strut{ f} \` , using the form:

:math:` \\strut{ f'_i(x, v) = f_i(x) + \\sum_k \\gamma_{ik}\, g_{ik}(x, v) +
u^{(a)}_i(x, v), } \`

where :math:` \\strut{ g_{ik}(x, v_0) = u^{(a)}_i(x, v_0) = 0} \` . This
would give the full emulator for :math:` \\strut{ f'_i(x, v)} \` as

:math:` \\strut{ f'_i(x, v) = \\sum_j \\beta_{ij}\, g_{ij}(x) + \\sum_k
\\gamma_{ik}\, g_{ik}(x, v) + u_i(x) + u^{(a)}_i(x, v) } \`,

where for convenience we define :math:` \\strut{ B' = \\{ \\beta_{ij},
\\gamma_{ik} \\}} \` and :math:` \\strut{ u' = u(x) + u^{(a)}(x,v)} \` .
Assessment of the new regression coefficients :math:` \\strut{ \\gamma_{ik}}
\` and stationary process :math:` \\strut{ u^{(a)}_i(x, v)} \` would come
from consideration of the specific improvements that :math:` \\strut{ f'}
\` incorporates. The reified emulator for :math:` \\strut{ f^+_i(x, v, w)}
\` would then be

:math:` \\strut{ f^+_i(x, v, w) = \\sum_j \\beta^+_{ij}\, g_{ij}(x) +
\\sum_k \\gamma^+_{ik}\, g_{ik}(x, v) + u^+_i(x) + u^+_i(x, v) +
u^+_i(x, v, w) } \`,

where we would now carefully apportion the uncertainty between each of
the random coefficients in the Reified emulator. Although this is a
complex task, we would carry out this procedure when the expert's
knowledge about improvements to the model is detailed enough that it
warrants inclusion in the analysis. An example of this procedure is
given in Goldstein, M. and Rougier, J. C. (2009). We can represent this
structure using the following Bayesian Belief Network, with :math:` \\strut{
B^+ = \\{ \\beta^+_{ij} , \\gamma^+_{ik} \\} } \` , and :math:` \\strut{
u^+ = u^+(x)+u^+(x,v)+u^+(x,v,w)} \` :

==========================================================================
|image1|
Figure 2: Bayesian belief network corresponding to Structured Reification.
==========================================================================

Additional Comments
-------------------

Once the emulator for :math:` \\strut{ f^+} \` has been specified, the
final step is to assess :math:` \\strut{ d^+} \` which links the Reified
model to reality. This is often a far simpler process than direct
assessment of model discrepancy, as the structured beliefs about the
difference between :math:` \\strut{ f} \` and reality :math:` \\strut{ y} \`
have already been accounted for in the link between :math:` \\strut{ f} \`
, :math:` \\strut{ f'} \` and :math:` \\strut{ f^+} \` . Reification also
provides a natural framework for incorporating multiple models, by the
judgement that each model will be informative about the Reified model,
which is then informative about reality :math:` \\strut{ y} \`.
Exchangeable Computer Models, as described in the discussion page
:ref:`DiscExchangeableModels<DiscExchangeableModels>`, provide a
further example of Reification. More details can be found in Goldstein,
M. and Rougier, J. C. (2009).

References
----------

Goldstein, M. and Rougier, J. C. (2009), "Reified Bayesian modelling and
inference for physical systems (with Discussion)", Journal of
Statistical Planning and Inference, 139, 1221-1239.

.. |image0| image:: /foswiki//pub/MUCM/MUCMToolkit/DiscReificationTheory/ReifNetwork2.png
   :width: 370px
   :height: 325px
.. |image1| image:: /foswiki//pub/MUCM/MUCMToolkit/DiscReificationTheory/ReifNetwork3.png
   :width: 410px
   :height: 360px
