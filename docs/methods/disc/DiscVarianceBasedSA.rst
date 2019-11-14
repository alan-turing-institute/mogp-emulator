.. _DiscVarianceBasedSA:

Discussion: Variance-based Sensitivity Analysis
===============================================

Description and background
--------------------------

The basic ideas of :ref:`sensitivity
analysis<DefSensitivityAnalysis>` (SA) are presented in the
topic thread on sensitivity analysis
(:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`).
We concentrate in the :ref:`MUCM<DefMUCM>` toolkit on probabilistic
SA, but the reasons for this choice and alternative approaches are
considered in the discussion page
:ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`. In
probabilistic SA, we assign a function :math:`\omega(x)` that is formally
treated as a probability density function for the inputs. The
interpretation and specification of :math:`\omega(x)` is considered in the
context of specific uses of SA; see the discussion pages on SA measures
for simplification
(:ref:`DiscSensitivityAndSimplification<DiscSensitivityAndSimplification>`)
and SA measures for output uncertainty
(:ref:`DiscSensitivityAndOutputUncertainty<DiscSensitivityAndOutputUncertainty>`).

Variance-based sensitivity analysis is a form of probabilistic
sensitivity analysis in which the impact of an input or a group of
inputs on a :ref:`simulator<DefSimulator>`'s output is measured
through variance and reduction in variance. We develop here the
principal measures of influence and sensitivity for individual inputs
and groups of inputs. Our development is initially in terms of a single
simulator output, before moving on to consider multiple outputs.

Notation
~~~~~~~~

The following notation and terminology is introduced in
:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`.

In accordance with the standard :ref:`toolkit
notation<MetaNotation>`, we denote the simulator by :math:`f` and
its inputs by :math:`\strut x`. The focus of SA is the relationship between
:math:`\strut x` and the simulator output(s) :math:`f(x)`. Since SA also
typically tries to isolate the influences of individual inputs, or
groups of inputs, on the output(s), we let :math:`x_j` be the j-th element
of :math:`\strut x` and will refer to this as the j-th input, for
:math:`j=1,2,\ldots,p`, where as usual :math:`p` is the number of inputs. If
:math:`\strut J` is a subset of the indices :math:`\{1,2,\ldots,p\}`, then
:math:`x_J` will denote the corresponding subset of inputs. For instance,
if :math:`J=\{2,6\}` then :math:`x_J=x_{\{2,6\}}` comprises inputs 2 and 6,
while :math:`x_j` is the special case of :math:`x_J` when :math:`J=\{j\}`.
Finally, :math:`x_{-j}` will denote the whole of the inputs :math:`\strut x`
*except* :math:`x_j`, and similarly :math:`x_{-J}` will be the set of all
inputs except those in :math:`x_J`.

Formally, :math:`\omega(x)` is a joint probability density function for all
the inputs. The marginal density function for input :math:`x_j` is denoted
by :math:`\omega_j(x_j)`, while for the group of inputs :math:`x_J` the
density function is :math:`\omega_J(x_J)`. The conditional distribution for
:math:`x_{-J}` given :math:`x_J` is :math:`\omega_{-J|J}(x_{-J}\,|\,x_J)`. Note,
however, that it is common for the distribution :math:`\strut\omega` to be
such that the various inputs are statistically independent. In this case
the conditional distribution :math:`\omega_{-J|J}` does not depend on
:math:`x_J` and is identical to the marginal distribution :math:`\omega_{-J}`.

Note that by assigning probability distributions to the inputs we
formally treat those inputs as random variables. Notationally, it is
conventional in statistics to denote random variables by capital
letters, and this distinction is useful also in probabilistic SA. Thus,
the symbol :math:`\strut X` denotes the set of inputs when regarded as
random (i.e. uncertain), while :math:`\strut x` continues to denote a
particular set of input values. Similarly, :math:`X_J` represents those
random inputs with subscripts in the set :math:`\strut J`, while :math:`x_J`
denotes an actual value for those inputs.

Discussion
----------

We present here the basic variance based SA measures. Some more
technical details may be found in the discussion page on the theory of
variance based SA
(:ref:`DiscVarianceBasedSATheory<DiscVarianceBasedSATheory>`).

The mean effect :math:`M_J(x_J)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we wish to see how varying the input :math:`x_j` influences the
simulator output :math:`f(x)`, we could choose values all the other inputs
(\(x_{-j}`) and run the simulator with these values fixed and with a
variety of different values of :math:`x_j`. However, the response of
:math:`f(x)` to varying :math:`x_j` will generally depend on the values we
choose for :math:`x_{-j}`. We therefore define the *mean effect* of
:math:`x_j` to be the average value of :math:`f(x)`, averaged over the
possible values of :math:`x_{-j}`. The appropriate averaging is to take the
expectation with respect to the conditional distribution of :math:`X_{-j}`
given :math:`x_j`. We denote the mean effect by

:math:`M_j(x_j) = {\mathrm E}[f(X)\,|\,x_j] = \\int f(x) \\,\omega_{-j|j}
(x_{-j}\,|\,x_j) \\,dx_{-j} \\,.`

This is a function of :math:`x_j`, and represents an average response of
the simulator output to varying :math:`x_j`. Simply plotting this function
gives a visual impression of how :math:`x_j` influences the output.

More generally, we can define the mean effect of a group of inputs:

:math:`M_J(x_J) = {\mathrm E}[f(X)\,|\,x_J] = \\int f(x) \\,\omega_{-J|J}
(x_{-J}\,|\,x_J) \\,dx_{-J} \\,.`

The mean effect of :math:`x_J` is a function of :math:`x_J`, and so is more
complex than the mean effect of a single input. However, it can reveal
more structure than the main effects of the individual inputs in the
group.

Interactions and main effects :math:`I_J(x_J)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let the average effect of changing :math:`x_1` from :math:`x_1^0` to
:math:`x_1^1` be denoted by :math:`a_1=M_1(x_1^1)-M_1(x_1^0)`, and similarly
let the average effect of changing :math:`x_2` from :math:`x_2^0` to
:math:`x_2^1` be denoted by :math:`a_2=M_2(x_2^1)-M_2(x_2^0)`. Now consider
changing both of these inputs, from :math:`(x_1^0,x_2^0)` to
:math:`(x_1^1,x_2^1)`. For a simple, smooth simulator, we might think that
the average effect of such a change might be :math:`a_1+a_2`. However, this
will generally not be the case. The actual average change will be
:math:`M_{\{1,2\}}(x_1^1,x_2^1)-M_{\{1,2\}}(x_1^0,x_2^0)`, and where this
is different from :math:`a_1+a_2` there is said to be an interaction
between :math:`x_1` and :math:`x_2`.

Formally, the interaction effect between inputs :math:`x_j` and :math:`x_{j'}`
is defined to be

:math:`I_{\{j,j'\}}(x_{\{j,j'\}}) = M_{\{j,j'\}}(x_{\{j,j'\}}) - I_j(x_j) -
I_{j'}(x_{j'}) - M\,,\qquad(1)`

where

:math:`M = \\mathrm{E}[f(X)] = \\int f(x)\, \\omega(x) dx`

is the overall expected value of the simulator output, and

:math:`I_j(x_j) = M_j(x_j) - M\,.\qquad(2)`

These definitions merit additional explanation! First, :math:`\strut M` is
the uncertainty mean, which is one of the quantities typically computed
in :ref:`uncertainty analysis<DefUncertaintyAnalysis>`. For our
purposes it is a reference point. If we do not specify the values of any
of the inputs, then :math:`\strut M` is the natural estimate for :math:`f(X)`.

If, however, we specify the value of :math:`x_j`, then the natural estimate
for :math:`f(X)` becomes :math:`M_j(x_j)`. We can think of this as the
reference point :math:`\strut M` plus a deviation :math:`I_j(x_j)` from that
reference point. We call :math:`I_j(x_j)` the *main effect* of :math:`x_j`.

(It is easy to confuse the mean effect :math:`M_j(x_j)` with the main
effect :math:`I_1(x_1)` - the two terms are obviously very similar - and
indeed some writers call :math:`M_j(x_j)` the main effect of :math:`x_j`. In
informal discussion, such imprecision in terminology is unimportant, but
the distinction is useful in formal analysis and we will endeavour to
use the terms "mean effect" and "main effect" precisely.)

Now if we specify both :math:`x_j` and :math:`x_{j'}`, equation (1) expresses
the natural estimate :math:`M_{\{j,j'\}}(x_{\{j,j'\}})` as the sum of (a)
the reference point :math:`\strut M`, (b) the two main effects
:math:`I_j(x_j)` and :math:`I_{j'}(x_{j'})`, and (c) the interaction effect
:math:`I_{\{j,j'\}}(x_{\{j,j'\}})`.

We can extend this approach to define interactions of three or more
inputs. The formulae become increasingly complex and the reader may
choose to skip the remainder of this subsection because such
higher-order interactions are usually quite small and are anyway hard to
visualise and interpret.

For the benefit of readers who wish to see the detail, however, we
define increasingly complex interactions recursively via the general
formula

:math:`I_J(x_J)=M_J(x_J) - \\sum_{J'\subset J}I_{J'}(x_{J'})\,,\qquad(3)`

where the sum is over all interactions for which :math:`\strut J'` is a
proper subset of :math:`\strut J`. Notice that when :math:`\strut J'` contains
only a single input index :math:`I_{J'}(x_{J'})` is a main effect, and by
convention we include in the sum the case where :math:`\strut J'` is the
empty set whose "interaction" term is just :math:`\strut M`.

By setting :math:`J=\{j\}`, equation (3) gives the main effect definition
(2), and with :math:`J=\{j.j'\}` it gives the two-input interaction
definition (1). A three-input interaction is obtained by taking the
corresponding three-input mean effect and subtracting from it (a) the
reference point :math:`\strut M`, (b) the three single-input main effects
and (c) the three two-input interactions. And so on.

The sensitivity variance :math:`V_J`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mean interactions are quite detailed descriptions of the effects of
individual inputs and groups of inputs, because they are functions of
those inputs. For many purposes, it is helpful to have a single figure
summary of how sensitive the output is to a given input. Does that input
have a "large" effect on the output? (The methods developed in the
discussion page on decision based SA
(:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`) are different from
those derived here because there we ask the question whether an input
has an "important" effect, in terms of influencing a decision.)

We measure the magnitude of an input's influence on the output by the
expected square of its main effect, or equivalently by the variance of
its mean effect. This definition naturally applies also to a group, so
we define generally

:math:`V_J = \\mathrm{Var}[M_J(X_J)] = \\int \\{M_J(x_J) -
M\}^2\,\omega_J(x_J)\,dx_J\,.`

This is called the sensitivity variance of :math:`x_J`. Although we have
derived this measure by thinking about how large an effect, on average,
varying the inputs :math:`x_J` has on the output, there is another very
useful interpretation.

Consider the overall variance

:math:`V=\mathrm{Var}[f(X)] = \\int \\{f(x) - M\}^2\, \\omega(x)\, dx\,.`

This is another measure that is commonly computed as part of an
uncertainty analysis. It expresses the overall uncertainty about
:math:`f(X)` when :math:`\strut X` is uncertain (and has the probability
distribution defined by :math:`\omega(x)`). If we were to learn the correct
values of the inputs comprising :math:`x_J`, then we would expect this to
reduce uncertainty about :math:`f(X)`. The variance conditional on learning
:math:`x_J` would be

:math:`w(x_J) = \\mathrm{Var}[f(X)\,|\,x_J] = \\int \\{f(x)-M_J(x_J)\}^2
\\,\omega_{-J|J}(x_{-J}\,|\,x_J) \\,dx_{-J}\,.`

Notice that this would depend on what value we discovered :math:`x_J` had,
which of course we do not know. A suitable measure of what the
uncertainty would be after learning :math:`x_J` is the expected value of
this conditional variance, i.e.

:math:`W_J = \\mathrm{E}[w(X_J)] = \\int w(x_J)\,\omega_J(x_J) \\,dx_J
\\,.`

It is shown in
:ref:`DiscVarianceBasedSATheory<DiscVarianceBasedSATheory>` that
:math:`V_J` is the amount by which uncertainty is reduced, i.e.

:math:`V_J = V - W_J\,.`

Therefore we have two useful interpretations of the sensitivity
variance, representing different ways of thinking about the sensitivity
of :math:`f(x)` to inputs :math:`x_J`:

#. :math:`V_J` measures the average magnitude of the mean effect
   :math:`M_J(x_J)`, and so describes the scale of the influence that
   :math:`x_J` has on the output, on average.
#. :math:`V_J` is the amount by which uncertainty about the output would be
   reduced, on average, if we were to learn the correct values for the
   inputs in :math:`x_J`, and so describes how much of the overall
   uncertainty is due to uncertainty about :math:`x_J`.

| The second of these is particularly appropriate when we are using SA
  to analyse uncertainty about the simulator output that is induced by
  uncertainty in the inputs. The first is often more relevant when we
  are using SA to understand the simulator (when :math:`\omega(x)` may be
  simply a weight function rather than genuinely a probability density
  function).

The sensitivity index :math:`S_J` and total sensitivity index :math:`T_J`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sensitivity variance :math:`V_J` is in units that are the square of the
units of the simulator output, and it is common to measure sensitivity
instead by a dimensionless index. The *sensitivity inde*\ x of :math:`x_J`
is its sensitivity variance :math:`V_J` expressed as a proportion of the
overall variance :math:`\strut V`:

:math:`S_J = V_J / V\,.`

Thus an index of 0.5 indicates that uncertainty about :math:`x_J` accounts
for half of the the overall uncertainty in the output due to uncertainty
in :math:`\strut x`. (The index is often multiplied by 100 so as to be
expressed as a percentage; for instance an index of 0.5 would be
referred to as 50%.)

However, there is another way to think about how much of the overall
uncertainty is attributable to :math:`x_J`. Instead of considering how much
uncertainty is reduced if we were to learn :math:`x_J`, we could consider
how much uncertainty *remains* after we learn the values of all the
other inputs, which is :math:`W_{-J}`. This is not the same as :math:`V_J`,
and in most situations is greater. So another index for :math:`S_J` is its
*total sensitivity index*

:math:`T_J = W_{-J} / V = 1-S_{-J}\,.`

The variance partition theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The relationship between these indices (and the reason why :math:`T_J` is
generally larger than :math:`S_J`) can be seen in a general theorem that is
proved in
:ref:`DiscVarianceBasedSATheory<DiscVarianceBasedSATheory>`. First
notice that for an individual :math:`x_j` the sensitivity variance :math:`V_j`
is not just the variance of the mean effect :math:`M_j(X_j)` but also of
its main effect :math:`I_j(X_j)` (since the main effect is just the mean
effect minus a constant). However, it is not true that for a group
:math:`x_J` of more than one input :math:`V_J` equals the variance of
:math:`I_J(X_J)`. If we define

:math:`V^I_J = \\mathrm{Var}[I_J(X_J)]`

then we have :math:`V^I_j=V_j` but otherwise we refer to :math:`V^I_J` as the
interaction variance of :math:`x_J`. Just as an interaction effect between
two inputs :math:`x_j` and :math:`x_{j'}` concerns aspects of the joint effect
of those two inputs that are not explained by their main effects alone,
their interaction variance concerns uncertainty that is attributable to
the joint effect that is not explained by their separate sensitivity
variances. Interaction variances are a useful summary of the extent of
interactions between inputs.

Specifically, the following result holds (under a condition to be
presented shortly)

:math:`V_{\{j,j'\}} = V_j + V_{j'} + V^I_{\{j,j'\}}\,.`

So the amount of uncertainty attributable to (and removed by learning)
both :math:`x_j` and :math:`x_{j'}` is the sum of the amounts attributable to
each separately (their separate sensitivity variances) plus their
interaction variance.

The condition for this to hold is that :math:`x_j` and :math:`x_{j'}` must be
statistically independent. This independence is a property that can be
verified from the probability density function :math:`\omega(x)`.

Generalising, suppose that all the inputs are mutually independent. This
means that their joint density function :math:`\omega(x)` reduces to the
product :math:`\prod_{j=1}^p\omega_j(x_j)` of their marginal density
functions. Independence often holds in practice, partly because it is
much easier to specify :math:`\omega(x)` by thinking about the uncertainty
in each input separately.

If the :math:`x_j`s are mutually independent, then

:math:`V_J = \\sum_{J'\subseteq J} V^I_{J'}\,.\qquad(4)`

Thus the sensitivity variance of a group :math:`x_J` is the sum of the
individual sensitivity variances of the inputs in the group plus all the
interaction variances between members of the group. (Notice that this
time the sum is over all :math:`\strut J'` that are subsets of :math:`\strut
J` including :math:`\strut J` itself.)

In particular, the total variance can be partitioned into the sum of all
the sensitivity and interaction variances:

:math:`V = \\sum_J V^I_J\,.`

This is the partition theorem that is proved in
:ref:`DiscVarianceBasedSATheory<DiscVarianceBasedSATheory>`. We now
consider what it means for the relationship between :math:`S_J` and
:math:`T_J`.

First consider :math:`S_J`. From the above equation (4), this is the sum of
the individual sensitivity indices :math:`S_j` for inputs :math:`x_j` in
:math:`X_J`, plus all the interaction variances between the inputs in
:math:`x_J`, also expressed as proportions of :math:`\strut V`.

On the other hand :math:`T_J` can be seen to be the sum of the individual
sensitivity indices :math:`S_j` plus all the interaction variances (divided
by :math:`\strut V`) that are *not* between inputs in :math:`x_{-J}`. The
difference is subtle, perhaps, but the interactions whose variances are
included in :math:`T_j` are all the ones included in :math:`S_J` plus
interactions that are between (one or more) inputs in :math:`x_J` and (one
or more) inputs outside :math:`x_J`. This is why :math:`T_J` is generally
larger than :math:`S_J`.

The difference between :math:`T_J` and :math:`S_J` is in practice an indicator
of the extent to which inputs in :math:`x_J` interact with inputs in
:math:`x_{-J}`. In particular, the difference between :math:`T_j` and :math:`S_j`
indicates the extent to which :math:`x_j` is involved in interactions with
other inputs.

Regression variances and coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another useful group of measures is associated with fitting simple
regression models to the simulator. Consider approximating :math:`f(x)`
using a regression model of the form

:math:`\hat f_g(x) = \\alpha + g(x)^\mathrm{T}\gamma \\,`

where :math:`g(x)` is a chosen vector of fitting functions and :math:`\alpha`
and :math:`\gamma` are parameters to be chosen for best fit (in the sense
of minimising expected squared error with respect to the distribution
:math::ref:`\omega(x)`). The general case is dealt with in
`DiscVarianceBasedSATheory<DiscVarianceBasedSATheory>`; here we
outline just the simplest, but in many ways the most useful, case.

Consider the case where :math:`g(x)=x_j`. Then the best fitting value of
:math:`\gamma` defines an average gradient of the effect of :math:`x_j`, and
is given by

:math:`\gamma_j = \\mathrm{Cov}[X_j,f(X)] / \\mathrm{Var}[X_j]\,.`

Using this fitted regression reduces uncertainty about :math:`f(X)` by an
amount

:math:`V^L_j = \\mathrm{Cov}[X_j,f(X)]^2 / \\mathrm{Var}[X_j]\,.`

This can be compared with the sensitivity variance of :math:`x_j`, which is
the reduction in uncertainty that is achieved by learning the value of
:math:`x_j`. The sensitivity variance :math:`V_j` will always be greater than
or equal to :math:`V^L_j`, and the difference between the two is a measure
of the degree of nonlinearity in the effect of :math:`x_j`.

Notice that the variance and covariance needed in these formulae are
evaluated using the distribution :math:`\omega(x)`. So
:math:`\mathrm{Var}[X_j]` is just the variance of the marginal distribution
of :math:`x_j`, and if :math:`\bar x_j` is the mean of that marginal
distribution then

:math:`\mathrm{Cov}[X_j,f(X)] = \\int x_j\,M_j(x_j)\,\omega_j(x_j)\,dx_j -
\\bar x_j\,M\,.`

Multiple outputs
~~~~~~~~~~~~~~~~

If :math:`f(x)` is a vector of :math:`\strut r` outputs, then all of the above
measures generalise quite simply. The mean and main effects are now
:math:`r\times 1` vector functions of their arguments, and all of the
variances become :math:`\strut r\times r` matrices. For example, the the
overall variance becomes the matrix

:math:`V=\mathrm{Var}[f(X)] = \\int \\{f(x) - M\}\{f(x) - M\}^\mathrm{T}\,
\\omega(x)\, dx\,.`

Note, however, that we do not consider matrix versions of :math:`S_J` and
:math:`T_J`, because it is not really meaningful to divide a sensitivity
variance matrix :math:`V_J` by the overall variance matrix :math:`\strut V`.

In practice, there is little extra understanding to be obtained by
attempting an SA of multiple outputs in this way beyond what can be
gained by SA of each output separately. Often, if the primary interest
is not in a single output then it can be defined in terms of a single
function of the outputs, and then SA carried out on that single function
is indicated. Some more discussion of SA on a function of :math:`f(x)`,
with an example of SA applied to whether :math:`f(x)` exceeds some
threshold, may be found in
:ref:`DiscSensitivityAndOutputUncertainty<DiscSensitivityAndOutputUncertainty>`.
