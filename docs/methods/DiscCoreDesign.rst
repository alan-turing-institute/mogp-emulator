.. _DiscCoreDesign:

Discussion: Technical issues in training sample design for the core problem
===========================================================================

Description and Background
--------------------------

A number of alternatives for the design of a training sample for the
:ref:`core problem<DiscCore>` are presented in the alternatives page
on training sample design for the core problem
(:ref:`AltCoreDesign<AltCoreDesign>`), but the emphasis there is on
presenting the simplest and most widely used approaches with a minimum
of technical detail. This discussion page goes into more detail about
the alternatives.

In particular,

-  :ref:`AltCoreDesign<AltCoreDesign>` assumes that the region of
   interest has been transformed to :math::ref:`[0,1]^p` by simple linear
   transformations of each `simulator<DefSimulator>` input. We
   discuss here the formulation of the region of interest and various
   issues around transforming it to a unit hypercube in this way.
-  :ref:`AltCoreDesign<AltCoreDesign>` presents a number of ways to
   generate designs that have a space-filling character. We discuss here
   the principles of optimal design, various simplified criteria for
   generating training samples, and how these are related to the
   space-filling designs.

Discussion
----------

The design region
~~~~~~~~~~~~~~~~~

The design region, which we denote by :math:`\cal X`, is a subset of the
space of possible input configurations that we will confine our training
design points :math:`x_1,x_2, \\ldots,x_n` to lie in.

 Specifying the region
^^^^^^^^^^^^^^^^^^^^^

The design region should cover the region over which it is desired to
:ref:`emulate<DefEmulator>` the simulator. This will usually be a
relatively small part of the whole space of possible input
configurations. Simulators are generally built to represent a wide range
of practical contexts, whereas the user is typically interested in just
one such context. For example, a simulator that simulates the hydrology
of rainwater falling in a river catchment will have inputs whose values
can be defined to represent a wide range of catchments, whereas the user
is interested in just one catchment. The range of input parameter values
of interest would then be just the values that might be appropriate to
that catchment. Because of uncertainty about the true values of the
inputs to describe that catchment, we will wish to build the emulator
over a range of plausible values for each uncertain input, but this will
still be a small part of the range of values that the simulator can
accept.

:math:`\cal{X}` is usually specified by thinking of a range of plausible
values for each input and then defining :math:`{\cal X}={\cal X}_1 \\times
{\cal X}_2 \\times \\ldots\times {\cal X}_p\,,` where :math:`{\cal X}_i`
is a finite interval of plausible values for input i. We then say that
:math:`\cal X` is rectangular. In practice, we may not be able to limit the
plausible values for a given input, in the sense that in theory that
input might conceivably take any value, but it would be extremely
surprising if it lay outside a well-defined and restricted region. Then
it would be inefficient to try to emulate the simulator over the whole
range of theoretical possible values of this input, and we would define
its :math:`{\cal X}_i` to cover only the range that the input is likely to
lie in. This is why we have used the phrase "range of plausible values."

It may be that some inputs are known or specified precisely, so that
some :math:`{\cal X}_i` could contain just a single value. However, we can
effectively ignore these inputs and act as if the simulator is defined
with these parameters fixed. Similarly, we can ignore any inputs whose
values must for this context equal known functions of other inputs.
Throughout the :ref:`MUCM<DefMUCM>` toolkit, the number of inputs
(denoted by :math:`p`) is always the number of inputs that can take a range
of values (even if all other inputs were to be fixed). So :math:`\cal X`
will always be of dimension :math:`p`.

However, :math:`\cal X` does not have to be rectangular. A simulator of
plant growth might include inputs describing the soil in terms of the
percentages of clay and sand. Not only must each percentage be between 0
and 100, but the sum of the two percentages must be less than or equal
to 100. This latter constraint imposes a non-rectangular region for
these two inputs. :ref:`AltCoreDesign<AltCoreDesign>` does not
discuss design over non-rectangular regions. Some possible generic
approaches are:

-  Generate a design over a larger rectangular region containing :math:`\cal
   X` and reject any points lying outside :math:`\cal X`. This would seem
   a sensible approach if :math:`\cal X` is "almost" rectangular, so that
   not many points will be rejected.
-  Develop specific design methods for non-rectangular regions. This is
   a topic that is being studied in MUCM, but such methods are not
   currently available in the toolkit.
-  Transform :math:`\cal X` to a rectangular region. In the example of the
   percentages of clay and sand in soil, two kinds of transformations
   come to mind. If we denote these two soil inputs by :math:`s_1` and
   :math:`s_2`, then the first transformation is to define the inputs to be
   :math:`s_1` and :math:`100\,s_2/(100-s_1)`. In terms of these two inputs,
   :math:`{\cal X}= [0,100]^2`. Non-linear one-to-one transformations like
   this are discussed below.

The second transformation option for the soil inputs example is to
generate a design over the unrestricted range of [0,100] for both
:math:`s_1` and :math:`s_2` and to reflect any point having :math:`s_1+s_2>100`
by replacing :math:`s_1` by :math:`100-s_1` and :math:`s_2` by :math:`100-s_2`. This
works because in this case there is a simple one-to-two mapping between
:math:`\cal X` and the rectangular region, but although this might be
indicative of tricks that could be applied in other situations it is not
a generic approach to the problem of non-rectangular regions.

 Transforming to a unit hypercube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`AltCoreDesign<AltCoreDesign>` assumes that :math:`\cal X` is
rectangular and that each :math:`{\cal X}_i` is transformed to :math:`[0,1]`
by a simple linear transformation. More generally, we could use
nonlinear transformations of individual inputs or, as we have seen,
transformations of two or more inputs together. Any one-to-one
transformation which turns :math:`\cal X` into :math:`[0,1]^p` could be used
in conjunction with the design procedures of
:ref:`AltCoreDesign<AltCoreDesign>` to produce a training sample
design. However, the transformation that is used has implications for
the optimality criteria considered below, and in particular for the
appropriateness of simple space-filling designs.

The transformation needs to be one-to-one so that we can transform each
design point back into the original input space in order to run the
simulator at those inputs. We denote the back transformation by :math:`t`,
so that the design point :math:`x_j` converts to the point :math:`t(x_j)` in
the space of the simulator inputs.

Optimality criteria
~~~~~~~~~~~~~~~~~~~

In principle, what makes a good design depends on what we know about the
simulator to begin with. Hence the modelling decisions that we take with
regard to the mean and covariance functions, plus what we also express
in terms of prior information about the
:ref:`hyperparameters<DefHyperparameter>` in those functions - see
the core threads: the thread for the analysis of the core model using
Gaussian process methods (:ref:`ThreadCoreGP<ThreadCoreGP>`) and the
thread for the Bayes lineaar emulation for the core model
(:ref:`ThreadCoreBL<ThreadCoreBL>`).

The formal way to develop an optimal design is using Bayesian decision
theory. This ideal is not feasible for the problem of choosing a
training sample for building an emulator. Nevertheless, we can bear in
mind the broad nature of such a solution when considering simplified
criteria.

 Principles of optimal design for a training sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A good design will enable us to build a good emulator, one that predicts
the simulator's output accurately. The aim of the design is to reduce
uncertainty about the simulator. Our uncertainty about the simulator
before obtaining the training sample data has two main components.

-  We are uncertain about the values of the hyperparameters - :math:`\beta`
   (the hyperparameters of the mean function), :math:`\sigma^2` (the
   variance parameter in the covariance function) and :math:`\delta` (the
   other hyperparameters in the covariance function). This uncertainty
   is expressed in their prior distributions or moments.
-  We would still be uncertain about the precise output that the
   simulator will produce for any given input configuration, even if we
   knew the hyperparameters. This uncertainty is expressed in the
   Gaussian process (in the fully :ref:`Bayesian<DefBayesian>`
   approach of :ref:`ThreadCoreGP<ThreadCoreGP>`), or in the
   equivalent :ref:`Bayes linear<DefBayesLinear>` second-order
   moments interpretation of the mean and covariance functions (in the
   approach of :ref:`ThreadCoreBL<ThreadCoreBL>`).

What features of the design will help us to learn about these things? If
we first consider the second kind of uncertainty, regarding the shape of
the simulator output as a function of its inputs, we should have design
points that are not too close together. This is because points very
close together are highly correlated, so that one could be predicted
well from the other. Having points too close together implies some
redundancy in being able to predict the function.

Good design to learn about the parameters :math:`\beta` of the mean
function depends on the form of the function, which is discussed in the
alternatives function on the emulator prior mean function
(:ref:`AltMeanFunction<AltMeanFunction>`). If we have a constant
mean, :math:`m(x)=\beta`, the location of the design points is irrelevant
and all designs are equally good. If we have a mean function of the form
:math:`m(x) = \\beta_1 + \\beta_2^T x` expressing a linear trend in each
input, a good design will concentrate points in the corners of the
design space, to learn best about the slope parameters :math:`\beta_2`. If
the mean function includes terms that are quadratic in the inputs, we
will concentrate design points on more of the corners and also in the
centre of the space.

To learn about :math:`\sigma^2`, again it is of little importance where the
design points are placed. To learn about :math:`\delta`, however, we will
generally need pairs of points that are at different distances apart,
from being very close together to being far apart.

 Basing design criteria on predictive variances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The primary objective of the design is that after we have run the model
at the training sample design points and fitted the emulator it will
predict the simulator output at other input points accurately. This
accuracy is determined by the posterior predictive variance at that
input configuration. In the procedure page for building a Gaussian
process emulator for the core problem
(:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`) we find two formulae for
the predictive variance conditional on hyperparameters. The full
Bayesian optimal design approach would require the *unconditional*
variances. These would be very complicated to compute and would depend
not only on the design but also on the simulator outputs that we
observed at the design points. These values are of course unknown at the
time of creating the design, and so the full Bayesian design process
would need to average over the prior uncertainty about those
observations. It is this that makes proper optimal design impractical
for this problem. We will instead base the design criteria on the
conditional variance formulae in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`. It is important to
recognise, however, that in doing so our design will not be chosen with
a view to learning about the parameters that we have conditioned on.

In the general case, conditional on the full set of hyperparameters
:math:`\theta=\{\beta,\sigma^2,\delta\}` we have the variance function
:math:`v^*(x,x) = \\sigma^2 c^{(1)}(x)`, where we define

:math:`c^{(1)}(x) = c(x,x) - c(x)^T A^{-1} c(x) \\,.`

When the mean function takes the linear form and we have weak prior
information on :math:`\beta` and :math:`\sigma^2`, then conditional only on
:math:`\delta` we have the variance function :math:`v^*(x,x) =
\\widehat\sigma^2 c^{(2)}(x)`, where now we define

:math:`c^{(2)}(x) = c^{(1)}(x)\, +\, c(x)^T A^{-1} H\left( H^T A^{-1}
H\right)^{-1}H^TA^{-1}c(x) \\,.`

In both cases we have a constant multiplier, :math:`\sigma^2` or its
estimate :math:`\widehat\sigma^2`. As discussed above, the details of the
design have little influence on how well :math:`\sigma^2` is estimated, so
we consider as the primary factor for choosing a design either
:math:`c^{(1)}(.)` or :math:`c^{(2)}(.)` as appropriate.

Notice that neither formulae involves the to-be-observed simulator
outputs from the design. The matrix :math::ref:`A` and the function :math:`c(.)`
are defined in `ProcBuildCoreGP<ProcBuildCoreGP>` as depending
only on the correlation function :math:`c(.,.)`, while the matrix :math:`H`
depends only on the assumed structure of the mean function. The only
hyperparameters that are required by either formula are the vector of
correlation function hyperparameters :math:`\delta`. It is therefore
possible to base design criteria on either :math:`c^{(1)}(x)` or
:math:`c^{(2)}(x)` if we are prepared to specify a prior estimate for
:math:`\delta`.

The difference between the two functions is that :math:`c^{(1)}(x)` arises
from the predictive covariance conditioned on all the hyperparameters,
and so basing a design criterion on this formula will ignore learning
about :math:`\beta`. In contrast, the second term in :math:`c^{(2)}(x)`
expresses specifically the learning about :math:`\beta` in the case of the
linear mean function. Neither allows for learning about :math:`\delta`.

 The effect of transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before discussing specific design criteria based on these functions, we
return to the assumption that the input space has been transformed to
the unit hypercube. This is important because we are proposing to use
formulae which depend on the form of the correlation function and, in
the case of :math:`c^{(2)}(x)` also on the form of the mean function. Both
of these functions are for the purposes of our design problem defined on
the unit cube, not on the original input space. If the correlation
function in the original input space is :math:`c^0(.,.)` then the
correlation function in the unit cube design space has the form
:math:`c(x,x^\prime) = c^0(t(x),t(x^\prime))`.

Now if the transformation is a simple linear one in each dimension, then
all the correlation functions considered in the alternatives page on
emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`) would have
the same form in the transformed space, with only the correlation
lengths (as expressed in :math:`\delta`) changing. Similarly, if the mean
function has the linear form then this form is also retained in the
transformed space. This is why in :ref:`AltCoreDesign<AltCoreDesign>`
it is assumed that such a transformation is all that is required to
achieve the unit hypercube design space. It is then not necessary to
discuss the fact that we have potentially different forms of correlation
and mean function in the transformed space.

For more complex cases, the distinction cannot be ignored. In the case
of the mean function, a belief that the simulator output would respond
roughly quadratically to a certain input would not then hold if we made
a nonlinear transformation of that input. Unless we can realistically
assume that the expected relationship between the output and the inputs
has the linear form :math:`h(x)^T\beta` in the *transformed design space,*
we cannot use :math:`c^{(2)}(.,.)`.

The correlation function will often be less sensitive to the
transformation, in the sense that we may not find it easy to say whether
a specific form (such as the Gaussian form, see
:ref:`AltCorrelationFunction<AltCorrelationFunction>`) would apply in
the original input space or in the transformed design space. Indeed,
transformation may make the simple stationary correlation functions in
:ref:`AltCorrelationFunction<AltCorrelationFunction>` more
appropriate (see also the discussion page on the Gaussian assumption
(:ref:`DiscGaussianAssumption<DiscGaussianAssumption>`)).

 Specific criteria
^^^^^^^^^^^^^^^^^

Having said that we wish to minimise the predictive variance, the
question arises: for which value(s) of :math:`x`? The usual answer is to
minimise the predictive variance integrated over the whole of the design
space. This gives us the two criteria

:math:`C_I^{(u)}(D) = \\int_{[0,1]^p} c^{(u)}(x) dx\,,`

for :math:`u=1,2`. These are the integrated predictive variance criteria.

The integration in the above formula gives equal weight to each point in
the design space. There may well be situations in which we are more
interested in achieving high accuracy over some regions of the space
than over others. Then we can define a weight function :math:`\omega(.)`
and consider the more general criteria

:math:`C_W^{(u)}(D) = \\int_{R^p} c^{(u)}(x) \\omega(x) dx\,,`

for :math:`u=1,2`. Notice now that we integrate not just over the unit
hypercube but over the whole of :math:`p`-dimensional space. This
recognises the fact that although we have constrained the design space
to cover all of the genuinely plausible values of the inputs we may
still have some interest in predicting outside that range. These are the
weighted predictive variance criteria.

How should we expect designs created under the various criteria to
differ? First we note that designs using :math:`C_I^{(2)}(D)` or
:math:`C_W^{(2)}(D)` take account of the possibility of learning about
:math:`\beta` and so can be expected to yield more design points towards
the edges of the design space than their counterparts using
:math:`C_I^{(1)}(D)` or :math:`C_W^{(1)}(D)`. Second, the weighted criteria
should produce more points in areas of high :math:`\omega(x)`. However,
both of these effects are moderated by the fact that points very close
together are wasteful, since they provide almost the same information.

Hence we may expect designs to differ appreciably under the various
criteria when the design size :math:`n` is small, so that extra points in
an area need not be so close together as to be redundant. But for large
designs, all of these criteria are likely to yield fairly evenly spread
designs.

All of these criteria are relatively computationally demanding to
implement in practice (for instance when using the optimised Latin
hypercube design method, as described in the procedure page
(:ref:`ProcOptimalLHC<ProcOptimalLHC>`)). Some theory using entropy
arguments shows that minimising the criterion :math:`C_I^{(1)}(D)` is very
similar to maximising the uncertainty in the design points, leading to
the entropy criterion (also known as the D-optimality criterion), see
:ref:`AltOptimalCriteria<AltOptimalCriteria>`.

:math:`C_E(D) = \| A \|\,,`

which is much quicker to compute. Whereas low values of the other
criteria are good, we aim for high values of :math:`C_E(D)`.

All of these criteria can be used in the optimised Latin hypercube
method, :ref:`ProcOptimalLHC<ProcOptimalLHC>`. This will usually
produce designs that are close to optimal according to the chosen
criterion, unless the optimal design is very far from evenly spread. In
that case, searching for the best Latin hypercube is less likely to
produce near-optimal designs, and other search criteria should be
employed.

 Prior choices of correlation hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As discussed above, implementation of any of the above criteria requires
a prior estimate of :math:`\delta`. In general this requires careful
thought, but it is simplified if we have a Gaussian or exponential power
form of correlation in the design space. For the Gaussian form we
require only estimates of the correlation lengths in each input
dimension, while for the exponential power form we also need estimates
of the power hyperparameters in each dimension. For the latter, power
parameters of 2 in each dimension reduce the exponential power form to
the Gaussian, and would be appropriate if we expect smooth
differentiability with respect to each input. (Note that to make this
choice for the design stage does not imply an assumption of a Gaussian
correlation function when the emulator is built.) Otherwise a value
between 1 and 2, e.g. 1.5, would be preferred.

Correlation length parameters are all relative to the [0,1] range of
values in each dimension. Typical values might be 0.5, suggesting a
relatively smooth response to an input over that range. A lower value,
e.g. 0.2, would be appropriate for an input that was thought to be
strongly influential.

The choice of correlation length parameters in particular can influence
the design. Assigning a lower correlation length to one input will tend
to produce a design with shorter distances between points in this
dimension.

 Space-filling designs
^^^^^^^^^^^^^^^^^^^^^

Unless we specify unequal correlation lengths, or use a weighted
criterion with a weight function that is very far from uniform over the
design space, then we can expect all of the design criteria to produce
designs with points that are more or less evenly spread over
:math:`[0,1]^p`. This leads to a further simplification in design, by
ignoring the formal predictive variance or entropy criteria above and
simply choosing a design that has this even spread property. Such
designs are called space-filling, and these are the design methods that
are presented as the approaches in
:ref:`AltCoreDesign<AltCoreDesign>`.

Additional Comments
-------------------

Further background on designs, particularly on model based :ref:`optimal
design<DefModelBasedDesign>`, can be found in the topic thread
:ref:`ThreadTopicExperimentalDesign<ThreadTopicExperimentalDesign>`.

This is an area of active research within MUCM. New findings and
guidance may be added here in due course.
