.. _ThreadGenericHistoryMatching:

Thread: History Matching
========================

Overview
--------

The principal user entry points to the MUCM toolkit are the various
threads, as explained in
:ref:`MetaToolkitStructure<MetaToolkitStructure>`. This thread takes
the user through a technique known as :ref:`history
matching<DefHistoryMatching>`, which is used to learn about the
inputs :math:`\strut{ x }` to a model :math:`\strut{ f(x) }` using
observations of the real system :math:`\strut{ z }`. As the history
matching process typically involves the use of expectations and
variances of :ref:`emulators<DefEmulator>`, we assume that the user
has successfully emulated the model using the Bayes Linear strategy as
detailed in :ref:`ThreadCoreBL<ThreadCoreBL>`. An associated
technique corresponding to a fully probabilistic emulator, as described
in :ref:`ThreadCoreGP<ThreadCoreGP>`, will be discussed in a future
release. Here we use the term model synonymously with the term
:ref:`simulator<DefSimulator>`.

The description of the link between the model and the real system is
vital in the history matching process, therefore several of the concepts
discussed in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
will be used here.

As we are not concerned with the details of emulation we will describe a
substantially more general case than covered by the :ref:`core
model<DiscCore>`. Assumptions we share with the core model are:

-  We are only concerned with one simulator.
-  The simulator is deterministic.

However, in contrast to the Core model we now assume:

-  We have observations of the real world process against which to
   compare the simulator.
-  We wish to make statements about the real world process.
-  The simulator can produce one, or more than one, output of interest.

The first two of these new assumptions are fundamental to this thread as
we will be comparing model outputs with real world measurements. We then
use this information to inform us about the model inputs :math:`\strut{ x
}`. The third point states that we will be dealing with both univariate
and multivariate cases. In this release we discuss the :ref:`Bayes Linear
case<ThreadCoreBL>`, where our beliefs about the simulator are
represented as a :ref:`second-order
specification<DefSecondOrderSpec>`. Other assumptions such as
whether simulator derivative information is available (which could have
been used in the emulation construction process), do not concern us
here.

Notation
--------

In accordance with standard :ref:`toolkit notation<MetaNotation>`, in
this page we use the following definitions:

-  :math:`\strut{ x }` - vector of inputs to the model
-  :math:`\strut{ f(x) }` - vector of outputs of the model function
-  :math:`\strut{ y }` - vector of the actual system values
-  :math:`\strut{ z }` - vector of observations of reality :math:`\strut{ y }`
-  :math:`\strut{ x^+ }` - \`best input'
-  :math:`\strut{ d }` - model discrepancy
-  :math:`\strut{ \\mathcal{X} }` - set of all acceptable inputs
-  :math:`\strut{ \\mathcal{X}_0 }` - whole input space
-  :math:`\strut{ \\mathcal{X}_j }` - reduced input space after
   :math:`\strut{j}` waves
-  :math:`\strut{ I(x) }` - implausibility measure

Motivation for History Matching
-------------------------------

It is often the case that when dealing with a particular model of a
physical process, observations of the real world system are available.
These observations :math:`\strut{ z }` can be compared with outputs of the
model :math:`\strut{ f(x) }`, and often a major question of interest is:
what can we learn about the inputs :math:`\strut{ x }` using the
observations :math:`\strut{ z }` and knowledge of the model :math:`\strut{ f(x)
}`.

History matching is a technique which seeks to identify regions of the
input space that would give rise to acceptable matches between model
output and observed data, a set of inputs that we denote as :math:`\strut{
\\mathcal{X} }`. Often large parts of the input space give rise to
model outputs that are very different from the observed data. The
strategy, as is described below, involves iteratively discarding such
\`implausible' inputs from further analysis, using straightforward,
intuitive criteria.

At each iteration this process involves: the construction of emulators
(which we will not discuss in detail here); the formulation of
:ref:`implausibility measures :math:`\strut{ I(x)
}`<DefImplausibilityMeasure>`; the imposing of cutoffs on the
implausibility measures, and the subsequent discarding of unwanted (or
implausible) regions of input space.

Often in computer model experiments, the vast majority (or even all) of
the input space would give rise to unacceptable matches to the observed
data, and it is these regions that the history matching process seeks to
identify and discard. Analysis of the often extremely small volume that
remains can be of major interest to the modeller. This might involve
analysing in which parts of the space acceptable matches can be found,
what are the dependencies between acceptable inputs and what is the
quality of matches that are possible. The goal here is just to rule out
the obviously bad parts: for a more detailed approach involving priors
and posterior distributions for the best input :math:`\strut{ x^+ }`, the
process known as calibration has been developed. This will be described
in a future release, including a comparison between the calibration and
history matching processes.

Implausibility Measures
-----------------------

The history matching approach is centred around the concept of an
:ref:`implausibility measure<DefImplausibilityMeasure>` which we now
introduce, for further discussion see
:ref:`AltImplausibilityMeasure<AltImplausibilityMeasure>`. An
implausibility measure :math:`\strut{ I(x) }` is a function defined over
the whole input space which, when large, suggests that there would be a
large disparity between the model output and the observed data.

We do not know the model outputs :math:`\strut{ f(x) }` corresponding to
every point :math:`\strut{ x }` in input space, as the model typically
takes too long to run. In order to construct such an implausibility
measure, we first build an emulator such as is described in (coreBL) in
order to obtain the expectation and variance of :math:`\strut{ f(x) }`. We
then compare the expected output :math:`\strut{ {\rm E}[f(x)] }` with the
observations :math:`\strut{ z }`. In the simplest case where :math:`\strut{
f(x) }` represents a single output and :math:`\strut{ z }` a single
observation, a possible form for the univariate implausibility measure
is:

:math:`\strut{ I^2(x) = \\frac{ ({\rm E}[f(x)] - z )^2}{ {\rm Var}[{\rm
E}[f(x)]-z] } = \\frac{ ({\rm E}[f(x)] - z )^2}{{\rm Var}[f(x)] + {\rm
Var}[d] + {\rm Var}[e]} }`

where :math:`\strut{ {\rm E}[f(x)] }` and :math:`\strut{ {\rm Var}[f(x)] }`
are the emulator expectation and variance respectively, :math::ref:`\strut{ d }`
is the `model discrepancy<DefModelDiscrepancy>`, discussed in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
and :math::ref:`\strut{ e }` is the observational error. The second equality
follows from the definition of the `best input<DefBestInput>`
approach (see :ref:`DiscBestInput<DiscBestInput>` for details).

The basic idea is that if :math:`\strut{ I(x) }` is high for some
:math:`\strut{ x }`, then even given all the uncertainties present in the
problem, we would still expect the output of the model to be a poor
match to the observed data :math:`\strut{ z }`. We can hence discard
:math:`\strut{ x }` as a potential member of the set :math:`\strut{
\\mathcal{X} }`.

As is discussed in
:ref:`AltImplausibilityMeasure<AltImplausibilityMeasure>`, there are
many possible choices of measure. If the function has many outputs then
one can define a univariate implausibility measure :math:`\strut{ I_{(i)}(x)
}` for each of the outputs labelled by :math:`\strut{ i }`. One can then
use the maximum implausibility :math:`\strut{ I_M(x) }` to discard input
space. It is also possible to define a full multivariate implausibility
measure :math:`\strut{ I_{MV}(x) }`, provided one has available suitable
multivariate versions of the model discrepancy :math::ref:`\strut{ d }` (see for
example `DiscStructuredMD<DiscStructuredMD>`), the observational
errors, and a multivariate emulator. (A multivariate emulator is not
essential if, for example, the user has an accurate multi-output
emulator: see
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`).

Implausibility measure are simple and intuitive, and are easily
constructed and used, as is described in the next section.

Imposing Implausibility Cutoffs
-------------------------------

The history matching process seeks to identify the set of all inputs
:math:`\strut{ \\mathcal{X} }` that would give rise to acceptable matches
between outputs and observed data. Rather that focus on identifying such
acceptable inputs, we instead discard inputs that are highly unlikely to
be members of :math:`\strut{ \\mathcal{X} }`.

This is achieved by imposing cutoffs on the implausibility measures. For
example, if we were dealing with the univariate implausibility measure
defined above, we might impose the cutoff :math:`\strut{ c }` and discard
from further analysis all inputs that do not satisfy the constraint
:math:`\strut{ I(x) \\le c }` This defines a new sub-volume of the input
space that we refer to as the non-implausible volume, and denote as
:math:`\strut{ \\mathcal{X}_1 }`. The choice of value for the cutoff
:math:`\strut{ c }` is obviously important, and various arguments can be
employed to determine sensible values, as are discussed in
:ref:`DiscImplausibilityCutoff<DiscImplausibilityCutoff>`. A common
method is to use Pukelsheim's (1994) three-sigma rule that states that
for any unimodal, continuous distribution 0.95 of the probability will
lie within a :math:`\strut{ \\pm 3 \\sigma }` interval. This suggests that
taking a value of :math:`\strut{ c=3 }` is a reasonable starting point for
a univariate measure.

Suitable cutoffs for each of the implausibility measures introduced in
:ref:`AltImplausibilityMeasure<AltImplausibilityMeasure>`, such as
:math:`\strut{ I_M(x) }` and :math:`\strut{ I_{MV}(x) }`, can be found through
similar considerations. This often involves analysing the fraction of
input space that would be removed for various sizes of cutoff (see
:ref:`DiscImplausibilityCutoff<DiscImplausibilityCutoff>`). In many
applications, large amounts of input space can be removed using
relatively conservative (i.e. large) choices of the cutoffs.

We apply such space reduction steps iteratively, as described in the
next section.

Iterative Approach to Input Space Reduction
-------------------------------------------

As opposed to attempting to identify the set of acceptable inputs
:math:`\strut{ \\mathcal{X} }` in one step, we instead employ an iterative
approach to input space reduction. At each iteration or *wave*, we
design a set of runs only over the current non-implausible volume,
emulate using these runs, calculate the implausibility measures of
interest and impose cutoffs to define a new (smaller) non-implausible
volume. This is referred to as refocusing.

The full iterative method can be summarised by the following algorithm.
At each iteration or wave:

#. A design for a space filling set of runs over the current
   non-implausible volume :math:`\strut{ \\mathcal{X}_j }` is created.
#. These runs (along with any non-implausible runs from previous waves)
   are used to construct a more accurate emulator defined only over the
   current non-implausible volume :math:`\strut{ \\mathcal{X}_j }`.
#. The implausibility measures are then recalculated over :math:`\strut{
   \\mathcal{X}_j }`, using the new emulator,
#. Cutoffs are imposed on the implausibility measures and this defines a
   new, smaller, non-implausible volume :math:`\strut{ \\mathcal{X}_{j+1}
   }` which should satisfy :math:`\strut{ \\mathcal{X} \\subset
   \\mathcal{X}_{j+1} \\subset \\mathcal{X}_{j} }`.
#. Unless the stopping criteria described below have been reached, or
   the computational resources exhausted, return to step 1.

At each wave the emulators become more accurate, and this allows further
reduction of the input space. Assuming sufficient computational
resources, the stopping criteria are achieved when, after a (usually
small) number of waves, the emulator variance becomes far smaller than
the other uncertainties present, namely the model discrepancy and
observational errors. At this point the algorithm is terminated. The
current non-implausible volume :math:`\strut{ \\mathcal{X}_j }` should be a
reasonable approximation to the acceptable set of inputs :math:`\strut{
\\mathcal{X} }`. For further details and discussion of why this method
works, and for full descriptions of the stopping criteria, see
:ref:`DiscIterativeRefocussing<DiscIterativeRefocussing>`.

A 1D Example
------------

For a simple, illustrative example of the iterative approach to history
matching, see :ref:`Exam1DHistoryMatch<Exam1DHistoryMatch>` where a
simple 1-dimensional model is matched to observed data using two waves
of refocussing.

Additional Comments, References and Links.
------------------------------------------

While the goal of this approach is to identify the set of acceptable
inputs :math:`\strut{ \\mathcal{X} }`, it is possible that this set is
empty. This possibility would be identified by the history matching
approach, and we would therefore suggest that history matching is
employed first, before other more detailed techniques are used. Once it
is established that the set :math:`\strut{ \\mathcal{X} }` is non-empty,
and once the location, size and structure of :math:`\strut{ \\mathcal{X} }`
have been analysed (which are often of major interest to the modellers),
then more detailed techniques such as probabilistic calibration can be
employed.

Note that if, at any wave we find that the set :math:`\strut{ \\mathcal{X}_k
}` is empty, then we would declare that :math:`\strut{ \\mathcal{X} }` is
empty also, and therefore that the simulator does not provide acceptable
matches to the observed data. Conversely, we can establish that
:math:`\strut{ \\mathcal{X} }` is non-empty by checking to see if any of
the runs we have used, in any of the waves, would pass all the
implausibility cutoffs. If so these runs are, by definition, members of
:math:`\strut{ \\mathcal{X} }`. If we have reached the stopping criteria
after k iterations and have not found any such runs, we can do a final
batch of runs provided :math:`\strut{ \\mathcal{X}_k }` is still non-empty.

The iterative refocussing strategy presented in this thread is a very
powerful method and has been successfully used to history match complex
models across a variety of application areas. These include oil
reservoir models (Craig et. al. 1996, 1997) and models of Galaxy
formation (Vernon et. al. 2010, Bower et. al. 2009).

Pukelsheim, F. (1994). “The three sigma rule.” The American
Statistician, 48: 88–91.

Craig, P. S., Goldstein, M., Seheult, A. H., and Smith, J. A. (1996).
“Bayes linear strategies for history matching of hydrocarbon
reservoirs.” In Bernardo, J. M., Berger, J. O., Dawid, A. P., and Smith,
A. F. M. (eds.), Bayesian Statistics 5, 69–95. Oxford, UK: Clarendon
Press.

Craig, P. S., Goldstein, M., Seheult, A. H., and Smith, J. A. (1997).
“Pressure matching for hydrocarbon reservoirs: a case study in the use
of Bayes linear strategies for large computer experiments.” In Gatsonis,
C., Hodges, J. S., Kass, R. E., McCulloch?, R., Rossi, P., and
Singpurwalla, N. D. (eds.), Case Studies in Bayesian Statistics, volume
3, 36–93. New York: Springer-Verlag.

Vernon, I., Goldstein, M., and Bower, R. (2010), “Galaxy Formation: a
Bayesian Uncertainty Analysis,” MUCM Technical Report 10/03

Bower, R., Vernon, I., Goldstein, M., et al. (2009), “The Parameter
Space of Galaxy Formation,” to appear in MNRAS; MUCM Technical Report
10/02
