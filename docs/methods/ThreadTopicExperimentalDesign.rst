.. _ThreadTopicExperimentalDesign:

---+ Thread: Experimental design
================================

This thread presents different ways of an selecting experimental
:ref:`design<DefDesign>`, namely a set of input combinations at which
to make computer runs, to construct an :ref:`emulator<DefEmulator>`.
We may use the term "design space" to mean the region of input space
from which the combinations may be selected and the term "design point"
for an actual combination. Designs separate into two basic classes:
general purpose designs, which may be used for a range of different
:ref:`simulators<DefSimulator>`, and designs which are chosen to be
optimal (in some sense) for a particular simulator.

For computer experiments to build an emulator the most used general
purpose designs are :ref:`space filling
designs<DefSpaceFillingDesign>`. :ref:`model based
designs<DefModelBasedDesign>` which are based on the principles
of optimal design but tailor-made for computer experiments.

Solving optimality problems requires optimisers, that is optimisation
algorithms. Exchange algorithms
(:ref:`ProcExchangeAlgorithm<ProcExchangeAlgorithm>`) are favoured in
which, naively, "bad" design points are exchanged for "good". They are
cheap to implement and very fast. They are similar in style to global
optimisation algorithms, particularly those versions which include a
random search element.

A special technique to approximate the process and its covariance
function, which provides a useful simplification, is the Karhunen-Loeve
expansion:
:ref:`DiscKarhunenLoeveExpansion<DiscKarhunenLoeveExpansion>` and
this can be adapted to experimental design.

An ideal is to be able to conduct sequential experiments in which one
can adapt present computer runs to data and analysis conducted
previously. Some methods are being developed in
:ref:`ProcASCM<ProcASCM>`.

Overview
========

In any scientific study experimental design should play an important
part. The terminology changes, somewhat, between areas and even within
statistics itself. Similar terms are: planning of experiments, design of
a training set, supervised learning, selective sampling, spatial
sampling and so on. Historically, "experiment" has always been a key
part of the scientific method. Francis Bacon (1620, Novum Organum):

*... the real order of experience begins by setting up a light, and then
shows the road by it, commencing with a regulated and digested, not a
misplaced and vague course of experiment, and hence deducing axioms, and
from those axioms new experiments...*

The subject came of age with the introduction of properly planned
agricultural field trials in the 1920s by R A Fisher. It then developed
into combinatorial design methods (Latin Squares etc), :ref:`factorial
design<DiscFactorialDesign>`, response surface design, :ref:`optimal
design<AltOptimalCriteria>` both classical and Bayesian.

In this thread we focus on methods of computer experimental design for
the purpose of building an emulator.

Developing a protocol for a computer experiments study
======================================================

A scientific study requires an experimental protocol, even if the
initial protocol is adapted to circumstances. Experiments need to be
planned. This is as much true of a computer experiment as of a physical
experiment. We divide this section into two subsections, the first
covers some essential planning issues, the second gives a simple
standard scheme or *protocol*.

Planning
--------

The following are some issues which are important when planning a
computer experiment.

#. The objective of the experiment. There are many, e.g. (i) to
   understand the input-output relationship (ii) to find the most
   important explanatory factors (iii) to find the inputs which place
   the output in some target region or achieve a given level (the
   inverse problem) (iv) to find the input which maximises the output
   and what the maximum value is.
#. A starting model. (i) What do we know already about the relationship
   between inputs and outputs? (ii) Can we rank the inputs in perceived
   order of importance? Are there regions of input space for which the
   output has highly variable output?
#. Input and output variables. A full description is essential: (i)
   units of measurement (ii) ranges (iii) discrete, continuous,
   functional, etc (iv) our ability to set levels, measure, record,
   store etc. In summary: each variable needs a full "CV".
#. Input variables (factors) (i) nominal (central) values, (ii) range,
   (ii) method of setting levels: by hand, automated, input files etc.
#. Experimental costs/resources. (i) run time (physical or computer
   experiments) (ii) budget etc. A good measure of cost is how many
   hours, to set up, run the computer, etc to obtain results of a unit
   of experimental activity.

A simple four-stage protocol
----------------------------

It is unwise to launch a study with one large experiment. The following
is a basic protocol. Each stage will need an experimental design and one
should only proceed to the next stage after analysing the results of the
previous stage. Analysis is only discussed in this thread to the extent
needed for design, but it is helpful to provide diagrammatic
representations of results e.g. (i) tables of which input affects which
output (ii) basic effect plots.

#. *Nominal experiment*. Set all inputs to their nominal values and
   generate the output(s). This provides a useful check on (i) the
   performability of a basic run (ii) a central input-to-output
   combination (iii) data on set-up time, run time, etc. By
   experimenting at the "centre" of the input space a useful bench-mark
   for future runs is provided.
#. *Initial*\ :ref:`screening<DefScreening>`\ *experiment.* One may
   use a formal screening design. The purpose is to identify input
   factors which significantly affect one or more outputs, with a view
   to not including (or keeping at their nominal values) the
   non-significant factors. Even keeping all input factors at nominal
   and moving just one factor of interest is useful, although
   inefficient as part of a larger experiment.
#. *Main experiment*. This involves the design and conduct of a larger
   scale experiment making use of (i) perceived significant inputs (ii)
   prior knowledge of possible models. It is here that a more
   sophisticated design for computer experiments may be used.
#. *Confirmatory experiment
   (*\ :ref:`validation<DiscCoreValidationDesign>`\ *experiment)*. At
   a basic level it is useful to have additional training runs as an
   overall check on the accuracy/validity of the emulator. If the
   experiments are a success they will confirm or disconfirm prior
   beliefs about relationships, discover new ones, achieve some optimum
   etc. It is often important to carry out a more focused confirmatory
   follow-up experiment. For example, if it is considered that a set of
   input values puts the output in a target region, then confirmatory
   runs can try to confirm this.

Main experiment design for an emulator
======================================

We now consider in some depth the design of the “main experiment” as
described above, with which to build an emulator. The set of design
points together with the output in this case is commonly referred to as
the :ref:`training sample<DefTrainingSample>`. General discussion on
the design of a training sample is given in the page
:ref:`DiscCoreDesign<DiscCoreDesign>`, and we provide here some more
technical background. We will return briefly to consideration of
screening and validation designs in the final section of this thread.

The most widely used training sample designs are general purpose
designs, particularly those that have a
:ref:`space-filling<DefSpaceFillingDesign>` property. Such designs
attempt to place the design points in the design space so that they are
well separated and cover the design space evenly. The rationale for such
designs rests on the fact that the simulator output is assumed to vary
smoothly as the inputs change, and so in the case of a
:ref:`deterministic<DefDeterministic>` simulator there is very little
extra information to be gained by placing two design points very close
to each other. Having design points very close together can also lead to
numerical difficulties (as discussed in the page
:ref:`DiscBuildCoreGP<DiscBuildCoreGP>`). Conversely, leaving large
“holes” in the design space risks missing important local behaviour of
the simulator.

General purpose designs have a long history in experimental design and
:ref:`DiscFactorialDesign<DiscFactorialDesign>` gives a short
introduction. One could consider a space-filling design as a very
special type of factorial design, again tailored to computer
experiments. In the same way that classical factorial designs give a
certain amount of robustness against different possible simple
polynomial models, so space-filling designs guard against, or prepare
for the presence of, different output features that may arise in
different parts of the design space.

Such general-purpose designs have been widely and successfully used in
computer experiments. But there are several reasons to look at more
sophisticated “tailor-made” designs. For instance, not having points
close together makes it more difficult to identify the form and
parameters of a suitable covariance function (see the discussion of
covariance functions in the page
:ref:`AltCorrelationFunction<AltCorrelationFunction>` and of
estimating their parameters in
:ref:`AltEstimateDelta<AltEstimateDelta>`). Also, sequential design
procedures may allow the main experiment to adapt to information in
earlier stages when planning later stages. (Although some non-random
space-filling designs presented in the page
:ref:`AltCoreDesign<AltCoreDesign>` may be used in a sequential way,
they are not adaptive.) As a result, there is growing interest in
:ref:`model-based<DefModelBasedDesign>` optimal designs for training
samples.

The Bayesian approach is very useful in underpinning the principals of
optimal design because it gives well-defined meaning to the increase in
precision or information expected from an experiment. It is also natural
because in :ref:`MUCM<DefMUCM>`, we choose Bayesian models to build
the emulator.

Model based optimal design is critically dependent on the criteria used.
One way to think of optimal design is as a special type of decision
problem, and like all decision problems some notion of optimality is
needed (in economics one would have a utility function whose expectation
is a risk function). There are well-known criteria which were first
introduced in (non-Bayesian) classical regression analysis but are now
fully adapted to the Bayesian setting. An example of a Bayesian
principal working is in understanding the expected again in information
from an experiment. All these matters are discussed in
:ref:`AltOptimalCriteria<AltOptimalCriteria>`. Further discussion of
basic optimal design for computer experiments can also be found in
:ref:`AltCoreDesign<AltCoreDesign>`.

In the same way that model-based optimal experimental design grew out of
a more decision-theoretical approach to factorial design in regression,
so optimal design for computer experiments is a second or even third
generation approach to experimental design. The methodology behind
optimal design for computer experiments remains, here, Bayes optimal
design, but two issues (at least) distinguish the emphasis of optimal
design for computer experiments from that for Bayes optimal design in
regression. The first is that the criteria are most often based on
prediction because the overall quality of the emulator fit is important.
Second, the covariance parameters appear in the Gaussian Process model
in a non-linear way (see
:ref:`AltCorrelationFunction<AltCorrelationFunction>`), making
optimal design for covariance estimation more intractable when the
covariance parameters are unknown.

-  *Optimisation*. Solving an optimality problem requires and
   optimisation algorithm. Exchange algorithms (see the procedure page
   :ref:`ProcExchangeAlgorithm<ProcExchangeAlgorithm>`) iterativley
   swap one or more points in the design for the same number of points
   in the candidate set, but outside the design, with the aim of
   exchanging "bad" points for "better" points. The algorithms are
   simple to implement and fast, but not guaranteed to converge to the
   globally best solution. More sophisticated algorithms such as branch
   and bound which give a global optimum (see ProcBranchAndBound?) are
   available but slower and harder tio implement.
-  *The Karhunen-Loeve expansion*. A promising way to handle the
   nonlinearity of the covariance function in its parameters is to use
   the Karhunen-Loeve expansion. This approach is described in more
   detail below.
-  *Sequential design*. We have already mentioned the potential value of
   sequential design and this is also discussed below.

Karhunen Loeve (K-L) method
---------------------------

This is a method for representing a Gaussian Process and its covariance
function as arising from a random regression with an infinite number of
regression functions; see
:ref:`DiscKarhunenLoeveExpansion<DiscKarhunenLoeveExpansion>`. These
function are "orthogonal" in a well-defined sense. By truncating the
series, and equivalently its covariance function, we obtain an
approximation to the process but one which makes it an ordinary random
regression and therefore amenable to standard Bayes optimal design
methods; see :ref:`AltOptimalCriteria<AltOptimalCriteria>`. To use
the K-L method one needs to compute the expansion numerically because
there are very few cases in which there is a closed form solution. The
K-L method is one way of avoiding the problems associated with optimal
design for covariance parameters which arise because of the
non-linearity. Another benefit is that one can see how the smoothness of
the process is split between different terms; typically slowly varying
terms lead to design points which are more extreme or concentrate on few
areas whereas high frequency terms tend to require designs points which
are spread inside the design space.

Sequential experiments
----------------------

Sequential methods in experimental design can be simple; the above
four-stage protocol can be considered as a type of sequential
experiment. Full sequential procedures use the data and the analysis
from previous experiments to select further experiments. They can be one
design point at a time or block sequential. The Bayes paradigm is very
useful in understanding sequential experimental design and in
:ref:`AltOptimalCriteria<AltOptimalCriteria>` there is a discussion.
The basic strategy is to update parameter estimates, of both the
"regression" and covariance parameters, and base the next design point
or block of design points on the updated assessment of the underlying
Gaussian process. As mentioned, criteria which depend on prediction
quality are favoured.

It is useful to think of sequential design as being partly adaptive in
the case where outputs play little or no role in the choice of the next
block of designs points and fully adaptive, where both inputs and
outputs are used. The partly adaptive material appears in
:ref:`ProcASCM<ProcASCM>`. Fully adaptive methods will appear in
later releases of the toolkit, using the partly adaptive methods as a
foundation.

Design for other toolkit areas
==============================

Screening design
----------------

Screening was discussed earlier in the context of the second stage of
the four-stage protocol. Screening methods, with the resulting
specialised designs, are considered in the topic thread
:ref:`ThreadTopicScreening<ThreadTopicScreening>`.

Validation design
-----------------

Validation was discussed in the context of the fourth stage of the
four-stage protocol. Suitable criteria and designs for validation are an
active topic of research and we expect to provide more discussion in a
later release of the toolkit. Some interim ideas are presented in
:ref:`DiscCoreValidationDesign<DiscCoreValidationDesign>`.

Simulation design
-----------------

Thi kind of design that arise in the toolkit is in the context of
simulation techniques for computing predictions and other more complex
tasks from emulators. As discussed in
:ref:`ProcSimulationBasedInference<ProcSimulationBasedInference>`,
the general simulation method involves drawing simulated realisations of
the simulator itself, and the associated design issue is discussed in
:ref:`DiscRealisationDesign<DiscRealisationDesign>`. This is another
area where more research is needed and we hope to report progress in
later releases of the toolkit.

Design for combined physical and computer experiments
-----------------------------------------------------

An outstanding problem is to design experiments which are a mixture of
computer experiments (simulation runs) and physical experiments. Some of
the issues come under the heading of
:ref:`calibration<DefCalibration>`. A simple protocol is to do
physical experiments to improve the predictions of constants, features
or simply the model itself where these are predicted by the emulator to
be poor (high discrepancy) or where the uncertainty is large (high
posterior variance). An ideal Bayesian approach is to combine the
emulator and real world model into a single modelling system, given a
full joint prior distribution. This model-based approach may eventually
lead to more coherent optimal design than simple protocols of the kind
just mentioned. The importance of this area cannot be underestimated.

Additional Comments, References, and Links
==========================================

The following books have some design material.

Thomas J. Santner, Brian J. Williams, William Notz. The design and
analysis of computer experiments. Springer, 2003

K. Fang, R. Lui and A.Sudjianto. Design and modelling for computer
experiments. Chapman and Hall/CRC, 2005.

A recent paper on computer/physical experiments is:

D. Romano (with A Giovagnoli) A sequential methodology for integrating
physical and computer experiments. Presentation at the Newton Institute.
http://www.newton.ac.uk/programmes/DOE/seminars/081515001.html
