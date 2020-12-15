.. _AltBasisFunctions:

Alternatives: Basis functions for the emulator mean
===================================================

Overview
--------

The process of building an :ref:`emulator<DefEmulator>` of a
:ref:`simulator<DefSimulator>` involves first specifying prior
beliefs about the simulator and then updating this using a :ref:`training
sample<DefTrainingSample>` of simulator runs. Prior
specification may be either using the fully
:ref:`Bayesian<DefBayesian>` approach in the form of a :ref:`Gaussian
process<DefGP>` or using the :ref:`Bayes
linear<DefBayesLinear>` approach in the form of first and second
order moments. The basics of building an emulator using these two
approaches are set out in the two core threads
(:ref:`ThreadCoreGP<ThreadCoreGP>`,
:ref:`ThreadCoreBL<ThreadCoreBL>`).

In either approach it is necessary to specify a mean function and
covariance function. Choice of the form of the emulator prior mean
function is addressed in :ref:`AltMeanFunction<AltMeanFunction>`. We
consider here the additional problem of specifying the forms of the
:ref:`basis functions<DefBasisFunctions>` :math:`h(x)` when the form of
the mean function is linear.

The Nature of the Alternatives
------------------------------

In order to specify an appropriate form of the global trend component,
we typically first identify a set of potential basis functions
appropriate to the simulator in question, and then select which elements
of that set would best describe the simulator's mean behaviour via the
methods described above. There are a huge number of possible choices for
the specific forms of :math:`h(x)`. Common choices include:

-  Monomials - to capture simple large-scale effects
-  Orthogonal polynomials - to exploit computational simplifications in
   subsequent calculations
-  Fourier/trigonometric functions - to adequately represent periodic
   output
-  A very fast approximate version of the simulator - to capture the
   gross simulator behaviour using an existing model

Choosing the Alternatives
-------------------------

When the mean function takes a linear form, an appropriate choice of the
basis functions :math:`h(x)` is required. This is particularly true for a
Bayes linear emulator as the emphasis is placed on a detailed structural
representation of the simulator's mean behaviour.

Typically, there are two primary methods for determining an appropriate
collection of trend basis functions:

#. prior information about the model can be used directly to specify an
   appropriate form for the mean function;
#. if the number of available model evaluations is very large, then we
   can empirically determine the form of the mean function from this
   data alone.

Expert information about :math:`h(x)` can be derived from a variety of
sources including, but not limited to, the following:

-  knowledge and experience with the computer simulator and its outputs;
-  beliefs about the behaviour of the actual physical system that the
   computer model simulates;
-  experience with similar computer models such as previous versions of
   the same simulator or alternative models for the same system;
-  series expansions of the generating equations underlying the computer
   model (or an appropriately simplified model form);
-  fast approximate versions of the computer model derived from
   simplifications to the current simulator.

If the prior information is weak relative to the available number of
model evaluations and the computer model is inexpensive to evaluate,
then we may choose instead to determine the form of the trend directly
from the model evaluations. This empirical approach is reasonable since
any Bayesian posterior would be dominated by the large volume of data.
Thus in such situations it is reasonable to apply standard statistical
modelling techniques. Empirical construction of the emulator mean is a
similar problem to traditional regression model selection. In this case,
methods such as stepwise model selection could be applied given a set of
potential trend basis functions. However, using empirical methods to
identify the form of the emulator mean function requires many more model
evaluations than are required to fit an emulator with known form and
hence is only applicable if a substantial number of simulator runs is
available. Empirical construction of emulators using cheap simulators is
a key component of :ref:`multilevel
emulation<DefMultilevelEmulation>` which will be described in a
subsequent version of the Toolkit.

Often, the majority of the global variation of the output from a
computer simulator :math:`f(x)` can be attributed to a relatively small
subset, :math:`x_A`, of the input quantities called the :ref:`active
inputs<DefActiveInput>`. In such cases, the emulator mean is
considered to be a function of only the active inputs combined with a
modified form of the covariance. Using this active input approach can
make substantial computational savings; see the discussion page on
active and inactive inputs
(:ref:`DiscActiveInputs<DiscActiveInputs>`) for further details. If
the simulator has a high-dimensional input space then the elicitation of
information about potential active inputs possibly after a suitable
transformation of the input space, then and the form of at least some of
the model effects can be very helpful in emulator construction (see
Craig *et al.* 1998).

References
----------

Craig, P. S., Goldstein, M., Seheult, A. H., and Smith, J. A. (1998)
"Constructing partial prior specifications for models of complex
physical systems," *Applied Statistics*, **47**:1, 37--53
