.. _AltMeanFunction:

Alternatives: Emulator prior mean function
==========================================

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
approaches are set out in the two core threads: the thread for the
analysis of core model using Gaussian process methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) and the thread for the Bayes
linear emulation for the core model
(:ref:`ThreadCoreBL<ThreadCoreBL>`).

In either approach it is necessary to specify a mean function and
covariance function. We consider here the various alternative forms of
mean function that are dealt with in the :ref:`MUCM<DefMUCM>`
toolkit. An extension to the case of a vector mean function as required
by the thread for the analysis of a simulator with multiple outputs
using Gaussian methods
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`)
can be found in a companion page to this one, dealing with alternatives
for multi-output mean functions
(:ref:`AltMeanFunctionMultivariate<AltMeanFunctionMultivariate>`).

Choosing the Alternatives
-------------------------

The mean function gives the prior expectation for the simulator output
at any given set of input values. We assume here that only one output is
of interest, as in the :ref:`core problem<DiscCore>`.

In general, the mean function will be specified in a form that depends
on a number of :ref:`hyperparameters<DefHyperparameter>`. Thus, if
the vector of hyperparameters for the mean function is :math:`\beta` then
we denote the mean function by :math:`m(\cdot)`, so that :math:`m(x)` is the
prior expectation of the simulator output for vector :math:`x` of input
values.

In principle, this should entail the analyst thinking about what
simulator output would be expected for every separate possible input
vector :math:`x`. In practice, of course, this is not possible. Instead,
:math:`m(\cdot)` represents the general shape of how the analyst expects
the simulator output to respond to changes in the inputs. The use of the
unknown hyperparameters allows the emulator to learn their values from
the training sample data. So the key task in specifying the mean
function is to think generally about how the output will respond to the
inputs.

Having specified :math:`m(\cdot)`, the subsequent steps involved in
building and using the emulator are described in
:ref:`ThreadCoreGP<ThreadCoreGP>` /
:ref:`ThreadCoreBL<ThreadCoreBL>`.

The Nature of the Alternatives
------------------------------

The linear form
~~~~~~~~~~~~~~~

It is usual, and convenient in terms of subsequent building and use of
the emulator, to specify a mean function of the form:

.. math::

   m(x) = \beta^T h(x)

where :math:`h(\cdot)` is a vector of (known) functions of :math:`x`,
known as :ref:`basis functions<DefBasisFunctions>`. This is called
the linear form of mean function because it corresponds to the general
linear regression model in statistical analysis. When the mean function
is specified to have the linear form, it becomes possible to carry out
subsequent analyses more simply. The number of elements of the vector
:math:`h(\cdot)` will be denoted by :math:`q`. These elementary functions are
called :ref:`basis functions<DefBasisFunctions>`.

There remains the choice of :math:`h(\cdot)`. We illustrate the flexibility
of the linear form first through some simple cases.

-  The simplest case is when :math:`q=1` and :math:`h(x)=1` for all
   :math:`x`. Then the mean function is :math:`m(x) = \beta`, where
   now :math:`\beta` is a scalar hyperparameter representing an unknown
   overall mean for the simulator output. This choice expresses no prior
   knowledge about how the output will respond to variation in the
   inputs.

-  Another simple instance is when :math:`h(x)^T=(1,x)`, so that
   :math:`q=1+p`, where :math:`p` is the number of inputs. Then
   :math:`m(x)=\beta_1 + \beta_2 x_1 + \ldots + \beta_{1+p}x_p`, which
   expresses a prior expectation that the simulator output will show a
   trend in response to each of the inputs, but there is no prior
   information to suggest any specific nonlinearity in those trends.

-  Where there is prior belief in nonlinearity of response, then
   quadratic or higher polynomial terms might be introduced into
   :math:`h(\cdot)`.

In principle, all of the kinds of linear regression models that are used
by statisticians are available for expressing prior expectations about
the simulator. Some further discussion of the choice of basis functions
is given in the alternatives page for basis functions for the emulator
mean (:ref:`AltBasisFunctions<AltBasisFunctions>`) and the discussion
page on the use of a structured mean function
(:ref:`DiscStructuredMeanFunction<DiscStructuredMeanFunction>`).

Other forms of mean function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where prior information suggests that the simulator will respond to
variation in its inputs in ways that are not captured by a regression
form, then it is possible to specify any other mean function.

For example,

.. math::

   m(x) = \beta_1 / (1+\beta_2 x_1) + \exp\left(\beta_3 x_2\right)

expresses a belief that as the first input, :math:`x_1` increases the
simulator output will flatten out in the way specified in the first
term, while as :math:`x_2` increases the output will increase (or decrease
if :math:`\beta_3 < 0`) exponentially. Such a mean function might be used
where the prior information about the simulator is suitably strong, but
this cannot be cast as a regression form. As a result, the analysis (as
required for building the emulator and using it for tasks such as
:ref:`uncertainty analysis<DefUncertaintyAnalysis>`) will become more
complex.

Mean functions appropriate for the multivariate output setting are
discussed in
:ref:`AltMeanFunctionMultivariate<AltMeanFunctionMultivariate>`.

Additional Comments, References, and Links
------------------------------------------

It is important to recognise that the emulator specification does not
say that the emulator will respond to its inputs in exactly the way
expressed in the mean function. The Gaussian process, or its Bayes
linear analogue, will allow the actual simulator output to take any form
at all, and given enough training data will adapt to the true form
regardless of what is specified in the prior mean. However, the emulator
will perform better the more accurately the mean function reflects the
actual behaviour of the simulator.

As already discussed, the form of the mean function specifies the shape
that we expect the output to follow as the inputs are varied, with the
hyperparameters :math:`\beta` being estimated from the training data to
identify the mean function fully. A fully Bayesian analysis will require
a prior distribution to be specified for :math:`\beta`, while a Bayes
linear analysis will require a slightly different form of prior
information. This step is addressed in the appropriate core thread,
:ref:`ThreadCoreGP<ThreadCoreGP>` or
:ref:`ThreadCoreBL<ThreadCoreBL>`.
