.. _ThreadVariantModelDiscrepancy:

Thread Variant - Linking Models to Reality using Model Discrepancy
==================================================================

Overview
--------

The principal user entry points to the MUCM toolkit are the various
threads, as explained in the Toolkit structure page
(:ref:`MetaToolkitStructure<MetaToolkitStructure>`). This thread
takes the user through possible techniques used to link the model to
reality. This is a vital step in order to incorporate observed data and
to make any kind of statement about reality itself. As the process of
linking model to reality is somewhat independent of the
:ref:`emulation<DefEmulator>` strategy used, we will assume that the
user has successfully emulated the model using one of the strategies
outlined in the main threads (see e.g.
:ref:`ThreadCoreGP<ThreadCoreGP>` or
:ref:`ThreadCoreBL<ThreadCoreBL>`). If the model is fast enough,
emulation may not even be required, but the process described below
should still be employed. Here we use the term model synonymously with
the term :ref:`simulator<DefSimulator>`.

As we are not concerned with the details of emulation we will describe a
substantially more general case than covered by the :ref:`core
model<DiscCore>`. Assumptions we share with the core model are:

-  We are only concerned with one simulator.
-  The simulator is deterministic.

However, in contrast to the Core model we now assume:

-  We have observations of the real world process against which to
   compare the simulator.
-  We wish to make statements about the real world process.
-  The simulator can produce one, or more than one output of interest.

The first two assumptions are the main justifications for this thread.
The third point states that we will be dealing with both univariate and
multivariate cases. We will also discuss both the fully Bayesian case,
where the simulator is represented as a Gaussian process
(:ref:`ThreadCoreGP<ThreadCoreGP>`), and the Bayes Linear case
(:ref:`ThreadCoreBL<ThreadCoreBL>`), where our beliefs about the
simulator are represented as a :ref:`second-order
specification<DefSecondOrderSpec>`. Other assumptions such as
whether simulator derivative information is available (which might have
been used in the emulation construction process), do not concern us
here.

Notation
--------

In accordance with :ref:`toolkit notation<MetaNotation>`, in this
page we use the following definitions:

-  :math:`\strut{x }` - vector of inputs to the model
-  :math:`\strut{f(x) }` - vector of outputs of the model function
-  :math:`\strut{y }` - vector of the actual system values
-  :math:`\strut{z }` - vector of observations of reality :math:`\strut{y }`
-  :math:`\strut{x^+ }` - the \`best input'
-  :math:`\strut{d }` - the model discrepancy

Model Discrepancy
-----------------

No matter how complex a particular model of a physical process is, there
will always be a difference between the outputs of the model and the
real process that the model is designed to represent. If we are
interested in making statements about the real system using results from
the model, we must incorporate this difference into our analysis.
Failure to do so could lead to grossly inaccurate predictions and
inferences regarding the structure of the real system. See the
discussion page on model discrepancy
(:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>`) where these
ideas are detailed further.

We assume that the simulator produces :math:`\strut{r }` outputs that we
are interested in comparing to real observations. One of the simplest
and most popular methods to represent the difference between model and
reality is that of the :ref:`Best Input Approach<DefBestInput>` which
defines the Model Discrepancy via:

:math:`\strut{ y \\;\; = \\;\; f(x^+) + d, }`

where :math:`\strut{y }`, :math:`\strut{f(x) }`, :math:`\strut{d }` are all
random :math:`\strut{r }`-vectors representing the system values, the
simulator outputs and the :ref:`Model
Discrepancy<DefModelDiscrepancy>` respectively. :math::ref:`\strut{x^+
}` is the vector of \`\ `Best Inputs<DefBestInput>`', which
represents the values that the input parameters take in the real system.
We consider :math:`\strut{d }` to be independent of :math:`\strut{x^+ }` and
uncorrelated with :math:`\strut{f }` and :math:`\strut{f^+ }` (in the Bayes
Linear Case) or independent of :math:`\strut{f }` (in the fully Bayesian
Case), where :math:`\strut{f^+=f(x^+) }`. Note that the :math:`\strut{r
}:ref:`-vector :math:`\strut{d }` may still posses a rich covariance structure,
which will need to be `assessed<DefAssessment>`. Although the
Best Input approach is often chosen for its simplicity, there are
certain subtleties in the definition of :math:`\strut{x^+ }` and in the
independence assumptions. A full discussion of this approach is given in
the discussion page on the best input approach
(:ref:`DiscBestInput<DiscBestInput>`), and also see
:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>` for further
general discussion on the need for a Model Discrepancy term.

More careful methods have been developed that go beyond the simple
assumptions of the Best Input Approach. One such method, known as
:ref:`Reification<DefReification>`, is described in the discussion
page :ref:`DiscReification<DiscReification>` with further theoretical
details given in :ref:`DiscReificationTheory<DiscReificationTheory>`.

Observation Equation
--------------------

Unfortunately, we are never able to measure the real system values
represented by the vector :math:`\strut{y }`. Instead, we can perform
measurements :math:`\strut{z }` of :math:`\strut{y }` that involve some
measurement error. A simple way to express the link between :math:`\strut{z
}` and :math:`\strut{y }` is using the observation equation:

:math:`\strut{ z \\; \\; = \\; \\; y + e }`

where we assume that the measurement error :math:`\strut{e }` is
uncorrelated with :math:`\strut{y }` (in the Bayes Linear case) and
independent of :math:`\strut{y }` (in the fully Bayesian case). It maybe
the case that :math:`\strut{z }` does not correspond exactly to :math:`\strut{y
}`; for example, :math:`\strut{z }` could correspond to either a subset or
some linear combination of the elements of the vector :math:`\strut{y }`.
Methods for dealing with these cases where :math:`\strut{z=Hy+e }`, for
some matrix :math:`\strut{H }`, and cases where :math:`\strut{z }` is a more
complex function of :math:`\strut{y }` are described in the discussion page
on the observation equation
(:ref:`DiscObservations<DiscObservations>`).

Assessing the Model Discrepancy
-------------------------------

In order to make statements about the real system :math::ref:`\strut{y }`, we
need to be able to `assess<DefAssessment>` the Model Discrepancy
:math:`\strut{d }`. Assessing or estimating :math:`\strut{d }` is a difficult
problem: as is discussed in
:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>` :math:`\strut{d
}` represents a statistical model of a difference which is in reality
very complex. Various strategies are available, the suitability of each
depending on the context of the problem.

The first is that of Expert assessment, where the modeller's beliefs
about the deficiencies of the model are converted into statistical
statements about :math::ref:`\strut{ d }` (see
`DiscExpertAssessMD<DiscExpertAssessMD>`). Such considerations
are always important, but they are of particular value when there is a
relatively small amount of observational data to compare the model
output to.

The second is the use of informal methods to obtain order of magnitude
assessments of :math::ref:`\strut{d }` (see
`DiscInformalAssessMD<DiscInformalAssessMD>`). These would often
involve the use of simple computer model experiments to assess the
contributions to the model discrepancy from particular sources (e.g.
forcing function uncertainty).

The third is the use of more formal statistical techniques to assess
:math:`\strut{d }`. These include Bayesian inference (for example, using
MCMC), Bayes Linear inference methods and Likelihood inference. Although
more difficult to implement, these methods have the benefit of rigour
(see :ref:`DiscFormalAssessMD<DiscFormalAssessMD>` for details). It
is worth noting that a full Bayesian inference would
:ref:`calibrate<DefCalibration>` the model and assess :math:`\strut{d }`
simultaneously.

Cases Where Discrepancy has Clearly Defined Structure.
------------------------------------------------------

Physical Structure
~~~~~~~~~~~~~~~~~~

The structure of the discrepancy vector corresponds to the underlying
structure of the output vector, and we often choose to make aspects of
this structure explicit in our notation. Often such structures are
physical in nature, for example various parts of the system could be
naturally labeled by their space-time location :math:`\strut{u }`. Then we
might define the model discrepancy via:

:math:`\strut{ y(u) \\;\;=\;\; f(u,x^+) + d(u) }`

where :math:`\strut{u }` labels the space-time location of the system,
model and model discrepancy. Note that there may still be multiple
outputs at each value of :math:`\strut{u }`.

Consideration of such structures is important as they suggest natural
ways of parameterising the covariance matrix of :math::ref:`\strut{d(u) }`, for
example using a `separable form<DefSeparable>`, and they can
also suggest building certain physical trends into :math:`\strut{{\rm E}[d]
}:ref:`. Further discussion and examples of structured model discrepancies
can be found in `DiscStructuredMD<DiscStructuredMD>`.

Exchangeable Models
~~~~~~~~~~~~~~~~~~~

In some situations a simulator may require, in addition to the usual
input parameters :math:`\strut{x }`, a specification of certain system
conditions. The most common example of a system condition is that of a
forcing function (e.g. rainfall in a flood model). Often there exists a
set of different system conditions (e.g. a set of different possible
realisations of rainfall over a fixed time period) that are considered
equivalent in some sense. It can then be appropriate to consider the
simulator, run at each of the choices of system condition, as a set of
Exchangeable Computer Models. In this case the structure of the model
discrepancy has a particular form, and methodology has been developed to
analyse this more complex situation and the subsequent link to reality,
as can be found in the discussion page on exchangeable models
(:ref:`DiscExchangeableModels<DiscExchangeableModels>`).

Additional Comments, References and Links.
------------------------------------------

This thread has described the importance of including model discrepancy,
and discussed methods of assessing such a term. In the next release,
several procedures will be described for which model discrepancy plays a
vital role. These will include Calibration, History Matching and
Prediction.
