.. _ThreadTopicScreening:

Thread: Screening
=================

Overview
--------

:ref:`Screening<DefScreening>` involves identifying the relevant
input factors that drive a :ref:`simulator's<DefSimulator>`
behaviour.

Screening, also known as variable selection in the machine learning
literature, is a research area with a long history. Traditionally,
screening has been applied to physical experiments where a number of
observations of reality are taken. One of the primary aims is to remove,
or reduce, the requirement to measure inconsequential quantities
(inputs) thus decreasing the time and expense required for future
experiments. More recently, screening methods have been developed for
computer experiments where a simulator is developed to model the
behaviour of a physical, or other, system. In this context, the
quantities represent the input variables and the benefit of reducing the
dimension of the input space is on the :ref:`emulator<DefEmulator>`
model complexity and training efficiency rather than on the cost of
actually obtaining the input values themselves.

With the increasing usage of ever more complex models in science and
engineering, dimensionality reduction of both input and output spaces of
models has grown in importance. It is typical, for example in complex
models, to have several tens or hundreds of input (and potentially
output) variables. In such high dimensional spaces, efficient algorithms
for dimensionality reduction are of paramount importance to allow
effective probabilistic analysis. For very high (say over 1000) sizes of
input and/or output spaces open questions remain as to what can be
achieved (see for example the variant thread
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>` for
a discussion of the emulation of multiple outputs, and procedure page
:ref:`ProcOutputsPrincipalComponents<ProcOutputsPrincipalComponents>`
for a description of the use of principal component analysis to reduce
the dimension of the output space in particular). Even in simpler
models, efficient application of screening methods can reduce the
computational cost and permit a focused investigation of the relevant
factors for a given model.

Screening is a constrained version of dimensionality reduction where a
subset of the original variables is retained. In the general
dimensionality reduction case, the variables may be transformed before
being used in the emulator, typically using a linear or non-linear
mapping.

Both :ref:`screening<DefScreening>` and :ref:`Sensitivity
Analysis<DefSensitivityAnalysis>` (SA) may be utilised to
identify variables with negligible total effects on the output
variables. They can provide results at various levels of granularity
from a simple qualitative ranking of the importance of the input
variables through to more exact quantitative results. SA methods provide
more accurate variable selection results but require larger number of
samples, and thus entail much higher computational cost. The class of SA
methods are examined separately, in the topic thread on sensitivity
analysis
(:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`).

The focus in this thread is on screening and thus largely qualitative
types of methods which typically require lower computational resources,
making them applicable to more complex models. Screening methods can be
seen as a form of pre-processing and the simulator evaluations used in
the screening activity can also be used to construct the emulator.

The benefits of screening are many fold:

#. Emulators are simpler; the reduced input space typically results in
   simpler models with fewer (hyper)parameters that are more efficient,
   both to estimate and use.
#. Experimental design is more efficient, in a sequential setting; the
   initial expense of applying screening is typically more than recouped
   since a lower dimensional input space can be filled with fewer design
   points.
#. Interpretability is improved; the input variables are not transformed
   in any way and thus the practitioner can immediately infer that the
   quantities represented in the discarded variables need not be
   estimated or measured in the future.

Screening can be employed as part of the emulator construction and in
practice is often applied prior to many of the other methods described
in the MUCM toolkit.

In this thread we restrict our attention to single outputs, i.e. for
multiple output simulators the outputs would need to be treated
independently, and then the active inputs for each output identified
separately.

In the alternatives page on screening methods
(:ref:`AltScreeningChoice<AltScreeningChoice>`) we provide more
details of the alternative approaches to screening that are possible and
discuss under what scenarios each screening method may be appropriate.

After the screening task is completed, the identified inactive factors
may be excluded from further analysis as detailed in the discussion page
on active and inactive inputs
(:ref:`DiscActiveInputs<DiscActiveInputs>`).

References
----------

Saltelli, A., Chan, K. and Scott, E. M. (eds.) (2000). :ref:`Sensitivity
Analysis<http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0471998923>`.
Wiley.
