.. _DiscBestInput:

Discussion: The Best Input Approach.
====================================

Description and Background
--------------------------

As introduced in the variant thread on linking models to reality
(:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`),
the most commonly used method of linking a model :math:`f` to
reality :math:`y` is known as the :ref:`Best
Input<DefBestInput>` approach. This page discusses the Best
Input assumptions, highlights the strengths and weaknesses of such an
approach and gives links to closely related discussions regarding the
:ref:`model discrepancy<DefModelDiscrepancy>` :math:`d` itself,
and to methods that go beyond the Best Input approach. Here we use the
term model synonymously with the term :ref:`simulator<DefSimulator>`.

Notation
~~~~~~~~

The following notation and terminology is introduced in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`.
In accordance with standard :ref:`toolkit notation<MetaNotation>`, in
this page we use the following definitions:

-  :math:`x` - inputs to the model
-  :math:`f(x)` - the model function
-  :math:`y` - the real system value
-  :math:`z` - an observation of reality :math:`y`
-  :math:`x^+` - the 'best input' (see below)
-  :math:`d` - the model discrepancy (see below)

The full definitions of :math:`x^+` and :math:`d` are given in
the discussion below.

Discussion
----------

A model is an imperfect representation of reality, and hence there will
always be a difference between model output :math:`f(x)` and system
value :math:`y`. Refer to
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
where this idea is introduced, and to the discussion page on model
discrepancy (:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>`)
for the importance of including this feature in our analysis.

It is of interest to try to represent this difference in the simplest
possible manner, and this is achieved by a method known as the Best
Input approach. This approach involves the notion that were we to know
the actual value, :math:`x^+`, of the system properties, then the
evaluation :math:`f(x^+)` would contain all of the information in
the model about system performance. In other words, we only allow
:math:`x^+` to vary because we do not know the appropriate value at
which to fix the input. This does not mean that we would expect perfect
agreement between :math:`f(x^+)` and :math:`y`. Although the
model could be highly sophisticated, it will still offer a necessarily
simplified account of the physical system and will most likely
approximate the numerical solutions to the governing equations (see
:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>` for more
details). The simplest way to view the difference between :math:`f^+
= f(x^+)` and :math:`y` is to express this as:

.. math::
   y = f^+ + d.

Note that as we are uncertain as to the values of :math:`y`,
:math:`f^+` and :math:`d`, they are all taken to be random
quantities. We consider :math:`d` to be independent of
:math:`x^+` and uncorrelated with :math:`f` and
:math:`f^+` (in the Bayes Linear Case) or independent of
:math:`f` (in the fully Bayesian Case). The Model Discrepancy
:math:`d` is the subject, in various forms, of
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`.
The definition of the model discrepancy given by the Best Input approach
is simple and intuitive, and is widely used in computer modelling
studies. It treats the best input :math:`x^+` in an analogous
fashion to the parameters of a statistical model and (in certain
situations) is in accord with the view that :math:`x^+` is an
expression of the true but unknown properties of the physical system.

In the case where each of the elements of :math:`x` correspond to a
well defined physical property of the real system, :math:`x^+` can
simply be thought of as representing the actual values of these physical
properties. For example, if the univariate input :math:`x`
represented the acceleration due to gravity at the Earths surface,
:math:`x^+` would represent the real system value of
:math:`g=9.81 ms^{-2}`.

It is often the case that certain inputs do not have a direct physical
counterpart, but are, instead, some kind of 'tuning parameter' inserted
into the model to describe approximately some physical process that is
either too complex to model accurately, or is not well understood. In
this case :math:`x^+` can be thought of as the values of these
tuning parameters that gives the best performance of the model function
:math:`f(x)` in some appropriately defined sense (for example, the
best agreement between :math:`f^+` and the observed data). See the
discussion pages on reification
(:ref:`DiscReification<DiscReification>`) and its theory
(:ref:`DiscReificationTheory<DiscReificationTheory>`) for further
details about tuning parameters.

In general, inputs can be of many different types: examples include
physical parameters, tuning parameters, aggregates of physical
quantities, control (or variable) inputs, or decision parameters. The
meaning of :math:`x^+` can be different for each type: for a
decision parameter or a control variable there might not even be a
clearly defined :math:`x^+`, as we might want to simultaneously
optimise the behaviour of :math:`f` for all possible values of the
decision parameter or the control variable.

The statement that the model discrepancy :math:`d` is
probabilistically independent of both :math:`f` and
:math:`x^+` (or uncorrelated with :math:`f` or
:math:`x^+` in the Bayes Linear case) is a simple and in many cases
largely reasonable assumption, that helps ensure the tractability of
subsequent calculations. It involves the idea that the modeller has made
all the improvements to the model that s/he can think of, and that
beliefs about the remaining inaccuracies of the model would not be
altered by knowledge of the function :math:`f` or the best input
:math:`x^+`; see further discussions in
:ref:`DiscReification<DiscReification>` and
:ref:`DiscReificationTheory<DiscReificationTheory>`.

Additional Comments
-------------------

Although useful for many applications, the Best Input approach does
break down in certain situations. If we have access to two models, the
second a more advanced version of the first, then we cannot use the Best
Input assumptions for both models. In this case :math:`x^+` would
be different for each model, and it would be unrealistic to
simultaneously impose the independence assumption on both models. An
approach which resolves this issue by modelling relationships across
models, known as Reification, is described in
:ref:`DiscReification<DiscReification>` with a more theoretical
treatment given in
:ref:`DiscReificationTheory<DiscReificationTheory>`.
