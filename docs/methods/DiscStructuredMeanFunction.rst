.. _DiscStructuredMeanFunction:

Discussion: Use of a structured mean function
=============================================

Overview
--------

There are differing views on an appropriate degree of complexity for an
emulator mean function, :math:`m_\beta(x)`, within the computer model
community. Both using very simple mean functions, such as a constant,
and using more structured linear forms have support. The purpose of this
page is to outline the advantages of including a structured mean
function into an emulator where appropriate.

Discussion
----------

When working with expensive computer models, the number of available
evaluations of the accurate computer model are heavily restricted and so
we will obtain poor coverage of the input space for a fixed design size.
In such settings, using a simple constant-only emulator mean function
will result in emulators which are only locally informative in the
vicinity of points which have been evaluated and will revert to their
constant prior mean value once we move into regions of input space which
are far from observed model runs. Conversely, a regression model
contains global terms which are informative across the whole input space
and so does not suffer from these problems of sparse coverage, even when
far from observed simulator runs

In cases where detailed prior beliefs exist about the qualitative global
behaviour of the function, then a logical approach is to construct an
appropriate mean function which captures this. Global aspects of
simulator behaviour such as monotonicity or particular structural forms
are hard to capture from limited simulator evaluations without explicit
modelling in the prior mean function. By exposing potential global
effects via our prior beliefs, this allows us to learn about these
effects and directly gain specific quantitative insight into the
simulator's global behaviour, something which would require extensive
sensitivity analysis with a constant-only mean function. Additionally,
by considering how the global model effects may change over time and
space (or more generally across different outputs) the structured mean
functions provide a natural route into emulation problems with
multivariate (spatio-temporal) output.

If we can explain much of the simulator's variation by using a
structured mean function, then this has the added effect of removing
many of the simulator's global effects from the stochastic residual
process. Indeed, Steinberg & Bursztyn (2004) demonstrated that a
stochastic process with a Gaussian correlation function incorporates
low-order polynomials as leading eigenfunctions. Thus by explicitly
modelling simple global behaviours of the simulator via a structured
mean function we remove the global effects which were hidden in the
"black box" of the stochastic process, and thus reduce the importance of
the stochastic residual process. By diminishing the influence of the
stochastic process, many calculations involving the emulator can be
substantially simplified -- in particular, the construction of
additional designs and the visualisation of the emulator when the
simulator output is high-dimensional. Furthermore, capturing much of the
simulator variation in the mean function reduces the importance of
challenging and intensive tasks such as the estimation of the
hyperparameters :math:`\delta` of the correlation function.

Overfitting
-----------

The primary potential disadvantage to using a structured mean function
is that there is the potential for overfitting of the simulator output.
If the mean function is overly complex relative to the number of
available model evaluations, such as having specified a large number of
basis functions or as the result of empirical construction, then our
emulator mean function may begin to fit the small-scale behaviours of
the residual process. The consequences of overfitting are chiefly poor
predictive performance in terms of both interpolation and extrapolation,
and over-confidence in our adjusted variances. For obvious reasons,
simple mean functions do not encounter this problem however they also do
not have the benefits of adequately expressing the global model effects.
The risk of overfitting can be best controlled via appropriate
diagnostics and validation and the use of an appropriate parsimonious
mean function.

References
----------

-  Steinberg, D. M., and Bursztyn, D. (2004) "Data Analytic Tools for
   Understanding Random Field Regression Models," *Technometrics*,
   **46**:4, 411-420
