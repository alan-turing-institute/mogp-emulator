.. _AltCorrelationFunction:

Alternatives: Emulator prior correlation function
=================================================

Overview
--------

The process of building an :ref:`emulator<DefEmulator>` of a
:ref:`simulator<DefSimulator>` involves first specifying prior
beliefs about the simulator and then updating this using a :ref:`training
sample<DefTrainingSample>` of simulator runs. Prior
specification may be either using the fully
:ref:`Bayesian<DefBayesian>` approach in the form of a :ref:`Gaussian
process<DefGP>` (GP) or using the :ref:`Bayes
linear<DefBayesLinear>` approach in the form of first and second
order moments. The basics of building an emulator using these two
approaches are set out in the two core threads: the thread for the
analysis of the core model using Gaussian process methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) and the thread for the Bayes
linear emulation for the core model
(:ref:`ThreadCoreBL<ThreadCoreBL>`).

In either approach it is necessary to specify a covariance function. The
formulation of a covariance function is considered in the discussion
page on the GP covariance function
(:ref:`DiscCovarianceFunction<DiscCovarianceFunction>`). Within the
:ref:`MUCM<DefMUCM>` toolkit, the covariance function is generally
assumed to have the form of a variance (or covariance matrix) multiplied
by a correlation function. We present here some alternative forms for
the correlation function :math:`c(\cdot,\cdot)`, dependent on
hyperparameters :math:`\delta`.

Choosing the Alternatives
-------------------------

The correlation function :math:`c(x,x')` expresses the correlation between
the simulator outputs at input configurations :math:`x` and :math:`x'`, and
represents the extent to which we believe the outputs at those two
points should be similar. In practice, it is formulated to express the
idea that we believe the simulator output to be a relatively smooth and
continuous function of its inputs. Formally, this means the correlation
will be high between points that are close together in the input space,
but low between points that are far apart. The various correlation
functions that we will consider in this discussion of alternatives all
have this property.

The Nature of the Alternatives
------------------------------

The Gaussian form
~~~~~~~~~~~~~~~~~

It is common, and convenient in terms of subsequent building and use of
the emulator, to specify a covariance function of the form

.. math::
   c(x,x') = \exp\left[-0.5(x-x')^TC(x-x')\right]

where :math:`C` is a diagonal matrix whose diagonal elements are the
inverse squares of the elements of the :math:`\delta` vector. Hence, if
there are :math:`p` inputs and the :math:`i`-th elements of the input vectors
:math:`x` and :math:`x'` and the :ref:`correlation
length<DefCorrelationLength>` vector :math:`\delta` are
respectively :math:`x_i`, :math:`x'_i` and :math:`\delta_i`, we can write

.. math::
   c(x,x')&=&\exp\left\{-\sum_{i=1}^p 0.5\left[(x_i - x'_i)/\delta_i\right]^2\right\} \\
          &=&\prod_{i=1}^p\exp\left\{-0.5\left[(x_i - x'_i)/\delta_i\right]^2\right\}.

This formula shows the role of the correlation length hyperparameter
:math:`\delta_i`. The smaller its value, the closer together :math:`x_i` and
:math:`x'_i` must be in order for the outputs at :math:`x` and :math:`x'` to be
highly correlated. Large/small values of :math:`\delta_i` therefore mean
that the output values are correlated over a wide/narrow range of the
:math:`i`-th input :math:`x_i`. See the discussion of "smoothness" at the end of
this page.

Because of its similarity to the density function of a normal
distribution, this is known as the Gaussian form of correlation
function.

A generalised Gaussian form
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second expression in equation (2) has the notable feature that the
correlation function across the whole :math:`x` space is a product of
correlation functions referring to each input separately. A correlation
function of this form is said to be :ref:`separable<DefSeparable>`. A
generalisation of the Gaussian form that does not entail separability is

.. math::
   c(x,x') = \exp\left[-0.5(x-x')^TM(x-x')\right]

where now :math:`M` is a symmetric matrix with elements in the vector
:math:`\delta`. So if there are :math:`p` inputs the :math:`\delta` vector has
:math:`p(p+1)/2` elements, whereas in the simple Gaussian form it has only
:math:`p` elements. The hyperparameters are now also more difficult to
interpret.

In practice, this generalised form has rarely been considered. Its many
extra hyperparameters are difficult to estimate and the greater
generality seems to confer little advantage in terms of obtaining good
emulation.

The exponential power form
~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative generalisation replaces (2) by

.. math::
   c(x,x')=\prod_{i=1}^p\exp\left[-\{|x_i -x'_i|/\delta_{1i}\}^{\delta_{2i}}\right]

where now in addition to correlation length parameters :math:`\delta_{1i}`
we have power parameters :math:`\delta_{2i}`. Hence :math:`\delta` has :math:`2p`
elements. This is called the exponential power form and has been widely
used in practice because it allows the expression of alternative kinds
of :ref:`regularity<DefRegularity>` in the simulator output. The
simple Gaussian form has :math:`\delta_{2i}=2` for every input, but values
less than 2 are also possible. The value of 2 implies that as input i is
varied the output will behave very regularly, in the sense that the
simulator output will be differentiable with respect to :math:`x_i`. In
fact it implies that the output will be differentiable with respect to
input i infinitely many times.

If :math:`1 < \delta_{2i} <2` then the output will be differentiable once
with respect to :math:`x_i` but not twice, while if the value is less than
or equal to 1 the output will not be differentiable at all (but will
still be continuous).

We would rarely wish to allow the power parameters to get as low as 1,
since it is hard to imagine any simulator whose output is not
differentiable with respect to one of its inputs at *any* point in the
parameter space. However, it is hard to distinguish between a function
that is once differentiable and one that is infinitely differentiable,
and allowing power parameters between 1 and 2 can give appreciable
improvements in emulator fit.

Matérn forms
~~~~~~~~~~~~

Another correlation function that is widely used in some applications is
the Matérn form, which for a one-dimensional :math:`x` is

.. math::
   c(x,x') = \frac{2^{1-\delta_2}}{\Gamma(\delta_2)}
             \left(\frac{x-x'}{\delta_1}\right)^{\delta_2} {\cal
             K}_{\delta_2}\left(\frac{x-x'}{\delta_1}\right)

where :math:`{\cal K}_{\delta_2}(\cdot)` is a modified Bessel function of
the third kind, :math:`\delta_1` is a correlation length parameter and
:math:`\delta_2` behaves like the power parameter in the exponential power
family, controlling in particular the existence of derivatives of the
simulator. (The number of derivatives is :math:`\delta_2` rounded up to the
next integer.)

There are natural generalisations of this form to :math:`x` having more
than one dimension.

Adding a nugget
~~~~~~~~~~~~~~~

The Gaussian form with nugget modifies the simple Gaussian form (1) to

.. math::
   c(x,x') = \nu I_{x=x'} + \exp\left[-0.5(x-x')^TC(x-x')\right]

where the expression :math:`I_{x=x'}` is 1 if :math:`x=x'` and is otherwise
zero, and where :math:`\nu` is a :ref:`nugget<DefNugget>` term. A nugget
can similarly be added to any other form of correlation function.

There are three main reasons for adding a nugget term in the correlation
function.

Nugget for computation
^^^^^^^^^^^^^^^^^^^^^^

The simple Gaussian form (1) and the generalised Gaussian form (3) allow
some steps in the construction and use of emulators to be simplified,
with resulting computational benefits. However, the high degree of
regularity (see discussion below) that they imply can lead to
computational problems, too.

One device that is sometimes used to address those problems is to add a
nugget. In this case, :math:`\nu` is not usually treated as a
hyperparameter to be estimated but is instead set at a small fixed
value. The idea is that this small modification is used to achieve
computational stability (in situations that will be set out in the
relevant pages of this toolkit) and ideally :math:`\nu` should be as small
as possible.

Technically, the addition of a nugget implies that the output is now not
even continuous anywhere, but as already emphasised this is simply a
computational device. If the nugget is small enough it should have
negligible effect on the resulting emulator.

Nugget for inactive inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^

When some of the available inputs of the simulator are treated as
:ref:`inactive<DefInactiveInput>` and are to be ignored in building
the emulator, then a nugget term may be added to represent the
unmodelled effects of the inactive inputs; see also the discussion on
active and inactive inputs
(:ref:`DiscActiveInputs<DiscActiveInputs>`).

In this case, :math:`\nu` would normally be treated as an unknown
hyperparameter, and so is added to the set :math:`\delta` of
hyperparameters in the correlation function. The nugget's magnitude will
depend on how much of the output's variation is due to the inactive
inputs. By the nature of inactive inputs, this should be relatively
small but its value will generally need to be estimated as a
hyperparameter.

Nugget for stochastic simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of a nugget term to represent the randomness in outputs from a
:ref:`stochastic<DefStochastic>` simulator is similar to the way it
is introduced for inactive inputs. A thread dealing with stochastic
simulators will be incorporated in a future release of the MUCM toolkit.

Other forms of correlation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In principle, one could consider a huge variety of other forms for the
correlation function, and some of these might be useful for special
circumstances. However, the above cases represent all those forms that
have been used commonly in practice. In the machine learning community
several other flexible correlation functions have been defined which are
claimed to have a range of desirable properties, in particular
non-stationary behaviour. A particular example which has not yet been
applied in emulation is the so called "neural-network correlation
function" which was developed by Chris Williams and can be thought of as
representing a sum of an infinite hidden layer multilayer perceptron,
described in `Gaussian Processes for Machine
Learning <http://www.gaussianprocess.org/gpml/>`__.

Additional Comments, References, and Links
------------------------------------------

The notion of regularity is important in constructing a suitable
correlation function. Regularity is defined in the
:ref:`MUCM<DefMUCM>` toolkit as concerning continuity and
differentiability, and in particular the more derivatives the simulator
has, the more regular it is. The Gaussian form always has infintely many
derivatives, and so expresses a strong belief in the simulator output
responding to its inputs in a very regular way. The Matérn form, on the
other hand, allows any finite positive number of derivatives, depending
on the :math:`\delta_2` hyperparameter. The exponential power form allows
for the simulator output to be infinitely differentiable, not
differentiable anywhere, or just once differentiable.

A function that is at least once differentiable is in practice almost
indistinguishable from one that is infinitely differentiable, so the
extra flexibility of the Matérn form may not be a material advantage.
Whenever we can be confident that the output is continuous and
differentiable the Gaussian form could be used, and this confers
computational benefits in the creation and use of emulators. When we
suspect that the output could respond more irregularly, so that it is
not differentiable everywhere, then the exponential power form is
recommended.

A related property is that of :ref:`smoothness<DefSmoothness>`, which
in the MUCM toolkit concerns how rapidly the simulator output can
"wiggle" as we vary the inputs. This is in practice controlled by
correlation length parameters, which are often referred to as smoothness
or roughness parameters. The higher the values of these hyperparameters,
the less likely the output is to "wiggle" over any given range of
inputs.

In all this discussion of correlation functions, it is important to
remember that a discontinuous mean function will cause the output to be
discontinuous regardless of any regularity specified for the correlation
function, and a wiggly mean function will cause the output to wiggle
regardless of any smoothness properties of the correlation function. We
must therefore think of the correlation function as describing the
behaviour of the simulator output *after* the mean function is
subtracted.

As already discussed, the form of the correlation function specifies how
smooth we expect the simulator output to be as the inputs are varied,
with the hyperparameters :math:`\delta` being estimated from the training
data to identify the correlation function fully. A fully Bayesian
analysis will require prior distributions to be specified for the
hyperparameters, whereas slightly different procedures apply in a Bayes
linear analysis. This step is addressed in the appropriate thread, e.g.
:ref:`ThreadCoreGP<ThreadCoreGP>` or
:ref:`ThreadCoreBL<ThreadCoreBL>`.
