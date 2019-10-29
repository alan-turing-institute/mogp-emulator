.. _ExamVariantWithDerivatives1Dim:

Example: A one dimensional emulator built with function output and derivatives
==============================================================================

Simulator description
---------------------

In this page we present an example of fitting an emulator to a
:math:`\strut{p=1}` dimensional simulator when in addition to learning the
function output of the simulator, we also learn the partial derivatives
of the output w.r.t the input. The simulator we use is the function,
:math:`\strut {\rm sin}(2x) + (\frac{x}{2})^2`. Although this is not a
complex simulator which takes an appreciable amount of computing time to
execute, it is still appropriate to use as an example. We restrict the
range of the input, :math:`\strut{x}`, to lie in :math:`[-5,5]` and the
behaviour of our simulator over this range is shown in Figure 1. The
simulator can be analytically differentiated to provide the relevant
derivatives: :math:`\strut 2\,{\rm cos}(2x) + (\frac{x}{2})`. For the
purpose of this example we define the adjoint as the function which when
executed returns both the simulator output and the derivative of the
simulator output w.r.t the simulator input.

=================================================
|image0|
Figure 1: The simulator over the specified region
=================================================

Design
------

We need to choose a design to select at which input points the simulator
is to be run and at which points we want to include derivative
information. An Optimised Latin Hypercube, which for one dimension is a
set of equidistant points on the space of the input variable, is chosen.
We select the following :math:`\strut{5}` design points:

:math:`\strut D = [-5, -2.5, 0, 2.5, 5]`.

We choose to evaluate the function output and the derivative at each of
the :math:`\strut 5` points and scale our design points to lie in :math:`
[0,1]`. This information is added to :math:`\strut D` resulting in the
design, :math:`\strut{\tilde{D}}` of length :math:`\strut \\tilde{n}`. A point
in :math:`\strut \\tilde{D}` has the form :math:`\strut (x,d)`, where
:math:`\strut d` denotes whether a derivative or the function output is to
be included at that point; for example :math:`\strut (x,d)=(x,0)` denotes
the function output at point :math:`\strut x` and :math:`\strut (x,d)=(x,1)`
denotes the derivative w.r.t to the first input at point :math:`\strut x`.

This results in the following design:

:math:` \\tilde{D} = [(x_1,d_1),(x_2,d_2),\cdots,(x_{10},d_{10})] = [(0,0),
(0.25,0), (0.5,0), (0.75,0), (1,0), (0,1), (0.25,1), (0.5,1), (0.75,1),
(1,1)]`

The output of the adjoint at these points, which make up our training
data, is:

:math:` \\tilde{f}(\tilde{D}) = [6.794, 2.521, 0, 0.604, 5.706, -41.78,
-6.827, 20, 18.17, 8.219]^{\rm T}`.

Note that the first :math:`\strut 5` elements of :math:`\strut \\tilde{D}`,
and therefore :math:`\strut \\tilde{f}(\tilde{D})`, are simply the same as
in the core problem. Elements :math:`\strut 6-10` correspond to the
derivatives. Throughout this example :math:`\strut n` will refer to the
number of distinct locations in the design (\(\strut 5`), while
:math:`\strut \\tilde{n}` refers to the total amount of data points in the
training sample. Since we include a derivative at all :math:`\strut n`
points w.r.t to the :math:`\strut p=1` input, :math:`\strut
\\tilde{n}=n(p+1)=10`.

Gaussian process setup
----------------------

In setting up the Gaussian process, we need to define the mean and the
covariance function. As we are including derivative information here we
need to ensure that the mean function is once differentiable and the
covariance function is twice differentiable. For the mean function, we
choose the linear form described in the alternatives page on emulator
prior mean function (:ref:`AltMeanFunction<AltMeanFunction>`), which
is :math:`h(x) = [1,x]^{\rm T}` and :math:`q=1+p = 2`. This corresponds to
:math:`\strut \\tilde{h}(x,0) = [1,x]^{\rm T}` and we also have
:math:`\frac{\partial}{\partial x}h(x) = [1]` and so :math:`\strut
\\tilde{h}(x,1) = [1]`.

For the covariance function we choose :math:`\sigma^2c(\cdot,\cdot),` where
the correlation function :math:`c(\cdot,\cdot)` has the Gaussian form
described in the alternatives page on emulator prior correlation
function (:ref:`AltCorrelationFunction<AltCorrelationFunction>`). We
can break this down into 3 cases, as described in the procedure page for
building a GP emulator with derivative information
(:ref:`ProcBuildWithDerivsGP<ProcBuildWithDerivsGP>`):

Case 1 is the correlation between points, so we have :math:`\strut
d_i=d_j=0` and :math:` c\{(x_i,0),(x_j,0)\}=\exp\left[-\{(x_i -
x_j)/\delta\}^2\right] \`.

Case 2 is the correlation between a point, :math:`\strut x_i`, and the
derivatives at point :math:`\strut x_j`, so we have :math:`d_i= 0` and
:math:`d_j\ne0`. Since in this example :math:`\strut p=1`, this amounts to
:math:`\strut d_j=1` and we have:

\\[\frac{\partial}{\partial x_j} c\{(x_i,0),(x_j,1)\} =
\\frac{2}{\delta^2}\left(x_i-x_j\right)\,\exp\left[-\{(x_i -
x_j)/\delta\}^2\right]\].

Case 3 is the correlation between two derivatives at points :math:`\strut
x_i` and :math:`\strut x_j`. Since in this example :math:`p=1`, the only
relevant correlation here is when :math::ref:`d_i=d_j=1` (which corresponds to
Case 3a in `ProcBuildWithDerivsGP<ProcBuildWithDerivsGP>`) and
is given by:

\\[\frac{\partial^2}{\partial x_i \\partial x_j} c\{(x_i,1),(x_j,1)\} =
\\left(\frac{2}{\delta^2} -
\\frac{4\left(x_i-x_j\right)^2}{\delta^4}\right)\,\exp\left[-\{(x_i -
x_j)/\delta\}^2\right].\]

Each Case provides sub-matrices of correlations and we arrange them as
follows: \\[\tilde{A}=\left(\begin{array}{c|c} {\rm Case}\; 1 & {\rm
Case}\; 2 \\\\ {\rm Case}\; 2 & {\rm Case}\; 3 \\\\
\\end{array}\right)\,,\] an :math:`\tilde{n}\times \\tilde{n}` matrix. The
matrix :math:`\strut \\tilde{A}` is symmetric and within :math:`\strut
\\tilde{A}` we have symmetric sub-matrices, Case 1 and Case 3. Case 1
is an :math::ref:`n \\times n=5 \\times 5` matrix and is exactly the same as in
the procedure page `ProcBuildCoreGP<ProcBuildCoreGP>`. Since we
are including the derivative at each of the :math:`\strut 5` design points,
Case 2 and 3 sub-matrices are also of size :math:`n \\times n=5 \\times 5`.

Estimation of the correlation length
------------------------------------

We need to estimate the correlation length :math:`\delta`. In this example
we will use the value of :math:`\delta` that maximises the posterior
distribution :math:`\pi^*_{\delta}(\delta)`, assuming that there is no
prior information on :math:`\delta`, i.e. :math:`\pi(\delta)\propto
\\mathrm{const}:ref:`. The expression that needs to be maximised is (from
`ProcBuildWithDerivsGP<ProcBuildWithDerivsGP>`)

:math:`\pi^*_{\delta}(\delta) \\propto
(\widehat\sigma^2)^{-(\tilde{n}-q)/2}|\tilde{A}|^{-1/2}\|
\\tilde{H}^{\rm T} \\tilde{A}^{-1} \\tilde{H}|^{-1/2}\,.`

where

:math:`\widehat\sigma^2 = (\tilde{n}-q-2)^{-1}\tilde{f}(\tilde{D})^{\rm
T}\left\{\tilde{A}^{-1} - \\tilde{A}^{-1} \\tilde{H}\left(
\\tilde{H}^{\rm T} \\tilde{A}^{-1} \\tilde{H}\right)^{-1}\tilde{H}^{\rm
T}\tilde{A}^{-1}\right\} \\tilde{f}(\tilde{D})\,.`

We have
:math:`\tilde{H}=[\tilde{h}(x_1,d_1),\ldots,\tilde{h}(x_{10},d_{10})]^{\rm
T},` where :math:`\strut \\tilde{h}(x,d)` and :math:`\strut \\tilde{A}` are
defined above in section Gaussian process setup.

Recall that in the above expressions the only term that is a function of
:math:`\delta` is the correlation matrix :math:`\strut \\tilde{A}`.

The maximum can be obtained with any maximisation algorithm and in this
example we used Nelder - Mead. The value of :math:`\strut \\delta` which
maximises the posterior is 0.183 and we will fix :math:`\strut \\delta` at
this value thus ignoring the uncertainty with it, as discussed in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`. We refer to this value of
:math:`\strut \\delta` as :math:`\strut \\hat\delta`. We have scaled the input
to lie in [0,1] and so in terms of the original input scale, :math:`\strut
\\hat\delta` corresponds to a smoothness parameter of 10 x 0.183 = 1.83

Estimates for the remaining parameters
--------------------------------------

The remaining parameters of the Gaussian process are :math:`\strut \\beta`
and :math:`\strut \\sigma^2`. We assume weak prior information on :math:`\strut
{\beta}` and :math:`\strut \\sigma^2 \` and so having estimated the
correlation length, the estimate for :math:`\sigma^2` is given by the
equation above in section Estimation of the correlation length, and the
estimate for :math:`\strut {\beta}` is

:math:`\hat{\beta}=\left( \\tilde{H}^{\rm T} \\tilde{A}^{-1}
\\tilde{H}\right)^{-1}\tilde{H}^{\rm T} \\tilde{A}^{-1}
\\tilde{f}(\tilde{D}) \\,.`

Note that in these equations, the matrix :math:`\strut \\tilde{A}` is
calculated using :math:`\strut \\hat{\delta}`. The application of the two
equations for :math:`\strut \\hat\beta` and :math:`\strut \\widehat\sigma^2`,
gives us in this example :math:`\hat{\beta} = [ 4.734, -2.046]^{\rm T}` and
:math:`\widehat\sigma^2 = 15.47`

Posterior mean and Covariance functions
---------------------------------------

The expressions for the posterior mean and covariance functions as given
in :ref:`ProcBuildWithDerivsGP<ProcBuildWithDerivsGP>` are

:math:`m^*(x) = h(x)^{\rm T}\widehat\beta + \\tilde{t}(x)^{\rm T}
\\tilde{A}^{-1} (\tilde{f}(\tilde{D})-\tilde{H}\widehat\beta)`

and

:math:`v^*(x_i,x_j) = \\widehat\sigma^2\{c(x_i,x_j)\, -\,
\\tilde{t}(x_i)^{\rm T} \\tilde{A}^{-1} \\tilde{t}(x_j)\, +\, \\left(
h(x_i)^{\rm T} - \\tilde{t}(x_i)^{\rm T} \\tilde{A}^{-1}\tilde{H}
\\right) \\left( \\tilde{H}^{\rm T} \\tilde{A}^{-1}
\\tilde{H}\right)^{-1} \\left( h(x_j)^{\rm T} - \\tilde{t}(x_j)^{\rm T}
\\tilde{A}^{-1}\tilde{H} \\right)^{\rm T} \\}\,.`

Figure 2 shows the predictions of the emulator for 100 points uniformly
spaced on the original scale. The solid, black line is the output of the
simulator and the blue, dashed line is the emulator mean :math:`m^*`
evaluated at each of the 100 points. The blue dotted lines represent 2
times the standard deviation about the emulator mean, which is the
square root of the diagonal of matrix :math:`v^*`. The black crosses show
the location of the design points where we have evaluated the function
output and the derivative to make up the training sample. The green
circles show the location of the validation data which is discussed in
the section below. We can see from Figure 2 that the emulator mean is
very close to the true simulator output and the uncertainty decreases as
we get closer the location of the design points.

===============================================================================================================================================================================================================
|image1|
Figure 2: The simulator (solid black line), the emulator mean (blue, dotted) and 95% confidence intervals shown by the blue, dashed line. Black crosses are design points, green circles are validation points.
===============================================================================================================================================================================================================

Validation
----------

In this section we validate the above emulator according to the
procedure page for validating a GP emulator
(:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`).

The first step is to select the validation design. We choose here 15
space filling points ensuring these points are distinct from the design
points. The validation points are shown by green circles in Figure 2
above and in the original input space of the simulator are:

:math:` [-4.8, -4.3, -3.6, -2.9, -2.2, -1.5, -0.8, -0.1, 0.6, 1.3, 2, 2.7,
3.4, 4.1, 4.8]`

and in the transformed space:

:math:`D^\prime = [x^\prime_1,x^\prime_2,\cdots,x^\prime_{15}] = [0.02,
0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49, 0.56, 0.63, 0.7, 0.77, 0.84,
0.91, 0.98]`.

Note that the prime symbol (\(\,\prime:ref:`) does not denote a derivative,
as in `ProcValidateCoreGP<ProcValidateCoreGP>` we use the prime
symbol to specify validation. We're predicting function output in this
example and so do not need validation derivatives; as such we have a
validation design :math:`\strut D^\prime` and not :math:`\strut
\\tilde{D}^\prime`. The function output of the simulator at these
validation points is

:math:`f(D^\prime) = [5.934, 3.888, 2.446, 2.567, 2.161, 0.421, -0.840,
-0.196, 0.102, 0.938, 0.243, 1.050, 3.384, 5.143, 5.586]^{\rm T}`

We then calculate the mean :math:`m^*(\cdot)` and variance
:math:`v^*(\cdot,\cdot)` of the emulator at each validation design point in
:math:`\strut D^\prime` and the difference between the emulator mean and
the simulator output at these points can be compared in Figure 2.

We also calculate standardised errors given in
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>` as
:math:`\frac{f(x^\prime_j)-m^*(x_j^\prime)}{\sqrt{v^*(x^\prime_j,x^\prime_j)}}\,`
and plot them in Figure 3.

====================================================================================
|image2|
Figure 3: Individual standardised errors for the prediction at the validation points
====================================================================================

Figure 3 shows that all the standardised errors lie between -2 and 2
providing no evidence of conflict between simulator and emulator.

We calculate the Mahalanobis distance as given in
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`:

:math:`M = (f(D^\prime)-m^*(D^\prime))^{\rm
T}(v^*(D^\prime,D^\prime))^{-1}(f(D^\prime)-m^*(D^\prime)) = 6.30`

when its theoretical mean is

:math:`{\rm E}[M] = n^\prime = 15` and variance, :math:`{\rm Var}[M] =
\\frac{2n^{\prime}(n^{\prime}+\tilde{n}-q-2)}{\tilde{n}-q-4} = 12.55^2`

We have a slightly small value for the Mahalanobis distance therefore,
but it is within one standard deviation of the theoretical mean. The
validation sample is small and so we would only expect to detect large
problems with this test. This is just an example and we would not expect
a simulator of a real problem to only have one input, but with our
example we can afford to run the simulator intensely over the specified
input region. This allows us to assess the overall performance of the
emulator and, as Figure 2 shows, the emulator can be declared as valid.

Comparison with an emulator built with function output alone
------------------------------------------------------------

We now build an emulator for the same simulator with all the same
assumptions, but this time leave out the derivative information to
investigate the effect of the derivatives and compare the results.

We obtain the following estimates for the parameters:

:math:`\hat\delta = 2.537, \\hat{\beta} = [ 82.06, 54.34]^{\rm T}` and
:math:`\widehat\sigma^2 = 49615`.

Figure 4 shows the predictions of this emulator for 100 points uniformly
spaced on the original scale. The solid, black line is the output of the
simulator and the red, dashed line is the emulator mean evaluated at
each of the 100 points. The red dotted lines represent 2 times the
standard deviation about the emulator mean. The black crosses show the
location of the design points where we have evaluated the function
output. We can see from Figure 4 that the emulator is not capturing the
behaviour of the simulator at all and further simulator runs are
required.

========================================================================================================================================================================
|image3|
Figure 4: The simulator (solid black line), the emulator mean (red, dotted) and 95% confidence intervals shown by the red, dashed line. Black crosses are design points.
========================================================================================================================================================================

We add 4 further design points, :math:` [-3.75, -1.25, 1.25, 3.75]`, and
rebuild the emulator, without derivatives as before. This results in new
estimates for the parameters, :math:`\hat\delta = 0.177, \\hat{\beta} = [
4.32, -2.07]^{\rm T}` and :math:`\widehat\sigma^2 = 15.44`, and Figure 5
shows the predictions of this emulator for the same 100 points. We now
see that the emulator mean closely matches the simulator output across
the specified range.

=============================================================================================================================================================================================================
|image4|
Figure 5: The simulator (solid black line), the emulator mean (red, dotted) and 95% confidence intervals shown by the red, dashed line. Black crosses are design points, green circles are validation points.
=============================================================================================================================================================================================================

We repeat the validation diagnostics using the same validation data and
obtain a Mahalanobis distance of 4.70, while the theoretical mean is 15
with standard deviation 14.14. As for the emulator with derivatives, a
value of 4.70 is bit small; however the standardised errors calculated
as before, and shown in Figure 6 below, provide no evidence of conflict
between simulator and emulator and the overall performance of the
emulator as illustrated in Figure 5 is satisfactory.

====================================================================================
|image5|
Figure 6: Individual standardised errors for the prediction at the validation points
====================================================================================

We have therefore now built a valid emulator without derivatives but
required 4 extra simulator runs to the emulator with derivatives, to
achieve this.

.. |image0| image:: images/ExamVariantWithDerivatives1Dim/Figure1.png
   :width: 350px
   :height: 276px
.. |image1| image:: /foswiki//pub/MUCM/MUCMToolkit/ExamVariantWithDerivatives1Dim/Figure2.png
   :width: 350px
   :height: 276px
.. |image2| image:: /foswiki//pub/MUCM/MUCMToolkit/ExamVariantWithDerivatives1Dim/Figure3.png
   :width: 350px
   :height: 276px
.. |image3| image:: /foswiki//pub/MUCM/MUCMToolkit/ExamVariantWithDerivatives1Dim/Figure4.png
   :width: 350px
   :height: 276px
.. |image4| image:: /foswiki//pub/MUCM/MUCMToolkit/ExamVariantWithDerivatives1Dim/Figure5.png
   :width: 350px
   :height: 276px
.. |image5| image:: /foswiki//pub/MUCM/MUCMToolkit/ExamVariantWithDerivatives1Dim/Figure6.png
   :width: 350px
   :height: 276px
