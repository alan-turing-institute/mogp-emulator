.. _ProcUAMultipleEmulators:

Procedure: Uncertainty analysis for a function of simulator outputs using multiple independent emulators
========================================================================================================

Description and Background
--------------------------

Where separate, independent :ref:`emulators<DefEmulator>` have been
built for different :ref:`simulator<DefSimulator>` outputs, there is
often interest in some function(s) of those outputs. In particular, we
may wish to conduct :ref:`uncertainty
analysis<DefUncertaintyAnalysis>` on a function of the outputs.
We assume that, whatever method was used to build each emulator the
corresponding toolkit thread also describes how to compute uncertainty
analysis for that output alone.

If the function of interest is denoted by :math:`f_0(x)`, then for
uncertainty analysis we regard the input vector :math:`x` as a random
variable :math:`X` having a probability distribution :math:`\omega(x)` to
describe our uncertainty about it. Then uncertainty analysis involves
characterising the probability distribution of :math:`f_0(X)` that is
induced by this distribution for :math:`X`. This distribution is known as
the uncertainty distribution. In particular, we are often interested in
the uncertainty mean and and variance

:math:`M_0 = \\textrm{E}[f_0(X)] = \\int_{\cal
X}f_0(x)\,\omega(x)\,{\textrm{d}}x`

and

:math:`V_0 = {\textrm{Var}}[f_0(X)] = \\int_{\cal X}(f_0(x) - M_0)^2
\\omega(x)\,{\textrm{d}}x\,.`

The traditional way to compute these quantities is by Monte Carlo
methods, drawing many random values of :math:`x` from its distribution
:math:`\omega(x)` and running the simulator(s) at each sampled input vector
to compute the resulting values of :math:`f_0(x)`, which then comprise a
sample from the uncertainty distribution. Then for instance :math:`M_0` may
be estimated by the mean of this sample. The accuracy of this estimate
may be quantified using its standard error, which can in principle be
made small by taking a very large sample. In practice, this approach is
often impractical and in any case the MUCM approach using emulators is
generally much more efficient.

The estimate of :math:`M_0` is its emulator (posterior) mean, which we
denote by :math:`\textrm{E}^*[M_0]`, while accuracy of this estimate is
indicated by the emulator variance :math:`{\textrm{Var}}^*[M_0]`.
Similarly, the emulator mean of :math:`V_0`, :math:`\textrm{E}^*[V_0]`, is the
MUCM estimate of :math:`V_0`.

The individual emulators may be :ref:`Gaussian process<DefGP>` (GP)
or :ref:`Bayes linear<DefBayesLinear>` (BL) emulators, although some
of the specific procedures given here will only be applicable to GP
emulators.

Inputs
------

-  Emulators for :math:`r` simulator outputs :math:`f_u(x)`,
   :math:`u=1,2,\cdots,r`
-  A function :math:`f_0(x)` of these outputs for which uncertainty
   analysis is required
-  A probability distribution :math:`\omega(.)` for the uncertain inputs

Outputs
-------

-  Estimation of the uncertainty mean :math:`M_0`, the uncertainty variance
   :math:`V_0` or other features of the uncertainty distribution.

Procedures
----------

Linear case
~~~~~~~~~~~

The simplest case is when the function :math:`f_0(x)` is linear in the
outputs. Thus,

:math:`f_0(x)=a + \\sum_{u=1}^r b_u f_u(x)\,,`

where :math:`a` and :math:`b_1,b_2,\cdots,b_r` are known constants. In the
linear case, the emulator mean and variance of :math:`M_0` may be computed
directly from uncertainty means and variances of the individual
emulators, and the emulator mean of :math:`V_0` requires only a little
extra computation.

Let the following be the results of uncertainty analysis of :math:`f_u(X)`,
:math:`u=1,2,\cdots,r`, where :math:`X` has the specified distribution
:math:`\omega(x)`.

-  :math:`\textrm{E}^*[M_u]`, the emulator mean of the uncertainty mean
   :math:`M_u`,
-  :math:`{\textrm{Var}}^*[M_u]`, the emulator variance of :math:`M_u`, and
-  :math:`\textrm{E}^*[V_u]`, the emulator mean of the uncertainty variance
   :math:`V_u`.

Then

:math:`\textrm{E}^*[M_0] = \\sum_{u=1}^r b_u \\textrm{E}^*[M_u]\,,`

:math:`{\textrm{Var}}^*[M_0] = \\sum_{u=1}^r b_u^2
{\textrm{Var}}^*[M_u]\,,`

:math:`\textrm{E}^*[V_0] = \\sum_{u=1}^r b_u^2 \\textrm{E}^*[V_u] +
2\sum_{u<w}(F_{uw}-\textrm{E}^*[M_u]\textrm{E}^*[M_w])\,.`

The only term in the above formulae that we now need to consider is

:math:`F_{uw} = \\int_{\cal X} \\textrm{E}^*[f_u(x)] \\textrm{E}^*[f_w(x)]
\\,\omega(x) \\,{\textrm{d}}x\,.`

For general emulator structures, this can be evaluated very easily and
quickly by simulation. We simply draw many random input vectors :math:`x`
from the distribution :math:`\omega(x)` and in each case evaluate the
product of the emulator (posterior) means of the two outputs at the
sampled input vector. Given a sufficiently large sample, we can equate
:math:`F_{uw}` to the sample mean of these products. Note that this Monte
Carlo computation does not involve running the original simulator(s),
and so is typically computationally feasible.

We can do better than this in a special case which arises commonly in
practice.

 Special case
^^^^^^^^^^^^

Suppose that for each :math:`u=1,2,\cdots,r`, the emulator of :math:`f_u(x)`
is a GP emulator built using the procedures of the core thread
:ref:`ThreadCoreGP<ThreadCoreGP>` and with the following
specifications:

#. Linear mean function with basis function vector :math:`h_u(x)`.
#. Weak prior information about the hyperparameters :math:`\beta` and
   :math:`\sigma^2`.

Furthermore, suppose that the distribution :math:`\omega` is the
(multivariate) normal distribution with mean (vector) :math:`m` and
precision matrix (the inverse of the variance matrix) :math:`B`.

The emulator will in general include a collection of :math:`M` sets of
values of the correlation hyperparameter matrix :math:`B`. We present below
the computation of :math:`F_{uw}` for given :math:`B`, which is therefore the
value if :math:`M=1`. If :math:`M>1` the :math:`M` resulting values should be
averaged.

Let :math:`\hat\beta_u`, :math:`c_u(x)` and :math:`e_u` be the :math:`\hat\beta`,
:math:`c(x)` and :math:`e` vectors for the :math:`\strut u`-th emulator as
defined in the procedure pages for building the GP emulator
(:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`) and carrying out
uncertainty analysis (:ref:`ProcUAGP<ProcUAGP>`). Then

:math:`F_{uw} = \\hat\beta_u^T Q_{uw} \\hat\beta_w +\hat\beta_u^T S_{uw} e_w
+ \\hat\beta_w^T S_{wu} e_u + e_u^T P_{uw} e_w\,,`

where the matrices :math:`Q_{uw}`, :math:`S_{uw}` and :math:`P_{uw}` are defined
as follows:

:math:`Q_{uw} = \\int_{\cal X} h_u(x) \\,h_w(x)^T \\,\omega(x)
\\,{\textrm{d}}x\,,`

:math:`S_{uw} = \\int_{\cal X} h_u(x) \\,c_w(x)^T \\,\omega(x)
\\,{\textrm{d}}x\,,`

:math:`P_{uw} = \\int_{\cal X} c_u(x) \\,c_w(x)^T \\,\omega(x)
\\,{\textrm{d}}x\,.`

Notice that elements of :math:`Q_{uw}` are just expectations of products of
basis functions with respect to the distribution :math:`\omega(x)`, and
will usually be trivial to compute in the same way as the matrix
:math::ref:`Q_p` in `ProcUAGP<ProcUAGP>`. Indeed, if all the emulators
are built with the same set of basis functions then :math:`Q_{uw}` is the
same for all :math::ref:`u,w` and equals the :math:`Q_p` matrix given in
`ProcUAGP<ProcUAGP>`.

Similarly, the matrix :math::ref:`S_{uw}` is the same as the matrix :math:`S_p` in
`ProcUAGP<ProcUAGP>` (for the :math:`w`-th emulator) except that
instead of its own basis function vector we have the vector :math:`h_u(x)`
from the other emulator. If they have the same basis functions, then
:math:`S_{uw}` is just the :math:`S_p` matrix for emulator :math:`w`.

Hence it remains only to specify the computation of :math:`P_{uw}`. This
will of course depend on the form of the correlation functions used in
building the two emulators.

First suppose that each emulator is built with a generalised Gaussian
correlation function (see the alternatives page on emulator prior
correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`)) which we
write for the :math:`u`-th emulator as

:math:`\exp\{(x-x')^T D_u (x-x')\}\,.`

Then the :math:`(k,\ell)` element of :math:`F_{uw}` is

:math:`F_{uw}^{k\ell} = \|B|^{1/2} \|2D_u+2D_w+B|^{-1/2} \\exp(-g/2)\,,`

where

:math:` \\begin{array}{r l} g =&2(m^*-x_k)^T D_u (m^*-x_k) +2(m^*-x_\ell)^T
D_w (m^*-x_\ell) \\\\ &\quad + (m^*-m)^T B (m^*-m) \\end{array}`

and

:math:`m^\* = (2D_u + 2D_w +B)^{-1}(2D_u x_k + 2D_w x_\ell +Bm)\,.`

Although the generalised Gaussian correlation structure is sometimes
used, it is more common to have the simple Gaussian correlation
structure in which each :math:`D_u` is diagonal. If :math:`B` is also diagonal
(so that the various inputs are independent) then the above formulae
simplify further.

Simulation-based computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GP emulators we have the option of computing uncertainty analysis
quantities by simulation-based methods. We generate a large number
:math:`N` of realisations from each of the :math:`r` emulators, using the
approach of the procedure page
:ref:`ProcSimulationBasedInference<ProcSimulationBasedInference>`.
For each set of realisations, we compute the desired uncertainty
analysis property :math:`Z`; for instance :math:`Z` might be :math:`M_0`,
:math:`V_0` or the probability that :math:`f_0(X)` exceeds some threshold.
This computation is simply done by Monte Carlo. We then have a sample of
values from the posterior distribution of :math:`Z`, from which for
instance we can compute the emulator mean as an estimate of :math:`Z` and
the emulator variance as a summary of emulator uncertainty about :math:`Z`.

A formal description of this procedure is as follows.

#. For :math:`s=1,2,\cdots,N`:

   #. Draw random realisations :math:`f_u^{(s)}(x)`, :math:`s=1.2.\cdots,r`,
      from the emulators
   #. Draw a large sample of random :math:`x` values from the distribution
      :math:`\omega(x)`
   #. For each such :math:`x`, compute :math:`f_0^{(s)}(x)` from the
      :math:`f_u^{(s)}(x)` values
   #. Compute :math:`Z^{(s)}` from the :math:`f_0^{(s)}(x)` values

#. From this large sample of :math:`Z^{(s)}` values, compute the emulator
   mean and variance, etc.
