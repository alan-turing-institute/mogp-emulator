.. _ProcOutputSample:

Procedure: Generate random outputs from an emulator at specified inputs
=======================================================================

Description and Background
--------------------------

This procedure describes how to randomly sample output values from an
:ref:`emulator<DefEmulator>`. This, of course, requires a fully
probabilistic emulator such as the :ref:`Gaussian process<DefGP>`
emulator described in the core thread
:ref:`ThreadCoreGP<ThreadCoreGP>`, rather than a :ref:`Bayes
linear<DefBayesLinear>` emulator. You should refer to
:ref:`ThreadCoreGP<ThreadCoreGP>`, the procedure page on building a
GP emulator (:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`), the variant
thread on analysing a simulator with multiple outputs using GP methods
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`)
and the procedure page on building multivariate GP emulators
(:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`) for
definitions of the various functions used in this procedure.

Inputs
------

-  An emulator
-  Emulator hyperparameters :math:` \\theta \`
-  A set of points :math:`D=( x_1',\ldots,x_{n'}' )` in the input space at
   which randomly sampled outputs are required.

Outputs
-------

-  A set of jointly sampled values from the distribution of :math:`f(D')=\{
   f(x_1'),\ldots,f(x_{n'}')\} \` .

Procedure
---------

Scalar output general case
~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Define

:math:` m^*(D')=\{m^*(x_1'),\ldots, m^*(x_{n'}')\}^{\rm T} \`.

2) Define

:math:` v^*(D',D') \`, the :math:` n' \\times n' \` matrix with :math:`i,j`
element given by :math:`v^*(x_i',x_j')`.

3) Generate the random vector :math:` \\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T}
\` from :math:`N_{n'}\{m^*(D'), v^*(D',D')\}`, the multivariate normal
distribution with mean vector :math:`m^*(D')` and variance matrix
:math:`v^*(D',D')`.

Multivariate output general case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a simulator :math:`f(.)` that produces a vector :math:`f(x)` of :math:`r`
outputs for a single input value :math:`x`. To clarify notation, we
generate the random vector :math:`F= \\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T}
\`, arranged as follows:

:math:`F=\left(\begin{array}{c} (\mbox{output 1 at input }x'_{1}) \\\\
\\vdots \\\\ (\mbox{output 1 at input }x'_{n'}) \\\\ \\vdots
\:math:`\mbox{output r at input }x'_{1})\\\ \\vdots \\\\ (\mbox{output r at
input }x'_{n'}) \\end{array} \\right). \`

1) Define

:math:` m^*(D')=\{m^*(x_1')^{\rm T},\ldots, m^*(x_{n'}')^{\rm T}\}^{\rm T}
\`.

In the multivariate case this is an :math:`n' \\times r \` matrix with each
of the :math:` r \` columns representing one of the simulator outputs. We
arrange this into an :math:`n' r\times 1 \` column vector by stacking the
:math:` r \` columns of the :math:`n' \\times r \` matrix into a single column
vector (the :math:`\mathrm{vec}` operation):

:math:` \\mu^*(D')=\mathrm{vec}\{m^*(D')\} \`.

2) Define

:math:` V`, the :math:` (n'r) \\times (n'r) \` matrix with :math:`i,j` element
defined as the posterior covariance between elements :math:`i` and :math:`j`
of :math:`F= \\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T} \`.

3) Generate the random vector :math:`F \` from :math:`N_{n'r}\{\mu^*(D'),V\}`,
the multivariate normal distribution with mean vector :math:`\mu^*(D')` and
variance matrix :math:` V`.

 Multivariate output general case with separable covariance function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose we have a :ref:`separable<DefSeparable>` covariance function
of the form :math:`\Sigma c(.,.) \`, where :math:`\Sigma \` is the output
space covariance matrix, and :math:`c(.,.)` is the input space covariance
function. Following the notation in
:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`, we write the
posterior covariance function as

:math:`\textrm{Cov}[f(x),f(x')|f(D)]=\Sigma c^*(x,x')=\Sigma\{c(x,x')
-c(x)^{\rm T} A^{-1}c(x)\} \`,

with :math:`A=c(D,D) \`. The posterior variance matrix :math:` V` of :math:`
\\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T} \` can then be written as

:math:`V=\Sigma \\otimes c^*(D',D') \`, where :math:`\otimes \` is the
kronecker product. We can now generate :math:`
\\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T} \` using the matrix normal
distribution:

1) Define :math:`U_{\Sigma}` to be the lower triangular square root of
:math:`\Sigma \`

2) Define :math:`U_c` to be the lower triangular square root of
:math:`c^*(D',D') \`

3) Generate :math:`Z_{n',r} \`: an :math:`n'\times r \` matrix of independent
standard normal random variables

4) The random draw from the distribution of :math:`f(D')` is given by
:math:`F=U_c Z_{n',r} U_\sigma`

:math:`F` is an :math:`n'\times r \` matrix, arranged as follows:

:math:` F=\left(\begin{array}{ccc}(\mbox{output 1 at input }x'_{1}) &
\\cdots &(\mbox{output r at input }x'_{1}) \\\\ \\vdots & & \\vdots \\\\
(\mbox{output 1 at input }x'_{n'}) & \\cdots & (\mbox{output r at input
}x'_{n'}) \\end{array}\right) \`.

Scalar output linear mean and weak prior case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Define

:math:` m^*(D')=\{m^*(x_1'),\ldots, m^*(x_{n'}')\}^{\rm T} \`.

2) Define

:math:` v^*(D',D') \`, the :math:` n' \\times n' \` matrix with :math:`i,j`
element given by :math:`v^*(x_i',x_j')`.

3) Generate the random vector :math:` \\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T}
\` from a multivariate student t distribution with mean vector :math:` m^\*
(D')`, variance matrix :math:`V^*(D',D')`, and :math:`n-q` degrees of
freedom.

As an alternative to sampling from the multivariate student t
distribution, you can first first sample a random value of :math:`
\\sigma^2` and then sample from a multivariate normal instead:

3a) Sample :math:` \\tau^2` from the :math:`
Gamma\{(n-q)/2,(n-q-2)\hat{\sigma}^2/2\}` distribution. Note the
parameterisation of the gamma distribution here: if :math:` W\sim
Gamma(a,b)` then the density function is :math:`
p(w)=\frac{b^a}{\Gamma(a)}w^{a-1}\exp(-bw) \`.

3b) Set :math:` \\sigma^2=1/\tau^2` and replace :math:` \\hat{\sigma}^2 \` by
:math:` 1/\tau^2` in the formula for :math:`v^*(x_i',x_j')`.

3c) Compute :math:` v^*(D',D') \`, the :math:` n' \\times n' \` matrix with
:math:`i,j` element given by :math:`v^*(x_i',x_j')`.

3d) Generate the random vector :math:` \\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm
T} \` from :math:`N_{n'}\{m^*(D'), v^*(D',D')\}`, the multivariate normal
distribution with mean vector :math:`m^*(D')` and variance matrix
:math:`v^*(D',D')`.

Multivariate output linear mean and weak prior case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a simulator :math:`f(.)` that produces a vector :math:`f(x)` of :math:`r`
outputs for a single input value :math:`x`. To clarify notation, we
generate the random vector :math:`F= \\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T}
\`, arranged as follows:

:math:`F=\left(\begin{array}{c} (\mbox{output 1 at input }x'_{1}) \\\\
\\vdots \\\\ (\mbox{output 1 at input }x'_{n'}) \\\\ \\vdots
\:math:`\mbox{output r at input }x'_{1})\\\ \\vdots \\\\ (\mbox{output r at
input }x'_{n'}) \\end{array} \\right). \`

1) Define

:math:` m^*(D')=\{m^*(x_1')^{\rm T},\ldots, m^*(x_{n'}')^{\rm T}\}^{\rm T}
\`.

In the multivariate case this is an :math:`n' \\times r \` matrix with each
of the :math:` r \` columns representing one of the simulator outputs. We
arrange this into an :math:`n' r\times 1 \` column vector by performing the
:math:`Vec \` operation:

:math:` \\mu^*(D')=\mathrm{vec}\{m^*(D')\} \`.

2) Define

:math:` V`, the :math:` (n'r) \\times (n'r) \` matrix with :math:`i,j` element
defined as the posterior covariance between elements :math:`i` and :math:`j`
of :math:`F= \\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T} \`.

3) Generate the random vector :math:`F \` from a multivariate student t
distribution with mean vector :math:` \\mu^*(D') \`, variance matrix :math:`
V`, and :math:`n-q` degrees of freedom.

Multivariate output linear mean, weak prior, and separable covariance function case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we have a :ref:`separable<DefSeparable>` covariance function
of the form :math:`\Sigma c(.,.) \`, where :math:`\Sigma \` is the output
space covariance matrix, and :math:`c(.,.)` is the input space covariance
function. Following the notation in
:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`, we write the
posterior covariance function as

:math:`\textrm{Cov}[f(x),f(x')|f(D)]=\widehat{\Sigma}
c^*(x,x')=\widehat\Sigma\,\left\{c(x,x^\prime)\, -\, c(x)^{\rm T} A^{-1}
c(x^\prime)\, +\, R(x) \\left( H^{\rm T} A^{-1} H\right)^{-1}
R(x^\prime)^{\rm T} \\right\}`,

with :math:`A=c(D,D) \` and :math:` R(x) = h(x)^{\rm T} - c(x)^{\rm T} A^{-1}H
\`. The posterior variance matrix :math:` V` of :math:`
\\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T} \` can then be written as

:math:`V=\widehat{\Sigma} \\otimes c^*(D',D') \`, where :math:`\otimes \` is
the kronecker product. We can now generate :math:`
\\{f(x_1'),\ldots,f(x_{n'}')\}^{\rm T} \` using the matrix student
:math:`t` distribution:

1) Define :math:`U_{\Sigma}` to be the lower triangular square root of
:math:`\widehat{\Sigma} \`

2) Define :math:`U_c` to be the lower triangular square root of
:math:`c^*(D',D') \`

3) Generate a :math:`n'\times r` matrix :math:`T_{n',r}` having a multivariate
t distribution with uncorrelated elements, in the following three
sub-steps.

3a) Generate :math:`Z_{n',r} \`: an :math:`n'\times r \` matrix of independent
standard normal random variables.

3b) Generate :math:` W\sim Gamma\{(n-q)/2,0.5\}`.

3c) Set :math:`T_{n',r}=\frac{1}{\sqrt{W/(n-q)}}Z_{n',r} \`.

Note the parameterisation of the gamma distribution here: if :math:`W\sim
Gamma(a,b)` then the density function is
:math:`p(w)=\frac{b^a}{\Gamma(a)}w^{a-1}\exp(-bw) \`.

4) The random draw from the distribution of :math:`f(D')` is given by
:math:`F=U_c T_{n',r} U_\sigma`

:math:`F` is an :math:`n'\times r \` matrix, arranged as follows:

:math:` F=\left(\begin{array}{ccc}(\mbox{output 1 at input }x'_{1}) &
\\cdots &(\mbox{output r at input }x'_{1}) \\\\ \\vdots & & \\vdots \\\\
(\mbox{output 1 at input }x'_{n'}) & \\cdots & (\mbox{output r at input
}x'_{n'}) \\end{array}\right) \`.

As an alternative to sampling standard :math:`t` variables directly we can
replace step 3 with the following.
