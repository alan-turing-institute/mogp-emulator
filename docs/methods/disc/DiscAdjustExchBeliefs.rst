.. _DiscAdjustExchBeliefs:

Discussion: Adjusting Exchangeable Beliefs
==========================================

Overview
--------

:ref:`Second-order exchangeability<DefSecondOrderExch>` judgements
about a collection of quantities can be used to adjust our beliefs and
make inference about population means. There are many methods which
address this problem, which can also be extended to the adjustment of
beliefs about population variances and covariances. This requires some
additional consideration of the specification of our prior beliefs as
well as the organisation of such an adjustment.

This concept is of particular relevance as we typically consider the
residual process from an :ref:`emulator<DefEmulator>` to have a
second-order exchageable form. This means that we can employ these
methods to learn about the population mean and variance of the emulator
residuals. This process is described in
:ref:`ProcBLVarianceLearning<ProcBLVarianceLearning>`, whereas this
page focuses on the general methodology.

This page assumes that the reader is familiar with the concepts of
:ref:`exchangeability<DefExchangeability>` and :ref:`second-order
exchangeability<DefSecondOrderExch>`. The analysis and
discussion will be from a :ref:`Bayes linear<DefBayesLinear>`
perspective.

The Second-Order Representation Theorem
---------------------------------------

Suppose that we have a collection of random variables
:math:`Y=\{Y_1,Y_2,\dots\}`, which we judge to be second-order exchangeable
(SOE). Second-order exchangeability induces the following belief
specification for :math:`Y`:

-  :math:`\text{E}[Y_i]=\mu`,
-  :math:`\text{Var}[Y_i]=\Sigma`,
-  :math:`\text{Cov}[Y_i,Y_j]=\Gamma`,

for all :math:`i\neq j` where :math:`\mu`, :math:`\Sigma`, and :math:`\Gamma` are
the population means, variances and covariances.

It can be shown that for an infinite, second-order exchangeable sequence
:math:`Y_i` can be decomposed into two components - a **mean** component
and a **residual** component. The justification for this decomposition
is the representation theorem for second-order exchangeable random
variables (see section 6.4 of Goldstein & Woof). Application of this
representation theorem imposes the following structure and belief
specifications:

#. :math:`Y_i=\mathcal{M}(Y)+\mathcal{R}_i(Y)\\ \\ (1)` - Each individual
   :math:`Y_i` is decomposed into the population mean vector,
   :math:`\mathcal{M}(Y)`, and its individual residual vector,
   :math:`\mathcal{R}_i(Y)`.
#. :math:`\text{E}[\mathcal{M}(Y)]=\mu`, and
   :math:`\text{Var}[\mathcal{M}(Y)]=\Gamma=\text{Cov}[Y_i,Y_j]` - The
   expectation of the mean component is the population mean of the
   :math:`Y_i`, and since the mean component is common to all :math:`Y_i` it
   is the source of all the covariance between the :math:`Y_i`,
#. :math:`\text{E}[\mathcal{R}_i(Y)]=0`,
   :math:`\text{Var}[\mathcal{R}_i(Y)]=\Sigma-\Gamma`, and
   :math:`\text{Cov}[\mathcal{R}_i(Y),\mathcal{R}_j(Y)]=0`, :math:`\forall
   i\neq j` - The residual component is individual to each :math:`Y_i` and
   has mean 0, constant variance, and zero correlation across the
   :math:`Y_i` -- ie the residuals are themselves second-order
   exchangeable.
#. :math:`\text{Cov}[\mathcal{M}(Y),\mathcal{R}_i(Y)]=0` - The mean and
   residual components are uncorrelated.

Note here that :math:`\mu`, :math:`\Sigma`, and :math:`\Gamma` are just belief
specifications - these quantities are known and fixed - however
:math:`\mathcal{M}[Y]` and the :math:`\mathcal{R}_j(Y)`s are uncertain and we
consider learning about :math:`\mathcal{M}(Y)` and the variance of the
:math:`\mathcal{R}_j(Y)`s.

Adjusting beliefs about the mean of a collection of exchangeable random vectors
-------------------------------------------------------------------------------

We now describe how we can use a sample of exchangeable data to adjust
our beliefs about the underlying population quantities. Continuing with
the notation of previous sections, our goal is to consider how our
beliefs about the population mean :math:`\mathcal{M}(Y)` are adjusted when
we observe a sample of :math:`n` values of our exchangeable random
quantities, :math:`D_n=(Y_1,\dots,Y_n)`.

It can be shown that when adjusting beliefs about :math:`\mathcal{M}(Y)` by
the entire sample :math:`D_n`, the adjustment of our beliefs by the sample
mean

:math:`\bar{Y}_n=\frac{1}{n}\sum_{i=1}^nY_i`

is sufficient - see Section 6.10 of the Goldstein & Wooff for details.
Since the :math:`Y_i` are exchangeable, we can express the sample mean,
from a sample of size :math:`n`, as

:math:` \\bar{Y}_n=\mathcal{M}(Y)+\bar{\mathcal{R}}_n(Y), \`

where :math:`\bar{\mathcal{R}}_n(Y)` is the mean of the :math:`n` residuals,
and so

-  :math:` \\text{E}[\bar{Y}_n]=\mu, \`
-  :math:` \\text{Var}[\bar{Y}_n]=\Gamma+\frac{1}{n}(\Sigma-\Gamma), \`
-  :math:` \\text{Cov}[\bar{Y}_n,\mathcal{M}(Y)]=\Gamma. \`

We can then apply the Bayes linear adjustment formulae directly to find
our adjusted beliefs about the population mean, :math:`\mathcal{M}(Y)`,
given the mean of a sample of size :math:`n` as

:math:` \\text{E}_n[\mathcal{M}(Y)] = \\mu +
\\Gamma\left(\Gamma+\frac{1}{n}(\Sigma-\Gamma)\right)^{-1}(\bar{Y}_n-\mu),
\`

with corresponding adjusted variance

:math:`
\\text{Var}_n[\mathcal{M}(Y)]=\Gamma-\Gamma\left(\Gamma+\frac{1}{n}(\Sigma-\Gamma)\right)^{-1}\Gamma.
\`

Adjusting beliefs about the variance of a collection of exchangeable random vectors
-----------------------------------------------------------------------------------

We can apply the same methodology to learn about the population variance
of a collection of exchangeable random quantities. In order to do so, we
must make one additional assumption (namely that the
:math:`\mathcal{R}_i(Y)^2` are SOE in addition to :math:`\mathcal{R}_i(Y)`),
and we require specifications of our uncertainty about the variances
expressed via fourth-order moments. Methods for choosing appropriate
prior specifications are discussed in the next section.

Suppose that :math:`Y_1,Y_2,\dots` is an infinite exchangeable sequence of
scalars as above, where :math:`\text{E}[Y_i]=\mu`,
:math:`\text{Var}[Y_i]=\sigma^2`, and :math:`\text{Cov}[Y_i,Y_j]=\gamma`. Then
we have the standard second-order exchangeability representation

:math:` Y_i=\mathcal{M}(Y)+\mathcal{R}_i(Y), \` where the
:math:`\mathcal{R}_1(Y),\mathcal{R}_2(Y),\dots` are SOE with variance
:math:`\text{Var}[\mathcal{R}_i(Y)]=\sigma^2-\gamma`. How we proceed from
here depends on whether we can consider the population mean
:math:`\mathcal{M}(Y)` to be known or unknown.

:math:`\mathcal{M}(Y)` known
~~~~~~~~~~~~~~~~~~~~~~~~~

In the case where the population mean, :math:`\mathcal{M}(Y)`, is known
then there is no uncertainty surrounding the value of :math:`\mu`. In which
case we can consider :math:`\text{Var}[\mathcal{M}(Y)]=\gamma=0`.

To learn about population variance using this methodology, we require an
appropriate exchangeability representation for an appropriate quantity.
Consider the :math:`V_i=[\mathcal{R}_i(Y)]^2=(Y_i-\mathcal{M}(Y))^2` which
are directly observable when :math:`\mathcal{M}(Y)` is known. If we assume
that this sequence of squared residuals :math:`V_1, V_2, \\dots` is also
SOE, then we have the representation

:math:`V_i=[\mathcal{R}_i(Y)]^2=\mathcal{M}(V)+\mathcal{R}_i(V),`

where, as before, :math:`\mathcal{M}(V)` is the population mean of the
:math:`V_i=(Y_i-\mathcal{M}(Y))^2` and is hence the **population variance**
of the :math:`Y_i`. To learn about :math:`\mathcal{M}(V)` (and hence to learn
about the population variance of the :math:`Y_i`) we require the following
belief specifications:

-  :math:`\text{E}[\mathcal{M}(V)] = \\omega_Y=\sigma^2` -- our expectation
   of the population variance of the :math:`Y_i`,
-  :math:`\text{Var}[\mathcal{M}(V)]=\omega_\mathcal{M}` -- our uncertainty
   associated with the population variance of :math:`Y_i`, which can be
   resolved by observation of additional data,
-  :math:`\text{Var}[\mathcal{R}_i(V)]=\omega_\mathcal{R}` -- irresolvable
   "residual uncertainty" in the :math:`V_i`,

and that the sequence of :math:`\mathcal{R}_i(V)` are uncorrelated with
zero mean. We can then use this representation and belief specification
to apply the adjustment described above to revise our beliefs about the
population variance :math:`\mathcal{M}(V)`. As the sample mean is
sufficient for the adjustment of beliefs in these situations and is
directly observable, we calculate and adjust by the sample mean of the
:math:`V_i`

:math:`\bar{V}_n=\bar{Y}_n^{(2)} = \\sum_{i=1}^n (Y_i-\mathcal{M}(Y))^2.`

We then evaluate our adjusted expectation and variance of the population
variance to be

:math:`\text{E}_{\bar{V}_n}[\mathcal{M}(V)] =
\\frac{\omega_\mathcal{M}\bar{V}_n +
\\frac{1}{n}\omega_\mathcal{R}\omega_Y}{\omega_\mathcal{M}+\frac{1}{n}\omega_\mathcal{R}},`

:math:`\text{Var}_{\bar{V}_n}[\mathcal{M}(V)] =
\\frac{\frac{1}{n}\omega_\mathcal{M}\omega_\mathcal{R}}{\omega_\mathcal{M}+\frac{1}{n}\omega_\mathcal{R}},`

where :math:`\text{E}_{\bar{V}_n}[\mathcal{M}(V)]` represents our adjusted
beliefs about the population variance of the :math:`Y_i`, and
:math:`\text{Var}_{\bar{V}_n}[\mathcal{M}(V)]` represents our remaining
uncertainty.

:math:`\mathcal{M}(Y)` unknown
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we have an identical setup as before, only now the population
mean of the :math:`Y_i`, :math:`\mathcal{M}(Y)`, is no longer known -- ie
:math:`\text{Var}[\mathcal{M}(Y)]=\gamma > 0`. Additionally, since the
population mean is now uncertain the quantities
:math:`V_i=(Y_i-\mathcal{M}(Y))^2` are no longer directly observable so we
can no longer calculate and adjust by :math:`\bar{V}_n`. Therefore, we must
construct an alternative adjustment using appropriate combinations of
observables which will be informative for the population variance.
Suppose that we take a sample of size :math:`n\geq 2`, and that we
calculate the sample variance, :math:`s^2` in the usual way. We then obtain
the following representation (see section 8.2 of Goldstein & Wooff for
details):

:math:`s^2=\mathcal{M}(V)+ T.`

We can then express our beliefs about :math:`s^2` as:

-  :math:`\text{E}[s^2]=\omega_Y`,
-  :math:`\text{Var}[s^2]=\omega_\mathcal{M}+\omega_T`,
-  :math:`\text{Cov}[s^2,\mathcal{M}(V)]=\omega_\mathcal{M}`,
-  and
   :math:`\text{Var}[T]=\omega_T=\frac{1}{n}\omega_\mathcal{R}+\frac{2}{n(n-1)}[\omega_\mathcal{M}+\omega_Y^2].`

Thus the sample variance, :math:`s^2`, can be related directly to the
population variance, :math:`\mathcal{M}(V)`, and so we can (in principle)
use the directly-observable :math:`s^2` to learn about the population
variance. Before we can make the adjustment, we must make some
additional assumptions that the residuals have certain fourth-order
uncorrelated properties - namely that for :math:`k\neq j\neq i`,
:math:`\mathcal{R}_j(Y)\mathcal{R}_k(Y)` is uncorrelated with
:math:`\mathcal{M}(Y)` and :math:`\mathcal{R}_i(Y)`, and that for :math:`k > j,w >
u` that :math:`\mathcal{R}_k(Y)\mathcal{R}_j(Y)` and
:math:`\mathcal{R}_e(Y)\mathcal{R}_u(Y)` are also uncorrelated. With these
assumptions and specifications, the adjusted expectation and variance of
the population variance given :math:`s^2` are given by

:math:` \\text{E}_{s^2}[\mathcal{M}(V)] =
\\frac{\omega_\mathcal{M}s^2+\omega_T\omega_Y}{\omega_\mathcal{M}+\omega_T},
\`

:math:` \\text{Var}_{s^2}[\mathcal{M}(V)] =
\\frac{\omega_\mathcal{M}\omega_T}{\omega_\mathcal{M}+\omega_T} \`

Choice of prior values
----------------------

For the first- and second-order quantities :math:`\mu`, :math:`\Sigma` and
:math:`\Gamma` in the case of the population mean update, and :math:`\omega_Y`
in the case of the population variance update we suggest relying on
standard belief specification or elicitation techniques. These
quantities are simply means, variances and covariances of observable
quantities so obtaining appropriate prior values via direct
specification or numerical investigation of related problems both
provide relevant methods of assessment.

The quantities which are most challenging to specify beliefs about are
the fourth-order quantities
:math:`\omega_\mathcal{M}=\text{Var}[\mathcal{M}(V)]`, and
:math:`\omega_\mathcal{R}=\text{Var}[\mathcal{R}(V)]` required by the
population variance update. Direct assessment or specification of these
quantities is challenging, so we briefly describe an heuristic method
for making such belief assessments.

Comparison to known distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us first consider
:math:`\omega_\mathcal{R}=\text{Var}[\mathcal{R}(V)]`. This quantity
represents our judgements about the shape of the distribution of the
:math:`Y_i`. One method of assessment is therefore to relate
:math:`\omega_\mathcal{R}` to the **kurtosis** of that distribution, and
then specify an appropriate value by comparison to the shape of other
known distributions.

Suppose that we consider that the population variance acts as a scale
parameter for the residuals :math:`\mathcal{R}_i(Y)` such that
:math:`\mathcal{R}_i(Y)=\sqrt{\mathcal{M}(V)}Z_i`, where the :math:`Z_i` are
independent standardised quantities with zero mean and unit variance
which are independent of the value of :math:`\mathcal{M}(V)`. Then
:math:`\mathcal{R}_i(V)=\mathcal{M}(V)(Z_i^2-1)`, and so :math:`
\\omega_\mathcal{R} = \\text{Var}[\mathcal{R}_i(V)] =
(\omega_\mathcal{M} + \\omega_Y^2)\text{Var}[Z_i^2] =
(\omega_\mathcal{M} + \\omega_Y^2)(\kappa-1), \` where
:math:`\kappa=\text{Kur}(Z_i)` is the kurtosis of :math:`Z_i`. If we believe
that the :math:`Z_i` were Gaussian in distribution, this would suggest a
value of :math:`\kappa=3`. Similarly, a Uniform distribution suggests
:math:`\kappa=1.8` and a t-distribution with unit variance and :math:`\nu`
degrees of freedom gives :math:`\kappa=3(\nu-1)/(\nu-4)`. In general,
higher values for :math:`\kappa` (and hence :math:`\omega_\mathcal{R}`),
increase the proportion of variance in :math:`\mathcal{M}(V)` which cannot
be resolved by observing data and so diminish the weight of the
observations in update formula.

Proportion of variance resolved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a choice for :math:`\omega_\mathcal{R}`, we now must determine an
appropriate value for :math:`\omega_\mathcal{M}` to complete our belief
specification. We briefly discuss two possible methods, the first
arising from considerations of the effectiveness of the update. Let us
write :math:`\omega_\mathcal{M}=c \\omega_Y^2 \` for some :math:` c > 0 \`,
then the problem reduces to selection of a value of :math:`c \`. We can try
to assess :math:` c \` by considering the proportion of uncertainty
remaining in the population variance after the update, which is given by

:math:`\frac{\text{Var}_{s^2}[\mathcal{M}(V)]}{\text{Var}[\mathcal{M}(V)]}=\frac{1}{1+\frac{n-1}{\phi}\frac{c}{c+1}}`

which decreases monotonically as a function of :math:`c`, and where we
define :math:`\phi` as

:math:`\phi=\frac{1}{n}\{(n-1)\text{Var}[Z_i^2]+2\} \`

which is fixed given :math:` n \` and :math:`\kappa \`. We can then explore
our attitudes to the implications of differing sample sizes; for
example, small values of :math:` c \` would suggest that any sample
information will rapidly reduce our remaining variance as a proportion
of the prior.

Equivalent sample size
~~~~~~~~~~~~~~~~~~~~~~

An alternative approach for specifying :math:`\omega_\mathcal{M} \` is to
make a direct judgement on the worth of the prior information via the
notion of equivalent sample size. We can express the adjusted
expectation of the population variance as

:math:`\text{E}_{s^2}[\mathcal{M}(V)]=\alpha s^2 + (1-\alpha)
\\text{E}[\mathcal{M}(V)] \`,

with

:math:`\alpha=\frac{\omega_\mathcal{M}}{\omega_\mathcal{M}+\omega_T}. \`

Suppose that we consider that the prior information we have about the
population variance is worth a notional sample size of :math:` m \`, and
that we collect observations of a sample of size :math:` n \`. In which
case, it would be reasonable to adjust our beliefs with a weighting
given by

:math:`\alpha=\frac{n}{n+m}. \`

By examining our beliefs about the relative merits of the prior and
sample information we can make an assessment for an appropriate value of
:math:`\omega_\mathcal{M} \`. In fact, this method of specification is
equivalent to the alternative method discussed above.

References
----------

\* Goldstein, M. and Wooff, D. A. (2007), Bayes Linear Statistics:
Theory and Methods, Wiley.
