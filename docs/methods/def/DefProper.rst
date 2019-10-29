.. _DefProper:

Definition of Term: Proper (or improper) distribution
=====================================================

A proper distribution is one that integrates (in the case of a
probability density function for a continuous random variable) or sums
(in the case of a probability mass function for a discrete random
variable) to unity. According to probability theory, all probability
distributions must have this property.

The concept is relevant in the context of so-called
:ref:`weak<DefWeakPrior>` prior distributions which are claimed to
represent (or approximate) a state of ignorance. Such distributions are
often improper in the following sense.

Consider a random variable that can take any positive value. One weak
prior distribution for such a random variable is the uniform
distribution that assigns equal density to all positive values. Such a
density function would be given by

:math:`\pi(x) = k`

for some positive constant :math:`k` and for all positive values :math:`x` of
the random variable. However, there is no value of :math:`k` for which this
is a proper distribution. If :math:`k` is not zero, then the density
function integrates to infinity, while if :math:`k=0` it integrates to
zero. For no value of :math:`k` does it integrate to one. So such a uniform
distribution simply does not exist.

Nevertheless, analysis can often proceed as if this distribution were
genuinely proper for some :math:`k`. That is, by using it as a prior
distribution and combining it with the evidence in the data using Bayes'
theorem, we will usually obtain a posterior distribution that is proper.
The fact that the prior distribution is improper is then unimportant; we
regard the resulting posterior distribution as a good approximation to
the posterior distribution that we would have obtained from any prior
distribution that was actually proper but represented very great prior
uncertainty. This use of weak prior distributions is discussed in the
definition of the weak prior distribution
(:ref:`DefWeakPrior<DefWeakPrior>`), and is legitimate provided that
the supposed approximation really applies. One situation in which it
would not is when the improper prior leads to an improper posterior.

Formally, we write the uniform prior as

:math:`\pi(x) \\propto 1\,,`

using the proportionality symbol to indicate the presence of an
unspecified scaling constant :math:`k` (despite the fact that no such
constant can exist). Other improper distributions are specified in like
fashion as being proportional to some function, implying a scaling
constant which nevertheless cannot exist. An example is another commonly
used weak prior distribution for a positive random variable,

:math:`\pi(x) \\propto x^{-1}\,,`

known as the log-uniform prior.
