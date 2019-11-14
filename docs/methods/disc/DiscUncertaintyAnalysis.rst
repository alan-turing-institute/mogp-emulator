.. _DiscUncertaintyAnalysis:

Discussion: Uncertainty analysis
================================

In uncertainty analysis, we wish to quantify uncertainty about simulator
outputs due to uncertainty about simulator inputs. We define :math:` X \`
to be the uncertain true inputs, and :math:` f(X) \` to be the
corresponding simulator output(s). In the emulator framework, :math:` f(.)
\` is also treated as an uncertain function, and it is important to
consider both uncertainty in :math:` X \` and uncertainty in :math:` f(.) \`
when investigating uncertainty about :math:` f(X) \`. In particular, it is
important to distinguish between the unconditional distribution of :math:`
f(X) \`, and the distribution of :math:` f(X) \` conditional on :math:` f(.)
\`. For example:

#. :math:` \\textrm{E}[f(X)] \` is the expected value of :math:` f(X) \`,
   where the expectation is taken with respect to both :math:` f(.) \` and
   :math:` X`. The value of this expectation can, in principle, be obtained
   for any emulator and input distribution.
#. :math:` \\textrm{E}[f(X)|f(.)] \` is the expected value of :math:` f(X) \`,
   where the expectation is taken with respect to :math:` X \` only as :math:`
   f(.)` is given. If :math:` f(.)` is a computationally cheap function,
   we could, for example, obtain the value of this expectation using
   Monte Carlo, up to an arbitrary level of precision. However, when :math:`
   f(.)` is computationally expensive such that we require an emulator
   for :math:` f(.)`, **this expectation is an uncertain quantity**. We are
   uncertain about the value of :math:` \\textrm{E}[f(X)|f(.)] \`, because
   we are are uncertain about :math:` f(.) \`.

There is no sense in which :math:` \\textrm{E}[f(X)] \` can be 'wrong': it
is simply a probability statement resulting from a choice of emulator
(good or bad) and input distribution. But an estimate of :math:`
\\textrm{E}[f(X)|f(.)] \:ref:` obtained using an emulator can be poor if we
have a poor emulator (in the `validation<DefValidation>` sense)
for :math:`f(.)`. Alternatively, we may be very uncertain about :math:`
\\textrm{E}[f(X)|f(.)] \` if we don't have sufficient training data for
the emulator of :math:`f(.)`. Hence in practice, the distinction is
important for considering *whether we have enough simulator runs for our
analysis of interest*.
