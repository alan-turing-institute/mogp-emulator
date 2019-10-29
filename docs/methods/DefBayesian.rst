.. _DefBayesian:

Definition of Term: Bayesian
============================

The adjective 'Bayesian' refers to ideas and methods arising in the
field of Bayesian Statistics.

Bayesian statistics is an approach to constructing statistical
inferences that is fundamentally and philosophically different from the
approach that is more commonly taught, known as frequentist statistics.

In Bayesian statistics, all inferences are probability statements about
the true, but unknown, values of the parameters of interest. The formal
interpretation of those probability statements is as the personal
beliefs of the person providing them.

The fact that Bayesian inferences are essentially personal judgements
has been the basis of heated debate between proponents of the Bayesian
approach and those who espouse the frequentist approach. Frequentists
claim that Bayesian methods are subjective and that subjectivity should
have no role in science. Bayesians counter that frequentist inferences
are not truly objective, and that the practical methods of Bayesian
statistics are designed to minimise the undesirable aspects of
subjectivity (such as prejudice).

As in any scientific debate, the arguments and counter-arguments are
many and complex, and it is certainly not the intention of this brief
definition to go into any of that detail. In the context of the MUCM
toolkit, the essence of an :ref:`emulator<DefEmulator>` is that it
makes probability statements about the outputs of a
:ref:`simulator<DefSimulator>`, and the frequentist approach does not
formally allow such statements. Therefore emulators are necessarily
Bayesian.

In the :ref:`MUCM<DefMUCM>` field, there is a recognised alternative
to the fully Bayesian approach of characterising uncertainty about
unknown parameters by complete probability distributions. This is the
:ref:`Bayes linear<DefBayesLinear>` approach, in which uncertainty is
represented only through means, variances and covariances.
