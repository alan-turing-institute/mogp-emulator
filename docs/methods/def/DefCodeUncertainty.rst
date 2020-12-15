.. _DefCodeUncertainty:

Definition of Term: Code uncertainty
====================================

An :ref:`emulator<DefEmulator>` is a probabilistic representation of
the output(s) of a :ref:`simulator<DefSimulator>`. For any given
input configuration the output is uncertain (unless this input
configuration was included in the :ref:`training
sample<DefTrainingSample>`). This uncertainty is known as code
uncertainty. The emulator defines exactly what the code uncertainty is
about the output(s) from any given configuration(s) of inputs.

Because the output(s) are uncertain, we are also uncertain about
properties of those outputs, such as the uncertainty mean in an
:ref:`uncertainty analysis<DefUncertaintyAnalysis>`. The code
uncertainty expressed in the emulator then implies code uncertainty
about the properties of interest. Statements that we can make about
those properties are therefore *inferences*. For instance, we may
estimate a property by its mean, which is evaluated with respect to its
code uncertainty distribution.
