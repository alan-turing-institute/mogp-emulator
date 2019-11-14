.. _DefActiveInput:

Definition of Term: Active input
================================

A :ref:`simulator<DefSimulator>` will typically have many inputs. In
general, the larger the number of inputs the more complex it becomes to
build a good :ref:`emulator<DefEmulator>`, and the larger the number
of simulator runs required for the :ref:`training
sample<DefTrainingSample>`. There is therefore motivation for
reducing the number of inputs.

Fortunately, it is also generally found that not all inputs affect the
output(s) of interest appreciably over the required range of input
variation. The number of inputs having appreciable impact on outputs for
the purpose of the analysis is often small. The term 'active input'
loosely means one which does have such an impact. More precisely, the
active inputs are those that are deemed to be inputs when building the
emulator.
