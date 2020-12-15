.. _DefSmoothness:

Definition of Term: Smoothness
==============================

The output of a :ref:`simulator<DefSimulator>` is regarded as a
function of its inputs. One consideration in building an
:ref:`emulator<DefEmulator>` for the simulator is whether the output
is expected to be a smooth function of the inputs.

Smoothness might be measured formally in a variety of ways. For
instance, we might refer to the expected number of times that the output
crosses zero in a given range, or the expected number of local maxima or
minima it will have in that range (which for a differentiable output
corresponds to the number of times its derivative crosses zero). In the
toolkit we do not adopt any specific measure of smoothness, but the
concept is an important one.
