.. _DefRegularity:

Definition of Term: Regularity
==============================

The output of a :ref:`simulator<DefSimulator>` is regarded as a
function of its inputs. One consideration in building an
:ref:`emulator<DefEmulator>` for the simulator is whether the output
is expected to be a continuous function of its inputs, and if so whether
it is differentiable. Within the :ref:`MUCM<DefMUCM>` toolkit, we
refer to the continuity and differentiability properties of a simulator
as regularity.

The minimal level of regularity would be a simulator whose output is
everywhere a continuous function of the inputs, but is not always
differentiable. A function which is differentiable everywhere has
greater degree of regularity the more times it is differentiable. The
most regular of functions are those that are differentiable infinitely
many times.
