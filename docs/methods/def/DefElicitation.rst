.. _DefElicitation:

Definition of Term: Elicitation
===============================

Probability distributions are needed to express uncertainty about
unknown quantities in various contexts. In the :ref:`MUCM<DefMUCM>`
toolkit, there are two principal areas where such probability
distributions may be required. One is to specify prior knowledge about
:ref:`hyperparameters<DefHyperparameter>` associated with building a
:ref:`Gaussian process<DefGP>` :ref:`emulator<DefEmulator>`, while
the other is to formulate uncertainty about inputs to the
:ref:`simulator<DefSimulator>` (for instance, for the purposes of
:ref:`uncertainty<DefUncertaintyAnalysis>` or
:ref:`sensitivity<DefSensitivityAnalysis>` analyses). These
probability distributions may be created in various ways, but the most
basic is to represent the knowledge/beliefs of a relevant expert.

The process of formulating an expert's knowledge about an uncertain
quantity as a probability distribution is called elicitation.
