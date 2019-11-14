.. _MetaHomePage:

The MUCM Toolkit, release 6
===========================

Welcome to the :ref:`MUCM<DefMUCM>` Toolkit. The toolkit is a
resource for people interested in quantifying and managing uncertainty
in the outputs of mathematical models of complex real-world processes.
We refer to such a model as a simulation model or a
:ref:`simulator<DefSimulator>`.

The toolkit is a large, interconnected set of webpages and one way to
use it is just to browse more or less randomly through it. However, we
have also provided some organised starting points and threads through
the toolkit.

-  We have an introductory tutorial on MUCM methods and uncertainty in
   simulator outputs :ref:`here<MetaToolkitTutorial>`.

-  You can read about the :ref:`toolkit
   structure<MetaToolkitStructure>`.

-  The various threads, each of which sets out in a series of steps how
   to use the MUCM approach to build an :ref:`emulator<DefEmulator>`
   of a simulator and to use it to address some standard problems faced
   by modellers and users of simulation models. This release contains
   the following threads:

   -  :ref:`ThreadCoreGP<ThreadCoreGP>`, which deals with the
      simplest emulation scenario, called the :ref:`core
      problem<DiscCore>`, using the :ref:`Gaussian
      process<DefGP>` approach;
   -  :ref:`ThreadCoreBL<ThreadCoreBL>`, which also deals with the
      core problem, but follows the :ref:`Bayes
      linear<DefBayesLinear>` approach. A simple guide to the
      differences between the two approaches can be found in the
      alternatives page on Gaussian process or Bayes Linear Emulator
      (:ref:`AltGPorBLEmulator<AltGPorBLEmulator>`);
   -  :ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`,
      which extends the core problem to address the case where we wish
      to emulate two or more simulator outputs;
   -  :ref:`ThreadVariantDynamic<ThreadVariantDynamic>`, which
      extends the core analysis in a different direction, where we wish
      to emulate the time series output of a dynamic simulator;
   -  :ref:`ThreadVariantTwoLevelEmulation<ThreadVariantTwoLevelEmulation>`,
      which considers the situation where we have two simulators of the
      same real-world phenomenon, a slow but relatively accurate
      simulator whose output we wish to emulate, and a quick and
      relatively crude simulator. This thread discusses how to use many
      runs of the fast simulator to build an informative prior model for
      the slow simulator, so that fewer training runs of the slow
      simulator are needed;
   -  :ref:`ThreadVariantWithDerivatives<ThreadVariantWithDerivatives>`,
      which extends the core analysis for the case where we can obtain
      derivatives of the simulator output with respect to the various
      inputs, to use as training data;
   -  :ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`,
      which deals with modelling the relationship between the simulator
      outputs and the real-world process being simulated. Recognising
      this :ref:`model discrepancy<DefModelDiscrepancy>` is a crucial
      step in making useful predictions from simulators, in calibrating
      simulation models and handling multiple models.
   -  :ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`,
      which deals with combining two or more emulators to produce
      emulation of some combination of the respective simulator outputs;
   -  :ref:`ThreadGenericEmulateDerivatives<ThreadGenericEmulateDerivatives>`,
      which shows how to use an emulator to predict the values of
      derivatives of the simulator output;
   -  :ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`,
      which deals with iteratively narrowing down the region of possible
      input values for which the simulator would produce outputs
      acceptably close to observed data. This topic is related to
      calibration, which will be addressed in a future release of the
      toolkit.
   -  :ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`,
      which is a topic thread providing more detailed background on the
      topic of sensitivity analysis, and linking together the various
      procedures for such techniques in the other toolkit threads.
   -  :ref:`ThreadTopicScreening<ThreadTopicScreening>`, which
      provides a broad view of the idea of
      :ref:`screening<DefScreening>` the simulator inputs to reduce
      their dimensionality.
   -  :ref:`ThreadTopicExperimentalDesign<ThreadTopicExperimentalDesign>`,
      which gives a detailed overview of the methods of experimental
      design that are relevant to MUCM, particularly those relating to
      the design of a training sample.

-  Another important feature of the toolkit is the MUCM Case Studies.
   The Case Studies are demonstrations of the MUCM methodology applied
   to address substantive challenges faced by users of real simulation
   models. The techniques that they use are all described in the toolkit
   and there are appropriate links from each Case Study to the relevant
   pages in the toolkit. The Case Studies generally are accessed from
   the page :ref:`MetaCaseStudies<MetaCaseStudies>`, and from the
   menu bar. Currently, this page links just to the first case study,
   but the intention is to add at least two more case studies in
   subsequent releases.

Later releases of the toolkit will add more threads and other material,
including more extensive examples to guide the toolkit user and further
Case Studies. In each release we also add more detail to some of the
existing threads; for instance, in this release we have a substantial
reworking of the variant thread on emulating multiple outputs, and also
new material on variance learning for Bayes Linear emulation.
