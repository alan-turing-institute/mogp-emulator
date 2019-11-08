.. _AltScreeningChoice:

Alternatives: Deciding which screening method to use
====================================================

We first provide a classification of :ref:`screening<DefScreening>`
methods, followed by a discussion under what settings each screening
method may be appropriate.

Classification of screening methods
-----------------------------------

Screening methods have been broadly categorised in the following
categories:

#. **Screening Design** methods. An experimental
   :ref:`design<DefDesign>` is constructed with the express aim of
   identifying :ref:`active<DefActiveInput>` factors. This approach
   is the classical statistical method, and is typically associated with
   the Morris method (see the procedure page on the Morris screening
   method (:ref:`ProcMorris<ProcMorris>`) which defines a design
   optimised to infer the global importance of
   :ref:`simulator<DefSimulator>` inputs to the simulator output)
#. **Ranking** methods. Input variables are ranked according to some
   measure of association between the simulator inputs and outputs.
   Typical measures considered are correlation, or partial correlation
   coefficients between simulator inputs and the simulator output. Other
   non-linear measures of association are possible, but these methods
   tend not to be widely used.
#. **Wrapper** methods. An emulator is used to assess the predictive
   power of subsets of variables. Wrapper methods can use a variety of
   search strategies:

   #. Forward selection where variables are progressively incorporated
      in larger and larger subsets.
   #. Backward elimination where variables are sequentially deleted from
      the set of active inputs, according to some scoring method, where
      the score is typically the root mean square prediction error of
      the simulator output (or some modification of this such as the
      Bayesian information criterion).
   #. Efroymsons algorithm also known as stepwise selection, proceeds as
      forward selection but after each variable is added, the algorithm
      checks if any of the selected variables can be deleted without
      significantly affecting the Residual Sum of Squares (RSS).
   #. Exhaustive search where all possible subsets are considered.
   #. Branch and Bound strategies eliminates subset choices as early as
      possible by assuming the performance criterion is monotonic, i.e.
      the score improves as more variables are added. For a discussion
      of the algorithm see its procedure page
      :ref:`ProcBranchAndBoundAlgorithm<ProcBranchAndBoundAlgorithm>`.

#. **Embedded** methods. For both variable ranking and wrapper methods,
   the emulator is considered a perfect black box. In embedded methods,
   the variable selection is integrated as part of the training of the
   emulator, although this might proceed in a sequential manner, to
   allow some benefits of the reduction in input variables considered.

In this thread we focus on methods most appropriate for computer
experiments that are the most general, i.e. the assumptions made are not
overly restrictive to a particular class of models.

Decision process
----------------

If the simulator is available and deterministic a screening design
approach, the Morris method (see :ref:`ProcMorris<ProcMorris>`), is
most appropriate where a one factor at a time (OAT) design is used to
identify active inputs. The Morris method is a very simple process,
which can be understood as the construction of a design to estimate the
expected value and variance (over the input space) of the partial
derivatives of the simulator output with respect the simulator inputs.
The method creates efficient designs to estimate these, thus to use the
method it will be necessary to evaluate the simulator over the Morris
design and the method cannot be reliably applied to data from other
designs. The restriction to deterministic simulators (where the output
is identical for repeated evaluations at the same inputs) is because
partial derivatives are likely to be rather sensitive to noise on the
simulator output, although this will depend to some degree on the signal
to noise ratio in the outputs.

If the simulator is not easily evaluated (maybe simply because we don't
have direct access to the code), or the training design has already been
created, then design based approaches to emulation are not possible and
the alternative methods described above must be considered. Also if the
simulator output has a random component for a fixed input, then the
below methods can be readily applied.

If the critical features of the simulator output can be captured by a
small set of fixed basis functions (often simply linear or low order
polynomials) then a regression (wrapper) analysis can be used to
identify the active inputs. One such procedure is described in the
procedure page for the empirical construction of a BL emulator
(:ref:`ProcBuildCoreBLEmpirical <ProcBuildCoreBLEmpirical>`)
and is an example of a wrapper method. Many other methods for variable
selection are possible and can be found in textbooks on statistics and
in papers on stepwise variable selection for regression models.

An alternative to the wrapper methods above is to employ an embedded
method, Automatic Relevance Determination (ARD), which is described in
the procedure page
(:ref:`ProcAutomaticRelevanceDetermination<ProcAutomaticRelevanceDetermination>`).
Automatic relevance determination essentially uses the estimates of the
input variable length scale hyperparameters in the emulator covariance
function to assess the relevance of each input to the overall emulator
model. The method has the advantage that the relatively flexible
Gaussian process model is employed to estimate the impact of each input,
as opposed to a linear in parameters regression model, but the cost is
increased computational complexity.

References
----------

Guyon, I. and A. Elisseeff (2003). *An introduction to variable and
feature selection*,
`http://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf <http://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf>`_.
Journal of Machine Learning Research 3, 1157 - 1182.
