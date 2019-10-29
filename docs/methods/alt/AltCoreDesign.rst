.. _AltCoreDesign:

Alternatives: Training Sample Design for the Core Problem
=========================================================

Overview
--------

An important step in the development of an
:ref:`emulator<DefEmulator>` is the creation of a :ref:`training
sample<DefTrainingSample>` of runs of the
:ref:`simulator<DefSimulator>`. This entails running the simulator at
each of a set of input points in order to see what output(s) the
simulator produces at each of those input configurations. The set of
input points is called the :ref:`design<DefDesign>` for the training
sample.

The design of training samples for building emulators falls within the
wide field of statistical experimental design. The topic thread on
experimental design
(:ref:`ThreadTopicExperimentalDesign<ThreadTopicExperimentalDesign>`)
contains much more detail and discussion about design issues in
:ref:`MUCM<DefMUCM>`, and on their relationship to conventional
statistical experimental designs. Even within the realm of the MUCM
toolkit there are many contexts in which designs are needed, each with
its own criteria and yielding different kinds of solutions. We consider
here the choice of design for a training sample to build an emulator for
the "core problem." The core problem is discussed in page
:ref:`DiscCore<DiscCore>`, and this page falls within the two core
threads, the thread for the analysis of the core model using Gaussian
process methods (:ref:`ThreadCoreGP<ThreadCoreGP>`) and the thread
for the Bayes linear emulation for the core model
(:ref:`ThreadCoreBL<ThreadCoreBL>`) - see the discussion of threads
in the toolkit structure page
(:ref:`MetaToolkitStructure<MetaToolkitStructure>`).

Choosing the Alternatives
-------------------------

| The training sample design needs to facilitate the development of an
  emulator that correctly and accurately predicts the actual simulator
  output over a desired region of the input space. By convention, we
  suppose that the input region of interest, :math:`\cal{X}`, is the unit
  cube in :math:`p` dimensions, :math:`[0,1]^p`. We assume that this
  has been achieved by transforming each of the :math:`p` simulator
  inputs individually by a simple linear transformation, and that the
  :math:`i`-th input corresponds to the :math:`i`-th dimension of
  :math:`\cal{X}`. (More
  extended discussion on technical issues in training sample design for
  the core problem can be found in page
  :ref:`DiscCoreDesign<DiscCoreDesign>`.)
| There are several criteria for a good design, which are also discussed
  in :ref:`DiscCoreDesign<DiscCoreDesign>`, but the most widely
  adopted approach in practice is to employ a design in which the points
  are spread more or less evenly and as far apart as possible within
  :math:`\cal{X}`. Such a design is called-space-filling. The detailed
  alternatives that are discussed below all have space-filling
  properties.

The Nature of the Alternatives
------------------------------

Factorial design
~~~~~~~~~~~~~~~~

Factorial design over :math:`\cal X` defines a set of points :math:`L_j` in
:math:`[0,1]` called levels for dimension :math:`j=1,2,\ldots,p`. Then if
the number of points in :math:`L_j` is
:math:`n_j` the full factorial design takes all :math:`n=\prod_{j=1}^p n_j`
combinations of the levels to produce a :math:`p`-dimensional grid
of points in :math:`\cal {X}`. If the levels are spread out over :math:`[0,1]`
in each dimension (so as to have a space-filling property in each
dimension), then the full factorial is a space-filling design. However,
if :math:`p` is reasonably large then even when the number of levels
takes the minimal value of two in each dimension the size :math:`n=2^p` of
the design becomes prohibitively large for practical simulators.

Fractional factorial designs consist of subsets of the full factorial
that generally have some pattern or symmetry properties, but although
some such designs can be realistically small it is usual to adopt
alternative ways to construct space-filling designs.

Optimised Latin hypercube designs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Latin hypercube (LHC) is a set of points chosen randomly subject to a
constraint that ensures that the values of each input separately are
spread evenly across :math:`[0,1]`. The procedure for constructing a LHC is
given in :ref:`ProcLHC<ProcLHC>`. LHCs are not guaranteed to be
space-filling in :math:`\cal{X}`, just in each dimension separately. It
is therefore usual to generate a large number of random LHCs and to
select from these the one that best satisfies a specified criterion.

One popular criterion is the minimum distance between any two points in
the design. Choosing the LHC with the maximal value of this criterion
helps to ensure that the design is well spread out over :math:`\cal{X}`,
and a LHC optimised according to this criterion is known as a maximin
LHC design. This and other criteria are discussed in
:ref:`DiscCoreDesign<DiscCoreDesign>`.

The procedure for generating an optimised LHC, according to any desired
criterion and in particular according to the maximin criterion, is given
in the procedure for generating an optimised Latin hypercube design
(:ref:`ProcOptimalLHC<ProcOptimalLHC>`).

Non-random space-filling design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A number of different sequences of numbers have been proposed that have
space-filling properties. The can be thought of as pseudo-random
sequences. The sequences use different algorithms to generate them, but
all have the property that they are potentially infinite in length, and
a design of :math:`n` points is obtained simply by taking the first
:math:`n` points in the sequence.

-  Lattice designs. A lattice is a special grid of :math:`n` points
   in :math:`[0,1]^d`. It is defined by :math:`d` generators, and each
   successive point is obtained by adding a constant (depending on the
   generator) to each coordinate and then reducing back to :math:`[0,1]`. If the
   generators are well-chosen the result can be a good space-filling
   design. The procedure for generating a lattice design, with
   suggestions on choice of generators, is given in the procedure for
   generating a lattice design
   (:ref:`ProcLatticeDesign<ProcLatticeDesign>`).

-  Weyl sequences. A Weyl sequence is similar to a lattice design in the
   way it is generated, but with generators that are irrational numbers.
   See the procedure for generating a Weyl design
   (:ref:`ProcWeylDesign<ProcWeylDesign>`).

-  Halton sequences. A Halton sequence also has a prime integer
   "generator" for each dimension, and each prime generates a sequence
   of fractions. For instance, the generator 2 produces the sequence
   :math:`{\scriptstyle\frac{1}{2}}, {\scriptstyle\frac{1}{4}},
   {\scriptstyle\frac{3}{4}}, {\scriptstyle\frac{1}{8}},
   {\scriptstyle\frac{5}{8}}, {\scriptstyle\frac{3}{8}},
   {\scriptstyle\frac{7}{8}}, {\scriptstyle\frac{1}{16}}, \ldots`.
   So if the :math:`i`-th dimension has generator 2 then these will be the :math:`i`-th
   coordinates of successive points in the Halton sequence. See the
   procedure for generating a Halton design
   (:ref:`ProcHaltonDesign<ProcHaltonDesign>`).

-  Sobol's sequence. The Sobol's sequence uses the same set of coordinates
   as a Halton sequence with generator 2 for each dimension, but then
   reorders them according to a complicated rule. If we used the Halton
   sequence in :math:`p=2` dimensions with generator 2 for both
   dimensions, we would get the sequence :math:`({\scriptstyle\frac{1}{2},\frac{1}{2}}),
   ({\scriptstyle\frac{1}{4},\frac{1}{4}}),
   ({\scriptstyle\frac{3}{4},\frac{3}{4}}), \ldots`, and so on, so
   that all the points would lie on the diagonal of :math:`[0,1]^2`. The
   Sobol's sequence reorders the coordinates of each successive block of
   :math:`2^i` points :math:`(i=0,1,2,\ldots)` in a LHC way. For instance, the
   Sobol's sequence for :math:`p=2` begins
   :math:`({\scriptstyle\frac{1}{2},\frac{1}{2}}),
   ({\scriptstyle\frac{1}{4}, \frac{3}{4}}),
   ({\scriptstyle\frac{3}{4},\frac{1}{4}}), \ldots`. The complexity of
   the algorithm is such that we do not provide an explicit procedure in
   the :ref:`MUCM<DefMUCM>` toolkit, but we are aware of two freely
   available algorithms (:ref:`disclaimer<MetaSoftwareDisclaimer>`).
   For users of the R programming language, we suggest the function
   ``runif.sobol(n,d)`` from the package ``fOptions``` in the `R
   repository <http://cran.r-project.org/>`__. The Sobol's sequence is
   sometimes known also as the LP-tau sequence, and the
   `GEM-SA <http://tonyohagan.co.uk/academic/GEM/>`__ software package
   also generates Sobol's designs under this name. For more explanation
   and insight into the Sobol's sequence, see the Sobol's sequence
   procedure page (:ref:`ProcSobolSequence<ProcSobolSequence>`).

Model based optimal design
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimal design seeks a design which maximises/minimises some function,
typically, of the covariance matrix of the parameters or predictions.
Different :ref:`optimality criteria<AltOptimalCriteria>` can be
chosen for the classical optimal design. Formal optimisation may lead to
space-filling designs but may also yield designs which are better
tailored to specific emulation requirements. There is more information
about model based optimal design in
:ref:`ThreadTopicExperimentalDesign<ThreadTopicExperimentalDesign>`.
In particular, MUCM is developing a sequential strategy to select the
design, called ASCM (:ref:`Adaptive Sampler for Complex
Models<ProcASCM>`), that will eventually make use of the
:ref:`Karhunen Loeve expansion<DiscKarhunenLoeveExpansion>` to
approximate the Gaussian process.

Additional Comments, References, and Links
------------------------------------------

Because the optimised LHC designs require very many random LHCs to be
generated in order to choose the best one, this kind of design takes
substantially longer to generate than the non-random sequences. They
also have the disadvantage that the procedure is random and so repeating
it to generate a new design with the same number of points and
dimensions will produce a different result.

Another advantage of the Weyl, Halton and Sobol's sequences is that we
can readily add further points to the design. This facilitates the idea
of sequential design, where the training set is steadily increased until
a sufficiently good and validated emulator is obtained.

On the other hand, the non-random designs can be difficult to tune to
get good space-filling properties; only the Sobol's sequence does not
require careful choice of a set of generators. These designs can also
produce clumps or ridges of points.

The Halton and Sobol's sequences are examples of low-discrepancy
sequences. Discrepancy is a measure of departure of a set of points from
a uniform spread over :math:`[0,1]^p` and these are some of a small number
of sequence generators that have been shown to have asymptotically
minimal discrepancy. For more details, and in particular for a full
description of the Sobol's sequence, see

-  Kuipers, L. and Niederreiter, H. (2005). Uniform distribution of
   sequences. Dover Publications, ISBN 0-486-45019-8

Finally, the optimised LHC designs have additional flexibility through
the optimality criterion that may allow them to adapt better to prior
information about the simulator; see
:ref:`DiscCoreDesign<DiscCoreDesign>`.
