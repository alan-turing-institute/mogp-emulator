.. _AltOptimalDesignAlgorithms:

Alternatives: Optimal Design Algorithms
=======================================

Overview
--------

Finding an optimal design is a special type of problem, but shares
features with more general optimisation problems.Typically the objective
function will have very many local optimal so that any of the general
methods of global optimization may be used such simulated annealing
(SA), genetic algorthims (GA) and global random search. Each of these
has a large litertature and many implementations. As with general
optimisation a reasonable strategy is to use an algorithm which is good
for unimodal problems, such as sequential quadratic programming (SQP),
but with multi-start. Another strategy is to start with a global method
and finish with a local method. Although stochastic algorithms such as
GA have theory which show that under certain conditions the algorithm
will converge in a probability sense to a global optimum, it is safer to
treat them as simply finding local optima.

Choosing the Alternatives
-------------------------

For small problems, that is, low dimension and low sample size the
Branch and Bound algorithm is guaranteed to converge to a global
optimum. The technical issue then is to find the "bounds" and
"branching" required in the specification of the algorithm. For Maximum
Entropy Sampling (MES) spectral bounds are available.

As pointed out, an optimum design problem can be thought of as an
optimal subset problem, so that exchange algorithms in which "bad"
points are exchanged for "good" are a natural are a natural method and
turn out to be fast and effective. But, as with general global
optimisation methods, they are not guaranteed to converge to an optimum
so multi-start is essential. Various versions are possible such as
exchanging more than one point and incorporating random search. In the
latter case exchange algorithms share have some of the flavour of global
optimisation algorithms which also have randomisation.

The notion of "greadiness" is useful in optimisation algorithms. A
typical type of greedy algorithms will move one factor at a time, to
find an optimum along that direction, cycling through the factors one at
a time. Only under very special consitions are such algorithms
guaranteed to converge, but combined with multi-start they can be
effective and they are easy to implement.On can think of an exchange
algorithms as a type of greedy algorithm.

When the problem has been made discrete by optimally choosing a design
from and candidate set it is appropriate to call an algorithm a
"combinatorial algorithm": one is choosing a combination of levels of
the factors.

The Nature of the Alternatives
------------------------------

We describe two algorithms

Exchange algorithms
~~~~~~~~~~~~~~~~~~~

This is described in more detail in
:ref:`ProcExchangeAlgorithm<ProcExchangeAlgorithm>`. The method is to
find an optimal design with fixed sample size :math:`\strut{n}`. Given a
large candidate set and an initial design, which is a subset of the
candidate set, points are exchanged between the design and the points in
the candidate set, but outside the design. This is done in a way to keep
the same size fixed between exchanges. During this exchange one tries to
improve the value of the design optimality criterion. Variations are to
exchange more than one point, that is a "block" of points. Since the
candidate set is large one may select points to be exchanged at random.

Branch and bound algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is descibed in more detail in
:ref:`ProcBranchAndBoundAlgorithm<ProcBranchAndBoundAlgorithm>`.
Branch and Bound searches on a tree of possible designs. The branching
tells us how to go up and down the tree. The bounding is a way to avoid
having to search a whole branch of the tree by using a bound to show
that all designs down a branch are worse than the current branch. In
this way the decision at each stage is to carry on down a "live" branch,
or jump to a new branch because the rest of a branch is no longer live.
The bounds also supply the stopping rule. Roughly, if the algorithm
stops because the bound tells us not to continue down a branch and there
are no other live branches available then it is straightfoward to see we
have a global optimum. The algorithm works well when the tree is set up
not to be too enormous and the bound is easy to compute: fast bounding,
not too many branches. But there is a tension: the tighter the bound the
more effective it is in eliminating branches but typically the harder to
find or compute. The hope is that there is a very natural bound and this
is often the case with design problems since there is usual matrix
theory involved one can use matrix inequalities of various kinds. The
optimum subset nature of the problem also lends itself to trees based on
unions and intersections of subsets.

Additional Comments, References, and Links
------------------------------------------

Global optimisation can be used as a generic term, which covers all
methods not tailored to particular classes of function, eg convex.

A. Zhigljavsi and A. Zilinskas. Stochastic Global Optimization.
Springer, (2008).
