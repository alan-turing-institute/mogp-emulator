.. _ProcBranchAndBoundAlgorithm:

Procedure: Branch and bound algorithm
=====================================

Description and Background
--------------------------

Branch and Bound algorithm is a commonly used technique in several
mathematical programming applications especially in combinatorial
problem when a problem is difficult to be solved directly. It is
preferred over many other algorithms because it reduces the amount of
search needed to find the optimal solution.

Branch and Bound creates a set of subproblems by dividing the space of
current problem into unexplored subspaces represented as nodes in a
dynamically generated search tree, which initially contains the root
(the original problem). Performing one iteration of this procedure is
based on three main components: selection of the node to process, bound
calculation, and branching, where branching is the partitioning process
at each node of the tree and bounding means finding lower and upper
bounds to construct a proof of optimality without exhaustive research.
The algorithm can be terminated whenever the difference between the
upper bound and the lower bound is smaller than the chosen :math:`\epsilon`
or if the set of live nodes is empty, i.e. there is no unexplored
parts of the solution space left, and the optimal solution is then the
one recorded as "current best".

The branch and bound procedure explained here is for finding a maximum
entropy design. The computations are based on the process covariance
matrix of a large candidate set of order :math:`N \times N`, stored as
a function. Following the Maximum Entropy Sampling principle, the goal
is to find the design of sample size :math:`\strut{n}` whose :math:`n
\times n` process covariance matrix has maximum determinant. Since the
design is a subset of the candidate set the design covariance matrix
will be a submatrix of the candidate set covariance matrix.

Inputs
------

#. Candidate set of :math:`N` points
#. A known covariance matrix (stored as a function) of the candidate
   points
#. :math:`E` the set of all eligible points
#. :math:`F` the set of all points forced to be in the design
#. :math:`\epsilon > 0` small chosen number
#. A counter :math:`k = 0`
#. An initial design of size :math:`S_0` of size :math:`n`
#. An optimality criterion :math:`I = \log \det C[S]` (the version
   described is for entropy).
#. General upper and lower bounds: :math:`U = Ub(C, F, E, s)`, :math:`L =
   Lb{(C, F, E, s)}`.

Outputs
-------

#. Global optimal design for the problem
#. Value of the objective
#. Various counts such as number of branchings made

Procedure
---------

#. Let :math:`k` stand for the iteration number, :math:`U_k`
   stand for the upper bound at :math:`k` th iteration,
   :math:`I`, the incumbent, i.e. for best current value of the
   target function and :math:`L` be the set of all unexplored
   subspaces
#. Remove the problem, tuple, with max :math:`Ub` from
   :math:`L` in order to be explored.
#. Branch the problem according to these conditions
#. Set :math:`k =k+1`

   -  If :math:`|F|+|E|-1>s`, compute :math:`Ub(C, F, E\setminus i, s)` and
      if :math:`Ub(C, F, E\setminus i, s)>U_k`, where :math:`i` is
      any index selected to be removed from :math:`E`, then add
      :math:`(C, F, E\setminus i, s)` to :math:`L`, else if
      :math:`|F|+|E|-1=s`, then set :math:`S=F \bigcup E \setminus i` and compute
      :math:`\log \det Cov[S]`, and if :math:`\log \det Cov[S] > I` then
      set :math:`I=\log \det Cov[S]` and the current design is :math:`S`.
   -  If :math:`|F|+1 < s`, compute :math:`Ub(C,F \bigcup i, E\setminus
      i, s)` and if :math:`Ub(C, F \bigcup i,E\setminus i, s) > U_k`,
      then add :math:`(C, F,E\setminus i, s)` to :math:`L` else if
      :math:`|F|+1 < s`, then set :math:`S = F \bigcup i` and compute
      :math:`\log \det S` and if :math:`\log \det Cov[S] > I` then set
      :math:`I = \log \det S` and the current design is
      :math:`S`.

#. Update :math:`U_{k+1}` to be the highest upper bound in :math:`L`.
#. Repeat steps 3,4,5 while :math:`U_k - I > \epsilon`.

Additional Comments, References, and Links
------------------------------------------

The upper bound used in the algorithm above are based on spectral bounds
for determinants: the determinant of a non-negative definite matrix (eg
:math:`C[S]`) is less than or equal to the product of the diagonal
elements, which is taken as the upper bound. In this version of the
branch and bound the current value of the objective function is taken as
the lower bound.

In principle the same algorithm can easily be applied to other optimal
design criteria if a general upper bounding function can be found. For
example for IMSEP (prediction MSE) we want to *minimise* the objective
and need a lower bound. The maximum eigen-value of the posterior
covariance matrix of the unsampled point in the candidate set can be
used, although expensive to compute.

The following is a main reference on branch and bound for maximum
entropy sampling.

W. K. Chun, J. Lee, and M. Queyranne. An exact algorithm for maximum
entropy sampling. *Oper. Res.*, 43(4):684-691, 1995

A general reference on combinatorial optimisation which contains useful
material on the Branch and Bound algorithm is:

Handbook of combinatorial optimization. Supplement Vol. B. Edited by
Ding-Zhu Du and Panos M. Pardalos. Springer-Verlag, New York, 2005.
