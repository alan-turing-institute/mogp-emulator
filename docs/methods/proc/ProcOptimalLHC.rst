.. _ProcOptimalLHC:

Procedure: Generate an optimised Latin hypercube design
=======================================================

Description and Background
--------------------------

A Latin hypercube (LHC) is a random set of points in :math:`[0,1]^p`
constructed so that for :math:`i=1,2,\ldots,p` the i-th coordinates of the
points are spread evenly along [0,1]. However, a single random LHC will
rarely have good enough space-filling properties in the whole of
:math:`[0,1]^p` or satisfy other desirable criteria. Therefore, it is usual
to generate many LHCs and then select the one having the best value of a
suitable criterion.

We present here the procedure for a general optimality criterion and
also include details of the most popular criterion, known as maximin. A
thorough consideration of optimality criteria may be found in the
discussion page on technical issues in training sample design for the
core problem (:ref:`DiscCoreDesign<DiscCoreDesign>`).

Inputs
------

-  Number of dimensions :math:`p`
-  Number of points desired :math:`n`
-  Number of LHCs to be generated :math:`N`
-  Optimality criterion :math:`C(D)`

Outputs
-------

-  Optimised LHC design :math:`D = \\{x_1, x_2, \\ldots, x_n\}`

Procedure
---------

Procedure for a general criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. For :math:`k=1,2,\ldots,N` independently, generate a random LHC :math:`D_k`
   using the procedure page on generating a Latin hypercube
   (:ref:`ProcLHC<ProcLHC>`) and evaluate the criterion :math:`C_k =
   C(D_k)`.
#. Let :math:`K=\arg\max\{D_k\}` (i.e. :math:`K` is the number of the LHC with
   the highest criterion value).
#. Set :math:`D=D_K`.

Note that the second step here assumes that high values of the criterion
are desirable. If low values are desirable, then change argmax to
argmin.

Maximin criterion
~~~~~~~~~~~~~~~~~

A commonly used criterion is

:math:`C(D) = \\min_{j\ne j^\prime}|x_j - x_{j^\prime}|` ,

where :math:`|x_j - x_{j^\prime}|` denotes a measure of distance between
the two points :math:`x_j` and :math:`x_{j^\prime}` in the design. The
distance measure is usually taken to be squared Euclidean distance: that
is, if :math:`u=(u_1,u_2,\ldots,u_p)` then we define

:math:`|u\| = u^T u = \\sum_{i=1}^p u_i^2\,.`

High values of this criterion are desirable.

Additional Comments
-------------------

Note that the resulting design will not truly be optimal with respect to
the chosen criterion because only a finite number of LHCs will be
generated.
