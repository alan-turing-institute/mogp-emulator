.. _ProcExchangeAlgorithm:

Procedure: Exchange Algorithm
=============================

Description and Background
--------------------------

The exchange algorithm has been used and modified by several authors to
construct a :math:`D`-optimal and other types of design. Fedorov (1972),
Mitchell and Miller (1970), Wynn (1970) , Mitchell (1974) and Atkinson
and Donev (1989) study versions of this algorithm. The main idea of all
versions is start with an initial feasible design and then greedily
modify the design by exchange until a satisfactory design is obtained.
The following steps are the general steps for an exchange algorithm.

Inputs
------

#. A large candidate set, such as a full factorial design (scaled
   integer grid) or a spacefilling design.
#. A initial design chosen, at random, or preferably, a space filling
   design.
#. An objective function :math:`M`, which is based on an
   optimal design criterion.

Outputs
-------

#. The optimal or near optimal design :math:`D=\{x_1,x_2, \cdots, x_n\}`.
#. The value of the target function at the obtained design

Procedure
---------

#. Start with the initial :math:`n` point design.
#. Compute the target function :math:`M`.
#. General step. Find a set of points of size :math:`k` from
   the current design and replace with another set of :math:`k`
   points chosen from the candidate set. If the value of :math:`M`,
   after this exchange, is improved, keep the new design and update
   :math:`M`. The set of points removed from the current design can
   be put back in the candidate set.
#. Repeat till the stopping rule applies. It may be that after many
   attempts at a good exchange there is no improvement or only a small
   improvement in the design. As for global optimisation algorithms it
   is usual to plot the value of the objective function against the
   number of improving exchanges, or simply the number of exchanges.

Additional Comments, References, and Links
------------------------------------------

#. Note that the algorithm will need an updating rule for the objective
   function which is preferable fast to compute. There is in principle
   no restriction on the objective function.
#. There are are many variations. One version is to first add
   :math:`k` optimally and sequentially to obtain a design with
   :math:`n+k` points and then remove :math:`k` optimally and
   sequentially. Such operation is called an *excursion*. The choice of
   the test points outside the current design at step 3 (above) may be
   computational slow in which case a fast strategy is to choose them at
   random. The same idea can be applied when we use an excursion method,
   at least for the forward part of the excursion.
#. The method can be used in Bayesian or classical optimal design.

A.C. Atkinson and A.N. Donev. The construction of exact D-optimal
designs with application to blocking response surface designs.
*Biometrika*, 76, 515-526, 1989.

H.P. Wynn. The sequential generation of D-optimal experimental designs,
*Ann. Math. Stat.*, 41, 1644-1655, 1970.

N. K. Nguyen and A.J. Miller, A review of exchange algorithms for
constructing descrete D-optimal designs, *J. of Computational Statistics
& Data Analysis* 14, 489-498, 1992.

T.J. Mitchell,An algorithm for the construction of D-optimal designs,
*Technometrics*, 20, 203-210, 1974.

T.J. Mitchell, and A. J. Miller, Use of 'design repair' to
construct designs for special linear models, *Math. Div. Ann. Progr.
Rept.*,130-131, 1970.

V. Fedorov. *Theory of Optimal Experiments*. Academic Press, New York
,1972.
