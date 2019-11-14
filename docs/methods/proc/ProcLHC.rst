.. _ProcLHC:

Procedure: Generate a Latin hypercube
=====================================

Description and Background
--------------------------

A Latin hypercube (LHC) is a random set of points in :math:`[0,1]^p`
constructed so that for :math:`i=1,2,\ldots,p` the i-th coordinates of the
points are spread evenly along [0,1].

Inputs
------

-  Number of dimensions :math:`p`
-  Number of points desired :math:`n`

Outputs
-------

-  LHC :math:`D = \\{x_1, x_2, \\ldots, x_n\}`

Procedure
---------

#. For each :math:`i=1,2,\ldots,p`, independently generate :math:`n` random
   numbers :math:`u_{i1}, u_{i2}, \\ldots, u_{in}` in [0,1] and a random
   permutation :math:`b_{i1}, b_{i2}, \\ldots, b_{in}` of the integers
   :math:`0,1,\ldots,n-1`.
#. For each :math:`i=1,2,\ldots,p` and :math:`j=1,2,\ldots,n` let :math:`x_{ij} =
   (b_{ij}+u_{ij})/n`.
#. For :math:`j=1,2,\ldots,n`, the j-th LHC point is :math:`x_j = (x_{1j},
   x_{2j}, \\ldots, x_{pj})`.

Additional Comments
-------------------

The construction of a LHC implies that the projection of the set of
points into the i-th dimension results in :math:`n` points that are evenly
spread over that dimension. If we project into two or more of the :math:`p`
dimensions, a LHC may also appear to be well spread (space-filling) in
that projection, but this is not guaranteed.
