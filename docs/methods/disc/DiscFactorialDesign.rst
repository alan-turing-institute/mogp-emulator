.. _DiscFactorialDesign:

Discussion: Factorial design
============================

Description and Background
--------------------------

Several toolkit operations call for a set of points to be specified in
the :ref:`simulator's<DefSimulator>` space of inputs, and the choice
of such a set is called a design. Factorial designs refers to general
class of design in which each (univarate) input, or factor, has a
prespecified set of levels. A design which consists of all combinations
of factor levels is called a *full factorial design*. For instance, if
there are two factors, the first with 3 levels and the second with 2
levels, then the full factorial design has 6 points. Another important
case is where :math:`\strut d` factors all have :math:`\strut k` levels, so
that the full factorial design has :math:`k^d` points. If the design is a
subset of a full factorial design then it is sometimes refered to as a
*fraction* (fraction of a full factorial).

A full factorial design makes a suitable *candidate set* of designs
points, that is to say a large set out of which one may select a subset
for the actual design used to build an emulator. In this sense factorial
desiogn is a generic term and covers most designs. An exception is where
points are selected at random to a certain precision.

#. Regular fractions. In this case every input (factor) takes
   :math:`\strut{k}` levels and :math:`\strut{k}` is a power of a prime
   number. In the :math:`\strut{k^d}` case the fractions would typically
   have :math:`\strut{n = k^{d-r}}`, for a suitable integer :math:`\strut{r}`.
#. Irregular fractions. These preserve some of the properties of regular
   fractions but are typically a one-off, or in small families. One of
   the most famous is the 12-point Plackett-Burman design, for up to 11
   factors, each with 2 levels.
#. Response surface designs. These are more adventurous that regular
   fractions, but may have nice symmetry properties. For example for
   :math:`\strut{d}=3` one may take a :math:`\strut{2^3}` full factorial with
   points :math:`(\pm 1,\pm 1, \\pm 1)` and add "star" points on the axes
   at :math:`(\pm c,0,0), (0, \\pm c,0), (0,0, \\pm c)`. Designs built from
   two or more types are called *composite*.
#. :ref:`Screening<DefScreening>` design. The classical versions of
   these design have links to group screening (e.g. pooling blood
   samples for fast elimination of disease free patients) and search
   problems such as binary search in computer science and information
   theory and games (e.g. find the bad penny out of twelve with a
   minimal number of balance weighings). Highly fractionated factorial
   designs are useful, though typically only have two or three levels.
   The Morris method (see the procedure page
   :ref:`ProcMorris<ProcMorris>`) is a kind of hybrid of factorial
   design and a space filling design.

Discussion
----------

Factorial design has typically been used, historically, to fit
polynomial models and some general principles are established in this
context.

#. A factor must have at least :math:`\strut{k}` levels to model a
   polynomial up to degree :math:`\strut{k-1}` in that factor.
#. Avoid aliasing: make sure that all the terms in the models one may be
   interested in can be estimated at the same time. Technically,
   aliasing means that two polynomials agree up to a constant on the
   design and therefore cannot be distinguished by the design; *e.g.*
   with the :math:`2^{3-1}` fraction :math:`(-1,-1,-1), (1,1,-1),(1,-1,1),
   (-1,1,1)` the terms :math:`x_1` and :math:`x_2x_3` are aliased.
#. Blocking. There may be a *blocking factor* which is not modeled
   explicitly but can be guarded against in the experiment. In physical
   experiments time and temperature are typical blocking factors
#. Replication. This is advantageous in physical experiments to obtain
   dependent estimates of the error variance :math:`\sigma^2`.

Additional Comments, References, and Links
------------------------------------------

Boukouvalas, A., Gosling, J.P. and Maruri-Aguilar, H., `An efficient
screening method for computer
experiments <http://wiki.aston.ac.uk/twiki/pub/AlexisBoukouvalas/WebHome/screenReport.pdf>`__.
NCRG Technical Report, Aston University (2010)

D R Cox and N Reid. The Theory Of The Design Of Experiments, CRC Press,
2000.

D-Z Du and F K Hwang. Combinatorial Group Testing and Applications,
World Scientific, 1993.

G. E. P. Box, J. S. Hunter and W. G. Hunter. Statistics for
Experimenters: Design, Innovation, and Discovery , 2nd Edition, Wiley,
2005.

G. Pistone, E. Riccomango, H. P. Wynn, Algebraic Statistics, CRC Press,
2001.

K S Banerjee. Weighing designs. Marcel Dekker, New York, 1975.
