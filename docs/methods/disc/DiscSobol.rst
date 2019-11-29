.. _DiscSobol:

Discussion: Sobol sequence
===========================

Description and background
--------------------------

A Sobol sequence is a sequence of points in a design space which has
the general property of being :ref:`space
filling<DefSpaceFillingDesign>`. The procedure page
:ref:`ProcSobolSequence<ProcSobolSequence>` provides a detailed
algorithm for constructing Sobol sequences based on Sobol's original
description, and a worked example. Sobol's construction is based on
bit-wise XOR operations between special generators called direction
numbers. The first impression is that the construction of a Sobol
sequence is a complex task. However, by constructiong a few Sobol
numbers by hand, using the procedure in
:ref:`ProcSobolSequence<ProcSobolSequence>`, it is easy to understand
the underlying structure.

Discussion
----------

For users of the R programming language, we suggest the function
``runif.sobol(n,d)`` from the package fOptions in the `R
repository <http://cran.r-project.org/>`_.

A well known concern about Sobol sequences is the correlation between
points in high dimensional sequences. A solution to this is by
scrambling the sequence. The R package above described allows for the
sequence to be scrambled, if desired. In this case, either of the two
functions described above can be called with the arguments ::

   (n,d,scrambling,seed,init)

The argument scrambling takes an integer which is 0 if no scrambling is
to be used, 1 for Owen type of scrambling, 2 for Faure-Tezuka scrambling
and 3 for a combination of both. The argument seed is the integer seed
to trigger the scrambling process and init is a logical flag which
allows the Sobol sequence to restart if set to ``TRUE``.
