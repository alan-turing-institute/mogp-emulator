.. _DefBasisFunctions:

Definition of Term: Basis functions
===================================

The mean function of an emulator, :math:`m(x)`, expresses the global
behaviour of the computer model and can be written in a variety of forms
as described in the alternatives page on emulator prior mean function
(:ref:`AltMeanFunction<AltMeanFunction>`). When the mean function
takes a linear form it can be expressed as :math:`m(x)=\beta^Th(x)`
where :math:`h(x)` is a :math:`q`-vector of known *basis functions* of :math:`x`
which describe global features of the simulator's behaviour and
:math:`\beta` is a vector of unknown coefficients.

The task of specifying appropriate forms for :math:`h(x)` is addressed in
the alternatives page on basis functions for the emulator mean
(:ref:`AltBasisFunctions<AltBasisFunctions>`).
