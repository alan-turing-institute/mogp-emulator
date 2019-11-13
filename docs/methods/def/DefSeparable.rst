.. _DefSeparable:

Definition of Term: Separable
=============================

An :ref:`emulator<DefEmulator>`'s correlation function specifies the
correlation between the outputs of a :ref:`simulator<DefSimulator>`
at two different points in its input space. The input space is almost
always multi-dimensional, since the simulator will typically have more
than one input. Suppose that there are :math:`p` inputs, so that a point in
the input space is a vector :math:`x` of :math:`p` elements
:math:`x_1,x_2,\ldots,x_p`. Then a correlation function :math:`c(x,x^\prime)`
specifies the correlation between simulator outputs at input vectors
:math:`x` and :math:`x^\prime`. The inputs are said to be separable if the
correlation function has the form of a product of one-dimensional
correlation functions:

.. math::
   c(x,x^\prime) = \prod_{i=1}^p c_i(x_i,x_i^\prime).

Specifying the correlation between points that differ in more than one
input dimension is potentially a very complex task, particularly because
of the constraints involved in creating a valid correlation function.
Separability is a property that greatly simplifies this task.

Separability is also used in the context of emulators with multiple
outputs. In this case the term 'separable' is typically used to denote
separability between the outputs and the inputs, i.e. that all the
outputs have the same correlation function :math:`c(x,x')` (which is often
itself separable as defined above). The general covariance then takes
the form:

.. math::
   \text{Cov}[f_u(x), f_{u'}(x')] = \sigma_{uu'}c(x,x')

where :math:`u` and :math:`u'` denote two different outputs and
:math:`\sigma_{uu'}` is a covariance between these two outputs.
