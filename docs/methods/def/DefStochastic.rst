.. _DefStochastic:

Definition of Term: Stochastic
==============================

A :ref:`simulator<DefSimulator>` is referred to as stochastic if
running it more than once with a given input configuration will produce
randomly different output(s).

*Example*: A simulator of a hospital admissions system has as inputs a
description of the system in terms of number of beds, average time spent
in a bed, demand for beds, etc. Its primary output is the number of days
in a month in which the hospital is full (all beds occupied). The
simulator generates random numbers of patients requiring admission each
day, and random lengths of stay for each patient. As a result, the
output is random. Running the simulator again with the same inputs
(number of beds, etc) will produce different random numbers of patients
needing admission and different random lengths of stay, and so the
output will also be random.
