# Multi-Output Gaussian Process Emulator

`mogp_emulator` is a Python package for fitting Gaussian Process Emulators to computer simulation results.
The code contains routines for fitting GP emulators to simulation results with a single or multiple target
values, optimizing hyperparameter values, and making predictions on unseen data. The library also implements
experimental design, dimension reduction, and calibration tools to enable modellers to understand complex
computer simulations.

[![Build Status](https://travis-ci.com/alan-turing-institute/mogp-emulator.svg?branch=master)](https://travis-ci.com/alan-turing-institute/mogp-emulator)
[![codecov](https://codecov.io/gh/alan-turing-institute/mogp-emulator/branch/master/graph/badge.svg)](https://codecov.io/gh/alan-turing-institute/mogp-emulator)
[![Documentation Status](https://readthedocs.org/projects/mogp-emulator/badge/?version=latest)](https://mogp-emulator.readthedocs.io/en/latest/?badge=latest)

## Installation

`mogp_emulator` requires Python version 3.6 or later. The code and all of its dependencies can be
installed via `pip`:

```bash
pip install mogp-emulator
```

Optionally, you may want to install some additional optional packages. `matplotlib` is useful for
visualising some of the results of the benchmarks, and `patsy` is highly recommended for users that
wish to parse R-style string formulas for specifying mean functions. These can be found in the
[requirements-optional.txt](requirements-optional.txt) file in the main repository.

## Documentation

The documentation is available at [readthedocs](https://mogp-emulator.readthedocs.io) for the current
builds of the `master` and `devel` versions, plus any previous tagged releases. The documentation
is available there in HTML, PDF, or e-Pub format.

## Getting Help

This package is under active development by the Research Engineering Group at the Alan Turing Institute
as part of several projects on Uncertainty Quantification. Questions about the code or any feedback on
the usability and features that you would find useful can be sent to Eric Daub
<[edaub@turing.ac.uk](mailto:edaub@turing.ac.uk)>. If you encounter any bugs or problems with installing
the software, please see the Issues tab on the Github page, and if the issue is not present, create a new one.

## Contributing

If you find this software useful, please consider taking part in its development! We aim to make
this a welcoming, collaborative, open-source project where users of any background or skill levels feel that they
can make valuable contributions. Please see the [Code of Conduct](CODE_OF_CONDUCT.md) for more on our
expectations for participation in this project, and our [Contributing Guidelines](CONTRIBUTING.md)
for how to get involved.