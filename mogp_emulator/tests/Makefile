# Makefile for running tests and benchmarks for multi-output emulator problems

# To get help, simply enter make into the shell to see a list of options

## help       : Display this message (same as make with no argument)
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## all        : Run unit tests and benchmarks
.PHONY: all
all: tests benchmarks

## tests      : Run unit tests
.PHONY: tests
tests:
	pytest

## benchmarks : Run the Branin, Rosenbrock (convergence), Tsunami (performance),
##            : MICE, and History Matching benchmarks
.PHONY: benchmarks
benchmarks: branin tsunami rosenbrock mice histmatch

## branin     : Run the 2D Branin tests of convergence. Produces the plots
##            : branin_2d_error.png and branin_2d_unc.png
.PHONY: branin
branin: branin_2d_error.png branin_2d_unc.png

## tsunami    : Run the benchmarks on a tsunami emulator problem. Produces the
##            : plot tsunami_scaling.png
.PHONY: tsunami
tsunami: tsunami_scaling.png

## rosenbrock : Run the Rosenbrock test of convergence. Produces the plots
##            : rosenbrock_error.png and rosenbrock_unc.png
.PHONY: rosenbrock
rosenbrock: rosenbrock_error.png rosenbrock_unc.png

## mice       : Run the MICE Experimental Design applied to the 2D Branin function.
##            : Produces the plots mice_error.png and mice_unc.png
.PHONY: mice
mice: mice_error.png mice_unc.png

## histmatch  : Run the History Matching demo showing 1D and 2D examples.
##            : Produces the plots histmatch_1D.png and histmatch_2D.png
.PHONY: histmatch
histmatch: histmatch_1D.png histmatch_2D.png

# commands above are shortcuts to the targets below

branin_2d_unc.png branin_2d_error.png: benchmark_branin.py
	python3 benchmark_branin.py
	
rosenbrock_unc.png rosenbrock_error.png: benchmark_rosenbrock.py
		python3 benchmark_rosenbrock.py
	
tsunami_scaling.png: benchmark_tsunami.py
	python3 benchmark_tsunami.py

mice_unc.png mice_error.png: benchmark_MICE.py
	python3 benchmark_MICE.py
	
histmatch_1D.png histmatch_2D.png: history_matching_demo.py
	python3 history_matching_demo.py