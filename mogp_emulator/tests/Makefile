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

## benchmarks : Run the Branin (convergence) and tsunami (performance) benchmarks
.PHONY: benchmarks
benchmarks: branin tsunami

## branin     : Run the 2D Branin tests of convergence. Produces the plots
##            : branin_2d_error.png and branin_2d_unc.png
.PHONY: branin
branin: branin_2d_error.png branin_2d_unc.png

## tsunami    : Run the benchmarks on a tsunami emulator problem. Produces the
##            : plot tsunami_scaling.png
.PHONY: tsunami
tsunami: tsunami_scaling.png

# commands above are shortcuts to the targets below

branin_2d_unc.png branin_2d_error.png: benchmark_branin.py
	python3 benchmark_branin.py
	
tsunami_scaling.png: benchmark_tsunami.py
	python3 benchmark_tsunami.py
