# Makefile for running tests and benchmarks for multi-output emulator problems

# To get help, simply enter make into the shell to see a list of options

## help       : Display this message (same as make with no argument)
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## all        : Run all benchmarks
.PHONY: all
all: benchmarks

## benchmarks : Run the Branin, Rosenbrock (convergence), Tsunami (performance),
##            : Pivoted Cholesky, MICE, dimension reduction,
##            : and History Matching benchmarks
.PHONY: benchmarks
benchmarks: branin tsunami rosenbrock pivot mice gKDR histmatch

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

## gKDR       : Run the dimension reduction benchmark
##            : Produces the plot benchmark_kdr_GP_loss.pdf
.PHONY: gKDR
gKDR: benchmark_kdr_GP_loss.pdf

## pivot      : Run the pivoted Cholesky decomposition benchmark
##            : Produces the plots pivot_error.png and pivot_unc.png
.PHONY: pivot
pivot: pivot_error.png pivot_unc.png

# commands above are shortcuts to the targets below

branin_2d_unc.png branin_2d_error.png: benchmark_branin.py
	python3 benchmark_branin.py

rosenbrock_unc.png rosenbrock_error.png: benchmark_rosenbrock.py
	python3 benchmark_rosenbrock.py

tsunami_scaling.png: benchmark_tsunami.py
	python3 benchmark_tsunami.py

mice_unc.png mice_error.png: benchmark_MICE.py
	python3 benchmark_MICE.py

histmatch_1D.png histmatch_2D.png: benchmark_historymatching.py
	python3 benchmark_historymatching.py

benchmark_kdr_GP_loss.pdf benchmark_kdr_GP_loss.npy: benchmark_kdr_GP.py
	python3 benchmark_kdr_GP.py

pivot_error.png pivot_unc.png: benchmark_pivot.py
	python3 benchmark_pivot.py
