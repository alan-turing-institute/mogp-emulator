# Makefile for running tests for multi-output emulator problems

# To get help, simply enter make into the shell to see a list of options

## help       : Display this message (same as make with no argument)
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## all        : Run unit tests and benchmarks
.PHONY: all
all: tests

## tests      : Run unit tests
.PHONY: tests
tests:
	pytest
