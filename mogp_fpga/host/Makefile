AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

OPT=-O2
CXXFLAGS=-Wall $(OPT)

square_exponential_test : square_exponential_test.o
	g++ -o square_exponential_test square_exponential_test.o $(AOCL_LINK_CONFIG)

square_exponential_test.o : square_exponential_test.cpp
	g++ $(CXXFLAGS) -c square_exponential_test.cpp $(AOCL_COMPILE_CONFIG)
