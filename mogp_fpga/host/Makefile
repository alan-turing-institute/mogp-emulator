AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

OPT=-O2
CXXFLAGS=-Wall $(OPT)

OBJECTS=kernel_tests.o
TARGET=kernel_tests

all : $(OBJECTS)
	g++ $(AOCL_LINK_CONFIG) -o $(TARGET) $(OBJECTS)

%.o : %.cpp
	g++ $(CXXFLAGS) $(AOCL_COMPILE_CONFIG) -c $< -o $@

clean:
	rm -f *.o
