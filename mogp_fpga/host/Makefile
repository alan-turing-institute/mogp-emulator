COMPILER=g++

AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

PYBIND_INCLUDES=$(shell python -m pybind11 --includes)
SUFFIX=$(shell python-config --extension-suffix)

OPT=-O2
DEBUG=-Wall
CXXFLAGS=-shared -std=c++14 -undefined -fPIC $(PYBIND_INCLUDES) $(OPT) $(DEBUG)

OBJECT=prediction$(SUFFIX)
TARGET=prediction.cpp

all: $(TARGET)
	$(COMPILER) $(CXXFLAGS) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) $(TARGET) -o $(OBJECT)

clean:
	rm -f *$(SUFFIX)
