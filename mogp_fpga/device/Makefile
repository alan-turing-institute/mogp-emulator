AOC = aoc
EMULATION_FLAGS = -march=emulator -legacy-emulator
SIMULATION_FLAGS = -march=simulator
REPORT_FLAGS = -rtl
TARGET = prediction

SOURCES = distance.cl \
          matrix_vector_product.cl \
          square_exponential.cl

report: $(SOURCES)
	 $(AOC) $(REPORT_FLAGS) $(SOURCES)
	mv square_exponential.aocr $(TARGET).aocr

emulation: $(SOURCES)
	$(AOC) $(EMULATION_FLAGS) $(SOURCES)
	mv square_exponential.aocx $(TARGET).aocx

simulation: $(SOURCES)
	$(AOC) $(SIMULATION_FLAGS) $(SOURCES)
	mv square_exponential.aocx $(TARGET).aocx

device: $(SOURCES)
	$(AOC) $(SOURCES)
	mv square_exponential.aocx $(TARGET).aocx


clean:
	rm -f *.aoco *.aocr
