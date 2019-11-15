AOC = aoc
EMULATION_FLAGS = -march=emulator -legacy-emulator
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

clean:
	rm -f *.aoco *.aocr
