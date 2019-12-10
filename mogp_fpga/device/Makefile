AOC = aoc
EMULATION_FLAGS = -march=emulator -legacy-emulator
SIMULATION_FLAGS = -march=simulator
REPORT_FLAGS = -rtl

SOURCES = prediction.cl

report: $(SOURCES)
	 $(AOC) $(REPORT_FLAGS) $(SOURCES)

emulation: $(SOURCES)
	$(AOC) $(EMULATION_FLAGS) $(SOURCES)

simulation: $(SOURCES)
	$(AOC) $(SIMULATION_FLAGS) $(SOURCES)

device: $(SOURCES)
	$(AOC) $(SOURCES)

clean:
	rm -rf *.aoco *.aocr *.temp
