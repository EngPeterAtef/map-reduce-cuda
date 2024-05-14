NVCC := nvcc
CXX := g++

TARGETS := kmeans
# SOURCES := kmeans.cu map_reduce.cu
SOURCES := map_reduce.cu
DEF_HEADER := random_generator.hpp
HEADERS := config.cuh $(DEF_HEADER)
CPU_SOURCES := cpu_kmeans.cpp
OBJECTS := $(SOURCES:.cu=.o)

all: $(TARGETS)

kmeans: $(SOURCES) $(HEADERS)
	$(NVCC) -O3 -dc $(SOURCES)
	$(NVCC) -O3 -o $@ $(OBJECTS)


clean:
	rm -f *.o $(TARGETS)
