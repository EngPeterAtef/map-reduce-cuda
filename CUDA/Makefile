NVCC := nvcc
CXX := g++

TARGETS := kmeans
# SOURCES := kmeans.cu map_reduce.cu
SOURCES := map_reduce.cu
# HEADERS := config.cuh random_generator.hpp
KMEANS_HEADERS := kmeans.cuh random_generator.hpp
OBJECTS := $(SOURCES:.cu=.o)

all: $(TARGETS)

kmeans: $(SOURCES) $(KMEANS_HEADERS)
	$(NVCC) -O3 -dc $(SOURCES)
	$(NVCC) -O3 -o $@ $(OBJECTS)


clean:
	rm -f *.o $(TARGETS)
