CXX := g++

TARGETS := mapreduce
SOURCES := map_reduce.cpp
HEADERS := config.hpp random_generator.hpp kmeans.hpp
OBJECTS := $(patsubst %.cpp,%.o,$(SOURCES))

all: $(TARGETS)


mapreduce: $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(SOURCES)

clean:
	rm -f *.o $(TARGETS)
