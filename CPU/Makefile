CXX := g++

TARGETS := mapreduce
SOURCES := map_reduce.cpp
HEADERS_KMEANS := random_generator.hpp kmeans.hpp
HEADERS_WORDCOUNT := wordcount.hpp
OBJECTS := $(SOURCES:.cpp=.o)

# all: $(TARGETS)


kmeans: $(SOURCES) $(HEADERS_KMEANS)
	$(CXX) -o $@ $(SOURCES)

wordcount: $(SOURCES) $(HEADERS_WORDCOUNT)
	$(CXX) -o $@ $(SOURCES)

clean:
	rm -f *.o $(TARGETS)
