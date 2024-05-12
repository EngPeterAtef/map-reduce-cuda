#ifndef MAP_REDUCE_HPP
#define MAP_REDUCE_HPP
#include <iostream>

// No. of input elements (Lines in text file)
unsigned long long NUM_INPUT;
// No. of pairs per input element
const int NUM_PAIRS = 1;
// Total No. of output values (K - No. of clusters)
const int NUM_OUTPUT = 10;

// No. of values in each line (Size of datapoint)
const int DIMENSION = 2;
// No. of iterations
const int ITERATIONS = 1000;

// Custom types
struct Vector2D
{
    int values[DIMENSION];
};

// Type declarations for input, output & key-value pairs
using input_type = Vector2D;  // Datapoint (or vector) read from the text file
using output_type = Vector2D; // Outputs are the cluster centroids

// So each point will get associated with a cluster (with id -> key)
using Mykey = int;        // Cluster that the point corresponds to
using MyValue = Vector2D; // Point associated with the cluster

// Pair type definition
struct MyPair
{
    Mykey key;
    MyValue value;

    // Printing for debugging
    friend std::ostream &operator<<(std::ostream &os, const MyPair &pair)
    {
        os << "Key: " << pair.key << ", Point: ";
        for (int i = 0; i < DIMENSION; i++)
            os << pair.value.values[i] << " ";
        os << "\n";
        return os;
    }
};
struct ShuffleAndSort_KeyPairOutput
{
    Mykey key;
    std::vector<MyValue> values;
};

/*
    Comparision operator for comparing between 2 KeyValuePairs
    Returns true if first pair has key less than the second
*/
struct PairCompare
{
    bool operator()(const MyPair &lhs, const MyPair &rhs)
    {
        return lhs.key < rhs.key;
    }
};

unsigned long long TOTAL_PAIRS;

void runMapReduce(const input_type *input, output_type *output);

#endif // MAP_REDUCE_HPP