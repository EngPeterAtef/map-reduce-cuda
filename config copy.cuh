#ifndef MAP_REDUCE_CUH
#define MAP_REDUCE_CUH

// No. of input elements (Lines in text file)
// TODO: Maybe make it variable, calculated from reading the text file
// No. of pairs per input element
const int NUM_PAIRS = 1;
// Total No. of output values (K - No. of clusters)
const int NUM_INPUT = 5;
const int NUM_OUTPUT = 15;
// No. of iterations
const int ITERATIONS = 1000;
// Custom types
struct Point
{
    dim3 values;
};
// Type declarations for input, output & key-value pairs
using input_type = dim3;  // Datapoint (or vector) read from the text file
using output_type = dim3; // Outputs are the cluster centroids

using key_type = int; // Cluster that the point corresponds to
using value_type = Point;

// Pair type definition
struct pair_type
{
    key_type key;
    value_type value;

    // Printing for debugging
    friend std::ostream &operator<<(std::ostream &os, const pair_type &pair)
    {
        os << "Key: " << pair.key << ", Point: ";
        os << pair.value.values.x << " " << pair.value.values.y;
        os << "\n";
        return os;
    }
};

/*
    Comparision operator for comparing between 2 KeyValuePairs
    Returns true if first pair has key less than the second
*/
struct KeyValueCompare
{
    __host__ __device__ bool operator()(const pair_type &lhs, const pair_type &rhs)
    {
        return lhs.key < rhs.key;
    }
};

// const uint64_cu TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;
const unsigned long long TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

void runMapReduce(const input_type *input, output_type *output);

#endif // MAP_REDUCE_CUH
