#ifndef WORDCOUNT_CUH
#define WORDCOUNT_CUH
#include <iostream>

// GPU parameters
const int MAP_BLOCK_SIZE = 512;
const int REDUCE_BLOCK_SIZE = 32;
int MAP_GRID_SIZE;
int REDUCE_GRID_SIZE;

// No. of input elements (Lines in text file)
unsigned long long NUM_INPUT;
// No. of pairs per input element
const int NUM_PAIRS = 1;
// Total No. of output values (K - No. of clusters)
const int NUM_OUTPUT = 3;

// No. of values in each line (Size of datapoint)
const int DIMENSION = 1;
// No. of iterations
const int ITERATIONS = 1;

// Custom types
struct Vector2D
{
    std::string values[DIMENSION];
    // ovveride << operator to print based on the variable DIMENSION
    friend std::ostream &operator<<(std::ostream &os, const Vector2D &vector)
    {
        for (int i = 0; i < DIMENSION; i++)
        {
            os << vector.values[i] << " ";
        }
        return os;
    }
};
using Mykey = std::string;
using MyValue = int;
// Pair type definition
struct MyPair
{
    Mykey key;
    MyValue value;
    // ovveride << operator to print
    friend std::ostream &operator<<(std::ostream &os, const MyPair &pair)
    {
        os << pair.key << " " << pair.value;
        return os;
    }
};
using input_type = Vector2D;
using output_type = MyPair;

/*
    Comparision operator for comparing between 2 KeyValuePairs
    Returns true if first pair has key less than the second
*/
struct PairCompare
{
    __host__ __device__ bool operator()(const MyPair &lhs, const MyPair &rhs)
    {
        return lhs.key < rhs.key;
    }
};
struct PairCompare2
{
    __host__ __device__ bool operator()(const MyPair &lhs, const MyPair &rhs)
    {
        return lhs.key <= rhs.key;
    }
};

unsigned long long TOTAL_PAIRS;

// functions definitions
extern __device__ void mapper(const input_type *input, MyPair *pairs, output_type *output);
extern __device__ void reducer(MyPair *pairs, size_t len, output_type *output);

__device__ void mapper(const input_type *input, MyPair *pairs, output_type *output)
{
    pairs->key = input->values[0];
    pairs->value = 1;
}

__device__ void reducer(MyPair *pairs, size_t len, output_type *output)
{

    // int new_value = 0;
    // int values_size = (int)pairs->values.size();
    // for (int i = 0; i < values_size; i++)
    // {
    //     new_value += pairs->values[i];
    // }
    // MyPair output_pair;
    // output_pair.key = pairs[0].key;
    // output_pair.value = new_value;
    // output.push_back(output_pair);
}

void initialize(input_type *input, output_type *output)
{
}
#endif // WORDCOUNT_CUH