#ifndef KMEANS_CUH
#define KMEANS_CUH
#include <iostream>
#include "random_generator.hpp"

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
const int DIMENSION = 2;
// No. of iterations
const int ITERATIONS = 10;

// Custom types
struct Vector2D
{
    int values[DIMENSION];
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
struct PairCompareLessEql
{
    __host__ __device__ bool operator()(const MyPair &lhs, const MyPair &rhs)
    {
        return lhs.key <= rhs.key;
    }
};

unsigned long long TOTAL_PAIRS;

__device__ __host__ unsigned long long
distance(const Vector2D &p1, const Vector2D &p2)
{
    unsigned long long dist = 0;
    for (int i = 0; i < DIMENSION; i++)
    {
        int temp = p1.values[i] - p2.values[i];
        dist += temp * temp;
    }

    return dist;
}

// functions definitions
extern __device__ void mapper(const input_type *input, MyPair *pairs, output_type *output);
extern __device__ void reducer(MyPair *pairs, size_t len, output_type *output);

__device__ void mapper(const input_type *input, MyPair *pairs, output_type *output)
{
    // Find centroid with min distance from the current point
    unsigned long long min_distance = ULLONG_MAX;
    int cluster_id = -1;

    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        unsigned long long dist = distance(*input, output[i]);
        if (dist < min_distance)
        {
            min_distance = dist;
            cluster_id = i;
        }
    }

    pairs->key = cluster_id;
    pairs->value = *input;
    // printf("Key: %d, Point: %d %d\n", pairs->key, pairs->value.values[0], pairs->value.values[1]);
}

/*
    Reducer to convert Key-Value pairs to desired output
    `len` number of pairs can be read starting from pairs, and output is stored in memory
*/
__device__ void reducer(MyPair *pairs, size_t len, output_type *output)
{
    // printf("Key: %d, Length: %llu\n", pairs[0].key, len);

    // Find new centroid
    unsigned long long new_values[DIMENSION]; // unsigned long long to avoid overflow
    for (int i = 0; i < DIMENSION; i++)
        new_values[i] = 0;

    for (size_t i = 0; i < len; i++)
    {
        for (int j = 0; j < DIMENSION; j++)
            new_values[j] += pairs[i].value.values[j]; // Wow, this is bad naming
    }

    // unsigned long long diff = 0;

    // Take the key of any pair
    int cluster_idx = pairs[0].key;
    for (int i = 0; i < DIMENSION; i++)
    {
        new_values[i] /= len;

        // diff += abs((int)new_values[i] - output[cluster_idx].values[i]);
        output[cluster_idx].values[i] = new_values[i];
    }

    // printf("Key: %d, Diff: %llu\n", cluster_idx, diff);
}

/*
    Initialize according to normal KMeans
    Choose K random data points as initial centroids
*/
void initialize(input_type *input, output_type *output)
{
    // Uniform Number generator for random datapoints
    UniformDistribution distribution(NUM_INPUT);

    // Now chose initial centroids
    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        int sample = distribution.sample();
        output[i] = input[sample];
    }
}
#endif // KMEANS_CUH