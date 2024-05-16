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
int NUM_OUTPUT = 10;

// No. of values in each line (Size of datapoint)
const int DIMENSION = 2;
// No. of iterations
const int ITERATIONS = 2;
const int MAX_WORD_SIZE = 10;
const int MAX_INPUT_SIZE = 101;

struct Vector2D
{
    char values[DIMENSION * MAX_WORD_SIZE]; // Single character array
    int len[DIMENSION];                     // Length of each word

    // Override << operator to print based on the variable DIMENSION
    friend std::ostream &operator<<(std::ostream &os, const Vector2D &vector)
    {
        os << "(";
        for (int i = 0; i < DIMENSION; i++)
        {
            for (int j = 0; j < MAX_WORD_SIZE; j++)
            {
                char currentChar = vector.values[i * MAX_WORD_SIZE + j];
                if (currentChar != '\0')
                    os << currentChar;
                else
                    break;
            }
            if (i != DIMENSION - 1)
            {
                os << ", ";
            }
        }
        os << ")";

        return os;
    }
};
struct ReadVector
{
    std::string values[DIMENSION];
    // ovveride << operator to print based on the variable DIMENSION
    friend std::ostream &operator<<(std::ostream &os, const ReadVector &vector)
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
using Mykey = char;       // Cluster that the point corresponds to
using MyValue = Vector2D; // Point associated with the cluster

// Pair type definition
struct MyPair
{
    Mykey key[MAX_WORD_SIZE];
    MyValue value;

    // Printing for debugging
    friend std::ostream &operator<<(std::ostream &os, const MyPair &pair)
    {
        os << "Key: " << pair.key << ", Point: ";
        os << pair.value;
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
    __host__ __device__ bool myStrCmpLess(const char *str1, const char *str2)
    {
        while (*str1 && *str2 && *str1 == *str2)
        {
            ++str1;
            ++str2;
        }
        return *str1 < *str2; // Return true only if the left-hand side is strictly less than the right-hand side
    }
    __host__ __device__ bool operator()(const MyPair &lhs, const MyPair &rhs)
    {
        // return lhs.key < rhs.key;
        return myStrCmpLess(lhs.key, rhs.key);
    }
};
struct PairCompareLessEql
{
    __host__ __device__ bool myStrCmpLessEqual(const char *str1, const char *str2)
    {
        while (*str1 && *str2 && *str1 == *str2)
        {
            ++str1;
            ++str2;
        }
        // If both strings are equal up to the end of one of them or both have ended,
        // return true (considered less or equal)
        if (*str1 == *str2)
            return true;
        // If one string has ended but the other hasn't, consider the one with the shorter length as less or equal
        if (!*str1 && *str2)
            return true;
        if (*str1 && !*str2)
            return false;
        // Otherwise, return the comparison result
        return *str1 <= *str2;
    }

    __host__ __device__ bool operator()(const MyPair &lhs, const MyPair &rhs)
    {
        // return lhs.key <= rhs.key;
        return myStrCmpLessEqual(lhs.key, rhs.key);
    }
};
struct PairCompareGreater
{
    __host__ __device__ bool myStrCmpGreater(const char *str1, const char *str2)
    {
        while (*str1 && *str2 && *str1 == *str2)
        {
            ++str1;
            ++str2;
        }
        // If both strings are equal up to the end of one of them or both have ended,
        // return false (not considered greater)
        if (*str1 == *str2)
            return false;
        // If one string has ended but the other hasn't, consider the one with the longer length as greater
        if (!*str1 && *str2)
            return false;
        if (*str1 && !*str2)
            return true;
        // Otherwise, return the comparison result
        return *str1 > *str2;
    }

    __host__ __device__ bool operator()(const MyPair &lhs, const MyPair &rhs)
    {
        // return lhs.key > rhs.key;
        return myStrCmpGreater(lhs.key, rhs.key);
    }
};

struct ShuffleAndSort_KeyPairOutput
{
    Mykey key[MAX_WORD_SIZE];
    MyValue values[MAX_INPUT_SIZE];
    int size = 0;
    // ovveride << operator to print
    friend std::ostream &operator<<(std::ostream &os, const ShuffleAndSort_KeyPairOutput &pair)
    {
        os << pair.key << ": [";
        for (int i = 0; i < pair.size; i++)
        {
            os << pair.values[i] << " ";
        }
        os << "]";
        return os;
    }
};
unsigned long long TOTAL_PAIRS;
__device__ __host__ int charPtrToInt(const char *str, int len)
{
    int val = 0;
    for (int i = 0; i < len; i++)
    {
        // printf("Char: %c\n", str[i]);
        val = val * 10 + (str[i] - '0');
    }
    return val;
}
__device__ void intToCharPtr(int val, int &len, char *str)
{
    int temp = val;
    int digits = 0;
    while (temp != 0)
    {
        temp /= 10;
        digits++;
    }
    len = digits;
    int index = digits - 1;
    if (val == 0)
    {
        str[0] = '0';
        str[1] = '\0';

        return;
    }
    while (val != 0)
    {
        str[index] = (val % 10) + '0';
        val /= 10;
        index--;
    }
    str[digits] = '\0';
}
__device__ __host__ unsigned long long
distance(const Vector2D &p1, const Vector2D &p2)
{
    unsigned long long dist = 0;
    for (int i = 0; i < DIMENSION; i++)
    {
        int p1_size = p1.len[i];
        int p2_size = p2.len[i];
        // convert char* to int
        // printf("P1: %d, P2: %d\n", p1_size, p2_size);
        int p1_val = charPtrToInt(&p1.values[i * MAX_WORD_SIZE], p1_size);
        int p2_val = charPtrToInt(&p2.values[i * MAX_WORD_SIZE], p2_size);
        // printf("P1: %d, P2: %d\n", p1_val, p2_val);

        int temp = p1_val - p2_val;
        dist += temp * temp;
    }

    return dist;
}

__device__ void mapper(const input_type *input, MyPair *pairs, output_type *output, int *NUM_OUTPUT_D)
{

    // Find centroid with min distance from the current point
    unsigned long long min_distance = ULLONG_MAX;
    int cluster_id = -1;

    for (int i = 0; i < *NUM_OUTPUT_D; i++)
    {
        // printf("Output: %d\n", output[i].len[0]);
        // printf("Input: %d\n", input->len[0]);
        unsigned long long dist = distance(*input, output[i]);
        if (dist < min_distance)
        {
            min_distance = dist;
            cluster_id = i;
        }
    }
    int lenClusterId;
    intToCharPtr(cluster_id, lenClusterId, (pairs->key));

    // pairs->key = clusterIdStr;
    // pairs->value = *input;
    for (int i = 0; i < DIMENSION; i++)
    {
        for (int j = 0; j < MAX_WORD_SIZE; j++)
        {
            pairs->value.values[i * MAX_WORD_SIZE + j] = input->values[i * MAX_WORD_SIZE + j];
        }
        pairs->value.len[i] = input->len[i];
    }
    // printf("Key: %s\n", pairs->key);
}

/*
    Reducer to convert Key-Value pairs to desired output
    `len` number of pairs can be read starting from pairs, and output is stored in memory
*/
__device__ void reducer(ShuffleAndSort_KeyPairOutput *pairs, output_type *output)
{
    // Find new centroid
    int new_values[DIMENSION];
    for (int i = 0; i < DIMENSION; i++)
        new_values[i] = 0;
    int values_len = pairs->size;
    for (int i = 0; i < values_len; i++)
    {
        for (int j = 0; j < DIMENSION; j++)
        {
            int p_size = pairs->values[i].len[j];
            int p_val = charPtrToInt(&pairs->values[i].values[j * MAX_WORD_SIZE], p_size);
            new_values[j] += p_val;
        }
    }

    // // Take the key of any pair
    // char *key = pairs[0].key;
    // // find key size
    // int key_size = 0;
    // while (key[key_size] != '\0')
    // {
    //     key_size++;
    // }

    // // int cluster_idx = pairs[0].key;
    // int cluster_idx = charPtrToInt(key, key_size);
    // printf("Cluster Idx: %d\n", cluster_idx);

    for (int i = 0; i < DIMENSION; i++)
    {
        new_values[i] /= values_len;
        intToCharPtr(new_values[i], output->len[i], &output->values[i * MAX_WORD_SIZE]);
        // printf("Output: %s\n", output[cluster_idx].values[i * MAX_WORD_SIZE]);
    }
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