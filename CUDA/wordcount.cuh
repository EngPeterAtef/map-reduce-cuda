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
int NUM_OUTPUT = 0;

// No. of values in each line (Size of datapoint)
const int DIMENSION = 1;
// No. of iterations
const int ITERATIONS = 1;
const int MAX_WORD_SIZE = 10;
const int MAX_INPUT_SIZE = 1000;

struct Vector2D
{
    char values[DIMENSION * MAX_WORD_SIZE]; // Single character array
    int len[DIMENSION];                     // Length of each word

    // Override << operator to print based on the variable DIMENSION
    friend std::ostream &operator<<(std::ostream &os, const Vector2D &vector)
    {
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
            os << " ";
        }
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
using input_type = Vector2D; // Datapoint (or vector) read from the text file

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
        os << "Key: " << pair.key << ", Value: ";
        os << pair.value;
        os << "\n";
        return os;
    }
};
struct MyOutputPair
{
    Mykey key[MAX_WORD_SIZE];
    Mykey value[MAX_WORD_SIZE];

    // Printing for debugging
    friend std::ostream &operator<<(std::ostream &os, const MyOutputPair &pair)
    {
        os << "Key: ";
        for (int i = 0; i < MAX_WORD_SIZE; i++)
        {
            char currentChar = pair.key[i];
            if (currentChar == '\0')
                break;
            os << currentChar;
        }
        os << ", ";
        os << "Value: ";
        for (int i = 0; i < MAX_WORD_SIZE; i++)
        {
            char currentChar = pair.value[i]; // Access value instead of key
            if (currentChar == '\0')
                break;
            os << currentChar;
        }

        return os;
    }
};
using output_type = MyOutputPair; // Outputs are the cluster centroids

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

unsigned long long TOTAL_PAIRS;
__device__ __host__ int charPtrToInt(const char *str, int len)
{
    int val = 0;
    // printf("inside func");
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
    // char *str = (char *)malloc((digits + 1) * sizeof(char));
    // char str[MAX_WORD_SIZE];
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
    // return str;
}

__device__ void mapper(const input_type *input, MyPair *pairs, output_type *output, int *NUM_OUTPUT_D)
{
    // copy input key to pair key
    for (int i = 0; i < DIMENSION; i++)
    {
        for (int j = 0; j < MAX_WORD_SIZE; j++)
        {
            pairs->key[i * MAX_WORD_SIZE + j] = input->values[i * MAX_WORD_SIZE + j];
        }
    }
    // pairs->key = input->values[0];
    // pairs->value = 1;
    pairs->value.values[0] = '1';
    pairs->value.len[0] = 1;
}

__device__ void reducer(ShuffleAndSort_KeyPairOutput *pairs, output_type *output)
{

    int values_size = pairs->size;
    for (int i = 0; i < MAX_WORD_SIZE; i++)
    {
        output->key[i] = pairs->key[i];
    }
    int len;
    intToCharPtr(values_size, len, output->value);
    // printf("Key: %s, Value: %s, idx %d\n", output->key, output->value, outputIdx);
}

void initialize(input_type *input, output_type *output)
{
}
#endif // WORDCOUNT_CUH