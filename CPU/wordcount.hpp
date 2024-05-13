#ifndef WORDCOUNT_HPP
#define WORDCOUNT_HPP
#include <iostream>
#include <string>
#include <climits>
#include <vector>

int num_mappers = 6;
int num_reducers = 6;
// No. of input elements (Lines in text file)
unsigned long long NUM_INPUT;
// No. of pairs per input element

// No. of values in each line (Size of datapoint)
const int DIMENSION = 2;
// No. of iterations
const int ITERATIONS = 1;
unsigned long long TOTAL_PAIRS;

// So each point will get associated with a cluster (with id -> key)
using Mykey = std::string; // Cluster that the point corresponds to
using MyValue = int;       // Point associated with the cluster
// Custom types
struct Vector2D
{
    std::string values[DIMENSION];
    // ovveride << operator to print based on the variable DIMENSION
};
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
struct ShuffleAndSort_KeyPairOutput
{
    Mykey key;
    std::vector<MyValue> values;
    // ovveride << operator to print
    friend std::ostream &operator<<(std::ostream &os, const ShuffleAndSort_KeyPairOutput &pair)
    {
        os << pair.key << ": [";
        for (int i = 0; i < pair.values.size(); i++)
        {
            os << pair.values[i] << " ";
        }
        os << "]";
        return os;
    }
};
// Type declarations for input, output & key-value pairs
using input_type = Vector2D; // Datapoint (or vector) read from the text file
using output_type = MyPair;  // Outputs are the cluster centroids

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

void initialize(const std::vector<input_type> &input, std::vector<output_type> &output)
{
}

void mapper(const input_type *input, MyPair *pairs, const std::vector<output_type> &output)
{
    pairs->key = input->values[0];
    pairs->value = 1;
}

void reducer(ShuffleAndSort_KeyPairOutput *pairs, std::vector<output_type> &output)
{
    int new_value = 0;
    int values_size = (int)pairs->values.size();
    for (int i = 0; i < values_size; i++)
    {
        new_value += pairs->values[i];
    }
    MyPair output_pair;
    output_pair.key = pairs[0].key;
    output_pair.value = new_value;
    output.push_back(output_pair);
}
#endif // WORDCOUNT_HPP