#ifndef WORDCOUNT_HPP
#define WORDCOUNT_HPP
#include <iostream>
#include <string>
#include <climits>
#include <vector>

int map_num_threads = 6;
int reduce_num_threads = 6;
// No. of input elements (Lines in text file)
unsigned long long NUM_INPUT;

// No. of values in each line (Size of datapoint)
const int DIMENSION = 2;
// No. of iterations
const int ITERATIONS = 1;
unsigned long long TOTAL_PAIRS;

using Mykey = std::string;
using MyValue = int;
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
using input_type = Vector2D;
using output_type = MyPair;

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