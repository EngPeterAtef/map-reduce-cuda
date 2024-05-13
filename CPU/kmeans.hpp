#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <iostream>
#include <string>
#include <climits>
#include <vector>

#include "random_generator.hpp"

int num_mappers = 12;
int num_reducers = 6;
// No. of input elements (Lines in text file)
unsigned long long NUM_INPUT;

// Number of output clusters K
const int NUM_OUTPUT = 10;

// No. of values in each line (Size of datapoint)
const int DIMENSION = 2;
// No. of iterations
const int ITERATIONS = 1000;
unsigned long long TOTAL_PAIRS;

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
void initialize(const std::vector<input_type> &input, std::vector<output_type> &output)
{
    // append the first 10 points to the output
    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        output_type temp;
        output.push_back(temp);
    }
    // Uniform Number generator for random datapoints
    UniformDistribution distribution(NUM_INPUT);

    // Now chose initial centroids
    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        int sample = distribution.sample();
        // std::cout << "inside loop " << i << std::endl;
        // std::cout << "Sample: " << sample << std::endl;
        // std::cout << "input Sample: " << input[sample].values[1] << std::endl;

        // output[i] = input[sample];
        for (int j = 0; j < DIMENSION; j++)
        {
            output[i].values[j] = input[sample].values[j];
            // std::cout << output[i].values[j] << " ";
        }
    }
}
unsigned long long distance(const Vector2D &p1, const Vector2D &p2)
{
    unsigned long long dist = 0;
    for (int i = 0; i < DIMENSION; i++)
    {
        // std::cout << "P1: " << p1.values[i] << " P2: " << p2.values[i] << std::endl;
        int p1_val = std::stoi(p1.values[i]);
        int p2_val = std::stoi(p2.values[i]);
        int temp = p1_val - p2_val;
        dist += temp * temp;
    }

    return dist;
}
void mapper(const input_type *input, MyPair *pairs, const std::vector<output_type> &output)
{
    // std::cout << "Mapping first" << std::endl;
    // Find centroid with min distance from the current point
    unsigned long long min_distance = ULLONG_MAX;
    // std::cout << "long long" << std::endl;
    int cluster_id = -1;

    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        // std::cout << "before dis" << std::endl;
        // std::cout << "Input: " << input->values[0] << " " << input->values[1] << std::endl;
        // std::cout << "Output: " << output[i].values[0] << " " << output[i].values[1] << std::endl;
        // std::cout << "Output: " << output->at(1).values[0] << " " << output->at(1).values[1] << std::endl;
        unsigned long long dist = distance(*input, output[i]);
        // std::cout << "Distance: " << dist << std::endl;
        if (dist < min_distance)
        {
            min_distance = dist;
            cluster_id = i;
        }
    }
    // std::cout << "Cluster ID: " << cluster_id << std::endl;
    pairs->key = cluster_id;
    pairs->value = *input;
    if (cluster_id == -1)
    {
        std::cout << "Error: Cluster ID not found" << std::endl;
    }
    // if (cluster_id == 0)
    // {
    //     std::cout << "Input: " << input->values[0] << " " << input->values[1] << std::endl;
    //     std::cout << "Output: " << output[0].values[0] << " " << output[0].values[1] << std::endl;
    //     std::cout << "Distance: " << min_distance << std::endl;
    // }
}

void reducer(ShuffleAndSort_KeyPairOutput *pairs, std::vector<output_type> &output)
{
    // printf("Key: %d, Length: %llu\n", pairs[0].key, len);

    // Find new centroid
    int new_values[DIMENSION];
    for (int i = 0; i < DIMENSION; i++)
    {
        new_values[i] = 0;
    }
    int values_size = (int)pairs->values.size();
    for (int i = 0; i < values_size; i++)
    {
        for (int j = 0; j < DIMENSION; j++)
        {
            // std::cout << "Value: " << pairs->values[i].values[j] << std::endl;
            int val_int = 0;
            try
            {
                val_int = std::stoi(pairs->values[i].values[j]);
            }
            catch (const std::exception &e)
            {
                // std::cerr << e.what() << '\n';
            }

            new_values[j] += val_int;
        }
    }

    // uint64_cu diff = 0;

    // Take the key of any pair
    int cluster_idx = pairs[0].key;
    for (int i = 0; i < DIMENSION; i++)
    {
        new_values[i] /= values_size;

        // diff += abs((int)new_values[i] - output[cluster_idx].values[i]);
        // convert to string
        std::string new_val = std::to_string(new_values[i]);
        output[cluster_idx].values[i] = new_val;
    }

    // printf("Key: %d, Diff: %llu\n", cluster_idx, diff);
}
#endif // KMEANS_HPP