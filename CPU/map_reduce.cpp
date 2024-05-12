#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <climits>
#include <pthread.h>
#include <vector>
#include <algorithm>
#include "config.hpp"
#include "random_generator.hpp"
void read_data(std::vector<input_type> &data, std::string filename)
{
    // Read data from text file
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cout << "Could not open file" << std::endl;
        return;
    }

    int num1, num2;

    // Read each line in the file
    while (file >> num1 >> num2)
    {
        // Create a vector to store the two numbers in the row
        input_type input;
        input.values[0] = num1;
        input.values[1] = num2;

        // Add the row to the data vector
        data.push_back(input);
    }

    // Close the file
    file.close();

    // for (int i = 0; i < inputNum; i++)
    // {
    //     std::cout << data[i][0] << " " << data[i][1] << std::endl;
    // }
}
unsigned long long distance(const Vector2D &p1, const Vector2D &p2)
{
    unsigned long long dist = 0;
    for (int i = 0; i < DIMENSION; i++)
    {
        int temp = p1.values[i] - p2.values[i];
        dist += temp * temp;
    }

    return dist;
}
void mapper(const input_type *input, MyPair *pairs, output_type *output)
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
}
// Define a struct to hold all parameters
struct MapParams
{
    input_type *data;
    output_type *output;
    std::vector<MyPair> *pairs;
    int start;
    int end;
};
void *map(void *args)
{
    // Cast the void pointer to the struct type
    MapParams *params = static_cast<MapParams *>(args);

    // Access the parameters
    input_type *data = params->data;
    output_type *output = params->output;
    std::vector<MyPair> *pairs = params->pairs;
    int start = params->start;
    int end = params->end;

    // // Define iterators for the slice
    // auto i = data.begin() + start; // Start index (inclusive)
    // auto j = data.begin() + end;   // End index (exclusive)

    // // Create a new vector containing the slice
    // std::vector<input_type> slicedVec(i, j);

    for (int i = start; i < end; i++)
    {
        // Call the mapper function
        // std::cout << "Mapping " << i << std::endl;
        mapper(&data[i], &pairs->at(i), output);
    }
    // mapper(&slicedVec, pair, output);

    return NULL;
}
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

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        printf("Requires 1 argument, name of input textfile\n");
        exit(1);
    }

    std::string filename = argv[1];
    // =================================================
    // ================== Read data ====================
    // =================================================
    std::vector<input_type> data;
    read_data(data, filename);
    int inputNum = (int)data.size();
    std::cout << "Number of input elements: " << inputNum << std::endl;
    NUM_INPUT = inputNum;
    TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

    // Allocate host memory
    size_t input_size = NUM_INPUT * sizeof(input_type);
    input_type *input = (input_type *)malloc(input_size);

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *)malloc(output_size);

    // copy from vector to array
    for (int i = 0; i < inputNum; i++)
    {
        input[i].values[0] = data[i].values[0];
        input[i].values[1] = data[i].values[1];
    }
    // choose initial centroids
    std::cout << "Initializing centroids" << std::endl;
    initialize(input, output);

    // =================================================
    // =================================================
    int num_mappers = 6;
    int num_reducers = 1;
    pthread_t mappers[num_mappers];
    pthread_t reducers[num_reducers];
    std::vector<MyPair> map_pairs(inputNum);

    // =================================================
    // ==================== Map ========================
    // =================================================
    // Create mappers
    for (int i = 0; i < num_mappers; i++)
    {
        int start = i * (inputNum / num_mappers);
        int end = (i + 1) * (inputNum / num_mappers);
        if (i == num_mappers - 1)
        {
            end = inputNum;
        }
        std::cout << "Mapper " << i << " start: " << start << " end: " << end << std::endl;
        // Create a struct instance with all parameters
        MapParams params;
        params.data = input;
        params.output = output;
        params.pairs = &map_pairs;
        params.start = start;
        params.end = end;
        // Create a thread for each mapper
        pthread_create(&mappers[i], NULL, map, (void *)&params);
    }
    // Wait for all mappers to finish
    for (int i = 0; i < num_mappers; i++)
    {
        pthread_join(mappers[i], NULL);
        std::cout << "Mapper " << i << " finished" << std::endl;
    }
    std::cout << "Mapping done" << std::endl;
    // print pairs
    // for (int i = 0; i < inputNum; i++)
    // {
    //     std::cout << map_pairs[i];
    // }

    // =================================================
    // ===================== Sort ======================
    // =================================================
    sort(map_pairs.begin(), map_pairs.end(), PairCompare());

    // =================================================
    // ============= Combine unique values =============
    // =================================================
    std::vector<ShuffleAndSort_KeyPairOutput> shuffle_output;
    for (int i = 0; i < inputNum; i++)
    {
        if (i == 0 || map_pairs[i].key != map_pairs[i - 1].key)
        {
            ShuffleAndSort_KeyPairOutput current_pair;
            current_pair.key = map_pairs[i].key;
            current_pair.values.push_back(map_pairs[i].value);
            shuffle_output.push_back(current_pair);
        }
        else
        {
            shuffle_output.back().values.push_back(map_pairs[i].value);
        }
    }
    // // Add the last pair to the output vector
    // shuffle_output.push_back(current_pair);
    // print pairs
    for (int i = 0; i < shuffle_output.size(); i++)
    {
        std::cout << "Key: " << shuffle_output[i].key << ", Values: ";
        for (int j = 0; j < 2; j++)
        {
            std::cout << shuffle_output[i].values[j].values[0] << " " << shuffle_output[i].values[j].values[1] << " ";
        }
        std::cout << std::endl;
    }

    // =================================================
    // =================== Reduce ======================
    // =================================================

    // =================================================
    // =================== Output ======================
    // =================================================

    // Free host memory
    std::cout << "Freeing memory" << std::endl;
    free(input);
    free(output);

    return 0;
}

// ===============================================================================
// ===============================================================================
// ===============================GPU IMPLEMENTATION==============================
// ===============================================================================
// ===============================================================================
