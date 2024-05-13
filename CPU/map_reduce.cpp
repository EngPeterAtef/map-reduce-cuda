#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
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

    std::string line;

    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            std::istringstream iss(line);
            input_type input;
            for (int i = 0; i < DIMENSION && (iss >> input.values[i]); ++i)
            {
                // Read DIMENSION values from the line
            }
            data.push_back(input); // Add the row to the data vector
        }
    }

    // Close the file
    file.close();
    // Print the data
    // int inputNum = (int)data.size();
    // for (int i = 0; i < inputNum; i++)
    // {
    //     for (int j = 0; j < DIMENSION; j++)
    //     {
    //         std::cout << data[i].values[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
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
// Define a struct to hold all parameters
struct MapParams
{
    std::vector<input_type> *data;
    std::vector<output_type> *output;
    std::vector<MyPair> *pairs;
    int start;
    int end;
};
void *map(void *args)
{
    // Cast the void pointer to the struct type
    MapParams *params = static_cast<MapParams *>(args);

    // Access the parameters
    std::vector<input_type> *data = params->data;
    std::vector<output_type> *output = params->output;
    std::vector<MyPair> *pairs = params->pairs;
    int start = params->start;
    int end = params->end;

    for (int i = start; i < end; i++)
    {
        // Call the mapper function
        // std::cout << "Mapping " << i << std::endl;
        mapper(&data->at(i), &pairs->at(i), *output);
    }

    return NULL;
}
struct ReduceParams
{
    std::vector<output_type> *output;
    std::vector<ShuffleAndSort_KeyPairOutput> *pairs;
    int start;
    int end;
};
void *reduce(void *args)
{
    // Cast the void pointer to the struct type
    ReduceParams *params = static_cast<ReduceParams *>(args);

    // Access the parameters
    std::vector<output_type> *output = params->output;
    std::vector<ShuffleAndSort_KeyPairOutput> *pairs = params->pairs;
    int start = params->start;
    int end = params->end;

    for (int i = start; i < end; i++)
    {
        // Call the reducer function
        // std::cout << "Reducing " << i << std::endl;
        reducer(&pairs->at(i), *output);
    }
    // reducer(&slicedVec, pair, output);

    return NULL;
}
void initialize(const std::vector<input_type> &input, std::vector<output_type> &output)
{
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

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        printf("Requires 1 argument, name of input textfile\n");
        exit(1);
    }

    std::string filename = argv[1];
    std::cout << "==========================================" << std::endl;
    std::cout << "Reading data from file: " << filename << std::endl;

    // =================================================
    // ================== Read data ====================
    // =================================================
    auto t_before_read = std::chrono::steady_clock::now();
    std::vector<input_type> data;
    read_data(data, filename);
    int inputNum = (int)data.size();
    std::cout << "==========================================" << std::endl;
    std::cout << "Number of input elements: " << inputNum << std::endl;
    NUM_INPUT = inputNum;
    TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

    // size_t output_size = NUM_OUTPUT * sizeof(output_type);
    // output_type *output = (output_type *)malloc(output_size);
    // use vectors to store output
    std::vector<output_type> output(NUM_OUTPUT);

    // copy from vector to array
    // for (int i = 0; i < inputNum; i++)
    // {
    //     for (int j = 0; j < DIMENSION; j++)
    //     {
    //         input[i].values[j] = data[i].values[j];
    //     }
    // }
    // free vector
    // data.clear();
    auto t_after_read = std::chrono::steady_clock::now();
    // choose initial centroids
    // std::cout << "==========================================" << std::endl;
    // std::cout << "Initializing centroids" << std::endl;
    initialize(data, output);
    // =================================================
    // ============= Setup Multi-threading =============
    // =================================================

    for (int k = 0; k < ITERATIONS; k++)
    {
        // std::cout << "==========================================" << std::endl;
        // std::cout << "Iteration: " << k << std::endl;

        // =================================================
        // ==================== Map ========================
        // =================================================

        pthread_t mappers[num_mappers];
        std::vector<MyPair> map_pairs(inputNum);
        // std::cout << "==========================================" << std::endl;
        // Create mappers
        for (int i = 0; i < num_mappers; i++)
        {
            int start = i * (inputNum / num_mappers);
            int end = (i + 1) * (inputNum / num_mappers);
            if (i == num_mappers - 1)
            {
                end = inputNum;
            }
            // std::cout << "Mapper " << i << " start: " << start << " end: " << end << std::endl;
            // Create a struct instance with all parameters
            MapParams params;
            params.data = &data;
            params.output = &output;
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
            // std::cout << "Mapper " << i << " finished" << std::endl;
        }
        // std::cout << "Mapping done" << std::endl;
        // print pairs
        // for (int i = 0; i < inputNum; i++)
        // {
        //     std::cout << map_pairs[i];
        // }

        // =================================================
        // ===================== Sort ======================
        // =================================================
        // std::cout << "==========================================" << std::endl;
        // std::cout << "Sorting pairs" << std::endl;
        sort(map_pairs.begin(), map_pairs.end(), PairCompare());
        // for (int i = inputNum; i >= 0; i--)
        // {
        //     std::cout << map_pairs[i];
        // }

        // =================================================
        // ============= Combine unique values =============
        // =================================================
        // std::cout << "==========================================" << std::endl;
        // std::cout << "Combining unique values" << std::endl;
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
        // for (int i = 0; i < shuffle_output.size(); i++)
        // {
        //     std::cout << "Key: " << shuffle_output[i].key << ", Values: ";
        //     for (int j = 0; j < 2; j++)
        //     {
        //         std::cout << "(" << shuffle_output[i].values[j].values[0] << ", " << shuffle_output[i].values[j].values[1] << ")"
        //                   << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // =================================================
        // =================== Reduce ======================
        // =================================================
        // std::cout << "==========================================" << std::endl;
        if (num_reducers > (int)shuffle_output.size())
        {
            num_reducers = (int)shuffle_output.size();
        }
        pthread_t reducers[num_reducers];
        // Create reducers
        for (int i = 0; i < num_reducers; i++)
        {
            int start = i * (shuffle_output.size() / num_reducers);
            int end = (i + 1) * (shuffle_output.size() / num_reducers);
            if (i == num_reducers - 1)
            {
                end = shuffle_output.size();
            }
            // std::cout << "Reducer " << i << " start: " << start << " end: " << end << std::endl;
            // Create a struct instance with all parameters
            ReduceParams params;
            params.output = &output;
            params.pairs = &shuffle_output;
            params.start = start;
            params.end = end;
            // Create a thread for each reducer
            pthread_create(&reducers[i], NULL, reduce, (void *)&params);
        }
        // Wait for all reducers to finish
        for (int i = 0; i < num_reducers; i++)
        {
            pthread_join(reducers[i], NULL);
            // std::cout << "Reducer " << i << " finished" << std::endl;
        }
        // std::cout << "Reducing done" << std::endl;
    }

    // =================================================
    // =================== Output ======================
    // =================================================
    std::cout << "==========================================" << std::endl;
    std::cout << "Final centroids: " << std::endl;
    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        std::cout << output[i].values[0] << " " << output[i].values[1] << std::endl;
    }
    auto t_final = std::chrono::steady_clock::now();
    // save output to file
    std::ofstream output_file;
    output_file.open(filename + "_output.txt");
    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        output_file << output[i].values[0] << " " << output[i].values[1] << std::endl;
    }
    output_file.close();

    std::cout << "==========================================" << std::endl;
    std::cout << "Map threads: " << num_mappers << ", Reduce threads: " << num_reducers << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Time taken to read data: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_after_read - t_before_read).count() << " ms" << std::endl;
    std::cout << "Time taken for map-reduce: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_final - t_after_read).count() << " ms" << std::endl;
    // // Free host memory
    // std::cout << "==========================================" << std::endl;
    // std::cout << "Freeing memory" << std::endl;
    // free(input);
    // free(output);

    return 0;
}