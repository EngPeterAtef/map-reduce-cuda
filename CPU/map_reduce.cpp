#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <climits>
#include <pthread.h>
#include <vector>
#include <algorithm>

// #include "kmeans.hpp"
#include "wordcount.hpp"

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
    // use vectors to store output
    std::vector<output_type> *output = new std::vector<output_type>;
    auto t_after_read = std::chrono::steady_clock::now();
    // initial
    // std::cout << "==========================================" << std::endl;
    initialize(data, *output);
    // =================================================
    // ============= Setup Multi-threading =============
    // =================================================
    int64_t t_map;
    int64_t t_sort;
    int64_t t_shuffle;
    int64_t t_reduce;

    for (int k = 0; k < ITERATIONS; k++)
    {
        // std::cout << "==========================================" << std::endl;
        // std::cout << "Iteration: " << k << std::endl;

        // =================================================
        // ==================== Map ========================
        // =================================================
        auto t_before_map = std::chrono::steady_clock::now();

        pthread_t mappers[map_num_threads];
        std::vector<MyPair> map_pairs(inputNum);
        std::vector<MapParams *> map_params_list(map_num_threads); // Store pointers to dynamically allocated MapParams instances

        // std::cout << "==========================================" << std::endl;
        // Create mappers
        for (int i = 0; i < map_num_threads; i++)
        {
            int start = i * (inputNum / map_num_threads);
            int end = (i + 1) * (inputNum / map_num_threads);
            if (i == map_num_threads - 1)
            {
                end = inputNum;
            }
            // std::cout << "Mapper " << i << " start: " << start << " end: " << end << std::endl;
            // Create a struct instance with all parameters
            // Dynamically allocate memory for each MapParams instance
            MapParams *mapParams = new MapParams;
            mapParams->data = &data;
            mapParams->output = output;
            mapParams->pairs = &map_pairs;
            mapParams->start = start;
            mapParams->end = end;
            map_params_list[i] = mapParams;
            // Create a thread for each mapper
            pthread_create(&mappers[i], NULL, map, (void *)mapParams);
        }
        // Wait for all mappers to finish
        for (int i = 0; i < map_num_threads; i++)
        {
            pthread_join(mappers[i], NULL);
            delete map_params_list[i];

            // std::cout << "Mapper " << i << " finished" << std::endl;
        }
        // std::cout << "Mapping done" << std::endl;
        // print pairs
        // for (int i = 0; i < inputNum; i++)
        // {
        //     std::cout << map_pairs[i];
        // }
        auto t_after_map = std::chrono::steady_clock::now();
        if (k == 0)
        {
            t_map = std::chrono::duration_cast<std::chrono::microseconds>(t_after_map - t_before_map).count();
        }
        else
        {
            t_map += std::chrono::duration_cast<std::chrono::microseconds>(t_after_map - t_before_map).count();
        }

        // =================================================
        // ===================== Sort ======================
        // =================================================
        // std::cout << "==========================================" << std::endl;
        // for (int i = 0; i < map_pairs.size(); i++)
        // {
        //     std::cout << map_pairs[i] << std::endl;
        // }
        // std::cout << "Sorting pairs" << std::endl;
        auto t_before_sort = std::chrono::steady_clock::now();
        sort(map_pairs.begin(), map_pairs.end(), PairCompare());
        auto t_after_sort = std::chrono::steady_clock::now();
        if (k == 0)
        {
            t_sort = std::chrono::duration_cast<std::chrono::microseconds>(t_after_sort - t_before_sort).count();
        }
        else
        {
            t_sort += std::chrono::duration_cast<std::chrono::microseconds>(t_after_sort - t_before_sort).count();
        }
        // for (int i = 0; i < map_pairs.size(); i++)
        // {
        //     std::cout << map_pairs[i] << std::endl;
        // }

        // =================================================
        // ============= Combine unique values =============
        // =================================================
        // std::cout << "==========================================" << std::endl;
        // std::cout << "Combining unique values" << std::endl;
        auto t_before_shuffle = std::chrono::steady_clock::now();
        std::vector<ShuffleAndSort_KeyPairOutput> *shuffle_output = new std::vector<ShuffleAndSort_KeyPairOutput>;
        for (int i = 0; i < inputNum; i++)
        {
            if (i == 0 || map_pairs[i].key != map_pairs[i - 1].key)
            {
                ShuffleAndSort_KeyPairOutput current_pair;
                current_pair.key = map_pairs[i].key;
                current_pair.values.push_back(map_pairs[i].value);
                shuffle_output->push_back(current_pair);
            }
            else
            {
                shuffle_output->back().values.push_back(map_pairs[i].value);
            }
        }
        auto t_after_shuffle = std::chrono::steady_clock::now();
        if (k == 0)
        {
            t_shuffle = std::chrono::duration_cast<std::chrono::microseconds>(t_after_shuffle - t_before_shuffle).count();
        }
        else
        {
            t_shuffle += std::chrono::duration_cast<std::chrono::microseconds>(t_after_shuffle - t_before_shuffle).count();
        }
        // // Add the last pair to the output vector
        // shuffle_output.push_back(current_pair);
        // print pairs
        // std::cout << "Shuffle output: " << std::endl;
        // for (int i = 0; i < shuffle_output->size(); i++)
        // {
        //     std::cout << (*shuffle_output)[i] << std::endl;
        // }

        // =================================================
        // =================== Reduce ======================
        // =================================================
        // std::cout << "==========================================" << std::endl;
        auto t_before_reduce = std::chrono::steady_clock::now();
        if (reduce_num_threads > (int)shuffle_output->size())
        {
            reduce_num_threads = (int)shuffle_output->size();
        }
        pthread_t reducers[reduce_num_threads];
        std::vector<ReduceParams *> reduce_params_list(reduce_num_threads); // Store pointers to dynamically allocated MapParams instances

        // Create reducers
        for (int i = 0; i < reduce_num_threads; i++)
        {
            int start = i * (shuffle_output->size() / reduce_num_threads);
            int end = (i + 1) * (shuffle_output->size() / reduce_num_threads);
            if (i == reduce_num_threads - 1)
            {
                end = shuffle_output->size();
            }
            // std::cout << "Reducer " << i << " start: " << start << " end: " << end << std::endl;
            // Create a struct instance with all parameters
            ReduceParams *reduceParams = new ReduceParams;
            reduceParams->output = output;
            reduceParams->pairs = shuffle_output;
            reduceParams->start = start;
            reduceParams->end = end;
            reduce_params_list[i] = reduceParams;
            // Create a thread for each reducer
            pthread_create(&reducers[i], NULL, reduce, (void *)reduceParams);
        }
        // Wait for all reducers to finish
        for (int i = 0; i < reduce_num_threads; i++)
        {
            pthread_join(reducers[i], NULL);
            delete reduce_params_list[i];

            // std::cout << "Reducer " << i << " finished" << std::endl;
        }
        auto t_after_reduce = std::chrono::steady_clock::now();
        if (k == 0)
        {
            t_reduce = std::chrono::duration_cast<std::chrono::microseconds>(t_after_reduce - t_before_reduce).count();
        }
        else
        {
            t_reduce += std::chrono::duration_cast<std::chrono::microseconds>(t_after_reduce - t_before_reduce).count();
        }
        delete shuffle_output;

        // std::cout << "Reducing done" << std::endl;
    }

    // =================================================
    // =================== Output ======================
    // =================================================
    std::cout << "==========================================" << std::endl;
    std::cout << "Final output: " << std::endl;
    int output_size = (int)(*output).size();
    for (int i = 0; i < output_size; i++)
    {
        std::cout << (*output)[i] << std::endl;
    }
    auto t_final = std::chrono::steady_clock::now();
    // save output to file
    std::ofstream output_file;
    output_file.open(filename + "_output.txt");
    for (int i = 0; i < output_size; i++)
    {
        output_file << (*output)[i] << std::endl;
    }
    output_file.close();

    std::cout << "==========================================" << std::endl;
    std::cout << "Map threads: " << map_num_threads << ", Reduce threads: " << reduce_num_threads << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Time taken to read data: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_after_read - t_before_read).count() << " ms" << std::endl;
    std::cout << "Time taken for map-reduce: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_final - t_after_read).count() << " ms" << std::endl;
    std::cout << "Time taken for map: " << t_map << "us" << std::endl;
    std::cout << "Time taken for sort: " << t_sort << "us" << std::endl;
    std::cout << "Time taken for shuffle: " << t_shuffle << "us" << std::endl;
    std::cout << "Time taken for reduce: " << t_reduce << "us" << std::endl;
    delete output;
    return 0;
}