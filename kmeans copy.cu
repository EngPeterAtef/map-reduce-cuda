#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "config.cuh"
#include "random_generator.hpp"

const bool SAVE_TO_FILE = true;

__device__ __host__ unsigned long long
distance(const dim3 &p1, const dim3 &p2)
{
    unsigned long long dist = 0;
    int x_diff = p1.x - p2.x;
    int y_diff = p1.y - p2.y;
    dist = x_diff * x_diff + y_diff * y_diff;

    return dist;
}

/*
    Mapper function for each input element
    Input is already stored in memory, and output pairs must be stored in the memory allocated
    Muliple pairs can be generated for a single input, but their number shouldn't exceed NUM_PAIRS
*/

__device__ void mapper(const input_type *input, pair_type *pairs, output_type *output)
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
    pairs->value.values = *input;
}

/*
    Reducer to convert Key-Value pairs to desired output
    `len` number of pairs can be read starting from pairs, and output is stored in memory
*/
__device__ void reducer(pair_type *pairs, int len, output_type *output)
{
    // printf("Key: %d, Length: %llu\n", pairs[0].key, len);

    // Find new centroid
    dim3 new_values;
    new_values.x = 0;
    new_values.y = 0;

    for (int i = 0; i < len; i++)
    {
        new_values.x += pairs[i].value.values.x;
        new_values.y += pairs[i].value.values.y;
    }

    // unsigned long long diff = 0;

    // Take the key of any pair
    int cluster_idx = pairs[0].key;
    new_values.x /= len;
    new_values.y /= len;
    output[cluster_idx] = new_values;
    // printf("Key: %d, Diff: %llu\n", cluster_idx, diff);
}

/*
    Initialize according to normal KMeans
    Choose K random data points as initial centroids
*/
void initialize(dim3 *input, dim3 *output, int NUM_INPUT, int NUM_OUTPUT)
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

/*
    Main function that runs a map reduce job.
*/
int main(int argc, char *argv[])
{
    using millis = std::chrono::milliseconds;
    using std::string;
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    if (argc != 2)
    {
        printf("Requires 1 argument, name of input textfile\n");
        exit(1);
    }

    string filename = argv[1];

    auto t_seq_1 = steady_clock::now();

    // Read data from text file
    std::ifstream file(filename);
    std::vector<std::vector<int>> data; // Vector of vectors to store the data

    if (!file.is_open())
    {
        std::cout << "Could not open file" << std::endl;
        return 1;
    }

    int num1, num2;

    // Read each line in the file
    while (file >> num1 >> num2)
    {
        // Create a vector to store the two numbers in the row
        std::vector<int> row = {num1, num2};

        // Add the row to the data vector
        data.push_back(row);
    }

    // Close the file
    file.close();

    int inputNum = (int)data.size();
    // NUM_INPUT = inputNum;
    // TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

    // Allocate host memory
    int inSize = inputNum * sizeof(dim3);
    dim3 *input = (dim3 *)malloc(inSize);

    int outSize = NUM_OUTPUT * sizeof(dim3);
    dim3 *output = (dim3 *)malloc(outSize);

    // copy from vector to array
    for (int i = 0; i < inputNum; i++)
    {
        input[i].x = data[i][0];
        input[i].y = data[i][1];
    }
    // Print the data vector to verify the contents
    std::cout << "Data read from file:" << std::endl;
    for (int i = 0; i < inputNum; i++)
    {
        std::cout << input[i].x << " " << input[i].y << std::endl;
    }

    // Now chose initial centroids
    initialize(input, output, inputNum, NUM_OUTPUT);
    // pp_initialize(input, output);

    auto t_seq_2 = steady_clock::now();

    // Run the Map Reduce Job
    runMapReduce(input, output);

    // Save output if required
    std::ofstream output_file;
    if (SAVE_TO_FILE)
    {
        string output_filename = filename + ".output";
        output_file.open(output_filename);
        if (!output_file.is_open())
        {
            std::cout << "Unable to open output file: " << output_filename;
            exit(1);
        }
    }

    printf("Centroids: \n");
    // Iterate through the output array
    for (int i = 0; i < NUM_OUTPUT; i++)
    {

        printf("%d ", output[i].x);
        printf("%d ", output[i].y);
        printf("\n");
        if (SAVE_TO_FILE)
            output_file << output[i].x << " ";
        output_file << output[i].y << " ";
        output_file << "\n";
    }

    // Free host memory
    free(input);
    free(output);

    auto t_seq_3 = steady_clock::now();

    auto time1 = duration_cast<millis>(t_seq_2 - t_seq_1).count();
    auto time2 = duration_cast<millis>(t_seq_3 - t_seq_2).count();
    auto total_time = duration_cast<millis>(t_seq_3 - t_seq_1).count();

    std::cout << "Time for CPU data loading + initialize: " << time1 << " milliseconds\n";
    std::cout << "Time for map reduce KMeans + writing outputs + free: " << time2 << " milliseconds\n";
    std::cout << "Total time: " << total_time << " milliseconds\n";

    return 0;
}
