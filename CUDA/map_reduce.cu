#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include "config.cuh"
#include "random_generator.hpp"
#include <cuda_runtime.h>

#define MAX_DEPTH 32

const bool SAVE_TO_FILE = true;

__device__ __host__
    uint64_cu
    distance(const Vector2D &p1, const Vector2D &p2)
{
    uint64_cu dist = 0;
    for (int i = 0; i < DIMENSION; i++)
    {
        int temp = p1.values[i] - p2.values[i];
        dist += temp * temp;
    }

    return dist;
}

/*
    Mapper function for each input element
    Input is already stored in memory, and output pairs must be stored in the memory allocated
    Muliple pairs can be generated for a single input, but their number shouldn't exceed NUM_PAIRS
*/
__device__ void mapper(const input_type *input, MyPair *pairs, output_type *output)
{
    // Find centroid with min distance from the current point
    uint64_cu min_distance = ULLONG_MAX;
    int cluster_id = -1;

    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        uint64_cu dist = distance(*input, output[i]);
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
    uint64_cu new_values[DIMENSION]; // uint64_cu to avoid overflow
    for (int i = 0; i < DIMENSION; i++)
        new_values[i] = 0;

    for (size_t i = 0; i < len; i++)
    {
        for (int j = 0; j < DIMENSION; j++)
            new_values[j] += pairs[i].value.values[j]; // Wow, this is bad naming
    }

    // uint64_cu diff = 0;

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

/*
    Main function that runs a map reduce job.
*/
int main(int argc, char *argv[])
{
    std::cout << "========================================" << "\n";
    std::cout << "============Device Properties===========" << "\n";
    std::cout << "========================================" << "\n";
    // Get device properties
    int deviceId = 0; // Device ID (usually 0 for the first GPU)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    // Print device properties
    printf("Device Name: %s\n", prop.name);
    printf("Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
    printf("Shared Memory Per Block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("Clock Rate: %d kHz\n", prop.clockRate);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
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
    NUM_INPUT = inputNum;
    TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

    MAP_GRID_SIZE = (NUM_INPUT + MAP_BLOCK_SIZE - 1) / MAP_BLOCK_SIZE;
    REDUCE_GRID_SIZE = (NUM_OUTPUT + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    std::cout << "========================================" << "\n";
    std::cout << "===========Map GPU Parameters===========" << "\n";
    std::cout << "========================================" << "\n";
    std::cout << "Block Size: " << MAP_BLOCK_SIZE << std::endl;
    std::cout << "Grid Size: " << MAP_GRID_SIZE << std::endl;
    std::cout << "========================================" << "\n";
    std::cout << "===========Reduce GPU Parameters========" << "\n";
    std::cout << "========================================" << "\n";
    std::cout << "Block Size: " << REDUCE_BLOCK_SIZE << std::endl;
    std::cout << "Grid Size: " << REDUCE_GRID_SIZE << std::endl;
    std::cout << "========================================" << "\n";
    std::cout << "Number of input elements: " << NUM_INPUT << std::endl;
    std::cout << "Number of pairs per input element: " << NUM_PAIRS << std::endl;
    std::cout << "Total number of pairs: " << TOTAL_PAIRS << std::endl;

    // Allocate host memory
    size_t input_size = NUM_INPUT * sizeof(input_type);
    input_type *input = (input_type *)malloc(input_size);

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *)malloc(output_size);

    // copy from vector to array
    for (int i = 0; i < inputNum; i++)
    {
        input[i].values[0] = data[i][0];
        input[i].values[1] = data[i][1];
    }
    // // Print the data vector to verify the contents
    // std::cout << "Data read from file:" << std::endl;
    // for (int i = 0; i < inputNum; i++)
    // {
    //     std::cout << input[i].values[0] << " " << input[i].values[1] << std::endl;
    // }
    std::cout << "Running Initialization..." << std::endl;
    // Now chose initial centroids
    initialize(input, output);

    auto t_seq_2 = steady_clock::now();
    std::cout << "Running Map-Reduce for " << ITERATIONS << " iterations..." << std::endl;
    // Run the Map Reduce Job
    runMapReduce(input, output);
    std::cout << "Number of output elements: " << NUM_OUTPUT << std::endl;

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

    std::cout << "========================================" << "\n";
    std::cout << "==============Final Output==============" << "\n";
    std::cout << "========================================" << "\n";
    // Iterate through the output array
    for (size_t i = 0; i < NUM_OUTPUT; i++)
    {
        for (int j = 0; j < DIMENSION; j++)
        {
            printf("%d ", output[i].values[j]);
            if (SAVE_TO_FILE)
                output_file << output[i].values[j] << " ";
        }

        printf("\n");
        if (SAVE_TO_FILE)
            output_file << "\n";
    }

    // Free host memory
    free(input);
    free(output);

    auto t_seq_3 = steady_clock::now();

    auto time1 = duration_cast<millis>(t_seq_2 - t_seq_1).count();
    auto time2 = duration_cast<millis>(t_seq_3 - t_seq_2).count();
    auto total_time = duration_cast<millis>(t_seq_3 - t_seq_1).count();
    std::cout << "========================================" << "\n";
    std::cout << "================Timings=================" << "\n";
    std::cout << "========================================" << "\n";
    std::cout << "Time for CPU data loading + initialize: " << time1 << " milliseconds\n";
    std::cout << "Time for map reduce KMeans + writing outputs + free: " << time2 << " milliseconds\n";
    std::cout << "Total time: " << total_time << " milliseconds\n";

    return 0;
}

// ===============================================================================
// ===============================================================================
// ===============================GPU IMPLEMENTATION==============================
// ===============================================================================
// ===============================================================================

// functions definitions
extern __device__ void mapper(const input_type *input, MyPair *pairs, output_type *output);
extern __device__ void reducer(MyPair *pairs, size_t len, output_type *output);

/*
    Mapping Kernel: Since each mapper runs independently of each other, we can
    give each thread its own input to process and a disjoint space where it can`
    store the key/value pairs it produces.
*/
__global__ void mapKernel(const input_type *input, MyPair *pairs, output_type *dev_output, uint64_cu *NUM_INPUT_D)
{
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x; // Global id of the thread
    // // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    // size_t step = blockDim.x * gridDim.x;

    // for (size_t i = threadId; i < *NUM_INPUT_D; i += step)
    // {
    //     // Input data to run mapper on, and the starting index of memory assigned for key-value pairs for this
    //     mapper(&input[i], &pairs[i * NUM_PAIRS], dev_output);
    // }
    if (threadId < *NUM_INPUT_D)
    {
        // Input data to run mapper on, and the starting index of memory assigned for key-value pairs for this
        mapper(&input[threadId], &pairs[threadId * NUM_PAIRS], dev_output);
    }
}

/*
    Call Mapper kernel with the required grid, blocks
*/
void runMapper(const input_type *dev_input, MyPair *dev_pairs, output_type *dev_output, uint64_cu *NUM_INPUT_D)
{
    mapKernel<<<MAP_GRID_SIZE, MAP_BLOCK_SIZE>>>(dev_input, dev_pairs, dev_output, NUM_INPUT_D);
    cudaDeviceSynchronize();
    // error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/*
    Reducer kernel
    Input is sorted array of keys (well, pairs)
    For each thread, find the keys that it'll work on and the range associated with each key
*/
__global__ void reducerKernel(MyPair *pairs, output_type *output, uint64_cu *TOTAL_PAIRS_D)
{
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x; // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    // size_t jump = blockDim.x * gridDim.x;

    // for (size_t i = threadId; i < NUM_OUTPUT; i += jump)
    if (threadId < NUM_OUTPUT)
    {
        // So now i is like the threadId that we need to run on
        // For each threadId, find the key associated with it (starting index, and the number of pairs)
        // And handle the case when there's no such key (no. of keys < no. of threads)
        size_t start_index = 0;            // Starting index of the key in the array of pairs
        size_t end_index = *TOTAL_PAIRS_D; // Ending index of the key in array of pairs
        size_t uniq_key_index = 0;         // In a list of unique sorted keys, the index of the key
        size_t value_size = 0;             // No. of pairs for this key

        // TODO: Can this be converted to a single pass over the entire array once?
        // Before the reducer
        // Store unique keys and their ranges
        for (size_t j = 1; j < *TOTAL_PAIRS_D; j++)
        {
            if (PairCompare()(pairs[j - 1], pairs[j]))
            {
                // The keys are unequal, therefore we have moved on to a new key
                if (uniq_key_index == threadId)
                {
                    // The previous key was the one associated with this thread
                    // And we have reached the end of pairs for that key
                    // So we now know the start and end for the key, no need to go through more pairs
                    end_index = j;
                    break;
                }
                else
                {
                    // Still haven't reached the key required
                    // Increae the uniq_key_index since it's a new key, and store its starting index
                    uniq_key_index++;
                    start_index = j;
                }
            }
        }

        // We can have that the thread doesn't need to process any key
        if (uniq_key_index != threadId)
        {
            return; // Enjoy, nothing to be done!
        }

        // Total number of pairs to be processed is end-start
        value_size = end_index - start_index;

        // Run the reducer
        reducer(&pairs[start_index], value_size, &output[threadId]);
    }
}

void runReducer(MyPair *dev_pairs, output_type *dev_output, uint64_cu *TOTAL_PAIRS_D)
{
    reducerKernel<<<REDUCE_GRID_SIZE, REDUCE_BLOCK_SIZE>>>(dev_pairs, dev_output, TOTAL_PAIRS_D);
    cudaDeviceSynchronize();
}
/*
this function that merges 2 arrays sequentially
params:
    int *array: the output array where to store the merged arrays
    int *leftArray: the left array
    int *rightArray: the right array
    int const left: the left index
    int subArrayOne: the size of the left array
    int subArrayTwo: the size of the right array
*/
__device__ void mergeSequential(MyPair *array, MyPair *leftArray, MyPair *rightArray, int subArrayOne, int subArrayTwo)
{
    auto indexOfSubArrayOne = 0, indexOfSubArrayTwo = 0;
    // the start of the merged array in the original array
    int indexOfMergedArray = 0;
    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo)
    {
        if (PairCompare2()(leftArray[indexOfSubArrayOne], rightArray[indexOfSubArrayTwo]))
        {
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else
        {
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }

    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne)
    {
        array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }

    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo)
    {
        array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
}

__device__ int coRank(MyPair *leftArray, MyPair *rightArray, int subArrayOne, int subArrayTwo, int k)
{
    int iLow = max(0, k - subArrayTwo);
    int iHigh = min(subArrayOne, k);
    // binary search
    while (true)
    {
        // initialize i with the mean of iLow and iHigh
        int i = (iLow + iHigh) / 2;
        int j = k - i;
        // our guess is too small, we must increase i
        if (i < subArrayOne && j > 0 && PairCompare()(leftArray[i], rightArray[j - 1]))
        {
            // i is too small, must increase it
            iLow = i + 1;
        }
        // our guess is too big, we must decrease i
        else if (i > 0 && j < subArrayTwo && PairCompare()(rightArray[j], leftArray[i - 1]))
        {
            // i is too big, must decrease it
            iHigh = i - 1;
        }
        else
        {
            // i is perfect in case leftArray[i] >= rightArray[j - 1] and leftArray[i - 1] <= rightArray[j]
            return i;
        }
    }
}

/*
    Merges two subarrays of array[].
    First subarray is arr[begin..mid]
    Second subarray is arr[mid+1..end]
    params:
        int *array: the output array where to store the merged arrays
        int const left: the left index
        int const mid: the middle index
        int const right: the right index
*/
__global__ void merge(MyPair array[], MyPair *leftArray, MyPair *rightArray, int *subArrayOne, int *subArrayTwo)
{
    int totalSize = *subArrayOne + *subArrayTwo;
    int numberThreads = blockDim.x * gridDim.x;
    // printf("numberThreads: %d\n", numberThreads);
    int amountPerThread = ceil((float)totalSize / (float)numberThreads);
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    // index of the output array that this thread will start merging from
    int k = threadId * amountPerThread;
    if (k < totalSize)
    {
        int i = coRank(leftArray, rightArray, *subArrayOne, *subArrayTwo, k);
        int j = k - i;
        // getting the size of each thread's segment
        // the next thread will start from kNext
        int kNext = min(k + amountPerThread, totalSize);
        int iNext = coRank(leftArray, rightArray, *subArrayOne, *subArrayTwo, kNext);
        int jNext = kNext - iNext;
        int iSegmentSize = iNext - i;
        int jSegmentSize = jNext - j;
        mergeSequential(&array[k], &leftArray[i], &rightArray[j], iSegmentSize, jSegmentSize);
    }
}

// Function to print an array
void printArray(MyPair A[], int size)
{
    for (int i = 0; i < size; i++)
        std::cout << A[i] << " ";
    std::cout << std::endl;
}
// begin is for left index and end is right index
// of the sub-array of arr to be sorted
void mergeSort(MyPair *array, int const begin, int const end, int n, MyPair *d_array)
{
    if (begin >= end)
    {
        return;
    }

    int mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid, n, d_array);
    mergeSort(array, mid + 1, end, n, d_array);
    // cout << "Merging: " << begin << " " << mid << " " << end << endl;
    // divide the array into 2 subarrays
    int const subArrayOne = mid - begin + 1;
    int const subArrayTwo = end - mid;
    int arraySize = end - begin + 1;

    // Create temp arrays
    auto *leftArray = new MyPair[subArrayOne],
         *rightArray = new MyPair[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[begin + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];
    // MyPair *d_array;
    MyPair *d_leftArray, *d_rightArray;
    int *d_subArrayOne, *d_subArrayTwo;
    // cudaMalloc(&d_array, n * sizeof(MyPair));
    cudaMalloc(&d_leftArray, subArrayOne * sizeof(MyPair));
    cudaMalloc(&d_rightArray, subArrayTwo * sizeof(MyPair));
    cudaMalloc(&d_subArrayOne, sizeof(int));
    cudaMalloc(&d_subArrayTwo, sizeof(int));
    cudaDeviceSynchronize();
    // copy the array to the device
    // cudaMemcpy(d_array, array, n * sizeof(MyPair), cudaMemcpyHostToDevice);
    // copy the leftArray and the rightArray to the device
    cudaMemcpy(d_leftArray, leftArray, subArrayOne * sizeof(MyPair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightArray, rightArray, subArrayTwo * sizeof(MyPair), cudaMemcpyHostToDevice);
    // copy the size of the leftArray and the rightArray to the device
    cudaMemcpy(d_subArrayOne, &subArrayOne, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_subArrayTwo, &subArrayTwo, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Query device properties to get the block size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numberBlocks = ceil((float)arraySize / (float)prop.maxThreadsPerBlock);
    merge<<<numberBlocks, prop.maxThreadsPerBlock>>>(&d_array[begin], d_leftArray, d_rightArray, d_subArrayOne, d_subArrayTwo);
    cudaDeviceSynchronize();
    // cout << "Array size: " << arraySize << endl;
    // copy the sorted array from the device to the host
    cudaMemcpy(array, d_array, n * sizeof(MyPair), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // free the memory
    // cudaFree(d_array);
    cudaFree(d_leftArray);
    cudaFree(d_rightArray);
    cudaFree(d_subArrayOne);
    cudaFree(d_subArrayTwo);
    delete[] leftArray;
    delete[] rightArray;
}
void runMapReduce(const input_type *input, output_type *output)
{
    // 1. Allocate memory on GPU for inputs, key-value pairs & centroids on GPU
    //      Note output memory is also allocated right now itself
    // 2. Copy inputs, initial centroids to GPU
    // 3. For the required iterations, do the following -
    //      1. Run Mapper kernel, which calls mapper function for the inputs decided for that thread
    //      2. Sort Key-Value pairs
    //      3. Reducer: Each thread gets a specific cluster, and finds the new centroids
    // 4. Copy output from GPU to host memory
    // 5. Free all GPU memory
    // Done! Finally

    uint64_cu *NUM_INPUT_D;
    uint64_cu *TOTAL_PAIRS_D;
    cudaMalloc(&NUM_INPUT_D, sizeof(uint64_cu));
    cudaMemcpy(NUM_INPUT_D, &NUM_INPUT, sizeof(uint64_cu), cudaMemcpyHostToDevice);
    cudaMalloc(&TOTAL_PAIRS_D, sizeof(uint64_cu));
    cudaMemcpy(TOTAL_PAIRS_D, &TOTAL_PAIRS, sizeof(uint64_cu), cudaMemcpyHostToDevice);

    // Pointers for input, key-value pairs & output on device
    input_type *dev_input;
    output_type *dev_output;
    MyPair *dev_pairs;

    // Allocate memory on GPU for input
    size_t input_size = NUM_INPUT * sizeof(input_type);
    cudaMalloc(&dev_input, input_size);

    // Allocate memory for key-value pairs
    size_t pair_size = TOTAL_PAIRS * sizeof(MyPair);
    cudaMalloc(&dev_pairs, pair_size);

    // Allocate memory for outputs
    // Since centroids are needed in K Means the entire time
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    cudaMalloc(&dev_output, output_size);

    // Allocate memory on host for older centroids
    // output_type *old_output = (output_type *) malloc(output_size);

    // Copy input datapoints to device
    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);

    // Copy initial centroids to device
    cudaMemcpy(dev_output, output, output_size, cudaMemcpyHostToDevice);

    // Now run K Means for the specified iterations
    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        // printf("Starting iteration %d\n", iter);

        // Copy older centroids
        // cudaMemcpy(old_output, dev_output, output_size, cudaMemcpyDeviceToHost);

        // Run mapper
        // This will run mapper kernel on all the inputs, and produces the key-value pairs
        // It also requires the current centroids, so pass `dev_output` as well
        runMapper(dev_input, dev_pairs, dev_output, NUM_INPUT_D);

        // Create Thrust device pointer from key-value pairs
        // thrust::device_ptr<MyPair> dev_pair_thrust_ptr(dev_pairs);

        // thrust::copy(dev_pair_thrust_ptr, dev_pair_thrust_ptr + TOTAL_PAIRS, std::ostream_iterator<MyPair>(std::cout, " "));

        // Sort Key-Value pairs based on Key
        // This should run on the device itself
        // thrust::sort(thrust::device, dev_pairs, dev_pairs + TOTAL_PAIRS, PairCompare());
        MyPair *host_pairs = (MyPair *)malloc(pair_size);
        cudaMemcpy(host_pairs, dev_pairs, pair_size, cudaMemcpyDeviceToHost);
        mergeSort(host_pairs, 0, NUM_INPUT - 1, NUM_INPUT, dev_pairs);
        // copy from the host to the device
        // cudaMemcpy(dev_pairs, host_pairs, pair_size, cudaMemcpyHostToDevice);
        // free(host_pairs);
        // // allocate memory for the sorted pairs
        // host_pairs = (MyPair *)malloc(pair_size);
        // cudaMemcpy(host_pairs, dev_pairs, pair_size, cudaMemcpyDeviceToHost);

        // // Print host_pairs
        // for (size_t i = 0; i < NUM_INPUT; i++)
        // {
        //     std::cout << host_pairs[i] << std::endl;
        // }
        free(host_pairs);

        // Run reducer kernel on key-value pairs
        runReducer(dev_pairs, dev_output, TOTAL_PAIRS_D);

        // Copy new centroids
        // cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);
    }
    // Copy outputs from GPU to host
    // Note host memory has already been allocated
    cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);

    // Free all memory allocated on GPU
    cudaFree(dev_input);
    cudaFree(dev_pairs);
    cudaFree(dev_output);
}