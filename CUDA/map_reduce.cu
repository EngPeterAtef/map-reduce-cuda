#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include "kmeans.cuh"
// #include "wordcount.cuh"
#define MAX_THREADS_PER_BLOCK 1024

using millis = std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::steady_clock;

void combineUniqueKeys(MyPair *host_pairs, ShuffleAndSort_KeyPairOutput *&dev_shuffle_output, int &output_size);
void runPipeline(input_type *input, output_type *&output);
void readData(input_type *&input, std::string filename, int &inputNum);
void saveData(const output_type *output, std::string filename);
void printDeviceProperties();
void printMapReduceGPUParams(int mapBlockSize, int mapGridSize, int reduceBlockSize, int reduceGridSize);
__global__ void mergeSortGPU(MyPair *arr, MyPair *temp, int n, int width);
__global__ void bitonicSortGPU(MyPair *arr, int j, int k);
void printArray(MyPair A[], int size);
float sort(MyPair *host_pairs, MyPair *dev_pairs);

int main(int argc, char *argv[])
{
    printDeviceProperties();
    if (argc != 2)
    {
        printf("Requires 1 argument, name of input textfile\n");
        exit(1);
    }

    std::string filename = argv[1];

    auto t_seq_1 = steady_clock::now();

    int inputNum;
    input_type *input;
    readData(input, filename, inputNum);
    NUM_INPUT = inputNum;
    TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

    MAP_GRID_SIZE = (NUM_INPUT + MAP_BLOCK_SIZE - 1) / MAP_BLOCK_SIZE;
    REDUCE_GRID_SIZE = (NUM_OUTPUT + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    printMapReduceGPUParams(MAP_BLOCK_SIZE, MAP_GRID_SIZE, REDUCE_BLOCK_SIZE, REDUCE_GRID_SIZE);
    std::cout << "========================================" << "\n";
    std::cout << "Number of input elements: " << NUM_INPUT << std::endl;
    std::cout << "Number of pairs per input element: " << NUM_PAIRS << std::endl;
    std::cout << "Total number of pairs: " << TOTAL_PAIRS << std::endl;

    // Allocate host memory

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *)malloc(output_size);

    std::cout << "Running Initialization..." << std::endl;
    // Now chose initial centroids
    initialize(input, output);

    auto t_seq_2 = steady_clock::now();
    std::cout << "Running Map-Reduce for " << ITERATIONS << " iterations..." << std::endl;
    // Run the Map Reduce Job
    runPipeline(input, output);
    std::cout << "Number of output elements: " << NUM_OUTPUT << std::endl;

    std::cout << "========================================" << "\n";
    std::cout << "==============Final Output==============" << "\n";
    std::cout << "========================================" << "\n";
    // Iterate through the output array
    // for (int i = 0; i < NUM_OUTPUT; i++)
    // {
    //     std::cout << output[i];
    //     std::cout << std::endl;
    // }

    saveData(output, filename);
    // Free host memory
    free(input);
    free(output);

    auto t_seq_3 = steady_clock::now();

    auto time1 = duration_cast<millis>(t_seq_2 - t_seq_1).count();
    auto time2 = duration_cast<millis>(t_seq_3 - t_seq_2).count();
    auto total_time = duration_cast<millis>(t_seq_3 - t_seq_1).count();
    // auto combiner_time = duration_cast<millis>(t_seq_5 - t_seq_4).count();
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

__global__ void mapKernel(const input_type *input, MyPair *pairs, output_type *dev_output, int *NUM_INPUT_D, int *NUM_OUTPUT_D)
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
        mapper(&input[threadId], &pairs[threadId * NUM_PAIRS], dev_output, NUM_OUTPUT_D);
    }
}

float runMapKernel(const input_type *dev_input, MyPair *dev_pairs, output_type *dev_output, int *NUM_INPUT_D, int *NUM_OUTPUT_D, cudaStream_t stream)
{
    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;
    cudaEventRecord(startGPU);

    mapKernel<<<MAP_GRID_SIZE, MAP_BLOCK_SIZE, 0, stream>>>(dev_input, dev_pairs, dev_output, NUM_INPUT_D, NUM_OUTPUT_D);
    // cudaDeviceSynchronize();
    cudaEventRecord(stopGPU);
    // Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
    // error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return millisecondsGPU;
}

__global__ void reduceKernel(ShuffleAndSort_KeyPairOutput *pairs, output_type *output, unsigned long long *TOTAL_PAIRS_D, int *NUM_OUTPUT_D)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x; // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    // size_t jump = blockDim.x * gridDim.x;

    // for (size_t i = threadId; i < NUM_OUTPUT; i += jump)
    // printf("Thread id: %d\n", threadId);
    if (threadId < *NUM_OUTPUT_D)
    {
        // Run the reducer
        reducer(&pairs[threadId], &output[threadId]);
        // printf("Key: %s, Value: %s\n", output[threadId].key, output[threadId].value);
    }
}

float runReduceKernel(ShuffleAndSort_KeyPairOutput *dev_pairs, output_type *dev_output, unsigned long long *TOTAL_PAIRS_D, int *NUM_OUTPUT_D)
{
    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;
    cudaEventRecord(startGPU);
    reduceKernel<<<REDUCE_GRID_SIZE, REDUCE_BLOCK_SIZE>>>(dev_pairs, dev_output, TOTAL_PAIRS_D, NUM_OUTPUT_D);
    // cudaDeviceSynchronize();
    cudaEventRecord(stopGPU);
    // Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
    return millisecondsGPU;
}
// Function to check if given number is a power of 2
bool isPowerOfTwo(int num)
{
    return num > 0 && (num & (num - 1)) == 0;
}

// Automated function to check if array is sorted
bool isSorted(MyPair *arr, int size)
{
    for (int i = 1; i < size; ++i)
    {
        if (PairCompare()(arr[i], arr[i - 1]))
            return false;
    }
    return true;
}
void runPipeline(input_type *input, output_type *&output)
{
    unsigned long long *NUM_INPUT_D;
    unsigned long long *TOTAL_PAIRS_D;
    int *NUM_OUTPUT_D;
    cudaMalloc(&NUM_INPUT_D, sizeof(unsigned long long));
    cudaMalloc(&TOTAL_PAIRS_D, sizeof(unsigned long long));
    cudaMalloc(&NUM_OUTPUT_D, sizeof(int));
    cudaDeviceSynchronize();

    cudaMemcpy(NUM_INPUT_D, &NUM_INPUT, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(TOTAL_PAIRS_D, &TOTAL_PAIRS, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(NUM_OUTPUT_D, &NUM_OUTPUT, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    size_t input_size = NUM_INPUT * sizeof(input_type);
    size_t pair_size = TOTAL_PAIRS * sizeof(MyPair);
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    // Pointers for input, key-value pairs & output on device
    input_type *dev_input;
    output_type *dev_output;
    MyPair *dev_pairs;

    // Allocate memory on GPU for input
    cudaMalloc(&dev_input, input_size);
    // Allocate memory for key-value pairs
    cudaMalloc(&dev_pairs, pair_size);
    // Allocate memory for outputs
    // Since centroids are needed in K Means the entire time
    cudaMalloc(&dev_output, output_size);
    // Copy initial centroids to device
    cudaMemcpy(dev_output, output, output_size, cudaMemcpyHostToDevice);
    // copy dev_input to host again to print
    // std::cout << "Printing input from dev" << std::endl;
    // input_type *host_input;
    // host_input = (input_type *)malloc(input_size);
    // cudaMemcpy(host_input, dev_input, input_size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < NUM_INPUT; i++)
    // {
    //     std::cout << host_input[i];
    //     std::cout << std::endl;
    // }
    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;
    // =================STREAMING=================
    const int numberOfStreams = 32;
    cudaStream_t streams[numberOfStreams];
    for (int i = 0; i < numberOfStreams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    // ==========================================
    // divide the data into segments
    int segment_size = (NUM_INPUT + numberOfStreams - 1) / numberOfStreams; // ceil
    std::cout << "Segment size: " << segment_size << std::endl;
    // Now run the algorithm for the specified iterations
    float mapGPUTime = 0, reduceGPUTime = 0, sortGPUTime = 0;

    MyPair *host_pairs;
    cudaMallocHost(&host_pairs, NUM_INPUT * sizeof(MyPair));
    // host_pairs = (MyPair *)malloc(NUM_INPUT * sizeof(MyPair));
    int64_t t_seq_combine;

    cudaEventRecord(startGPU);
    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        float temp = 0;
        float iterationTime = 0;
        std::cout << "Iteration: " << iter << std::endl;
        for (int s = 0; s < numberOfStreams; s++)
        {
            int start = s * segment_size;
            int end = (start + segment_size) < NUM_INPUT ? (start + segment_size) : NUM_INPUT;
            int segment_input_size = end - start;
            std::cout << "Stream: " << s << " Start: " << start << " End: " << end << " Segment size: " << segment_input_size << std::endl;
            int *segment_input_size_d;
            if (iter == 0)
            {
                // Copy the size of the segment to device
                cudaMallocAsync(&segment_input_size_d, sizeof(int), streams[s]);
                cudaMemcpyAsync(segment_input_size_d, &segment_input_size, sizeof(int), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(&dev_input[start], &input[start], segment_input_size * sizeof(input_type), cudaMemcpyHostToDevice, streams[s]);
            }

            // ================== MAP ==================
            temp = runMapKernel(&dev_input[start], &dev_pairs[start], dev_output, segment_input_size_d, NUM_OUTPUT_D, streams[s]);
            // Print the time of the runs
            std::cout << "\n\nIteration:" << iter << " Stream:" << s << " Map function GPU Time: " << temp << " ms" << std::endl;
            mapGPUTime += temp;
            iterationTime += temp;
        }
        cudaDeviceSynchronize();
        // ================== SORT ==================
        // std::cout << "Start Sort" << std::endl;
        // thrust::sort(thrust::device, dev_pairs, dev_pairs + TOTAL_PAIRS, PairCompare());
        // cudaMemcpy(host_pairs, dev_pairs, NUM_INPUT * sizeof(MyPair), cudaMemcpyDeviceToHost);
        temp = sort(host_pairs, dev_pairs);
        std::cout << "\n\nIteration:" << iter << " Sort function GPU Time: " << temp << " ms" << std::endl;
        sortGPUTime += temp;
        iterationTime += temp;
        // for (int i = 0; i < TOTAL_PAIRS; i++)
        // {
        //     std::cout << host_pairs[i];
        // }
        // std::cout << std::endl;
        // ============= Combine unique keys =============
        auto t_seq_4 = steady_clock::now();
        ShuffleAndSort_KeyPairOutput *dev_shuffle_output;
        int output_size;
        combineUniqueKeys(host_pairs, dev_shuffle_output, output_size);
        auto t_seq_5 = steady_clock::now();
        if (iter == 0)
            t_seq_combine = duration_cast<millis>(t_seq_5 - t_seq_4).count();
        else
            t_seq_combine += duration_cast<millis>(t_seq_5 - t_seq_4).count();

        // print shuffle output
        // for (int i = 0; i < shuffle_output->size(); i++)
        // {
        //     std::cout << shuffle_output->at(i);
        //     std::cout << std::endl;
        // }

        // allocate output if it was not allocated
        // check if output is allocated
        if (NUM_OUTPUT == 0)
        {
            NUM_OUTPUT = output_size;
            output = (output_type *)malloc(output_size * sizeof(output_type));
            // allocate dev_output
            cudaMalloc(&dev_output, NUM_OUTPUT * sizeof(output_type));
            // copy num output to device
            cudaMemcpy(NUM_OUTPUT_D, &NUM_OUTPUT, sizeof(int), cudaMemcpyHostToDevice);
            REDUCE_GRID_SIZE = (NUM_OUTPUT + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
        }

        // ================== REDUCE ==================
        temp = runReduceKernel(dev_shuffle_output, dev_output, TOTAL_PAIRS_D, NUM_OUTPUT_D);
        std::cout << "\n\nIteration " << iter << " Reduce function GPU Time: " << temp << " ms" << std::endl;
        reduceGPUTime += temp;
        iterationTime += temp;
        cudaFree(dev_shuffle_output);
        // print the time of the iteration
        std::cout << "\n\nIteration " << iter << " Total GPU Time: " << iterationTime << " ms" << std::endl;
    }
    // free the memory
    cudaDeviceSynchronize();
    cudaEventRecord(stopGPU);
    // Calculate Elapsed GPU time
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
    // Copy outputs from GPU to host
    cudaMemcpy(output, dev_output, NUM_OUTPUT * sizeof(output_type), cudaMemcpyDeviceToHost);

    // Free all memory allocated on GPU
    cudaFreeHost(host_pairs);
    cudaFree(dev_input);
    cudaFree(dev_pairs);
    cudaFree(dev_output);
    cudaFree(NUM_INPUT_D);
    cudaFree(TOTAL_PAIRS_D);
    cudaFree(NUM_OUTPUT_D);

    // show the total time of each kernel
    std::cout << "\n\nTotal Map GPU Time: " << mapGPUTime << " ms" << std::endl;
    std::cout << "\n\nTotal Sort GPU Time: " << sortGPUTime << " ms" << std::endl;
    std::cout << "\n\nTotal Reduce GPU Time: " << reduceGPUTime << " ms" << std::endl;
    std::cout << "\n\nTotal GPU Time: " << millisecondsGPU << " ms" << std::endl;
    std::cout << "Time for combine unique keys: " << t_seq_combine << " milliseconds\n";
    // printf("Time for combine unique keys: %f ms\n", t_seq_combine);
}

// ===============================================================
// ==========================SORT=================================
// ===============================================================
// GPU Kernel Implementation of Bitonic Sort
__global__ void bitonicSortGPU(MyPair *arr, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (PairCompareGreater()(arr[i], arr[ij]))
            {
                MyPair temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else
        {
            if (PairCompare()(arr[i], arr[ij]))
            {
                MyPair temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

// Device function for recursive Merge
__device__ void mergeSequential(MyPair *arr, MyPair *temp, int left, int middle, int right)
{
    int i = left;
    int j = middle;
    int k = left;

    while (i < middle && j < right)
    {
        if (PairCompareLessEql()(arr[i], arr[j]))
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < middle)
        temp[k++] = arr[i++];
    while (j < right)
        temp[k++] = arr[j++];

    for (int x = left; x < right; x++)
        arr[x] = temp[x];
}

// GPU Kernel for Merge Sort
__global__ void mergeSortGPU(MyPair *arr, MyPair *temp, int n, int width)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int left = tid * width;
    int middle = left + width / 2;
    int right = left + width;

    if (left < n && middle < n)
    {
        mergeSequential(arr, temp, left, middle, right);
    }
}

float sort(MyPair *host_pairs, MyPair *dev_pairs)
{

    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;

    // Set number of threads and blocks for kernel calls
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid = (NUM_INPUT + threadsPerBlock - 1) / threadsPerBlock;

    // Create GPU based arrays
    MyPair *gpuTemp;
    MyPair *gpuArrbiton;
    MyPair *gpuArrmerge;
    // Allocate memory on GPU
    cudaMalloc(&gpuTemp, NUM_INPUT * sizeof(MyPair));
    cudaMalloc(&gpuArrbiton, NUM_INPUT * sizeof(MyPair));
    cudaMalloc(&gpuArrmerge, NUM_INPUT * sizeof(MyPair));

    // Copy the input array to GPU memory
    cudaMemcpy(gpuArrmerge, dev_pairs, NUM_INPUT * sizeof(MyPair), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuArrbiton, dev_pairs, NUM_INPUT * sizeof(MyPair), cudaMemcpyDeviceToDevice);

    int choice = 2; // init with merge sort
    // std::cout << "\nSelect the type of sort:";
    // std::cout << "\n\t1. Merge Sort";
    // std::cout << "\n\t2. Bitonic Sort";
    // std::cout << "\nEnter your choice: ";
    // std::cin >> choice;
    // if (choice < 1 || choice > 2)
    // {
    //     while (choice != 1 || choice != 2)
    //     {
    //         std::cout << "\n!!!!! WRONG CHOICE. TRY AGAIN. YOU HAVE ONLY 2 DISTINCT OPTIONS-\n";
    //         std::cin >> choice;

    //         if (choice == 1 || choice == 2)
    //             break;
    //     }
    // }

    if (choice == 1)
    {
        std::cout << "\n--------------------------------------------------------------\nMERGE SORT SELECTED\n--------------------------------------------------------------";

        // Call GPU Merge Kernel and time the run
        cudaEventRecord(startGPU);
        for (int wid = 1; wid < NUM_INPUT; wid *= 2)
        {
            mergeSortGPU<<<threadsPerBlock, blocksPerGrid>>>(gpuArrmerge, gpuTemp, NUM_INPUT, wid * 2);
        }
        cudaEventRecord(stopGPU);

        // Transfer sorted array back to CPU
        cudaMemcpy(host_pairs, gpuArrmerge, NUM_INPUT * sizeof(MyPair), cudaMemcpyDeviceToHost);

        // Calculate Elapsed GPU time
        cudaEventSynchronize(stopGPU);
        cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
    }
    else
    {
        std::cout << "\n--------------------------------------------------------------\nBITONIC SORT SELECTED\n--------------------------------------------------------------";

        // bitonic sort
        if (isPowerOfTwo(NUM_INPUT))
        {

            int j, k;

            // Time the run and call GPU Bitonic Kernel
            cudaEventRecord(startGPU);
            for (k = 2; k <= NUM_INPUT; k <<= 1)
            {
                for (j = k >> 1; j > 0; j = j >> 1)
                {
                    bitonicSortGPU<<<blocksPerGrid, threadsPerBlock>>>(gpuArrbiton, j, k);
                }
            }
            cudaEventRecord(stopGPU);

            // Transfer Sorted array back to CPU
            cudaMemcpy(host_pairs, gpuArrbiton, NUM_INPUT * sizeof(MyPair), cudaMemcpyDeviceToHost);
            cudaEventSynchronize(stopGPU);
            cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
        }
        else
        {
            std::cout << "Size of array is not a power of 2. Please enter a power of 2 to use Bitonic Sort" << std::endl;
        }
    }
    std::cout << "\n\nSorted GPU array: ";
    // printArray(host_pairs, NUM_INPUT);
    if (isSorted(host_pairs, NUM_INPUT))
        std::cout << "\n\nSORT CHECKER RUNNING - SUCCESFULLY SORTED GPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;
    // Print the time of the runs
    // std::cout << "\n\nSorting GPU Time: " << millisecondsGPU << " ms" << std::endl;
    // End
    cudaFree(gpuArrmerge);
    cudaFree(gpuArrbiton);
    cudaFree(gpuTemp);
    return millisecondsGPU;
}

// ===============================================================
// ====================COMBINE UNIQUE KEYS========================
// ===============================================================
void combineUniqueKeys(MyPair *host_pairs, ShuffleAndSort_KeyPairOutput *&dev_shuffle_output, int &output_size)
{
    std::vector<ShuffleAndSort_KeyPairOutput> *shuffle_output = new std::vector<ShuffleAndSort_KeyPairOutput>();

    for (int i = 0; i < NUM_INPUT; i++)
    {
        bool isSame = strcmp(host_pairs[i].key, host_pairs[i - 1].key) == 0;
        if (i == 0 || !isSame)
        {
            ShuffleAndSort_KeyPairOutput current_pair;
            for (int k = 0; k < DIMENSION; k++)
            {
                for (int j = 0; j < MAX_WORD_SIZE; j++)
                {
                    // copy key
                    current_pair.key[j] = host_pairs[i].key[j];
                    // copy value
                    current_pair.values[0].values[k * MAX_WORD_SIZE + j] = host_pairs[i].value.values[k * MAX_WORD_SIZE + j];
                }
                current_pair.values[0].len[k] = host_pairs[i].value.len[k];
            }
            current_pair.size = 1;
            shuffle_output->push_back(current_pair);
        }
        else
        {
            for (int k = 0; k < DIMENSION; k++)
            {
                for (int j = 0; j < MAX_WORD_SIZE; j++)
                {
                    shuffle_output->back().values[shuffle_output->back().size].values[k * MAX_WORD_SIZE + j] = host_pairs[i].value.values[k * MAX_WORD_SIZE + j];
                }
                shuffle_output->back().values[shuffle_output->back().size].len[k] = host_pairs[i].value.len[k];
            }
            shuffle_output->back().size++;
        }
    }
    output_size = shuffle_output->size();
    // for (int i = 0; i < shuffle_output->size(); i++)
    // {
    //     std::cout << shuffle_output->at(i);
    //     std::cout << std::endl;
    // }
    // allocate memory for the output
    cudaMalloc(&dev_shuffle_output, output_size * sizeof(ShuffleAndSort_KeyPairOutput));
    cudaMemcpy(dev_shuffle_output, shuffle_output->data(), output_size * sizeof(ShuffleAndSort_KeyPairOutput), cudaMemcpyHostToDevice);
    // free the memory
    shuffle_output->clear();
    delete shuffle_output;
}

// ===============================================================
// ==========================UTILS================================
// ===============================================================
void stringToCharArray(const std::string &str, char *&charArray, char *&dev_charArray, int &length)
{
    // Convert std::string to C-style string
    const char *cstr = str.c_str();

    // Determine length of C-style string
    length = 0;
    while (cstr[length] != '\0')
    {
        length++;
    }

    // Allocate memory for the C-style array to store the characters
    charArray = (char *)malloc((length + 1) * sizeof(char)); // +1 for the null terminator

    // Check if memory allocation was successful
    if (charArray == nullptr)
    {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }

    // Copy characters from C-style string to the array
    for (int i = 0; i <= length; i++)
    {
        charArray[i] = cstr[i];
    }
    // copy to device
    // cudaMalloc(&dev_charArray, (length + 1) * sizeof(char));
    // cudaMemcpy(dev_charArray, charArray, (length + 1) * sizeof(char), cudaMemcpyHostToDevice);
}
void readData(input_type *&input, std::string filename, int &inputNum)
{
    std::vector<ReadVector> data;

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
            ReadVector tempInput;
            for (int i = 0; i < DIMENSION && (iss >> tempInput.values[i]); ++i)
            {
                // Read DIMENSION values from the line
            }
            data.push_back(tempInput); // Add the row to the data vector
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
    // Copy from vector to array
    inputNum = (int)data.size();
    size_t input_size = inputNum * sizeof(input_type);
    input = (input_type *)malloc(input_size);
    // copy from vector to array
    for (int i = 0; i < inputNum; i++)
    {
        for (int j = 0; j < DIMENSION; j++)
        {
            char *newCharList;
            char *dev_newCharList;
            int len;
            stringToCharArray(data[i].values[j], newCharList, dev_newCharList, len);
            for (int k = 0; k < len + 1; k++)
            {
                input[i].values[j * MAX_WORD_SIZE + k] = newCharList[k];
            }
            input[i].len[j] = len;
            // free the memory
            free(newCharList);
        }
    }
    data.clear();
    // print the input
    // for (int i = 0; i < inputNum; i++)
    // {
    //     std::cout << input[i];
    //     std::cout << std::endl;
    // }
}
void saveData(const output_type *output, std::string filename)
{
    // Save output if required
    std::ofstream output_file;

    std::string output_filename = filename + ".output";
    output_file.open(output_filename);
    if (!output_file.is_open())
    {
        std::cout << "Unable to open output file: " << output_filename;
        return;
    }
    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        output_file << output[i] << "\n";
    }
    output_file.close();
}
void printDeviceProperties()
{
    std::cout << "========================================" << "\n";
    std::cout << "============Device Properties===========" << "\n";
    std::cout << "========================================" << "\n";
    // Get device properties
    int deviceId = 0; // Device ID (usually 0 for the first GPU)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    // Print device properties
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate << " kHz" << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
}
void printMapReduceGPUParams(int mapBlockSize, int mapGridSize, int reduceBlockSize, int reduceGridSize)
{
    std::cout << "========================================" << "\n";
    std::cout << "===========Map GPU Parameters===========" << "\n";
    std::cout << "========================================" << "\n";
    std::cout << "Block Size: " << mapBlockSize << std::endl;
    std::cout << "Grid Size: " << mapGridSize << std::endl;
    std::cout << "========================================" << "\n";
    std::cout << "===========Reduce GPU Parameters========" << "\n";
    std::cout << "========================================" << "\n";
    std::cout << "Block Size: " << reduceBlockSize << std::endl;
    std::cout << "Grid Size: " << reduceGridSize << std::endl;
}
// Function to print an array
void printArray(MyPair A[], int size)
{
    for (int i = 0; i < size; i++)
        std::cout << A[i] << " ";
    std::cout << std::endl;
}