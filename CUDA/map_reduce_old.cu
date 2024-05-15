#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include "kmeans.cuh"
// #include "wordcount.cuh"

using millis = std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::steady_clock;

void mergeSort(MyPair *array, int const begin, int const end, int n, MyPair *d_array);
void combineUniqueKeys(MyPair *host_pairs, ShuffleAndSort_KeyPairOutput *&dev_shuffle_output, int &output_size);
void runPipeline(input_type *input, output_type *&output);
void readData(input_type *&input, std::string filename, int &inputNum);
void saveData(const output_type *output, std::string filename);
void printDeviceProperties();
void printMapReduceGPUParams(int mapBlockSize, int mapGridSize, int reduceBlockSize, int reduceGridSize);

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
    for (int i = 0; i < NUM_OUTPUT; i++)
    {
        std::cout << output[i];
        std::cout << std::endl;
    }

    saveData(output, filename);
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

__global__ void mapKernel(const input_type *input, MyPair *pairs, output_type *dev_output, unsigned long long *NUM_INPUT_D, int *NUM_OUTPUT_D)
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

void runMapKernel(const input_type *dev_input, MyPair *dev_pairs, output_type *dev_output, unsigned long long *NUM_INPUT_D, int *NUM_OUTPUT_D)
{
    mapKernel<<<MAP_GRID_SIZE, MAP_BLOCK_SIZE>>>(dev_input, dev_pairs, dev_output, NUM_INPUT_D, NUM_OUTPUT_D);
    cudaDeviceSynchronize();
    // error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
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

void runReduceKernel(ShuffleAndSort_KeyPairOutput *dev_pairs, output_type *dev_output, unsigned long long *TOTAL_PAIRS_D, int *NUM_OUTPUT_D)
{
    reduceKernel<<<REDUCE_GRID_SIZE, REDUCE_BLOCK_SIZE>>>(dev_pairs, dev_output, TOTAL_PAIRS_D, NUM_OUTPUT_D);
    cudaDeviceSynchronize();
}

void runPipeline(input_type *input, output_type *&output)
{
    unsigned long long *NUM_INPUT_D;
    unsigned long long *TOTAL_PAIRS_D;
    cudaMalloc(&NUM_INPUT_D, sizeof(unsigned long long));
    cudaMemcpy(NUM_INPUT_D, &NUM_INPUT, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMalloc(&TOTAL_PAIRS_D, sizeof(unsigned long long));
    cudaMemcpy(TOTAL_PAIRS_D, &TOTAL_PAIRS, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    int *NUM_OUTPUT_D;
    cudaMalloc(&NUM_OUTPUT_D, sizeof(int));
    cudaMemcpy(NUM_OUTPUT_D, &NUM_OUTPUT, sizeof(int), cudaMemcpyHostToDevice);

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

    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);

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

    // Copy initial centroids to device
    cudaMemcpy(dev_output, output, output_size, cudaMemcpyHostToDevice);

    // Now run K Means for the specified iterations
    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        MyPair *host_pairs;
        // ================== MAP ==================
        // TODO: USE STREAMING
        runMapKernel(dev_input, dev_pairs, dev_output, NUM_INPUT_D, NUM_OUTPUT_D);

        // ================== SORT ==================
        // thrust::sort(thrust::device, dev_pairs, dev_pairs + TOTAL_PAIRS, PairCompare());
        host_pairs = (MyPair *)malloc(pair_size);
        cudaMemcpy(host_pairs, dev_pairs, pair_size, cudaMemcpyDeviceToHost);
        mergeSort(host_pairs, 0, NUM_INPUT - 1, NUM_INPUT, dev_pairs);
        // for (int i = 0; i < TOTAL_PAIRS; i++)
        // {
        //     std::cout << host_pairs[i];
        // }
        // std::cout << std::endl;
        // ============= Combine unique keys =============
        ShuffleAndSort_KeyPairOutput *dev_shuffle_output;
        int output_size;
        combineUniqueKeys(host_pairs, dev_shuffle_output, output_size);
        // // print shuffle output
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
        runReduceKernel(dev_shuffle_output, dev_output, TOTAL_PAIRS_D, NUM_OUTPUT_D);
        // free the memory
        free(host_pairs);
        cudaFree(dev_shuffle_output);
    }
    // Copy outputs from GPU to host
    cudaMemcpy(output, dev_output, NUM_OUTPUT * sizeof(output_type), cudaMemcpyDeviceToHost);

    // Free all memory allocated on GPU
    cudaFree(dev_input);
    cudaFree(dev_pairs);
    cudaFree(dev_output);
    cudaFree(NUM_INPUT_D);
    cudaFree(TOTAL_PAIRS_D);
    cudaFree(NUM_OUTPUT_D);
}

// ===============================================================
// ========================MERGE SORT=============================
// ===============================================================
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
        if (PairCompareLessEql()(leftArray[indexOfSubArrayOne], rightArray[indexOfSubArrayTwo]))
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