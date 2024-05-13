#include <cuda_runtime.h>
#include <iostream>
#include "config.cuh"
#include <fstream>
#include <chrono>
#include <vector>
#include <thrust/sort.h>

#define MAX_DEPTH 64
/*
    Merge sort kernel
*/
__global__ void mergesort_kernel(MyPair *data, MyPair *dataAux, int begin, int end, int depth)
{
    int middle = (end + begin) / 2;
    int i0 = begin;
    int i1 = middle;
    int index;
    int n = end - begin;

    // Used to implement recursions using CUDA parallelism.
    cudaStream_t s, s1;
    // if the length is less than 2, return
    if (n < 2)
    {
        return;
    }

    // Create a new block to sort the left part.
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    mergesort_kernel<<<1, 1, 0, s>>>(data, dataAux, begin, middle, depth + 1);
    cudaStreamDestroy(s);

    // Create a new block to sort the right part.
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    mergesort_kernel<<<1, 1, 0, s1>>>(data, dataAux, middle, end, depth + 1);
    cudaStreamDestroy(s1);

    // Merges children's generated partition.
    // Does the merging using the auxiliary memory.
    for (index = begin; index < end; index++)
    {
        // if (i0 < middle && (i1 >= end || data[i0] <= data[i1]))
        if (i0 < middle && (i1 >= end || PairCompare2()(data[i0], data[i1])))
        {
            dataAux[index] = data[i0];
            i0++;
        }
        else
        {
            dataAux[index] = data[i1];
            i1++;
        }
    }

    // Copies from the auxiliary memory to the main memory.
    // Note that each thread operates a different partition,
    // and the auxiliary memory has exact the same size of the main
    // memory, so the threads never write or read on the same
    // memory position concurrently, since one must wait it's children
    // to merge their partitions.
    for (index = begin; index < end; index++)
    {
        data[index] = dataAux[index];
    }
}
/*
    Merge sort function called by the host
    params: a - array to be sorted
            n - size of the array
    returns: void
*/
void gpumerge_sort(MyPair *gpuData, int n, MyPair *sort_out)
{
    MyPair *gpuAuxData;
    int left = 0;
    int right = n;
    // pront left and right
    printf("Left: %d, Right: %d\n", left, right);
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    // Allocate GPU memory.
    cudaMalloc((void **)&gpuAuxData, n * sizeof(MyPair));

    // Launch on device
    mergesort_kernel<<<1, 1>>>(gpuData, gpuAuxData, left, right, 0);
    cudaDeviceSynchronize();

    // Copy from device to host
    // cudaMemcpy(gpuData, gpuData, n * sizeof(MyPair), cudaMemcpyDeviceToHost);
    cudaMemcpy(sort_out, gpuData, n * sizeof(MyPair), cudaMemcpyDeviceToHost);

    printf("Sorted data\n");
    for (int i = 0; i < TOTAL_PAIRS; i++)
    {
        std::cout << sort_out[i];
    }
    cudaFree(gpuAuxData);
    cudaFree(gpuData);
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

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
        input[i].values[0] = data[i][0];
        input[i].values[1] = data[i][1];
    }

    MyPair *dev_pairs;
    MyPair *host_pairs;
    host_pairs = (MyPair *)malloc(TOTAL_PAIRS * sizeof(MyPair));
    for (int i = 0; i < inputNum; i++)
    {
        MyPair pair;
        pair.key = i % 5;
        pair.value = input[i];
        host_pairs[i] = pair;
    }
    // Allocate memory for key-value pairs
    size_t pair_size = TOTAL_PAIRS * sizeof(MyPair);
    cudaMalloc(&dev_pairs, pair_size);
    cudaMemcpy(dev_pairs, host_pairs, pair_size, cudaMemcpyHostToDevice);
    std::cout << "Total number of pairs: " << TOTAL_PAIRS << std::endl;
    MyPair *sort_out = (MyPair *)malloc(TOTAL_PAIRS * sizeof(MyPair));
    gpumerge_sort(dev_pairs, TOTAL_PAIRS, sort_out);
    // printf("Shuffle and sort output\n");
    // print the output of the sort function
    // for (int i = 0; i < TOTAL_PAIRS; i++)
    // {
    //     std::cout << sort_out[i];
    // }
    return 0;
}