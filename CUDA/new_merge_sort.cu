#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include "config.cuh"
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <vector>
#include <fstream>
using namespace std;
#define MAX_THREADS_PER_BLOCK 1024

// Bitonic Sort for CPU
void bitonicSortCPU(MyPair *arr, int n)
{
    for (int k = 2; k <= n; k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)
        {
            for (int i = 0; i < n; i++)
            {
                int ij = i ^ j;

                if (ij > i)
                {
                    if ((i & k) == 0)
                    {
                        if (PairCompareGreater()(arr[i], arr[ij]))
                        {
                            MyPair temp;
                            temp.key = arr[i].key;
                            temp.value = arr[i].value;
                            arr[i].key = arr[ij].key;
                            arr[i].value = arr[ij].value;
                            arr[ij].key = temp.key;
                            arr[ij].value = temp.value;
                        }
                    }
                    else
                    {
                        if (PairCompare()(arr[i], arr[ij]))
                        {
                            MyPair temp;
                            temp.key = arr[i].key;
                            temp.value = arr[i].value;
                            arr[i].key = arr[ij].key;
                            arr[i].value = arr[ij].value;
                            arr[ij].key = temp.key;
                            arr[ij].value = temp.value;
                        }
                    }
                }
            }
        }
    }
}

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
                MyPair temp;
                temp.key = arr[i].key;
                temp.value = arr[i].value;
                arr[i].key = arr[ij].key;
                arr[i].value = arr[ij].value;
                arr[ij].key = temp.key;
                arr[ij].value = temp.value;
            }
        }
        else
        {
            if (PairCompare()(arr[i], arr[ij]))
            {
                MyPair temp;
                temp.key = arr[i].key;
                temp.value = arr[i].value;
                arr[i].key = arr[ij].key;
                arr[i].value = arr[ij].value;
                arr[ij].key = temp.key;
                arr[ij].value = temp.value;
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
__global__ void MergeSortGPU(MyPair *arr, MyPair *temp, int n, int width)
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

// CPU Merge Recursive Call function
void mergeCPU(MyPair *arr, MyPair *temp, int left, int mid, int right)
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right)
    {
        if (PairCompareLessEql()(arr[i], arr[j]))
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];

    while (j <= right)
        temp[k++] = arr[j++];

    for (int idx = left; idx <= right; ++idx)
        arr[idx] = temp[idx];
}

// CPU Implementation of Merge Sort
void mergeSortCPU(MyPair *arr, MyPair *temp, int left, int right)
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    mergeSortCPU(arr, temp, left, mid);
    mergeSortCPU(arr, temp, mid + 1, right);

    mergeCPU(arr, temp, left, mid, right);
}

// Function to print array
void printArray(MyPair *arr, int size)
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
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

// MAIN PROGRAM
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Requires 1 argument, name of input textfile\n");
        exit(1);
    }
    string filename = argv[1];

    // Read data from text file
    std::ifstream file(filename);
    std::vector<std::vector<int>> data; // Vector of vectors to store the data
    cout << "abl al file" << endl;
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
    int size = (int)data.size();

    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "CUDA MERGE AND BITONIC SORT IMPLEMENTATION" << std::endl;
    std::cout << "A Performance Comparison of These 2 Sorts in CPU vs GPU" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    int choice;
    std::cout << "\nSelect the type of sort:";
    std::cout << "\n\t1. Merge Sort";
    std::cout << "\n\t2. Bitonic Sort";
    std::cout << "\nEnter your choice: ";
    std::cin >> choice;

    if (choice < 1 || choice > 2)
    {
        while (choice != 1 || choice != 2)
        {
            std::cout << "\n!!!!! WRONG CHOICE. TRY AGAIN. YOU HAVE ONLY 2 DISTINCT OPTIONS-\n";
            std::cin >> choice;

            if (choice == 1 || choice == 2)
                break;
        }
    }

    if (choice == 1)
    {
        std::cout << "\n--------------------------------------------------------------\nMERGE SORT SELECTED\n--------------------------------------------------------------";
    }
    else
    {
        std::cout << "\n--------------------------------------------------------------\nBITONIC SORT SELECTED\n--------------------------------------------------------------";
    }

    std::cout << "\n--------------------------------------------------------------\nSELECTED SORT PROCESS UNDERWAY\n--------------------------------------------------------------";

    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;

    // Initialize CPU clock counters
    clock_t startCPU, endCPU;

    // Set number of threads and blocks for kernel calls
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Create CPU based Arrays
    MyPair *arr = new MyPair[size];
    MyPair *carr = new MyPair[size];
    MyPair *temp = new MyPair[size];

    for (int i = 0; i < size; i++)
    {
        MyPair pair;
        pair.key = i % NUM_OUTPUT;
        pair.value.values[0] = data[i][0];
        pair.value.values[1] = data[i][1];
        arr[i] = pair;
        carr[i] = pair;
    }
    // Create GPU based arrays
    MyPair *gpuTemp;
    MyPair *gpuArrbiton;
    MyPair *gpuArrmerge;
    // Allocate memory on GPU
    cudaMalloc(&gpuTemp, size * sizeof(MyPair));
    cudaMalloc(&gpuArrbiton, size * sizeof(MyPair));
    cudaMalloc(&gpuArrmerge, size * sizeof(MyPair));

    // Copy the input array to GPU memory
    cudaMemcpy(gpuArrmerge, arr, size * sizeof(MyPair), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuArrbiton, arr, size * sizeof(MyPair), cudaMemcpyHostToDevice);

    // Main If else block
    if (choice == 1)
    {
        // Call GPU Merge Kernel and time the run
        cudaEventRecord(startGPU);
        for (int wid = 1; wid < size; wid *= 2)
        {
            MergeSortGPU<<<threadsPerBlock, blocksPerGrid>>>(gpuArrmerge, gpuTemp, size, wid * 2);
        }
        cudaEventRecord(stopGPU);

        // Transfer sorted array back to CPU
        cudaMemcpy(arr, gpuArrmerge, size * sizeof(MyPair), cudaMemcpyDeviceToHost);

        // Calculate Elapsed GPU time
        cudaEventSynchronize(stopGPU);
        cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);

        // Time the CPU and call CPU Merge Sort
        startCPU = clock();
        mergeSortCPU(carr, temp, 0, size - 1);
        endCPU = clock();
    }
    else
    {
        int j, k;

        // Time the run and call GPU Bitonic Kernel
        cudaEventRecord(startGPU);
        for (k = 2; k <= size; k <<= 1)
        {
            for (j = k >> 1; j > 0; j = j >> 1)
            {
                bitonicSortGPU<<<blocksPerGrid, threadsPerBlock>>>(gpuArrbiton, j, k);
            }
        }
        cudaEventRecord(stopGPU);

        // Transfer Sorted array back to CPU
        cudaMemcpy(arr, gpuArrbiton, size * sizeof(MyPair), cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stopGPU);
        cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);

        // Time the run and call CPU Bitonic Sort
        startCPU = clock();
        bitonicSortCPU(carr, size);
        endCPU = clock();
    }

    // Calculate Elapsed CPU time
    double millisecondsCPU = static_cast<double>(endCPU - startCPU) / (CLOCKS_PER_SEC / 1000.0);

    // Display sorted GPU array
    std::cout << "\n\nSorted GPU array: ";
    // printArray(arr, size);

    // Display sorted CPU array
    std::cout << "\nSorted CPU array: ";
    if (size <= 100)
    {
        printArray(carr, size);
    }
    else
    {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    // Run the array with the automated isSorted checker
    if (isSorted(arr, size))
        std::cout << "\n\nSORT CHECKER RUNNING - SUCCESFULLY SORTED GPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;

    if (isSorted(carr, size))
        std::cout << "SORT CHECKER RUNNING - SUCCESFULLY SORTED CPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;

    // Print the time of the runs
    std::cout << "\n\nGPU Time: " << millisecondsGPU << " ms" << std::endl;
    std::cout << "CPU Time: " << millisecondsCPU << " ms" << std::endl;

    // Destroy all variables
    delete[] arr;
    delete[] carr;
    delete[] temp;

    // End
    cudaFree(gpuArrmerge);
    cudaFree(gpuArrbiton);
    cudaFree(gpuTemp);

    std::cout << "\n------------------------------------------------------------------------------------\n||||| END. YOU MAY RUN THIS AGAIN |||||\n------------------------------------------------------------------------------------";
    return 0;
}