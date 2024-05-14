#include <cuda_runtime.h>
#include <iostream>
#include "config.cuh"
#include <fstream>
#include <chrono>
#include <vector>
// C++ program for Merge Sort
using namespace std;

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
__device__ void mergeSequential(int *array, int *leftArray, int *rightArray, int subArrayOne, int subArrayTwo)
{
    auto indexOfSubArrayOne = 0, indexOfSubArrayTwo = 0;
    // the start of the merged array in the original array
    int indexOfMergedArray = 0;
    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo)
    {
        if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo])
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

__device__ int coRank(int *leftArray, int *rightArray, int subArrayOne, int subArrayTwo, int k)
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
        if (i < subArrayOne && j > 0 && leftArray[i] < rightArray[j - 1])
        {
            // i is too small, must increase it
            iLow = i + 1;
        }
        // our guess is too big, we must decrease i
        else if (i > 0 && j < subArrayTwo && leftArray[i - 1] > rightArray[j])
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
__global__ void merge(int array[], int *leftArray, int *rightArray, int *subArrayOne, int *subArrayTwo)
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
void printArray(int A[], int size)
{
    for (int i = 0; i < size; i++)
        cout << A[i] << " ";
    cout << endl;
}
// begin is for left index and end is right index
// of the sub-array of arr to be sorted
void mergeSort(int *array, int const begin, int const end, int n)
{
    if (begin >= end)
    {
        return;
    }

    int mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid, n);
    mergeSort(array, mid + 1, end, n);
    // cout << "Merging: " << begin << " " << mid << " " << end << endl;
    // divide the array into 2 subarrays
    int const subArrayOne = mid - begin + 1;
    int const subArrayTwo = end - mid;
    int arraySize = end - begin + 1;

    // Create temp arrays
    auto *leftArray = new int[subArrayOne],
         *rightArray = new int[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[begin + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];
    int *d_array;
    int *d_leftArray, *d_rightArray;
    int *d_subArrayOne, *d_subArrayTwo;
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMalloc(&d_leftArray, subArrayOne * sizeof(int));
    cudaMalloc(&d_rightArray, subArrayTwo * sizeof(int));
    cudaMalloc(&d_subArrayOne, sizeof(int));
    cudaMalloc(&d_subArrayTwo, sizeof(int));
    cudaDeviceSynchronize();
    // copy the array to the device
    cudaMemcpy(d_array, array, n * sizeof(int), cudaMemcpyHostToDevice);
    // copy the leftArray and the rightArray to the device
    cudaMemcpy(d_leftArray, leftArray, subArrayOne * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightArray, rightArray, subArrayTwo * sizeof(int), cudaMemcpyHostToDevice);
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
    cudaMemcpy(array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // free the memory
    cudaFree(d_array);
    cudaFree(d_leftArray);
    cudaFree(d_rightArray);
    cudaFree(d_subArrayOne);
    cudaFree(d_subArrayTwo);
    delete[] leftArray;
    delete[] rightArray;
}

// UTILITY FUNCTIONS
void mergeGpu(int *a, int *b, int *c, int sizeA, int sizeB)
{
    int *d_a, *d_b, *d_c;
    int *d_sizeA, *d_sizeB;
    cudaMalloc(&d_a, sizeA * sizeof(int));
    cudaMalloc(&d_b, sizeB * sizeof(int));
    cudaMalloc(&d_c, (sizeA + sizeB) * sizeof(int));
    cudaMalloc(&d_sizeA, sizeof(int));
    cudaMalloc(&d_sizeB, sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(d_a, a, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizeA, &sizeA, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizeB, &sizeB, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Query device properties to get the block size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int numberBlocks = ceil((float)(sizeA + sizeB) / (float)prop.maxThreadsPerBlock);
    merge<<<numberBlocks, prop.maxThreadsPerBlock>>>(d_c, d_a, d_b, d_sizeA, d_sizeB);
    cudaDeviceSynchronize();
    // copy data from device to host
    cudaMemcpy(c, d_c, (sizeA + sizeB) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
}
void testMerge()
{
    // create array of size 2500
    int n = 2500;
    // take n as input
    cout << "Enter the size of the array: ";
    cin >> n;
    int *a = (int *)malloc(n * sizeof(int));
    int *b = (int *)malloc(n * sizeof(int));
    int *c = (int *)malloc(2 * n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        a[i] = i + 1;
        b[i] = i + 1;
    }
    // printArray(a, n);
    cout << "Size: " << n << endl;
    // cout << "Given array is \n";
    // printArray(a, n);
    mergeGpu(a, b, c, n, n);
    // mergeSort(a, 0, n - 1);

    cout << "\nSorted array is \n";
    printArray(c, 2 * n);
}

int main()
{
    // testMerge();
    // create array of size 2500
    int n = 2500;
    // take n as input
    cout << "Enter the size of the array: ";
    cin >> n;
    int *a = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % (n / 2);
    }
    // printArray(a, n);
    // cout << "Size: " << n << endl;
    // cout << "Given array is \n";
    // printArray(a, n);

    mergeSort(a, 0, n - 1, n);
    cudaDeviceReset();

    cout << "\nSorted array is \n";
    printArray(a, n);
    return 0;
}
