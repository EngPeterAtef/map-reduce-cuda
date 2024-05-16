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

// Function to print an array
void printArray(MyPair A[], int size)
{
    for (int i = 0; i < size; i++)
        cout << A[i] << " ";
    cout << endl;
}
// begin is for left index and end is right index
// of the sub-array of arr to be sorted
void mergeSort(MyPair *array, int const begin, int const end, int n)
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
    auto *leftArray = new MyPair[subArrayOne],
         *rightArray = new MyPair[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[begin + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];
    MyPair *d_array;
    MyPair *d_leftArray, *d_rightArray;
    int *d_subArrayOne, *d_subArrayTwo;
    cudaMalloc(&d_array, n * sizeof(MyPair));
    cudaMalloc(&d_leftArray, subArrayOne * sizeof(MyPair));
    cudaMalloc(&d_rightArray, subArrayTwo * sizeof(MyPair));
    cudaMalloc(&d_subArrayOne, sizeof(int));
    cudaMalloc(&d_subArrayTwo, sizeof(int));
    cudaDeviceSynchronize();
    // copy the array to the device
    cudaMemcpy(d_array, array, n * sizeof(MyPair), cudaMemcpyHostToDevice);
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
    cudaFree(d_array);
    cudaFree(d_leftArray);
    cudaFree(d_rightArray);
    cudaFree(d_subArrayOne);
    cudaFree(d_subArrayTwo);
    delete[] leftArray;
    delete[] rightArray;
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
    int inputNum = (int)data.size();

    NUM_INPUT = inputNum;
    TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

    // Allocate host memory
    size_t input_size = NUM_INPUT * sizeof(input_type);
    input_type *input = (input_type *)malloc(input_size);

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *)malloc(output_size);
    cout << "b3d al file" << endl;

    // copy from vector to array
    for (int i = 0; i < inputNum; i++)
    {
        input[i].values[0] = data[i][0];
        input[i].values[1] = data[i][1];
    }
    cout << "ba3d al malloc" << endl;
    MyPair *pairs;
    pairs = (MyPair *)malloc(TOTAL_PAIRS * sizeof(MyPair));
    for (int i = 0; i < inputNum; i++)
    {
        MyPair pair;
        pair.key = i % NUM_OUTPUT;
        pair.value = input[i];
        pairs[i] = pair;
    }
    // Allocate memory for key-value pairs
    cout << "inputNum: " << inputNum << endl;
    mergeSort(pairs, 0, inputNum - 1, inputNum);
    cudaDeviceReset();

    cout << "\nSorted array is \n";
    if (isSorted(pairs, inputNum))
    {
        cout << "Sorted\n";
    }
    else
    {
        cout << "Not Sorted\n";
    }
    // printArray(pairs, inputNum);
    return 0;
}
