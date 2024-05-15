#include <iostream>
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
__device__ void merge(int array[], int *leftArray, int *rightArray, int subArrayOne, int subArrayTwo)
{
    int totalSize = subArrayOne + subArrayTwo;
    int numberThreads = blockDim.x * gridDim.x;
    // printf("numberThreads: %d\n", numberThreads);
    int amountPerThread = ceil((float)totalSize / (float)numberThreads);
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("threadId: %d\n", threadId);
    // index of the output array that this thread will start merging from
    int k = threadId * amountPerThread;

    if (k < totalSize)
    {
        int i = coRank(leftArray, rightArray, subArrayOne, subArrayTwo, k);
        int j = k - i;
        // getting the size of each thread's segment
        // the next thread will start from kNext
        int kNext = min(k + amountPerThread, totalSize);
        int iNext = coRank(leftArray, rightArray, subArrayOne, subArrayTwo, kNext);
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

// __global__ void mergeSortKernel(int *array, int *begin, int *end, int *n)
__global__ void mergeSortKernel(int *array, int *d_begin, int *d_end, int *d_n)
{
    int begin = *d_begin;
    int end = *d_end;
    if (begin >= end)
    {
        return;
    }
    int mid = begin + (end - begin) / 2;
    int *d_mid = new int(mid);
    int arraySize = end - begin + 1;
    // divide the array into 2 subarrays
    int subArrayOne = mid - begin + 1;
    int subArrayTwo = end - mid;
    if (threadIdx.x == 0)
    {
        // mergeSortKernel<<<1, 2>>>(array, d_begin, d_mid, d_n);
        mergeSortKernel<<<1, 1>>>(array, d_begin, d_mid, d_n);
    }
    else
    {
        // mergeSortKernel<<<1, 2>>>(array, d_mid, d_end, d_n);
        mergeSortKernel<<<1, 1>>>(array, d_mid, d_end, d_n);
    }
    // synchronize the threads
    __syncthreads();

    // Create temp arrays
    int *leftArray = new int[subArrayOne],
        *rightArray = new int[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < subArrayOne; i++)
        {
            leftArray[i] = array[begin + i];
            // printf("leftArray[%d]: %d\n", i, leftArray[i]);
        }
    }
    else
    {
        for (int j = 0; j < subArrayTwo; j++)
        {
            rightArray[j] = array[mid + 1 + j];
            // printf("rightArray[%d]: %d\n", j, rightArray[j]);
        }
    }

    __syncthreads();
    // merge the temp arrays back into array[left..right]
    // int numberBlocks = ceil((float)arraySize / 1024);
    // merge<<<numberBlocks, 1024>>>(&array[begin], leftArray, rightArray, &subArrayOne, &subArrayTwo);
    int *tempArray = new int[arraySize];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < subArrayOne; i++)
        {
            tempArray[i] = array[begin + i];
        }
    }
    else
    {
        for (int j = 0; j < subArrayTwo; j++)
        {
            tempArray[j + subArrayOne] = array[mid + 1 + j];
        }
    }
    merge(tempArray, leftArray, rightArray, subArrayOne, subArrayTwo);
    __syncthreads();
    // copy the merged array back to the original array
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < subArrayOne; i++)
        {
            array[begin + i] = tempArray[i];
        }
    }
    else
    {
        for (int j = 0; j < subArrayTwo; j++)
        {
            array[mid + 1 + j] = tempArray[j + subArrayOne];
        }
    }
    __syncthreads();
    // free the memory
    delete[] leftArray;
    delete[] rightArray;
    delete[] tempArray;
    // print
    // if (threadIdx.x == 0)
    // {
    //     printf("Merging begin: %d, end: %d\n", begin, end);
    //     // for (int i = 0; i < arraySize; i++)
    //     // {
    //     //     printf("%d ", array[begin + i]);
    //     // }
    //     // printf("\n");
    // }
    // __syncthreads();
}
// begin is for left index and end is right index
// of the sub-array of arr to be sorted
void mergeSort(int *array, int const begin, int const end, int n)
{
    if (n <= 1)
    {
        return;
    }
    int *d_array;
    int *d_begin, *d_end, *d_n;
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMalloc(&d_begin, sizeof(int));
    cudaMalloc(&d_end, sizeof(int));
    cudaMalloc(&d_n, sizeof(int));
    cudaDeviceSynchronize();
    // copy the array to the device
    cudaMemcpy(d_array, array, n * sizeof(int), cudaMemcpyHostToDevice);
    // copy the begin and end to the device
    cudaMemcpy(d_begin, &begin, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, &end, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Query device properties to get the block size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // int threadNum = prop.maxThreadsPerBlock;
    int threadNum = 2;
    int numberBlocks = ceil((float)n / (float)threadNum);
    // TODO: 2asam al array 3la al threads
    cout << "before kernel\n";
    mergeSortKernel<<<1, 1>>>(d_array, d_begin, d_end, d_n);
    // mergeSortKernel<<<1, threadNum>>>(d_array, begin, end, n);
    cudaDeviceSynchronize();
    cout << "after kernel\n";
    // copy the sorted array from the device to the host
    cudaMemcpy(array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // free the memory
    cudaFree(d_array);
    cudaFree(d_begin);
    cudaFree(d_end);
    cudaFree(d_n);
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
