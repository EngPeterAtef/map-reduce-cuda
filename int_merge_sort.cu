#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define MAX_DEPTH 1024
__global__ void simple_mergesort(int *data, int *dataAux, int begin, int end, int depth)
{
    int middle = (end + begin) / 2;
    int i0 = begin;
    int i1 = middle;
    int index;
    int n = end - begin;

    // Used to implement recursions using CUDA parallelism.
    cudaStream_t s, s1;

    if (n < 2)
    {
        return;
    }

    // Launches a new block to sort the left part.
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    simple_mergesort<<<1, 1, 0, s>>>(data, dataAux, begin, middle, depth + 1);
    cudaStreamDestroy(s);

    // Launches a new block to sort the right part.
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    simple_mergesort<<<1, 1, 0, s1>>>(data, dataAux, middle, end, depth + 1);
    cudaStreamDestroy(s1);

    // Merges children's generated partition.
    // Does the merging using the auxiliary memory.
    for (index = begin; index < end; index++)
    {
        if (i0 < middle && (i1 >= end || data[i0] <= data[i1]))
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
void gpumerge_sort(int *a, int n)
{
    int *gpuData;
    int *gpuAuxData;
    int left = 0;
    int right = n;
    // Query device properties to get the block size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_threads_per_block = prop.maxThreadsPerBlock;

    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    // Allocate GPU memory.
    cudaMalloc((void **)&gpuData, n * sizeof(int));
    cudaMalloc((void **)&gpuAuxData, n * sizeof(int));
    cudaMemcpy(gpuData, a, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch on device
    simple_mergesort<<<1, max_threads_per_block>>>(gpuData, gpuAuxData, left, right, 0);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(a, gpuData, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuAuxData);
    cudaFree(gpuData);
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

int main()
{
    // create array of size 5000
    int a[10];
    for (int i = 0; i < 10; i++)
    {
        a[i] = 10 - i;
    }
    int n = sizeof(a) / sizeof(a[0]);
    gpumerge_sort(a, n);
    for (int i = 0; i < n; i++)
    {
        printf("%d ", a[i]);
    }
    return 0;
}