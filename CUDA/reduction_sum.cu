#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void sum_reduction_kernel(float *input, float *output, int n)
{
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int start = blockIdx.x * blockDim.x * 2;

    // Load data from global memory to shared memory with coalescing
    shared_mem[tid] = input[start + tid];
    shared_mem[tid + blockDim.x] = input[start + blockDim.x + tid];

    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (tid == 0)
    {
        atomicAdd(output, shared_mem[0]);
    }
}

int main()
{
    const int ARRAY_SIZE = 1024;
    const int GRID_SIZE = (ARRAY_SIZE + (BLOCK_SIZE)-1) / (BLOCK_SIZE);

    // Initialize array
    float *h_input = new float[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, sizeof(float) * ARRAY_SIZE);
    cudaMalloc((void **)&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    // Initialize output value to 0
    cudaMemset(d_output, 0, sizeof(float));

    // Launch kernel
    sum_reduction_kernel<<<GRID_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, ARRAY_SIZE);

    // Copy result back to host
    float final_sum;
    cudaMemcpy(&final_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum of array elements: %f\n", final_sum);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;

    return 0;
}
