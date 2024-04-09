#include <iostream>
#include <math.h>
#include "cuda_runtime.h"

// Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

// function to add the elements of two arrays
// parallel GPU  // MultiThread by CUDA
__global__
void add_block(int n, float* x, float* y)
{
    //Thread index of the current Thread
    int index = threadIdx.x;
    // blockDim number of threads in the block
    int stride = blockDim.x;
    
    printf("ThreadIdx" + threadIdx.x);
    std::cout << "ThreadIdx" << threadIdx.x << std::endl;
    printf("BlockIdx" + blockIdx.x);

    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}


// parallel GPU  // MultiThread by CUDA in grid-stride loop https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fcuda-pro-tip-write-flexible-kernels-grid-stride-loops%2F
__global__
void add_grid(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}


int main(void)
{
    int N = 1 << 20; // 1M elements

    /*
        float* x = new float[N];
        float* y = new float[N];
    */
    // Allocate Unified Memory -- accessible from CPU or GPU
    float* x, * y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));


    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    // Run kernel on 1M elements on the GPU
    add <<<1, 1 >>> (N, x, y);

    // 2nd Run kernel on 1M elements on the GPU
    add_block <<<1, 256 >> > (N, x, y);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // 3rd Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add_grid <<<numBlocks, blockSize >>> (N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    /*
    delete[] x;
    delete[] y;
    */
    cudaFree(x);
    cudaFree(y);

    return 0;
}