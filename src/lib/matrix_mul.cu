#include <cuda_runtime.h>
#include "matrix_mul.cuh"

__global__ void matMulKernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matMul(const float* A, const float* B, float* C, int n) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n*n*sizeof(float));
    cudaMalloc(&d_B, n*n*sizeof(float));
    cudaMalloc(&d_C, n*n*sizeof(float));

    cudaMemcpy(d_A, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n*n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(n, n);
    dim3 numBlocks(1, 1);
    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
