#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(int *A, int *B, int *C, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
int main() {
    int N = 1000; // Number of rows in A and C
    int K = 2000; // Number of columns in A and rows in B
    int M = 1000; // Number of columns in B and C

    size_t sizeA = N * K * sizeof(int);
    size_t sizeB = K * M * sizeof(int);
    size_t sizeC = N * M * sizeof(int);

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (int *)malloc(sizeA);
    h_b = (int *)malloc(sizeB);
    h_c = (int *)malloc(sizeC);

    // Initialize matrices
    for (int i = 0; i < N * K; i++) h_a[i] = rand() % 100;
    for (int i = 0; i < K * M; i++) h_b[i] = rand() % 100;

    // Allocate device memory
    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_c, sizeC);

    // Copy matrices to the device
    cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice);

    // Determine the number of threads and blocks
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(ceil(M/threadsPerBlock.x), ceil(N/threadsPerBlock.y));

    double start = getTime();
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, K, M);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    double end = getTime();

    // Copy the result back to host
    cudaMemcpy(h_c, d_c, sizeC, cudaMemcpyDeviceToHost);

    // Print some results
    printf("Result Matrix:\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%d ", h_c[i * M + j]);
        }
        printf("\n");
    }

    printf("Execution Time: %.6f seconds\n", end - start);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
