#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(int *M, int *N, int *P, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < Width) && (col < Width)) {
    int Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[row * Width + k] * N[k * Width + col];
    }
    P[row * Width + col] = Pvalue;
  }
}

// CUDA Kernel function to multiply the matrices is assumed to be defined elsewhere

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main() {
  int N = 1 << 10;
  size_t size = N * N * sizeof(int);

  // host copies of matrices a, b & c
  int *h_a = (int *)malloc(size);
  int *h_b = (int *)malloc(size);
  int *h_c = (int *)malloc(size);
  int *d_a, *d_b, *d_c; // device copies of a, b, & c

  // Setup input values
  for (int i = 0; i < N * N; i++) {
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with N blocks
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

  double start = getTime();
  MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize(); // Wait for the GPU to finish
  double end = getTime();

  // Copy result back to host
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  printf("Result Matrix:\n");
  // Print a subset of the result matrix
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      printf("%d ", h_c[i * N + j]);
    }
    printf("\n");
  }

  printf("Execution Time: %.6f seconds\n", end - start);

  free(h_a); free(h_b); free(h_c);

  return 0;
}
