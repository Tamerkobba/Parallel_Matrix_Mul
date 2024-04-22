#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

_global_ void matrixMul(int *a, int *b, int *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main() {
  int N = 2048;
  size_t size = N * N * sizeof(int);

  // host copies of matrices a, b & c
  int *h_a = (int *)malloc(size);
  int *h_b = (int *)malloc(size);
  int *h_c = (int *)malloc(size);

  // device copies of matrices a, b & c
  int *d_a, *d_b, *d_c;

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Setup input values
  for (int i = 0; i < N * N; i++) {
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }

  // Copy inputs to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Launch matrixMul kernel on GPU
  int THREADS = 64;
  int BLOCKS = N / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  double start = getTime();
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
  double end = getTime();

  // Copy result back to host
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
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