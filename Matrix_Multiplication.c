#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void matrixMul(int *a, int *b, int *c, int N, int K, int M) {
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < M; col++) {
      c[row * M + col] = 0;
      for (int k = 0; k < K; k++) {
        c[row * M + col] += a[row * K + k] * b[k * M + col];
      }
    }
  }
}

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main() {
  int N = 1000; // Number of rows in A and C
  int K = 2000;  // Number of columns in A and rows in B
  int M = 1000;  // Number of columns in B and C

  size_t sizeA = N * K * sizeof(int);
  size_t sizeB = K * M * sizeof(int);
  size_t sizeC = N * M * sizeof(int);

  // host copies of matrices a, b & c
  int *h_a = (int *)malloc(sizeA);
  int *h_b = (int *)malloc(sizeB);
  int *h_c = (int *)malloc(sizeC);

  // Setup input values
  for (int i = 0; i < N * K; i++) {
    h_a[i] = rand() % 100;
  }
  for (int i = 0; i < K * M; i++) {
    h_b[i] = rand() % 100;
  }

  double start = getTime();
  matrixMul(h_a, h_b, h_c, N, K, M);
  double end = getTime();

  printf("Result Matrix:\n");
  // Print a subset of the result matrix
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      printf("%d ", h_c[i * M + j]);
    }
    printf("\n");
  }

  printf("Execution Time: %.6f seconds\n", end - start);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
