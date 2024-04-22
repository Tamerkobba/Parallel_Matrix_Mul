#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void matrixMul(int *a, int *b, int *c, int N) {
  #pragma acc data copyin(a[0:N*N], b[0:N*N]) copy(c[0:N*N])
  {
    #pragma acc kernels
    for (int row = 0; row < N; row++) {
      for (int col = 0; col < N; col++) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
          sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
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
  int N = 1 << 10; // 1024
  size_t size = N * N * sizeof(int);

  int *h_a = (int *)malloc(size);
  int *h_b = (int *)malloc(size);
  int *h_c = (int *)malloc(size);

  for (int i = 0; i < N * N; i++) {
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }

  double start = getTime();
  matrixMul(h_a, h_b, h_c, N);
  double end = getTime();

  printf("Result Matrix:\n");

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      printf("%d ", h_c[i * N + j]);
    }
    printf("\n");
  }

  printf("Execution Time: %.6f seconds\n", end - start);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
