#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

void MatrixMul(int *A, int *B, int *C, int N, int K, int M) {
    #pragma acc kernels loop independent copyin(A[0:N*K], B[0:K*M]) copyout(C[0:N*M]) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int sum = 0;
            #pragma acc loop reduction(+:sum) vector
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

int main() {
    int N = 1000; // Number of rows in A and C
    int K = 2000; // Number of columns in A and rows in B
    int M = 1000; // Number of columns in B and C

    size_t sizeA = N * K * sizeof(int);
    size_t sizeB = K * M * sizeof(int);
    size_t sizeC = N * M * sizeof(int);

    int *h_a, *h_b, *h_c;

    // Allocate host memory
    h_a = (int *)malloc(sizeA);
    h_b = (int *)malloc(sizeB);
    h_c = (int *)malloc(sizeC);

    // Initialize matrices
    for (int i = 0; i < N * K; i++) h_a[i] = rand() % 100;
    for (int i = 0; i < K * M; i++) h_b[i] = rand() % 100;

    double start = getTime();
    #pragma acc data copyin(h_a[0:N*K], h_b[0:K*M]) copyout(h_c[0:N*M])
    {
        MatrixMul(h_a, h_b, h_c, N, K, M);
    }
    double end = getTime();

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
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
