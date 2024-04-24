#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_SIZE 32

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

void MatrixMulTiled(int *A, int *B, int *C, int N, int K, int M) {
    #pragma acc data copyin(A[0:N*K], B[0:K*M]) copyout(C[0:N*M])
    {
        #pragma acc parallel loop
        for (int ii = 0; ii < N; ii += TILE_SIZE) {
            #pragma acc loop
            for (int jj = 0; jj < M; jj += TILE_SIZE) {
                #pragma acc loop
                for (int kk = 0; kk < K; kk += TILE_SIZE) {
                    int i_max = ii + TILE_SIZE > N ? N : ii + TILE_SIZE;
                    int j_max = jj + TILE_SIZE > M ? M : jj + TILE_SIZE;
                    int k_max = kk + TILE_SIZE > K ? K : kk + TILE_SIZE;
                    for (int i = ii; i < i_max; i++) {
                        for (int j = jj; j < j_max; j++) {
                            int sum = C[i * M + j];
                            for (int k = kk; k < k_max; k++) {
                                sum += A[i * K + k] * B[k * M + j];
                            }
                            C[i * M + j] = sum;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int N = 1000;  // Number of rows in A and C
    int K = 2000;  // Number of columns in A and rows in B
    int M = 1000;  // Number of columns in B and C

    size_t sizeA = N * K * sizeof(int);
    size_t sizeB = K * M * sizeof(int);
    size_t sizeC = N * M * sizeof(int);

    int *h_a = (int *)malloc(sizeA);
    int *h_b = (int *)malloc(sizeB);
    int *h_c = calloc(N * M, sizeof(int));  // Use calloc to initialize to zero

    // Initialize matrices
    for (int i = 0; i < N * K; i++) h_a[i] = rand() % 100;
    for (int i = 0; i < K * M; i++) h_b[i] = rand() % 100;

    double start = getTime();
    MatrixMulTiled(h_a, h_b, h_c, N, K, M);
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
