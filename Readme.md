# Parallelizing Matrix Multiplication using GPUs
Tamer Kobba
202104873
# Introduction
Matrix multiplication is an essential operation in various scientific computations, including deep learning and image processing. It involves the multiplication of a matrix with dimensions I x J (I rows by J columns), matrix M, and a matrix with dimensions J x K, matrix N, resulting in a matrix with dimensions I x K, matrix P. Each element of the output matrix P is the inner product of a row of M and a column of N. To parallelize this operation on GPUs, we utilize CUDA and OpenACC programming models.

# Basic CUDA Implementation
The CUDA programming model allows for direct control over thread and block distribution on GPU hardware. In our implementation, each thread calculates a single element in the result matrix `P`, which involves computing the dot product of a row from matrix `M` and a column from matrix `N`.

## Key Concepts:
- **Matrix Element Calculation**: Each GPU thread is responsible for computing one element of the output matrix. For the element at position `(row, col)`, the thread calculates the dot product of the `row`-th row of `M` and the `col`-th column of `N`.

- **Thread and Block Assignment**: Threads are organized into a two-dimensional grid of blocks, each containing a two-dimensional array of threads. The `row` and `col` indices for each thread are derived from its position within the block and the grid.

- **Execution Blocks**: The dimensions of the grid and blocks are calculated based on the size of the matrices to ensure coverage of all elements in matrix `P`. Blocks are often sized as `16x16`, meaning each block computes a `16x16` sub-section of `P`.

# Tiled Implementation Using Shared Memory

1. **Division into Tiles**: Matrix multiplication is divided into smaller sections, or "tiles," enabling the operation to be processed in manageable chunks that fit into the fast shared memory on each GPU block.

2. **Shared Memory Allocation**: Each block of threads allocates shared memory for a tile from each matrix (`s_a` for matrix `M` and `s_b` for matrix `N`). The size of these shared memory arrays (`TILE_WIDTH x TILE_WIDTH`) must fit within the GPU's shared memory limits while accommodating the entire tile.

3. **Loading Tiles into Shared Memory**:
   - Each thread loads one element from both matrices `M` and `N` into shared memory.
   - Threads load elements based on their block’s position relative to the entire matrix, ensuring correct data alignment for the multiplication.

4. **Synchronization**:
   - Threads synchronize using `__syncthreads()` after loading the data into shared memory and before starting the computation. This synchronization ensures that all required data is loaded and prevents data hazards.

5. **Performing Multiplication on Tiles**:
   - Threads compute the output by multiplying corresponding elements from `s_a` and `s_b` and accumulating the results in a local variable `Cvalue`.
   - This process iterates over the width of the tile to complete the dot product calculation for each element.

6. **Handling Multiple Tiles**:
   - For matrices larger than the tile size, the kernel processes the matrices in segments. It iteratively loads new tiles into shared memory and repeats the multiplication process.
   - Threads must synchronize after processing each tile to ensure all computations are complete before loading new data.

This tiled approach minimizes global memory access, which is a common bottleneck in GPU computations due to its relatively slow access speeds. By maximizing the use of shared memory, the kernel reduces the time spent on memory operations, thereby enhancing performance.


# Github Link
[https://github.com/Tamerkobba/Parallel_Matrix_Mul](https://github.com/Tamerkobba/Parallel_Matrix_Mul)

# Performance
# Sequential
We took the average time after running the program 10 times
Avergae Sequential time =24.06 seconds
# Parallel
I tested a variety of configurations regarding the number of threads per block and the number of blocks per grid:(We assumed `N`=1000 and `M`=2000)

I ensured that Blocks per grid should be enough to cover r all elements you want to compute for matrix multiplication

and for varying the threads per blocks I used such  (8x8), (16x16), (32x32)....



I just modify the values for `a` and `b` and test out 
# Cuda
```c 
dim3 threadsPerBlock(a,b);
dim3 blocksPerGrid(ceil(M/threadsPerBlock.x), ceil(N/threadsPerBlock.y));
```
## Basic

### Speedup Factor

| Configuration Name            | Threads per Block | Blocks per Grid    |   Speed-Up Factor|
|-------------------------------|-------------------|--------------------|-----------------|
| Ultra-Minimal Thread Blocks   | 2x2 (4 threads)   | (1000, 500)        | 168.859     |
| Very Small Thread Blocks      | 4x4 (16 threads)  | (500, 250)         | 441.75      |
| Small Thread Blocks           | 8x8 (64 threads)  | (250, 125)         |  874.785592      |
| Moderate Thread Blocks        | 16x16 (256 threads)| (125, 63)          | 1301.369      |
| Medium Thread Blocks          | 32x32 (1024 threads)| (63, 32)          |  1507.5846      |
| Large Thread Blocks           | 64x64 (4096 threads)| (32, 16)          |`wrong output gave matrix of zeros`    |



### Efficiency

| Configuration Name            | Threads per Block | Blocks per Grid    | Efficiency Measurement |
|-------------------------------|-------------------|--------------------|-----------------|
| Ultra-Minimal Thread Blocks   | 2x2 (4 threads)   | (1000, 500)        | 8.44295e-5      |
| Very Small Thread Blocks      | 4x4 (16 threads)  | (500, 250)         | 0.000220875   |
| Small Thread Blocks           | 8x8 (64 threads)  | (250, 125)         |  0.000437      |
| Moderate Thread Blocks        | 16x16 (256 threads)| (125, 63)          | 0.0006506    |
| Medium Thread Blocks          | 32x32 (1024 threads)| (63, 32)          |  0.0007537923    |
| Large Thread Blocks           | 64x64 (4096 threads)| (32, 16)          |`wrong output gave matrix of zeros`       |


### Scalability


## Tiled
For the tiled I modified the kernel configurations to be like so
```c
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(ceil((M + TILE_WIDTH - 1) / TILE_WIDTH), ceil((N + TILE_WIDTH - 1) / TILE_WIDTH));
```
### Speedup factor


| Configuration Name            | Threads per Block | Blocks per Grid    |   Speed-Up Factor|
|-------------------------------|-------------------|--------------------|-----------------|
| Ultra-Minimal Thread Blocks   | 2x2 (4 threads)   | (1000, 500)        | 57.913      |
| Very Small Thread Blocks      | 4x4 (16 threads)  | (500, 250)         | 207.9555      |
| Small Thread Blocks           | 8x8 (64 threads)  | (250, 125)         |  311.03   |
| Moderate Thread Blocks        | 16x16 (256 threads)| (125, 63)          | 1762.61  |
| Medium Thread Blocks          | 32x32 (1024 threads)| (63, 32)          |  1823.31       |
| Large Thread Blocks           | 64x64 (4096 threads)| (32, 16)          |`wrong output gave matrix of zeros`      |



### Efficiency
| Configuration Name            | Threads per Block | Blocks per Grid    | Efficiency Measurement |
|-------------------------------|-------------------|--------------------|-----------------|
| Ultra-Minimal Thread Blocks   | 2x2 (4 threads)   | (1000, 500)        | 2.59565e-5     |
| Very Small Thread Blocks      | 4x4 (16 threads)  | (500, 250)         | 0.0001039     |
| Small Thread Blocks           | 8x8 (64 threads)  | (250, 125)         | 0.000155515    |
| Moderate Thread Blocks        | 16x16 (256 threads)| (125, 63)          | 0.000881305     |
| Medium Thread Blocks          | 32x32 (1024 threads)| (63, 32)          |  0.00091166     |
| Large Thread Blocks           | 64x64 (4096 threads)| (32, 16)          |`wrong output gave matrix of zeros`       |
### Scalability
# OpenACC
## Basic
`S(P)=1.3131544`

## Tiled
## Speedup factor
`S(P)=19.3958741`
# Comparison Tiled and basic
## CUDA
### Methodologies
- **Basic Implementation**: Direct multiplication without utilizing shared memory, potentially causing high global memory traffic.
- **Tiled Implementation**: Utilizes shared memory to reduce global memory access, expected to enhance performance by reducing memory fetch times.

## Speed-Up Factor

The speed-up factor measures the improvement in execution time compared to a baseline. Higher values indicate better performance.

| Configuration Name            | Basic Speed-Up Factor | Tiled Speed-Up Factor |
|-------------------------------|-----------------------|-----------------------|
| Ultra-Minimal Thread Blocks   | 168.859               | 57.913                |
| Very Small Thread Blocks      | 441.75                | 207.9555              |
| Small Thread Blocks           | 874.785592            | 311.03                |
| Moderate Thread Blocks        | 1301.369              | 1762.61               |
| Medium Thread Blocks          | 1507.5846             | 1823.31               |
| Large Thread Blocks           | Error (zeros output)  | Error (zeros output)  |

- **Tiled Implementation** generally shows higher speed-up factors in larger configurations, indicating better utilization of GPU resources through shared memory.
- Both implementations fail with the largest thread block size due to likely exceeding shared memory limits, resulting in incorrect outputs.

## Efficiency

Efficiency is measured as the speed-up per unit of computing resource. Higher values are better, indicating more effective use of GPU threads.

| Configuration Name            | Basic Efficiency      | Tiled Efficiency      |
|-------------------------------|-----------------------|-----------------------|
| Ultra-Minimal Thread Blocks   | 8.44295e-5            | 2.59565e-5            |
| Very Small Thread Blocks      | 0.000220875           | 0.0001039             |
| Small Thread Blocks           | 0.000437              | 0.000155515           |
| Moderate Thread Blocks        | 0.0006506             | 0.000881305           |
| Medium Thread Blocks          | 0.0007537923          | 0.00091166            |
| Large Thread Blocks           | Error (zeros output)  | Error (zeros output)  |

### Analysis
- **Tiled Implementation** shows better efficiency especially in moderate and medium configurations due to reduced memory overhead and faster access times.

### Observations
- Both methods scale similarly up to medium thread blocks but experience issues with the largest size due to possible memory constraints.
- **Tiled Implementation** shows better scalability in terms of maintaining performance gains with increasing grid sizes, likely benefiting from reduced memory traffic and better data locality.

The Tiled Implementation using shared memory provides significant advantages in terms of speed-up and efficiency for moderate to medium-sized thread blocks, making it a preferable choice for matrix multiplication tasks on GPUs. Both implementations require careful management of memory and thread configurations to avoid performance degradation or computational errors at larger scales.

## OpenACC
# Comparison of Matrix Multiplication Implementations using OpenACC

This section compares two implementations of matrix multiplication: a straightforward implementation using OpenACC parallelism and a tiled version. Both methods aim to optimize matrix multiplication on parallel computing architectures, specifically leveraging GPU acceleration with OpenACC directives.

## Basic OpenACC Implementation

- **Method**: Utilizes basic loop parallelism with OpenACC to distribute matrix multiplication operations across GPU threads.
- **Code Structure**: Three nested loops iterate over the rows and columns of the matrices, with the innermost loop performing the multiplication and accumulation for each element.
- **Parallelization**: The outer two loops are marked with `#pragma acc loop independent` to enable parallel execution, and the innermost loop uses a reduction clause to sum the products.
- **Performance Metric**: The execution time recorded results in a speed-up factor (`S(P)`) of `1.3131544`. This indicates a modest improvement over serial execution, but suggests that there might be memory bandwidth or computation limitations affecting performance.

## Tiled OpenACC Implementation

- **Method**: Implements tiling to improve data locality and cache usage, which are critical for performance on GPUs. Tiling breaks down the matrix into smaller sub-matrices or "tiles", processed in chunks.
- **Code Structure**: Similar to the basic implementation but includes an additional layer of loops to handle the tiles. Each tile's boundaries are calculated before processing to handle edge cases where tiles do not fit perfectly into the matrix dimensions.
- **Parallelization**: Uses `#pragma acc parallel loop` for the outer loops handling the tiles, ensuring that each tile is processed in parallel, and reduces global memory accesses by focusing on local sections of the matrix at a time.
- **Performance Metric**: The tiled approach significantly improves performance, achieving a speed-up factor (`S(P)`) of `19.3958741`. This substantial increase highlights the effectiveness of tiling in optimizing memory access patterns and reducing cache misses.

The tiled implementation outperforms the basic parallel approach by a significant margin, demonstrating the benefits of optimizing data locality in matrix multiplication on GPUs. By reducing the number of global memory accesses and improving cache usage, the tiled method maximizes the throughput of the GPU, making it highly effective for large matrix sizes. This approach is particularly beneficial in environments where memory bandwidth is a limiting factor for performance.

# Conclusion

Throughout this report, we have explored various implementations of matrix multiplication on GPUs, utilizing both CUDA and OpenACC programming models. Our analysis has covered basic and tiled implementations, examining their performance across different configurations.

In CUDA, the basic implementation provided direct control over the computation but suffered from significant memory traffic issues, leading to less efficient execution. The tiled implementation, which utilized shared memory, demonstrated superior performance by minimizing global memory access and maximizing data locality. This approach showed especially notable improvements in speed-up and efficiency in moderate to medium-sized thread blocks, highlighting the benefits of shared memory in managing larger matrix computations effectively.

The OpenACC implementations further reinforced these findings, where the basic approach showed limited improvement over sequential execution, while the tiled version achieved a significant increase in speed-up factor. This underscores the effectiveness of tiling in optimizing GPU resource usage and enhancing computational speed.

Both CUDA and OpenACC have proven to be powerful tools for parallelizing matrix multiplication on GPUs. The choice of implementation—whether basic or tiled—depends on the specific requirements of the application and the hardware capabilities available. For applications demanding high performance and efficient memory usage, the tiled approach is clearly advantageous, providing substantial improvements in speed and efficiency.

This report highlights the importance of choosing the right GPU programming model and optimization technique for maximizing performance in scientific computing tasks. As GPUs continue to evolve, the potential for further optimization and performance gains remains significant, promising even faster and more efficient computation for complex mathematical operations like matrix multiplication.
