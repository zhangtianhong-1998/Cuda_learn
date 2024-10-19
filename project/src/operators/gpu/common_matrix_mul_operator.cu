#include "common_matrix_mul_operator.cuh"
namespace cudaoperators
{
    // 矩阵乘法的CUDA内核函数：C = A * B
    __global__ void mmul_common(const float *A, const float *B, float *C, int ds) 
    {

        int idx = threadIdx.x + blockDim.x * blockIdx.x; // 计算当前线程的x索引（全局x坐标）

        int idy = threadIdx.y + blockDim.y * blockIdx.y; // 计算当前线程的y索引（全局y坐标）

        if ((idx < ds) && (idy < ds)) // 防止越界
        {
            float temp = 0;
            for (int i = 0; i < ds; i++) // 对应行和列的点积操作
                temp += A[idy * ds + i] * B[i * ds + idx];   // 计算A的第idy行和B的第idx列的点积
                
            C[idy * ds + idx] = temp;
        }

    }
  
__global__ void mmul(const float *A, const float *B, float *C, int ds, int block_size) 
{
    // Declare dynamic shared memory as a 1D array
    extern __shared__ float shared_mem[];

    // As and Bs are pointers to different parts of shared memory
    float *As = shared_mem;  // First part of shared memory for As
    float *Bs = &shared_mem[block_size * block_size]; // Second part for Bs

    // Calculate global thread indices
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // 全局的列索引
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // 全局的行索引

    if ((idx < ds) && (idy < ds)) 
    {
        float temp = 0;

        // Loop over submatrices of A and B
        for (int i = 0; i < ds / block_size; i++) 
        {
            // Load data into shared memory from global memory
            // Manual 2D indexing for As and Bs using 1D array
            As[threadIdx.y * block_size + threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
            Bs[threadIdx.y * block_size + threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

            // Synchronize threads to ensure all data is loaded
            __syncthreads();

            // Compute partial sum
            for (int k = 0; k < block_size; k++) 
            {
                temp += As[threadIdx.y * block_size + k] * Bs[k * block_size + threadIdx.x];
            }

            // Synchronize threads again before loading new submatrix
            __syncthreads();
        }

        // Write the computed value to global memory
        C[idy * ds + idx] = temp;
    }
}


  void common_matrix_mul_operator_cu(const float *A, const float *B, float *C, int row, int col) 
  {

    int block_size = 3;
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float); // As and Bs

    dim3 block(block_size, block_size);
    dim3 grid((row + block.x - 1) / block.x, (col + block.y - 1) / block.y);

    mmul<<<grid, block, shared_mem_size>>>(A, B, C, row, block_size);

  }
}