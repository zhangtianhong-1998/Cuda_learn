#include "common_matrix_mul_operator.cuh"
namespace cudaoperators
{
    // 矩阵乘法的CUDA内核函数：C = A * B
    __global__ void mmul(const float *A, const float *B, float *C, int ds) 
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


  void common_matrix_mul_operator_cu(const float *A, const float *B, float *C, int row, int col) 
  {

    const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block

    dim3 block(block_size, block_size);  // 每个块包含32x32的线程

    dim3 grid((row + block.x - 1) / block.x, (row + block.y - 1) / block.y);  // 计算网格大小，确保覆盖全部元素

    mmul<<<grid, block>>>(A, B, C, row);  // 启动CUDA内核，执行矩阵乘法s

    // 这里需要传递的是指针
    // common_matrix_mul_operator_cu<<<block_num, thread_num>>>(size, input1, input2, output);

  }
}