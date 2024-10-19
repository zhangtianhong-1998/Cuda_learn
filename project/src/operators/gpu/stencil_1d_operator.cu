#include "stencil_1d_operator.cuh"
#include <stdio.h>
namespace cudaoperators
{
  __global__ void stencil_1d_operator(const int* in, int* out, int block_size, int padding) 
  {
    extern __shared__ int temp[];  // 动态分配的共享内存
    int globla_index = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + padding;

    // Read input elements into shared memory
    temp[lindex] = in[globla_index];
    
    if (threadIdx.x < padding) 
    {
        temp[lindex - padding] = in[globla_index - padding];  // Possible out of bounds access here
        temp[lindex + block_size] = in[globla_index + block_size];  // Possible out of bounds access here
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -padding; offset <= padding; offset++)
        result += temp[lindex + offset];

    // Store the result
    out[globla_index] = result;
  }

  void stencil_1d_operator_cu(int *in, int *out, int arraySize, int padding) 
  {
    int block_size = 8;
    int grid_size = arraySize / block_size;

    printf("arraySize: %d, grid_size: %d, padding: %d\n", arraySize, grid_size, padding);

    int shared_mem_size = block_size + 2 * padding;
    // 这里需要传递的是指针
    stencil_1d_operator<<<grid_size, block_size, shared_mem_size>>>(in + padding, out + padding, block_size, padding);

  }
} 
