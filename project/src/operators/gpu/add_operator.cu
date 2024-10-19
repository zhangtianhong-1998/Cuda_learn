#include "add_operator.cuh"

namespace cudaoperators
{
  __global__ void add_operator(int32_t size, const int* in1, const int* in2, int* out) 
  {
    int32_t global_index = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (global_index >= size) 
    {
      return;
    }
    
    float in_val1 = in1[global_index];
    float in_val2 = in2[global_index];
    
    out[global_index] = in_val1 + in_val2;
  }

  void add_operator_cu(int* input1, int* input2, int* output, const int size) 
  {

    int32_t thread_num = 512;
    int32_t block_num = (size + thread_num - 1) / thread_num;

    // 这里需要传递的是指针
    add_operator<<<block_num, thread_num>>>(size, input1, input2, output);

  }
} 
