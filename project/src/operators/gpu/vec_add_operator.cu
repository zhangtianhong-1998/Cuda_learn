#include "vec_add_operator.cuh"

namespace cudaoperators
{
  template <typename T>
  __global__ void vec_add_operator_native(const T* input1, const T* input2, T* output, int size) 
  {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (global_index >= size) 
    {
      return;
    }
    
    output[global_index] = input1[global_index] + input2[global_index];
  }

  // grid-stride loop 的形式
  // vector add kernel: C = A + B
  template <typename T>
  __global__ void vec_add_operator(const T* A, const T* B, T* C, int ds)
  {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < ds; idx += gridDim.x * blockDim.x) // a grid-stride loop
      C[idx] = A[idx] + B[idx]; // do the vector (element) add here
  }

  // 在 vec_add_operator.cu 中显式实例化模板
  template void cudaoperators::vec_add_operator_cu<float>(float* input1, float* input2, float* output, const int size);
  template void cudaoperators::vec_add_operator_cu<double>(double* input1, double* input2, double* output, const int size);
  template void cudaoperators::vec_add_operator_cu<int>(int* input1, int* input2, int* output, const int size);
  
  template <typename T>
  void vec_add_operator_cu(T* input1, T* input2, T* output, const int size)
  {

    int32_t block_num = 32;
    int32_t thread_num = 32;

    vec_add_operator<T><<<block_num, thread_num>>>(input1, input2, output, size);
  }
} 

