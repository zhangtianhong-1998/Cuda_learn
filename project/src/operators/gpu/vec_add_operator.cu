#include "vec_add_operator.cuh"
#include <cfloat>   // 包含浮点数的最大最小值
#include <limits>   // 包含 std::numeric_limits
#include <type_traits> // 包含 std::is_same

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
    int32_t thread_num = 256;

    vec_add_operator<T><<<block_num, thread_num>>>(input1, input2, output, size);
  }

//-----------向量规约-----------------------------
  template <typename T>
  __global__ void vec_sum(T* gdata, T* out, const int size)
  {
    extern __shared__ char share_mem[];
    T *sdata = (T*)share_mem; 
    
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    T val = 0;

    unsigned mask = 0xFFFFFFFFU;
    
    int lane = threadIdx.x % warpSize; // warp 中的第几个线程
    
    int warpID = threadIdx.x / warpSize; //第几个warp
    
    while (idx < size) 
    {  // grid stride loop to load 
        val += gdata[idx];
        idx += gridDim.x * blockDim.x;  
    }

    // 第一步规约
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    
    if (lane == 0) sdata[warpID] = val; // 更新到warp中的第0个位置
    __syncthreads(); // put warp results in shared mem

    // hereafter, just warp 0
    if (warpID == 0)
    {
        // 线程在块的内部 ，从共享内存中取数据
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;
        // final warp-shuffle reduction
        for (int offset = warpSize/2; offset > 0; offset >>= 1) 
          val += __shfl_down_sync(mask, val, offset);
          
        if  (tid == 0) atomicAdd(out, val);
    }
  }

 // 在 vec_add_operator.cu 中显式实例化模板
  template void cudaoperators::vec_sum_operator_cu<float>(float* input, float* output, const int size);
  template void cudaoperators::vec_sum_operator_cu<double>(double* input, double* output, const int size);
  template void cudaoperators::vec_sum_operator_cu<int>(int* input, int* output, const int size);
  template <typename T>
  void vec_sum_operator_cu(T* input, T* output, const int size)
  {
    int32_t block_num = 32;
    int32_t thread_num = 256;
    int32_t warp_size = 32;
    vec_sum<T><<<block_num, thread_num, warp_size>>>(input, output, size);
  }

//-----------沿向量维度取最大值（规约实现----------------------------
// 因为每种类型的最小值由特化的 get_min_value 函数提供，编译器只会实例化合适的类型，这样就避免了不兼容的类型赋值，消除了警告。
  template <typename T>
  __device__ T get_min_value();
  template <>
  __device__ int get_min_value<int>() { return INT_MIN; }
  template <>
  __device__ float get_min_value<float>() { return -FLT_MAX; }
  template <>
  __device__ double get_min_value<double>() { return -DBL_MAX; }
  template <typename T>
  __global__ void vec_max(T* gdata, T* out, const int size)
  {
    extern __shared__ char share_mem[];
    T *sdata = (T*)share_mem; 
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
      
    // 使用特化的设备函数来获取最小值，避免不兼容类型赋值
    T val = get_min_value<T>();

    // grid stride loop to load data
    while (idx < size) 
    {
        val = max(val, gdata[idx]);
        idx += gridDim.x * blockDim.x;
    }

    // 将每个线程的结果存入共享内存
    sdata[tid] = val;
    __syncthreads();

    // 使用标准共享内存规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    // 将结果写回全局内存
    if (tid == 0) out[blockIdx.x] = sdata[0];
  }

 // 在 vec_add_operator.cu 中显式实例化模板
  template void cudaoperators::vec_max_operator_cu<float>(float* input, float* output, const int size);
  template void cudaoperators::vec_max_operator_cu<double>(double* input, double* output, const int size);
  template void cudaoperators::vec_max_operator_cu<int>(int* input, int* output, const int size);
  template <typename T>
  void vec_max_operator_cu(T* input, T* output, const int size)
  {
    int32_t block_num = 32;
    int32_t thread_num = 256;
    int32_t warp_size = 32;

    vec_max<T><<<block_num, thread_num, warp_size>>>(input, output, size); // 对于每个block求取最大 局部最大
    // d_a的
    vec_max<T><<<1, thread_num, warp_size>>>(output, input, block_num); // 进一步求取全局最大
  }
} 

