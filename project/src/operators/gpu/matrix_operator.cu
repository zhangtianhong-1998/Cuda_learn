#include "matrix_operator.cuh"
namespace cudaoperators
{
    
// -----------------------------------矩阵乘法----------------------------------
    // 矩阵乘法的CUDA内核函数：C = A * B
    template <typename T>
    __global__ void mmul_common(const T *A, const T *B, T *C, int ds) 
    {

        int idx = threadIdx.x + blockDim.x * blockIdx.x; // 计算当前线程的x索引（全局x坐标）

        int idy = threadIdx.y + blockDim.y * blockIdx.y; // 计算当前线程的y索引（全局y坐标）

        if ((idx < ds) && (idy < ds)) // 防止越界
        {
            T temp = 0;
            for (int i = 0; i < ds; i++) // 对应行和列的点积操作
                temp += A[idy * ds + i] * B[i * ds + idx];   // 计算A的第idy行和B的第idx列的点积
                
            C[idy * ds + idx] = temp;
        }

    }
    template <typename T>
    __global__ void mmul(const T *A, const T *B, T *C, int ds, int block_size) 
    {
        // Declare dynamic shared memory without a type qualifier
        extern __shared__ char shared_mem[];  // Use char to declare the memory as untyped
        
        // Cast shared memory to the correct type and split it into As and Bs
        T *As = (T*)shared_mem;  // First part of shared memory for As
        T *Bs = (T*)&As[block_size * block_size]; // Second part for Bs

        // Calculate global thread indices
        int idx = threadIdx.x + blockDim.x * blockIdx.x; // 全局的列索引
        int idy = threadIdx.y + blockDim.y * blockIdx.y; // 全局的行索引

        if ((idx < ds) && (idy < ds)) 
        {
            T temp = 0;

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

  template void cudaoperators::common_matrix_mul_operator_cu<float>(const float* A, const float* B, float* C, int row, int col);
  template void cudaoperators::common_matrix_mul_operator_cu<double>(const double* A, const double* B, double* C,  int row, int col);
  template void cudaoperators::common_matrix_mul_operator_cu<int>(const int* A, const int* B, int* C, int row, int col);

  template <typename T>
  void common_matrix_mul_operator_cu(const T *A, const T *B, T *C, int row, int col) 
  {

    int block_size = 3;
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(T); // As and Bs

    dim3 block(block_size, block_size);
    dim3 grid((row + block.x - 1) / block.x, (col + block.y - 1) / block.y);

    mmul<T><<<grid, block, shared_mem_size>>>(A, B, C, row, block_size);

  }

// -----------------------------------矩阵行和-----------------------------------

    // // 矩阵行和 优化前
    // template <typename T>
    // __global__ void row_sums(const T* A, T* sums, size_t ds)
    // {
    //     int idx = threadIdx.x + blockDim.x * blockIdx.x; // create typical 1D thread index from built-in variables
    //     if (idx < ds)
    //     {
    //         T sum = 0;
    //         for (size_t i = 0; i < ds; i++)
    //         sum += A[idx*ds+i];      
    //         sums[idx] = sum;
    //     }
    // }
    // 优化后
    template <typename T>
    __global__ void row_sums(const T* A, T* sums, size_t ds)
    {
        int idx = blockIdx.x; // 实际是矩阵的列号

        if (idx < ds)
        {
            extern __shared__ char share_mem[];

            T *sdata = (T*)share_mem; 

            int tid = threadIdx.x;
            
            sdata[tid] = 0.0f;

            size_t tidx = tid;

            while (tidx < ds) 
            {  // block stride loop to load data
                sdata[tid] += A[idx * ds + tidx];
                tidx += blockDim.x;  
            }

            for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
            {
                __syncthreads();
                if (tid < s)  // parallel sweep reduction
                    sdata[tid] += sdata[tid + s];
            }

            if (tid == 0) atomicAdd(&sums[idx], sdata[0]);
        }
    }

    template void cudaoperators::matrix_row_sum_operator_cu<float>(const float* A, float* sum, const int row);
    template void cudaoperators::matrix_row_sum_operator_cu<double>(const double* A, double* sum,  const int row);
    template void cudaoperators::matrix_row_sum_operator_cu<int>(const int* A, int* sum,  const int row);

    template <typename T>
    void matrix_row_sum_operator_cu(const T* A, T* sum, const int row) 
    {
        int block_size = 256;
        // int grid = (row + block_size - 1) / block_size;
        int grid = row;
        
        row_sums<T><<<grid, block_size, block_size * sizeof(T)>>>(A, sum, row);

    }

// -----------------------------------矩阵列和-----------------------------------
    // template <typename T>
    // __global__ void column_sums(const T *A, T *sums, size_t ds)
    // {
    //     int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //     if (idx < ds)
    //     {
    //         T sum = 0.0f;
    //         for (size_t i = 0; i < ds; i++)
    //             sum += A[idx + ds * i];      
    //         sums[idx] = sum;
    //     }
    // }
    template <typename T>
    __global__ void column_sums(const T* A, T* sums, size_t ds)
    {
        int idx = blockIdx.x; // 实际是矩阵的行号
        if (idx < ds)
        {
            extern __shared__ char share_mem[];

            T *sdata = (T*)share_mem; 

            int tid = threadIdx.x;
            
            sdata[tid] = 0.0f;

            size_t tidx = tid;
            // block stride loop to load data
            while (tidx < ds)  
            {  
                sdata[tid] += A[idx  + tidx * ds];

                tidx += blockDim.x;  
            }

            for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
            {
                __syncthreads();
                // parallel sweep reduction
                if (tid < s)  
                    sdata[tid] += sdata[tid + s];
            }

            if (tid == 0) atomicAdd(&sums[idx], sdata[0]);
        }
    }

    template void cudaoperators::matrix_col_sum_operator_cu<float>(const float* A, float* sum, const int col);
    template void cudaoperators::matrix_col_sum_operator_cu<double>(const double* A, double* sum, const int col);
    template void cudaoperators::matrix_col_sum_operator_cu<int>(const int* A, int* sum,  const int col);

    template <typename T>
    void matrix_col_sum_operator_cu(const T* A, T* sum, const int col) 
    {
        int block_size = 256;
        // int grid = (col + block_size - 1) / block_size;
        // column_sums<T><<<grid, block_size>>>(A, sum, col);
        int grid = col;
        column_sums<T><<<grid, block_size, block_size * sizeof(T)>>>(A, sum, col);
    }

}