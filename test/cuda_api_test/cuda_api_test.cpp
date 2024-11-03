#include <glog/logging.h>
#include <gtest/gtest.h>
#include "utils.h"
#include "../../project/src/operators/interface.h"
#include <iostream>
#include <algorithm>

TEST(test_cuda_api, cuda_Managed) 
{
    const size_t N = 8ULL*1024ULL*1024ULL;  

    float *d_A, *d_sum;
    int blocks = 32;

    checkCudaError(cudaMallocManaged((void**)&d_A, N * sizeof(float)), "cudaMalloc for d_A");
    checkCudaError(cudaMallocManaged((void**)&d_sum, blocks * sizeof(float)), "cudaMalloc for d_A");
 
    float max_val = 5.0f;
    for (int i = 0; i < N; i++)  // initialize matrix in host memory
        d_A[i] = 1.0f;

    d_A[100] = max_val;

    cudaMemPrefetchAsync(d_A, N * sizeof(float), 0);
    cudaMemPrefetchAsync(d_sum, blocks * sizeof(float), 0);

    //计算
    moperators::get_vec_max_operator<float>(mbase::DeviceType::Device)(d_A, d_sum, N);

    cudaMemPrefetchAsync(d_sum, blocks * sizeof(float), cudaCpuDeviceId);
    
    cudaDeviceSynchronize();
    // 检查

    if(abs(max_val - *d_sum) < 1e-5)
    {
        LOG(INFO) << "Max reduction output:\t" << *d_sum;
        LOG(INFO) << "\t\t "<< "Host Max: " << max_val;
    }
    else
    {
        LOG(INFO) << "Max reduction Fault:\t" << *d_sum;
        LOG(INFO) << "\t\t "<< "Host Max: " << max_val;
    }

    cudaFree(d_sum);
    cudaFree(d_A);
}