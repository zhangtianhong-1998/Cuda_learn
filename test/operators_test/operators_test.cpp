#include <glog/logging.h>
#include <gtest/gtest.h>
#include <ctime>
#include "utils.h"
#include "../../project/src/operators/interface.h"
#include <stdio.h>


TEST(test_op, vec_add) 
{
    const int N = 32 * 1048576;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c; // CUDA 指针
    int size = N * sizeof(float); // 向量维数 

    // 分配 CPU 内存
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // 分配 GPU 内存，并检查错误
    checkCudaError(cudaMalloc((void**)&d_a, size), "cudaMalloc for d_a");
    checkCudaError(cudaMalloc((void**)&d_b, size), "cudaMalloc for d_b");
    checkCudaError(cudaMalloc((void**)&d_c, size), "cudaMalloc for d_c");

    // 生成随机数组
    // generateRandomArray(a, N);
    // generateRandomArray(b, N);

    for (int i = 0; i < N; i++)
    {  // initialize vectors in host memory
        a[i] = rand()/(float)RAND_MAX;
        b[i] = rand()/(float)RAND_MAX;
        c[i] = 0;
    }
    // 复制数据到设备
    checkCudaError(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice), "cudaMemcpy for d_a");
    checkCudaError(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice), "cudaMemcpy for d_b");

    // 调用 CUDA 内核，传递 N 而不是 size
    moperators::get_vec_add_operator<float>(mbase::DeviceType::Device)(d_a, d_b, d_c, N);

    // 检查内核执行错误
    checkCudaError(cudaGetLastError(), "Kernel execution");

    // 将结果从设备复制回主机
    checkCudaError(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy for d_c");

    // 打印部分结果
    for (int i = 0; i < N; ++i) 
    {
        if(c[i] != (a[i] + b[i]))
        {
            std::cout <<"Vec A :"<< a[i] << std::endl;
            std::cout <<"Vec B :"<< b[i] << std::endl;
            std::cout <<"Vec C :"<< c[i] << std::endl;
        }
    }

    // 释放内存
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


TEST(test_op, vec_sum) 
{
    const size_t N = 8ULL * 1024ULL * 1024ULL;  
    // const int N = 4;
    float *h_A, *h_sum, *d_A, *d_sum;

    h_A = new float[N];  // allocate space for data in host memory
    h_sum = new float;

    for (int i = 0; i < N; i++)  // initialize matrix in host memory
        h_A[i] = 1.0f;

    // 分配 GPU 内存，并检查错误
    checkCudaError(cudaMalloc((void**)&d_A, N * sizeof(float)), "cudaMalloc for d_A");
    checkCudaError(cudaMalloc((void**)&d_sum, sizeof(float)), "cudaMalloc for d_sum");

    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));
    //计算
    moperators::get_vec_sum_operator<float>(mbase::DeviceType::Device)(d_A, d_sum, N);
    // 拷贝
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    // 检查
    float sum = 0.0;
    for (int i = 0; i < N; i++)  // initialize matrix in host memory
        sum += h_A[i];

    if((sum - *h_sum) < 1e-5)
    {
        LOG(INFO) << "Sums correct!\t"<< "Device sum: " << *h_sum;
        LOG(INFO) << "\t\t "<< "Host sum: " << sum;
    }
    else
    {
        LOG(INFO) << "Sums fault!\n "<< "Device sum: " << *h_sum;
        LOG(INFO) << "\t\t "<< "Host sum: " << sum;
    }
    cudaFree(d_sum);
    cudaFree(d_A);
    free(h_sum);
    free(h_A);
}


TEST(test_op, vec_max) 
{
    const size_t N = 8ULL*1024ULL*1024ULL;  

    float *h_A, *h_sum, *d_A, *d_sum;

    float max_val = 5.0f;
    
    h_A = new float[N];  // allocate space for data in host memory
    h_sum = new float;

    for (int i = 0; i < N; i++)  // initialize matrix in host memory
        h_A[i] = 1.0f;

    h_A[100] = max_val;

    int blocks = 32;
    // 分配 GPU 内存，并检查错误
    checkCudaError(cudaMalloc((void**)&d_A, N * sizeof(float)), "cudaMalloc for d_A");
    checkCudaError(cudaMalloc((void**)&d_sum, blocks * sizeof(float)), "cudaMalloc for d_sum");

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    //计算
    moperators::get_vec_max_operator<float>(mbase::DeviceType::Device)(d_A, d_sum, N);
    // 拷贝
    cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    // 检查

    if(abs(max_val - *h_sum) < 1e-5)
    {
        LOG(INFO) << "Max reduction output:\t" << *h_sum;
        LOG(INFO) << "\t\t "<< "Host Max: " << max_val;
    }
    else
    {
        LOG(INFO) << "Max reduction Fault:\t" << *h_sum;
        LOG(INFO) << "\t\t "<< "Host Max: " << max_val;
    }

    cudaFree(d_sum);
    cudaFree(d_A);
    free(h_sum);
    free(h_A);
}
