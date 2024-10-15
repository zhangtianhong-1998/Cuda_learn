#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include "../utils.h"
#include "../../project/src/operators/interface.h"
#define N (20480 * 20480)  // 设置较大的 N


TEST(test_op, log_test) 
{
    std::cout<<"123456"<<std::endl;
}

TEST(test_op, add) 
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // CUDA 指针
    int size = N * sizeof(int); // 向量维数 

    // 分配 GPU 内存，并检查错误
    checkCudaError(cudaMalloc((void**)&d_a, size), "cudaMalloc for d_a");
    checkCudaError(cudaMalloc((void**)&d_b, size), "cudaMalloc for d_b");
    checkCudaError(cudaMalloc((void**)&d_c, size), "cudaMalloc for d_c");

    // 分配 CPU 内存
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // 生成随机数组
    generateRandomArray(a, N);
    generateRandomArray(b, N);

    // 复制数据到设备
    checkCudaError(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice), "cudaMemcpy for d_a");
    checkCudaError(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice), "cudaMemcpy for d_b");

    // 调用 CUDA 内核，传递 N 而不是 size
    moperators::get_add_operator(mbase::DeviceType::Device)(d_a, d_b, d_c, N);

    // 检查内核执行错误
    checkCudaError(cudaGetLastError(), "Kernel execution");

    // 将结果从设备复制回主机
    checkCudaError(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy for d_c");

    // 打印部分结果
    for (int i = 0; i < 10; ++i)  // 打印前10个结果
    {
        std::cout << c[i] << std::endl;
    }

    // 释放内存
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}