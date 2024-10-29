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


