#include <glog/logging.h>
#include <gtest/gtest.h>
#include "utils.h"
#include "../../project/src/operators/interface.h"
#include <iostream>
#include <algorithm>

TEST(test_op, com_matrix_mul) 
{
    const int DSIZE = 9;

    const float A_val = 3.0f;
    const float B_val = 2.0f;

    // 主机端（CPU）内存指针
    float *h_A, *h_B, *h_C;
    // 设备端（GPU）内存指针
    float *d_A, *d_B, *d_C;
    // 用于记录时间
    clock_t t0, t1, t2;

    double t1sum=0.0;  // 初始化时间
    double t2sum=0.0;  // 计算时间

    // 开始计时
    t0 = clock();

    h_A = new float[DSIZE*DSIZE];

    h_B = new float[DSIZE*DSIZE];

    h_C = new float[DSIZE*DSIZE];
    
    for (int i = 0; i < DSIZE * DSIZE; i++)
    {
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }
    // 初始化结束，记录时间
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;  // 计算初始化时间

    std::cout<<"Init took %f seconds.  Begin compute\n"<<t1sum<<std::endl;  // 输出初始化时间
    // 为设备端分配内存，并将主机端数据复制到设备端
    // 分配 GPU 内存，并检查错误
    checkCudaError(cudaMalloc((void**)&d_A, DSIZE*DSIZE*sizeof(float)), "cudaMalloc for d_a");
    checkCudaError(cudaMalloc((void**)&d_B, DSIZE*DSIZE*sizeof(float)), "cudaMalloc for d_b");
    checkCudaError(cudaMalloc((void**)&d_C, DSIZE*DSIZE*sizeof(float)), "cudaMalloc for d_c");

    checkCudaError(cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy for d_a");
    checkCudaError(cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy for d_b");

    moperators::get_com_matrix_mul_operator<float>(mbase::DeviceType::Device)(d_A, d_B, d_C, DSIZE, DSIZE);
    
    // CUDA处理的第二步完成（计算完成）

    // 将结果从设备端复制回主机端
    checkCudaError(cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy for d_b");

    // 计算完成，记录时间
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;  // 计算计算时间
    std::cout<<"Done. Compute took %f seconds\n"<<t2sum<<std::endl;  // 输出计算时间

    // CUDA处理的第三步完成（结果返回主机）

    for (int i = 0; i < DSIZE * DSIZE; i++)  // 遍历所有元素进行验证
        if (h_C[i] != A_val * B_val * DSIZE) 
        {  // 如果计算结果不正确，输出错误信息
            std::cout<<"mismatch at index %d, was: %f, should be: %f\n" << i << h_C[i] << A_val*B_val*DSIZE<<std::endl;
        }
    std::cout << "Success!\n" <<std::endl;  // 如果验证成功，输出成功信息

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}


bool validate(const float* data, size_t sz)
{
    float expected_value = static_cast<float>(sz); // 将传入的数组大小 sz 转换为浮点数类型 float，并存储在变量 expected_value 中。

    // 使用 std::find_if 找到第一个不符合条件的元素
    const float* result = std::find_if(data, data + sz, [expected_value](float value) {
        return value != expected_value;
    });

    if (result != data + sz) 
    {
        size_t index = result - data;  // 计算结果的索引
        std::cout << "results mismatch at " << index << ", was: " << *result 
                  << ", should be: " << expected_value << std::endl;
        return false;
    }

    return true;
}


TEST(test_op, matrix_row_sum_and_col_sum) 
{
    const size_t DSIZE = 16384;      // matrix side dimension

    float *h_A, *h_sums, *d_A, *d_sums;

    h_A = new float[DSIZE*DSIZE];  // allocate space for data in host memory
    h_sums = new float[DSIZE]();

    for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
        h_A[i] = 1.0f;
    for (int i = 0; i < DSIZE; i++)  // initialize matrix in host memory
        h_sums[i] = 0.0f;

    checkCudaError(cudaMalloc((void**)&d_A, DSIZE*DSIZE*sizeof(float)), "cudaMalloc for d_a");
    checkCudaError(cudaMalloc((void**)&d_sums, DSIZE*sizeof(float)), "cudaMalloc for d_sums");

    checkCudaError(cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy for d_a");
    checkCudaError(cudaMemcpy(d_sums, h_sums, DSIZE*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy for d_sums");

    moperators::get_matrix_row_sum_operator<float>(mbase::DeviceType::Device)(d_A, d_sums, DSIZE);

    checkCudaError(cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy for h_sums");

    if (!validate(h_sums, DSIZE)) 
    {
        LOG(INFO) << "row sums Fault!\n";
    }else{
        LOG(INFO) <<"row sums correct!\n";
    }

// ----------------------------------------------列求和

    checkCudaError(cudaMemset(d_sums, 0, DSIZE*sizeof(float)), "cudaMemset for d_sums");

    moperators::get_matrix_col_sum_operator<float>(mbase::DeviceType::Device)(d_A, d_sums, DSIZE);

    checkCudaError(cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy for d_sums");
    if (!validate(h_sums, DSIZE)) 
    {
        LOG(INFO) << "col sums Fault!\n";
    }else{
        LOG(INFO) <<"col sums correct!\n";
    }

    // 释放内存
    free(h_A);
    free(h_sums);
    cudaFree(d_A);
    cudaFree(d_sums);


}