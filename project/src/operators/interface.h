#ifndef INTERFACE_H
#define INTERFACE_H
#include <base/base.h>
#include "gpu/add_operator.cuh"
#include "gpu/common_matrix_mul_operator.cuh"
#include "gpu/stencil_1d_operator.cuh"

namespace moperators 
{
    // 定义函数指针类型，用于指向加法内核函数
    typedef void (*AddKernel)(int* input1, int* input2, int* output, const int size);

    // 根据设备类型返回不同的加法内核
    AddKernel get_add_operator(mbase::DeviceType device_type);

    typedef void (*com_matrix_mul_operator)(const float *A, const float *B, float *C, int row, int col);

    // 根据设备类型返回不同的加法内核
    com_matrix_mul_operator get_com_matrix_mul_operator(mbase::DeviceType device_type);


    typedef void (*stencil_1d_operator)(int *in, int *out, int arraySize, int padding);

    // 根据设备类型返回不同的加法内核
    stencil_1d_operator get_stencil_1d_operator(mbase::DeviceType device_type);
}
#endif  // INTERFACE_H
