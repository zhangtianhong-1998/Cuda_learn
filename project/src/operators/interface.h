#ifndef INTERFACE_H
#define INTERFACE_H
#include <base/base.h>
#include "gpu/add_operator.cuh"

namespace moperators 
{
    // 定义函数指针类型，用于指向加法内核函数
    typedef void (*AddKernel)(int* input1, int* input2, int* output, const int size);

    // 根据设备类型返回不同的加法内核
    AddKernel get_add_operator(mbase::DeviceType device_type);
}
#endif  // INTERFACE_H
