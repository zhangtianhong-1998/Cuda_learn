#ifndef INTERFACE_H
#define INTERFACE_H
#include <base/base.h>
#include "gpu/vec_add_operator.cuh"
#include "gpu/matrix_operator.cuh"
#include "gpu/stencil_1d_operator.cuh"
#include <functional>
namespace moperators 
{
    //---------------向量加法计算核心---------------------------------------------
    // 泛型 AddKernel 定义，使用模板来支持不同的数据类型
    template <typename T>
    using add_operator = std::function<void(T* input1, T* input2, T* output, const int size)>;

    // 根据设备类型返回不同的加法内核函数指针，支持泛型
    template <typename T>
    add_operator<T> get_vec_add_operator(mbase::DeviceType device_type);

    template <typename T>
    add_operator<T> get_vec_add_operator(mbase::DeviceType device_type) 
    {
        if (device_type == mbase::DeviceType::HOST) 
        {
        // 如果是 HOST，返回 nullptr，意味着没有使用 GPU
        return nullptr;
        } 
        else if (device_type == mbase::DeviceType::Device) 
        {
            return [](T* input1, T* input2, T* output, int size) {

                // 调用模板实例化的内核函数
                cudaoperators::vec_add_operator_cu(input1, input2, output, size);
            };
        }
        else 
        {
        // 如果传入的设备类型未知，记录错误并返回 nullptr
        LOG(FATAL) << "Unknown device type for get a add operator.";
        return nullptr;
        }
    }
//*----------------------向量归约和-----------------
    template <typename T>
    using vec_sum_operator = std::function<void(T* input, T* output, const int size)>;

    // 根据设备类型返回不同的加法内核函数指针，支持泛型
    template <typename T>
    vec_sum_operator<T> get_vec_sum_operator(mbase::DeviceType device_type);

    template <typename T>
    vec_sum_operator<T> get_vec_sum_operator(mbase::DeviceType device_type) 
    {
        if (device_type == mbase::DeviceType::HOST) 
        {
        // 如果是 HOST，返回 nullptr，意味着没有使用 GPU
        return nullptr;
        } 
        else if (device_type == mbase::DeviceType::Device) 
        {
            return [](T* input, T* output, int size) {

                // 调用模板实例化的内核函数
                cudaoperators::vec_sum_operator_cu(input, output, size);
            };
        }
        else 
        {
        // 如果传入的设备类型未知，记录错误并返回 nullptr
        LOG(FATAL) << "Unknown device type for get a add operator.";
        return nullptr;
        }
    }
//*----------------------向量维度最大化-----------------
    template <typename T>
    using vec_max_operator = std::function<void(T* input, T* output, const int size)>;

    // 根据设备类型返回不同的加法内核函数指针，支持泛型
    template <typename T>
    vec_max_operator<T> get_vec_max_operator(mbase::DeviceType device_type);

    template <typename T>
    vec_max_operator<T> get_vec_max_operator(mbase::DeviceType device_type) 
    {
        if (device_type == mbase::DeviceType::HOST) 
        {
        // 如果是 HOST，返回 nullptr，意味着没有使用 GPU
        return nullptr;
        } 
        else if (device_type == mbase::DeviceType::Device) 
        {
            return [](T* input, T* output, int size) {

                // 调用模板实例化的内核函数
                cudaoperators::vec_max_operator_cu(input, output, size);
            };
        }
        else 
        {
        // 如果传入的设备类型未知，记录错误并返回 nullptr
        LOG(FATAL) << "Unknown device type for get a add operator.";
        return nullptr;
        }
    }
    //---------------END ---------------------------------------------
    //---------------矩阵乘法算子泛型---------------------------------------------
    template <typename T>
    using com_matrix_mul_operator = std::function<void(const T* A, const T* B, T* C, int row, int col)>;

    template <typename T>
    com_matrix_mul_operator<T> get_com_matrix_mul_operator(mbase::DeviceType device_type);

    // 普通的乘法算子
    template <typename T>
    com_matrix_mul_operator<T> get_com_matrix_mul_operator(mbase::DeviceType device_type) 
    {
        if (device_type == mbase::DeviceType::HOST) 
        {

            return nullptr;
        } 
        else if (device_type == mbase::DeviceType::Device) 
        {
            return [](const T* A, const T* B, T* C, int row, int col) {

                // 调用模板实例化的内核函数
                cudaoperators::common_matrix_mul_operator_cu(A, B, C, row, col);
            };
        } 
        else 
        {
            // 如果传入的设备类型未知，记录错误并返回 nullptr
            LOG(FATAL) << "Unknown device type for get a common_matrix_mul operator.";
            return nullptr;
        }
    }
    //---------------END ---------------------------------------------
    //---------------矩阵按行求和---------------------------------------------
    template <typename T>
    using matrix_row_sum_operator = std::function<void(const T* A, T* sum, const int row)>;

    template <typename T>
    matrix_row_sum_operator<T> get_matrix_row_sum_operator(mbase::DeviceType device_type);

    // 普通的乘法算子
    template <typename T> 
    matrix_row_sum_operator<T> get_matrix_row_sum_operator(mbase::DeviceType device_type) 
    {
        if (device_type == mbase::DeviceType::HOST) 
        {
            return nullptr;
        } 
        else if (device_type == mbase::DeviceType::Device) 
        {
            return [](const T* A, T* sum, const int row) 
            {
                cudaoperators::matrix_row_sum_operator_cu(A, sum, row);
            };
        } 
        else 
        {
            // 如果传入的设备类型未知，记录错误并返回 nullptr
            LOG(FATAL) << "Unknown device type for get a matrix_row_sum_operator operator.";
            return nullptr;
        }
    }
    //---------------END ---------------------------------------------
    //---------------矩阵按列求和---------------------------------------------
    template <typename T>
    using matrix_col_sum_operator = std::function<void(const T* A, T* sum, const int col)>;

    template <typename T>
    matrix_col_sum_operator<T> get_matrix_col_sum_operator(mbase::DeviceType device_type);

    template <typename T>
    matrix_col_sum_operator<T> get_matrix_col_sum_operator(mbase::DeviceType device_type) 
    {
        if (device_type == mbase::DeviceType::HOST) 
        {
            return nullptr;
        } 
        else if (device_type == mbase::DeviceType::Device) 
        {
            return [](const T* A, T* sum, const int col) 
            {
                cudaoperators::matrix_col_sum_operator_cu(A, sum, col);
            };
        } 
        else 
        {
            LOG(FATAL) << "Unknown device type for get a matrix_col_sum_operator operator.";
            return nullptr;
        }
    }
    //---------------END ---------------------------------------------


    typedef void (*stencil_1d_operator)(int *in, int *out, int arraySize, int padding);

    // 根据设备类型返回不同的加法内核
    stencil_1d_operator get_stencil_1d_operator(mbase::DeviceType device_type);
}
#endif  // INTERFACE_H
