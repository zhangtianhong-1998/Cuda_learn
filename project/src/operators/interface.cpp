#include "interface.h"


namespace moperators 
{


  // 普通的乘法算子
  stencil_1d_operator get_stencil_1d_operator(mbase::DeviceType device_type)
  {
    if (device_type == mbase::DeviceType::HOST) 
    {
      // 如果是 HOST，返回 nullptr，意味着没有使用 GPU
      return nullptr;
    } 
    else if (device_type == mbase::DeviceType::Device) 
    {
      // 如果是 GPU 设备，返回指向 add_operator_cu 的函数指针
      return cudaoperators::stencil_1d_operator_cu;
    } 
    else 
    {
      // 如果传入的设备类型未知，记录错误并返回 nullptr
      LOG(FATAL) << "Unknown device type for get a common_matrix_mul operator.";
      return nullptr;
    }
  }

}
