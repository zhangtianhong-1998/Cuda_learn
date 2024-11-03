#ifndef BASE_H_
#define BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>


namespace mbase 
{
    // 设备类型
    enum class DeviceType : uint8_t 
    {
        Unknown = 0,
        HOST = 1,
        Device = 2,
    };

    enum class DataType : uint8_t {
        Unknown = 0,
        Float32 = 1,
        Int8 = 2,
        Int32 = 3,
        Float64 = 4,
    };
}

#endif  // BASE_H_
