#include <glog/logging.h>
#include <gtest/gtest.h>
#include <filesystem>  // 添加此头文件

int main(int argc, char* argv[]) 
{
    testing::InitGoogleTest(&argc, argv);

    FLAGS_log_dir = "./log/";

    // 创建 log 文件夹
    std::filesystem::create_directories(FLAGS_log_dir);

    google::InitGoogleLogging("Project");


    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start Test...\n";

    return RUN_ALL_TESTS();
}
