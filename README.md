# 这是一个从零学习Cuda的记录
这是我得bilibili主页
https://space.bilibili.com/347417212?spm_id_from=333.999.0.0

## 0. cmake安装
```
sudo apt install g++
sudo apt install gcc
sudo apt install cmake
```
## 1. googletest 安装
https://github.com/google/googletest
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 
sudo make install 
```

## 2. googletest 安装
https://github.com/google/glog
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF ..
make -j8 
sudo make install 
```

## 3. cuda安装
略

## 参考项目
- https://github.com/zjhellofss/KuiperLLama
- https://github.com/zjhellofss/KuiperInfer
