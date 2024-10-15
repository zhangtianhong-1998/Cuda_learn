#include <glog/logging.h>
#include "utils.h"
#include <iostream>

void log_out()
{
    LOG(INFO) << "test LOG";
}


void checkCudaError(cudaError_t err, const char* action) 
{
    if (err != cudaSuccess) 
    {
        std::cerr << "CUDA Error during: " << action << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void generateRandomArray(int*& a, int n) 
{
    a = new int[n];
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < n; ++i) 
    {
        a[i] = rand() % 100;
    }
}