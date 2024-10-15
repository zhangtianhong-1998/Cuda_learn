#ifndef TEST_LOG
#define TEST_LOG
#include <cuda_runtime_api.h>

void log_out();

void checkCudaError(cudaError_t err, const char* action);

void generateRandomArray(int*& a, int n);
#endif  // TEST_CU_CUH
