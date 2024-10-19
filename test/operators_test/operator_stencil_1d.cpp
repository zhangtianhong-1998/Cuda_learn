#include <glog/logging.h>
#include <gtest/gtest.h>
#include "utils.h"
#include "../../project/src/operators/interface.h"
#include <algorithm>
// Utility function for printing the contents of an array
void printArray(int *array, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

void fill_ints(int *x, int n) 
{
  std::fill_n(x, n, 1);
}

TEST(test_op, operator_stencil_1d) 
{
    int arraySize = 8;

    int padding = 3;
    int real_size = arraySize + 2 * padding;
    int* h_in;
    int* h_out;

    // Alloc space for host copies and setup values
    h_in = (int *)malloc(real_size * sizeof(int)); 
    h_out = (int *)malloc(real_size * sizeof(int)); 
    fill_ints(h_in, real_size);
    fill_ints(h_out, real_size);
    // Device arrays
    int *d_in, *d_out;
    cudaMalloc((void **)&d_in, real_size * sizeof(int));
    cudaMalloc((void **)&d_out, real_size * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_in, h_in, real_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_out, h_out, real_size * sizeof(int) , cudaMemcpyHostToDevice);
    
    // 计算
    moperators::get_stencil_1d_operator(mbase::DeviceType::Device)(d_in, d_out, arraySize, padding);

    // Copy the result back to host
    cudaMemcpy(h_out, d_out, real_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the output
    printf("Input Array:\n");
    printArray(h_in, real_size);

    printf("Output Array:\n");
    printArray(h_out, real_size);

    free(h_in); 
    free(h_out);
    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}