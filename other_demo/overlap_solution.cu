#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

// 定义可修改的常量和类型
typedef float ft;                     // 定义`ft`为`float`类型的别名，便于在代码中修改精度。
const int chunks = 64;                // 定义数据分块数量，用于并行处理。
const size_t ds = 1024*1024*chunks;   // 定义数据集大小，每个分块有1024*1024个元素。
const int count = 22;                 // 定义滑动窗口的大小，用于计算高斯PDF的平均值。
const int num_streams = 8;            // 定义CUDA流的数量，用于异步处理。

const float sqrt_2PIf = 2.5066282747946493232942230134974f;
const double sqrt_2PI = 2.5066282747946493232942230134974;


__device__ float gpdf(float val, float sigma) 
{
  return expf(-0.5f * val * val) / (sigma * sqrt_2PIf);
}

__device__ double gpdf(double val, double sigma) 
{
  return exp(-0.5 * val * val) / (sigma * sqrt_2PI);
}


// compute average gaussian pdf value over a window around each point
// __restrict__：优化指示符，告诉编译器指针所指的内存不重叠，便于优化内存访问。
__global__ void gaussian_pdf(const ft * __restrict__ x, ft * __restrict__ y, const ft mean, const ft sigma, const int n) 
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) 
  {
    ft in = x[idx] - (count / 2) * 0.01f; // 相当于将窗口中心放在 x[idx]，窗口宽度为 count * 0.01f
    ft out = 0;
    // 循环 count 次，对应窗口内的 count 个点。
    // 计算标准化变量：temp = (in - mean) / sigma。
    // 累加高斯PDF值：调用 gpdf(temp, sigma)，将结果累加到 out。
    // 移动到下一个窗口点：in += 0.01f，步长为 0.01f。
    for (int i = 0; i < count; i++) 
    {
      ft temp = (in - mean) / sigma;
      out += gpdf(temp, sigma);
      in += 0.01f;
    }
    y[idx] = out / count;
  }
}


// error check macro
#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)

// host-based timing
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start) {
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

int main() 
{
  ft *h_x, *d_x, *h_y, *h_y1, *d_y;

  cudaHostAlloc(&h_x,  ds*sizeof(ft), cudaHostAllocDefault);
  cudaHostAlloc(&h_y,  ds*sizeof(ft), cudaHostAllocDefault);
  cudaHostAlloc(&h_y1, ds*sizeof(ft), cudaHostAllocDefault);

  cudaMalloc(&d_x, ds*sizeof(ft));
  cudaMalloc(&d_y, ds*sizeof(ft));
  
  cudaCheckErrors("allocation error");

  cudaStream_t streams[num_streams];

  for (int i = 0; i < num_streams; i++) 
  {
    cudaStreamCreate(&streams[i]);
  }

  cudaCheckErrors("stream creation error");

  gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds); // warm-up

  for (size_t i = 0; i < ds; i++) 
  {
    h_x[i] = rand() / (ft)RAND_MAX;
  }

  cudaDeviceSynchronize();

  unsigned long long et1 = dtime_usec(0);

  cudaMemcpy(d_x, h_x, ds * sizeof(ft), cudaMemcpyHostToDevice);
  gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds);
  cudaMemcpy(h_y1, d_y, ds * sizeof(ft), cudaMemcpyDeviceToHost);
  cudaCheckErrors("non-streams execution error");

  et1 = dtime_usec(et1);
  std::cout << "non-stream elapsed time: " << et1/(float)USECPSEC << std::endl;

// #ifdef USE_STREAMS
  // cudaMemset(d_y, 0, ds * sizeof(ft));

  unsigned long long et = dtime_usec(0);

  for (int i = 0; i < chunks; i++) 
  { //depth-first launch
    cudaMemcpyAsync(d_x + i * (ds / chunks), h_x + i * (ds / chunks), (ds / chunks) * sizeof(ft), cudaMemcpyHostToDevice, streams[i % num_streams]);

    gaussian_pdf<<<((ds / chunks) + 255) / 256, 256, 0, streams[i % num_streams]>>>(d_x + i * (ds / chunks), d_y + i * (ds / chunks), 0.0, 1.0, ds / chunks);
    
    cudaMemcpyAsync(h_y + i * (ds / chunks), d_y + i * (ds / chunks), (ds / chunks) * sizeof(ft), cudaMemcpyDeviceToHost, streams[i % num_streams]);
  }

  cudaDeviceSynchronize();

  cudaCheckErrors("streams execution error");

  et = dtime_usec(et);

  for (int i = 0; i < ds; i++) 
  {
    if (h_y[i] != h_y1[i]) 
    {
      std::cout << "mismatch at " << i << " was: " << h_y[i] << " should be: " << h_y1[i] << std::endl;
      return -1;
    }
  }

  std::cout << "streams elapsed time: " << et/(float)USECPSEC << std::endl;
// #endif

  return 0;
}
