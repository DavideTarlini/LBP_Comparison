#ifndef LBP_CUDA_CUH
#define LBP_CUDA_CUH

#include <cuda_runtime.h>
#include <vector>
#include <array>

__global__ void _lbp_kernel(int* image, int* result, int rows, int cols);
__global__ void _histogram_kernel(int* lbp_image, int* histogram, int rows, int cols);
unsigned int* _histogram_cuda(int* lbp_image, int cols, int rows);
unsigned int* get_LBP_hist_cuda(int* image, int rows, int cols);

#endif