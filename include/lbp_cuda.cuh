#ifndef LBP_CUDA_CUH
#define LBP_CUDA_CUH

#include <cuda_runtime.h>
#include "../include/util.h"

__global__ void _lbp_kernel_t(int* image, int* result, int rows, int cols);
__global__ void _lbp_kernel(int* image, int* result, int rows, int cols);
__global__ void _lbp_h_kernel(int* image, unsigned int* histogram, int rows, int cols);
__global__ void _histogram_kernel(int* lbp_image, int* histogram, int rows, int cols);
unsigned int* _histogram_cuda(int* lbp_image, int cols, int rows);
results get_LBP_hist_cuda(int* image, int rows, int cols);

#endif