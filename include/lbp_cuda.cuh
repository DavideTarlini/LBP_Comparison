#ifndef LBP_CUDA_CUH
#define LBP_CUDA_CUH

#include <cuda_runtime.h>
#include "../include/util.h"

__global__ void _padd_kernel(int* image, int* padd_img, int rows, int cols);
__global__ void _lbp_h_kernel(int* image, int* padd_image, unsigned int* histogram, int rows, int cols);
results get_LBP_hist_cuda(int* image, int rows, int cols);

#endif