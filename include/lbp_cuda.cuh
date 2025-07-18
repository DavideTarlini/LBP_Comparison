#ifndef LBP_CUDA_CUH
#define LBP_CUDA_CUH

#include <cuda_runtime.h>
#include "../include/util.h"

__global__ void _padd_kernel(int* image, int* padd_img, int rows, int cols);
__global__ void _lbp_h_kernel_shared(int* image, unsigned int* histogram, int rows, int cols);
__global__ void _lbp_h_kernel_non_shared(int* image, unsigned int* histogram, int rows, int cols);
results get_LBP_hist_cuda_shared(int* image, int rows, int cols, int lbp_y, int lbp_x);
results get_LBP_hist_cuda_non_shared(int* image, int rows, int cols, int lbp_y, int lbp_x);

#endif