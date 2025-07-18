#include "../include/lbp_cuda.cuh"
#include <iostream>

__global__ void _padd_kernel(int* image, int* padd_img, int rows, int cols){
    const int r = blockIdx.y*blockDim.y + threadIdx.y;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    if( r < rows && c < cols){
        padd_img[(r+1)*(cols+2) + (c+1)] = image[(r*cols) + c];
    }
}

__global__ void _lbp_h_kernel_shared(int* image, unsigned int* histogram, int rows, int cols) {
    __shared__ unsigned int shared_hist[256];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        shared_hist[i] = 0;
    __syncthreads();

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        int padded_pos = (r + 1) * (cols + 2) + (c + 1);
        unsigned int center = image[padded_pos];

        unsigned int lbp_value = 0;
        lbp_value += (image[r*(cols+2) + c] >= center) * 128;
        lbp_value += (image[r*(cols+2) + (c+1)] >= center) * 64;
        lbp_value += (image[r*(cols+2) + (c+2)] >= center) * 32;
        lbp_value += (image[(r+1)*(cols+2) + (c+2)] >= center) * 16;
        lbp_value += (image[(r+2)*(cols+2) + (c+2)] >= center) * 8;
        lbp_value += (image[(r+2)*(cols+2) + (c+1)] >= center) * 4;
        lbp_value += (image[(r+2)*(cols+2) + c] >= center) * 2;
        lbp_value += (image[(r+1)*(cols+2) + c] >= center);

        atomicAdd(&shared_hist[lbp_value], 1);
    }

    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        atomicAdd(&histogram[i], shared_hist[i]);
}

__global__ void _lbp_h_kernel_non_shared(int* image, unsigned int* histogram, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        int padded_pos = (r + 1) * (cols + 2) + (c + 1);
        unsigned int center = image[padded_pos];

        unsigned int lbp_value = 0;
        lbp_value += (image[r*(cols+2) + c] >= center) * 128;
        lbp_value += (image[r*(cols+2) + (c+1)] >= center) * 64;
        lbp_value += (image[r*(cols+2) + (c+2)] >= center) * 32;
        lbp_value += (image[(r+1)*(cols+2) + (c+2)] >= center) * 16;
        lbp_value += (image[(r+2)*(cols+2) + (c+2)] >= center) * 8;
        lbp_value += (image[(r+2)*(cols+2) + (c+1)] >= center) * 4;
        lbp_value += (image[(r+2)*(cols+2) + c] >= center) * 2;
        lbp_value += (image[(r+1)*(cols+2) + c] >= center);

        atomicAdd(&histogram[lbp_value], 1);
    }
}

results get_LBP_hist_cuda_shared(int* image, int rows, int cols, int lbp_y, int lbp_x){
    unsigned int* histogram = new unsigned int[256];
    int* d_img;
    int* d_padd_img;
    unsigned int* d_hist; 

    cudaMalloc((void **) &d_img, rows*cols*sizeof(int));
    cudaMemcpy(d_img, image, rows*cols*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_padd_img, (rows+2)*(cols+2)*sizeof(int));
    cudaMemset(d_padd_img, 0, (rows+2)*(cols+2)*sizeof(int));

    cudaMalloc((void **) &d_hist, 256*sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256*sizeof(unsigned int));
    

    int p_y = lbp_y;
    int p_x = lbp_x;
    dim3 dimBlock_p(p_x, p_y, 1);
    dim3 dimGrid_p(ceil(float(cols)/p_x), ceil(float(rows)/p_y), 1);

    auto tp1 = std::chrono::high_resolution_clock::now();
    _padd_kernel<<<dimGrid_p, dimBlock_p>>>(d_img, d_padd_img, rows, cols);
    cudaDeviceSynchronize();
    auto tp2 = std::chrono::high_resolution_clock::now();
    auto err1 = cudaGetLastError();

    if (err1 != cudaSuccess) {
        printf("CUDA error during shared kernel launch: %s\n", cudaGetErrorString(err1));
        cudaFree(d_img);
        cudaFree(d_padd_img);
        cudaFree(d_hist);

        exit(EXIT_FAILURE);
    }

    dim3 dimBlock_lbp(lbp_x, lbp_y, 1);
    dim3 dimGrid_lbp(ceil(float(cols)/lbp_x), ceil(float(rows)/lbp_y), 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    _lbp_h_kernel_shared<<<dimGrid_lbp, dimBlock_lbp>>>(d_padd_img, d_hist, rows, cols);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto err2 = cudaGetLastError();
    
    if (err2 != cudaSuccess) {
        printf("CUDA error during shared kernel launch: %s\n", cudaGetErrorString(err2));
        cudaFree(d_img);
        cudaFree(d_padd_img);
        cudaFree(d_hist);

        exit(EXIT_FAILURE);
    }

    cudaMemcpy(histogram, d_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_padd_img);
    cudaFree(d_hist);

    results res;
    res.histogram = histogram;
    res.time_pad = tp2 - tp1;
    res.time_lbp = t2 - t1;
    res.grid_size = ceil(float(cols)/lbp_x)*ceil(float(rows)/lbp_y)*lbp_x*lbp_y;

    return res;
}

results get_LBP_hist_cuda_non_shared(int* image, int rows, int cols, int lbp_y, int lbp_x){
    unsigned int* histogram = new unsigned int[256];
    int* d_img;
    int* d_padd_img;
    unsigned int* d_hist; 

    cudaMalloc((void **) &d_img, rows*cols*sizeof(int));
    cudaMemcpy(d_img, image, rows*cols*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_padd_img, (rows+2)*(cols+2)*sizeof(int));
    cudaMemset(d_padd_img, 0, (rows+2)*(cols+2)*sizeof(int));

    cudaMalloc((void **) &d_hist, 256*sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256*sizeof(unsigned int));
    

    int p_y = lbp_y;
    int p_x = lbp_x;
    dim3 dimBlock_p(p_x, p_y, 1);
    dim3 dimGrid_p(ceil(float(cols)/p_x), ceil(float(rows)/p_y), 1);

    auto tp1 = std::chrono::high_resolution_clock::now();
    _padd_kernel<<<dimGrid_p, dimBlock_p>>>(d_img, d_padd_img, rows, cols);
    cudaDeviceSynchronize();
    auto tp2 = std::chrono::high_resolution_clock::now();
    auto err1 = cudaGetLastError();

    if (err1 != cudaSuccess) {
        printf("CUDA error during non shared kernel launch: %s\n", cudaGetErrorString(err1));
        cudaFree(d_img);
        cudaFree(d_padd_img);
        cudaFree(d_hist);

        exit(EXIT_FAILURE);
    }

    dim3 dimBlock_lbp(lbp_x, lbp_y, 1);
    dim3 dimGrid_lbp(ceil(float(cols)/lbp_x), ceil(float(rows)/lbp_y), 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    _lbp_h_kernel_non_shared<<<dimGrid_lbp, dimBlock_lbp>>>(d_padd_img, d_hist, rows, cols);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto err2 = cudaGetLastError();
    
    if (err2 != cudaSuccess) {
        printf("CUDA error during  non shared kernel launch: %s\n", cudaGetErrorString(err2));
        cudaFree(d_img);
        cudaFree(d_padd_img);
        cudaFree(d_hist);

        exit(EXIT_FAILURE);
    }

    cudaMemcpy(histogram, d_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_padd_img);
    cudaFree(d_hist);

    results res;
    res.histogram = histogram;
    res.time_pad = tp2 - tp1;
    res.time_lbp = t2 - t1;
    res.grid_size = ceil(float(cols)/lbp_x)*ceil(float(rows)/lbp_y)*lbp_x*lbp_y;

    return res;
}

