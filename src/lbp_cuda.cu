#include "../include/lbp_cuda.cuh"
#include <iostream>

__global__ void _lbp_h_kernel(int* image, unsigned int* histogram, int rows, int cols){
    __shared__ unsigned int shared_hist[256];

    const int r = blockIdx.y*blockDim.y + threadIdx.y;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    if((threadIdx.y*blockDim.x) + threadIdx.x < 256){
        shared_hist[(threadIdx.y*blockDim.x) + threadIdx.x] = 0;
    }
    __syncthreads();

    if(r < rows && c < cols){
        const int padded_pos = (r+1)*(cols+2) + (c+1);
        unsigned int lbp_value = 0;

        const unsigned int center = image[padded_pos];
  
        lbp_value |= (image[r*(cols+2) + c] >= center) << 7;
        lbp_value |= (image[r*(cols+2) + (c+1)] >= center) << 6;
        lbp_value |= (image[r*(cols+2) + (c+2)] >= center) << 5;
        lbp_value |= (image[(r+1)*(cols+2) + (c+2)] >= center) << 4;
        lbp_value |= (image[(r+2)*(cols+2) + (c+2)] >= center) << 3;
        lbp_value |= (image[(r+2)*(cols+2) + (c+1)] >= center) << 2;
        lbp_value |= (image[(r+2)*(cols+2) + c] >= center) << 1;
        lbp_value |= (image[(r+1)*(cols+2) + c] >= center) << 0;

        atomicAdd(&shared_hist[lbp_value], 1);
    }

    __syncthreads();
    
    if((threadIdx.y*blockDim.x) + threadIdx.x < 256){
        atomicAdd(&histogram[(threadIdx.y*blockDim.x) + threadIdx.x], shared_hist[(threadIdx.y*blockDim.x) + threadIdx.x]);
    }
}

results get_LBP_hist_cuda(int* image, int rows, int cols){
    int* padded_img = new int[(rows+2)*(cols+2)];

    for(int i=0; i<rows+2; i++){
        for(int j=0; j<cols+2; j++){
            if(i == 0 || i == rows+1 || j == 0 || j == cols+1) 
                padded_img[(i*(cols+2)) + j] = 0;
            else
                padded_img[((i)*(cols+2)) + (j)] = image[((i-1)*cols) + (j-1)];
        }
    }

    unsigned int* histogram = new unsigned int[256];
    int* d_img;
    unsigned int* d_hist;
    int padded_size = (rows+2)*(cols+2)*sizeof(int); 
    int s_lbp = 32;
    std::cout<< ceil(float(cols)/s_lbp) << "  " << ceil(float(rows)/s_lbp) << "\n\n";
    dim3 dimBlock_lbp(s_lbp, s_lbp, 1);
    dim3 dimGrid_lbp(ceil(float(cols)/s_lbp), ceil(float(rows)/s_lbp), 1);

    cudaMalloc((void **) &d_img, padded_size);
    cudaMemcpy(d_img, padded_img, padded_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_hist, 256*sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256*sizeof(unsigned int));
    
    auto t1 = std::chrono::high_resolution_clock::now();
    _lbp_h_kernel<<<dimGrid_lbp, dimBlock_lbp>>>(d_img, d_hist, rows, cols);
    auto err = cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    if (err != cudaSuccess) {
        printf("CUDA error during kernel launch: %s\n", cudaGetErrorString(err));
        cudaFree(d_img);
        cudaFree(d_hist);
        delete[] padded_img;

        exit(EXIT_FAILURE);
    }else{
        cudaMemcpy(histogram, d_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaFree(d_img);
        cudaFree(d_hist);
        delete[] padded_img;

        results res;
        res.histogram = histogram;
        res.time = t2 - t1;

        return res;
    }
}