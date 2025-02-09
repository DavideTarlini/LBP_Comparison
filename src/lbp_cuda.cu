#include "../include/lbp_cuda.cuh"
#include <iostream>

/*#define TILE_SIZE (34*34)
__global__ void _lbp_kernel_t(int* image, int* result, int rows, int cols){
    __shared__ unsigned int tile[TILE_SIZE];
    const int r = blockIdx.y*blockDim.y + threadIdx.y;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    if(r < rows && c < cols){
        const int padded_local_pos = (threadIdx.y+1)*(blockDim.x+2) + (threadIdx.x+1);
        const int global_pos = r*cols + c;
        const int global_padded_pos = (r+1)*(cols+2) + (c+1);
        unsigned int lbp_value = 0;

        tile[padded_local_pos] = image[global_padded_pos];

        if(threadIdx.y == 0){
            tile[(threadIdx.y)*(blockDim.x+2) + (threadIdx.x+1)] = image[(r)*(cols+2) + (c+1)];
        }else if (threadIdx.y == blockDim.y)
        {
            tile[(threadIdx.y+2)*(blockDim.x+2) + (threadIdx.x+1)] = image[(r+2)*(cols+2) + (c+1)];
        }

        if(threadIdx.x == 0){
            tile[(threadIdx.y+1)*(blockDim.x+2) + (threadIdx.x)] = image[(r+1)*(cols+2) + (c)];
        }else if (threadIdx.x == blockDim.x)
        {
            tile[(threadIdx.y+1)*(blockDim.x+2) + (threadIdx.x+2)] = image[(r+1)*(cols+2) + (c+2)];
        }

        unsigned int center = tile[padded_local_pos];
        
        __syncthreads();

        lbp_value |= (tile[(threadIdx.y)*(blockDim.x+2) + (threadIdx.x)] >= center) << 7;
        lbp_value |= (tile[(threadIdx.y)*(blockDim.x+2) + (threadIdx.x+1)] >= center) << 6;
        lbp_value |= (tile[(threadIdx.y)*(blockDim.x+2) + (threadIdx.x+2)] >= center) << 5;
        lbp_value |= (tile[(threadIdx.y+1)*(blockDim.x+2) + (threadIdx.x+2)] >= center) << 4;
        lbp_value |= (tile[(threadIdx.y+2)*(blockDim.x+2) + (threadIdx.x+2)] >= center) << 3;
        lbp_value |= (tile[((threadIdx.y+2)*(blockDim.x+2) + (threadIdx.x+1))] >= center) << 2;
        lbp_value |= (tile[(threadIdx.y+2)*(blockDim.x+2) + (threadIdx.x)] >= center) << 1;
        lbp_value |= (tile[(threadIdx.y+1)*(blockDim.x+2) + (threadIdx.x)] >= center) << 0;
        

        result[global_pos] = lbp_value;
    }
}*/

__global__ void _lbp_kernel(int* image, int* result, int rows, int cols){
    const int r = blockIdx.y*blockDim.y + threadIdx.y;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    if(r < rows && c < cols){
        const int pos = r*cols + c;
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
        

        result[pos] = lbp_value;
    }
}

__global__ void _lbp_h_kernel(int* image, unsigned int* histogram, int rows, int cols){
    __shared__ unsigned int shared_hist[256];

    const int r = blockIdx.y*blockDim.y + threadIdx.y;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int pos = r*cols + c;

    if(pos < 256){
        atomicExch(&shared_hist[pos], 0);
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

    if(pos <256){
        atomicAdd(&histogram[pos], shared_hist[pos]);
    }
}

__global__ void _histogram_kernel(int* lbp_image, unsigned int* histogram, int rows, int cols){
    __shared__ unsigned int shared_hist[256];

    const int r = blockIdx.y*blockDim.y + threadIdx.y;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int pos = r*cols + c;

    if(pos < 256){
        atomicExch(&shared_hist[pos], 0);
    }

    if(r < rows && c < cols){
        atomicAdd(&shared_hist[lbp_image[pos]], 1);
    }

    __syncthreads();

    //__shared__ unsigned int counter;
    //counter = 0;
    
    //int count = atomicAdd(&counter, 1);
    if(pos <256){
        atomicAdd(&histogram[pos], shared_hist[pos]);
    }
}

unsigned int* _histogram_cuda(int* lbp_image, int rows, int cols){
    unsigned int* histogram = new unsigned int[256];

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            histogram[lbp_image[(i*cols) + j]] += 1;
        }
    }

    return histogram; 
}

results get_LBP_hist_cuda(int* image, int rows, int cols){
    int* padded_img = new int[(rows+2)*(cols+2)];

    for(int i=0; i<rows+2; i++){
        for(int j=0; j<cols+2; j++){
            if(i == 0 || i == rows+1 || j == 0 || j == cols+1) 
                padded_img[(i*cols) + j] = 0;
            else
                padded_img[((i)*cols) + (j)] = image[((i-1)*cols) + (j-1)];
        }
    }

    unsigned int* histogram = new unsigned int[256];


    int* d_img;
    unsigned int* d_hist;
    int padded_size = (rows+2)*(cols+2)*sizeof(int); 
    int s_lbp = 1024;
    dim3 dimBlock_lbp(s_lbp, s_lbp, 1);
    dim3 dimGrid_lbp(ceil(cols/s_lbp), (rows/s_lbp), 1);

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
        exit(EXIT_FAILURE);
    }else{
        cudaMemcpy(histogram, d_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaFree(d_img);
        cudaFree(d_hist);

        results res;
        res.histogram = histogram;
        res.time = t2 - t1;

        return res;
    }
}