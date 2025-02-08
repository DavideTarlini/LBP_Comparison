#include "../include/lbp_cuda.cuh"

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

__global__ void _histogram_kernel(int* lbp_image, unsigned int* histogram, int rows, int cols){
    const int r = blockIdx.y*blockDim.y + threadIdx.y;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    if(r < rows && c < cols){
        const int pos = r*cols + c;
        atomicAdd(&histogram[lbp_image[pos]], 1);
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

unsigned int* get_LBP_hist_cuda(int* image, int rows, int cols){
    int* padded_img = new int[(rows+2)*(cols+2)];

    for(int i=0; i<rows+2; i++){
        for(int j=0; j<cols+2; j++){
            if(i == 0 || i == rows+1 || j == 0 || j == cols+1) 
                padded_img[(i*cols) + j] = 0;
            else
                padded_img[((i)*cols) + (j)] = image[((i-1)*cols) + (j-1)];
        }
    }


    /***** LBP *****/
    int padded_size = (rows+2)*(cols+2)*sizeof(int); 
    int res_size = rows*cols*sizeof(int);
    int* d_img;
    int* d_res;

    cudaMalloc((void **) &d_img, padded_size);
    cudaMalloc((void **) &d_res, res_size);
    cudaMemcpy(d_img, padded_img, padded_size, cudaMemcpyHostToDevice);

    int s_lbp = 1024;
    dim3 dimBlock_lbp(s_lbp, s_lbp, 1);
    dim3 dimGrid_lbp(ceil(cols/s_lbp), (rows/s_lbp), 1);

    _lbp_kernel<<<dimGrid_lbp, dimBlock_lbp>>>(d_img, d_res, rows, cols);

    auto err = cudaDeviceSynchronize();
    cudaFree(d_img);

    if (err != cudaSuccess) {
        printf("CUDA error during kernel launch: %s\n", cudaGetErrorString(err));
        cudaFree(d_res);
        exit(EXIT_FAILURE);
    }

    //int* result = new int[rows*cols];
    //cudaMemcpy(result, d_res, res_size, cudaMemcpyDeviceToHost);

    /***** Histogram *****/
    unsigned int* histogram = new unsigned int[256];
    unsigned int* d_hist;
    cudaMalloc((void **) &d_hist, 256*sizeof(unsigned int));

    int s_h = 1024;
    dim3 dimBlock_h(s_h, s_h, 1);
    dim3 dimGrid_h(ceil(cols/s_h), (rows/s_h), 1);

    _histogram_kernel<<<dimGrid_h, dimBlock_h>>>(d_res, d_hist, rows, cols);

    cudaMemcpy(histogram, d_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_res);
    cudaFree(d_hist);
    

    //auto histogram = _histogram_cuda(result, rows, cols);
    
    return histogram;
}