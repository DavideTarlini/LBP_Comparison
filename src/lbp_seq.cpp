#include "../include/lbp_seq.h"

int _lbp_seq(int* image, int r, int c, int cols){
    int lbp_value = 0;
    int pos = (r+1)*(cols+2) + (c+1);
    int center = image[pos];

    lbp_value |= (image[r*(cols+2) + c] >= center) << 7;
    lbp_value |= (image[r*(cols+2) + (c+1)] >= center) << 6;
    lbp_value |= (image[r*(cols+2) + (c+2)] >= center) << 5;
    lbp_value |= (image[(r+1)*(cols+2) + (c+2)] >= center) << 4;
    lbp_value |= (image[(r+2)*(cols+2) + (c+2)] >= center) << 3;
    lbp_value |= (image[(r+2)*(cols+2) + (c+1)] >= center) << 2;
    lbp_value |= (image[(r+2)*(cols+2) + c] >= center) << 1;
    lbp_value |= (image[(r+1)*(cols+2) + c] >= center) << 0;

    return lbp_value;
}

unsigned int* _histogram_seq(int* lbp_image, int rows, int cols){
    unsigned int* histogram = new unsigned int[256];

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            histogram[lbp_image[(i*cols) + j]] += 1;
        }
    }

    return histogram; 
}

unsigned int* get_LBP_hist_seq(int* image, int rows, int cols){
    int* padded_img = new int[(rows+2)*(cols+2)];

    for(int i=0; i<rows+2; i++){
        for(int j=0; j<cols+2; j++){
            if(i == 0 || i == rows+1 || j == 0 || j == cols+1) 
                padded_img[(i*cols) + j] = 0;
            else
                padded_img[((i)*cols) + (j)] = image[((i-1)*cols) + (j-1)];
        }
    }

    int* result = new int[rows*cols];

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            result[(i*cols) + j] = _lbp_seq(padded_img, i, j, cols);
        }
    }

    auto histogram = _histogram_seq(result, rows, cols);
    
    return histogram;
}