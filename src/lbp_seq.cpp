#include "../include/lbp_seq.h"

#include <iostream>
#include <math.h>

int _lbp_seq(int* image, int r, int c, int cols){
    int lbp_value = 0;
    int pos = (r+1)*(cols+2) + (c+1);
    int center = image[pos];

    lbp_value += (image[r*(cols+2) + c] >= center)*128;
    lbp_value += (image[r*(cols+2) + (c+1)] >= center)*64;
    lbp_value += (image[r*(cols+2) + (c+2)] >= center)*32;
    lbp_value += (image[(r+1)*(cols+2) + (c+2)] >= center)*16;
    lbp_value += (image[(r+2)*(cols+2) + (c+2)] >= center)*8;
    lbp_value += (image[(r+2)*(cols+2) + (c+1)] >= center)*4;
    lbp_value += (image[(r+2)*(cols+2) + c] >= center)*2;
    lbp_value += (image[(r+1)*(cols+2) + c] >= center);

    return lbp_value;
}

results get_LBP_hist_seq(int* image, int rows, int cols){
    int* padded_img = new int[(rows+2)*(cols+2)];
    unsigned int* histogram = new unsigned int[256];
    for(int i=0; i<256; i++){
        histogram[i] = 0;
    }

    auto tp1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<rows+2; i++){
        for(int j=0; j<cols+2; j++){
            if(i == 0 || i == rows+1 || j == 0 || j == cols+1) 
                padded_img[(i*(cols+2)) + j] = 0;
            else
                padded_img[((i)*(cols+2)) + (j)] = image[((i-1)*cols) + (j-1)];
        }
    }
    auto tp2 = std::chrono::high_resolution_clock::now();

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            histogram[_lbp_seq(padded_img, i, j, cols)] += 1;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    
    results res;
    res.histogram = histogram;
    res.time_pad = tp2 - tp1;
    res.time_lbp = t2 - t1;
    delete[] padded_img;

    return res;
}