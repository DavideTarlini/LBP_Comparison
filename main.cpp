#include <iostream>
#include <random>
#include <chrono>

#include "include/lbp_seq.h"
#include "include/lbp_cuda.cuh"

void experiment(int i, int* img, int rows, int cols, std::vector<double> global_exp_res, std::vector<double> ker_exp_res, bool print_hist){
    auto t1 = std::chrono::high_resolution_clock::now();                
    auto res_seq = get_LBP_hist_seq(img, rows, cols);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_t = t2 - t1;

    t1 = std::chrono::high_resolution_clock::now();  
    auto res_cuda = get_LBP_hist_cuda(img, rows, cols);
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cuda_t = t2 - t1;

    std::cout<< "---------- run " << i << " ---------- \n";
    std::cout<< "Image size: " << rows*cols << "\n";

    bool equal = true;
    for(int k=0; k<256; k++){
        if(res_seq.histogram[k] != res_cuda.histogram[k]){
            equal = false;
            break;
        }
    }

    if (print_hist)
    {
        std::cout<< "Histogram equality check: " << equal << "\n";
        for(int k=0; k<256; k++){
            std::cout<< res_seq.histogram[k] << " ";
        }
    }

    std::cout<< "\n";

    std::cout<< seq_t.count() << "  GL\n";
    std::cout<< cuda_t.count() << " GL\n";
    std::cout<< seq_t.count() / cuda_t.count() << " GL\n";
    std::cout<< res_seq.time.count() << "  KER\n";
    std::cout<< res_cuda.time.count() << "  KER\n";
    std::cout<< res_seq.time.count() / res_cuda.time.count() << " KER\n\n";

    global_exp_res.push_back(seq_t.count() / cuda_t.count());
    ker_exp_res.push_back(res_seq.time.count() / res_cuda.time.count());

    delete[] res_seq.histogram;
    delete[] res_cuda.histogram;
}

int main(int argc, char const *argv[])
{   
    int rows = 15000;
    int cols = 15000;

    std::random_device rd;  
    std::mt19937 gen(rd());
    auto dist = std::uniform_int_distribution<>(0, 255);

    int* linear_img = new int[rows*cols];
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            linear_img[(i*cols) + j] = dist(gen);
        }
    }
    
    std::vector<double> global_exp_res;
    std::vector<double> ker_exp_res;
    
    experiment(0, linear_img, rows, cols, global_exp_res, ker_exp_res, true);

    delete[] linear_img;

    return 0;
}
