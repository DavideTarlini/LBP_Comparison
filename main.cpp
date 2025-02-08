#include <vector>
#include <array>
#include <iostream>
#include <random>
#include <chrono>

#include "include/lbp_seq.h"
#include "include/lbp_cuda.cuh"

int main(int argc, char const *argv[])
{   
    int rows = 16000;
    int cols = 16000;

    std::random_device rd;  
    std::mt19937 gen(rd());
    auto dist = std::uniform_int_distribution<>(0, 255);

    int* linear_img = new int[rows*cols];
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            linear_img[(i*cols) + j] = dist(gen);
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();                
    auto res_seq = get_LBP_hist_seq(linear_img, rows, cols);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_t = t2 - t1;

    t1 = std::chrono::high_resolution_clock::now();  
    auto res_cuda = get_LBP_hist_cuda(linear_img, rows, cols);
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cuda_t = t2 - t1;

    std::cout<< seq_t.count() << "\n";
    std::cout<< cuda_t.count() << "\n";
    std::cout<< seq_t.count() / cuda_t.count();

    return 0;
}
