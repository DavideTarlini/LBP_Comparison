#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric>

#include "include/lbp_seq.h"
#include "include/lbp_cuda.cuh"

int main(int argc, char const *argv[])
{     
    std::ofstream sp_file("speedups.csv");

    std::random_device rd;  
    std::mt19937 gen(rd());
    auto dist = std::uniform_int_distribution<>(0, 255);

    int img_dim[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    int block_dim[] = {4, 8, 16, 32};
    
    for(int k = 0; k < sizeof(img_dim)/sizeof(img_dim[0]); k++){
        int rows = img_dim[k];
        int cols = img_dim[k];

        for (int i = 0; i < sizeof(block_dim)/sizeof(block_dim[0]); i++){
            std::vector<double> global_exp_res1;
            std::vector<double> pad_exp_res1;
            std::vector<double> lbp_exp_res1;
            std::vector<double> global_exp_res2;
            std::vector<double> pad_exp_res2;
            std::vector<double> lbp_exp_res2;

            for (int j = 0; j < 100; j++){
                int* linear_img = new int[rows * cols];
                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < cols; col++){
                        linear_img[row * cols + col] = dist(gen);
                    }
                }                

                auto t1 = std::chrono::high_resolution_clock::now();                
                auto res_seq = get_LBP_hist_seq(linear_img, rows, cols);
                auto t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> seq_t = t2 - t1;

                t1 = std::chrono::high_resolution_clock::now();  
                auto res_cuda_ns = get_LBP_hist_cuda_non_shared(linear_img, rows, cols, block_dim[i], block_dim[i]);
                t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> cuda_nst = t2 - t1;

                t1 = std::chrono::high_resolution_clock::now();  
                auto res_cuda_s = get_LBP_hist_cuda_shared(linear_img, rows, cols, block_dim[i], block_dim[i]);
                t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> cuda_st = t2 - t1; 

                std::cout<< "---------- run " << j << " ---------- \n";
                std::cout<< "Image size: " << rows*cols << "\n";

                bool equal = true;
                for(int k=0; k<256; k++){
                    if(res_seq.histogram[k] != res_cuda_s.histogram[k]){
                        equal = false;
                        break;
                    }
                }

                if (!equal)
                {
                    std::cout<< "Not Equal!!!\n";
                }

                std::cout<< "\n";

                delete[] res_seq.histogram;
                delete[] res_cuda_ns.histogram;
                delete[] res_cuda_s.histogram;

                global_exp_res1.push_back(seq_t.count() / cuda_nst.count());
                global_exp_res2.push_back(seq_t.count() / cuda_st.count());
                
                pad_exp_res1.push_back(res_seq.time_pad.count() / res_cuda_ns.time_pad.count());
                pad_exp_res2.push_back(res_seq.time_pad.count() / res_cuda_s.time_pad.count());

                lbp_exp_res1.push_back(res_seq.time_lbp.count() / res_cuda_ns.time_lbp.count());
                lbp_exp_res2.push_back(res_seq.time_lbp.count() / res_cuda_s.time_lbp.count());

                delete[] linear_img;
            }

            double mean_global1 = std::accumulate(global_exp_res1.begin(), global_exp_res1.end(), 0.0)/global_exp_res1.size();
            double mean_pad1 = std::accumulate(pad_exp_res1.begin(), pad_exp_res1.end(), 0.0)/pad_exp_res1.size();
            double mean_lbp1 = std::accumulate(lbp_exp_res1.begin(), lbp_exp_res1.end(), 0.0)/lbp_exp_res1.size();
            double mean_global2 = std::accumulate(global_exp_res2.begin(), global_exp_res2.end(), 0.0)/global_exp_res2.size();
            double mean_pad2 = std::accumulate(pad_exp_res2.begin(), pad_exp_res2.end(), 0.0)/pad_exp_res2.size();
            double mean_lbp2 = std::accumulate(lbp_exp_res2.begin(), lbp_exp_res2.end(), 0.0)/lbp_exp_res2.size();

            sp_file << img_dim[k] << ","
                    << block_dim[i] << ","
                    << mean_global1 << ", "
                    << mean_global2 << ", "
                    << mean_pad1 << ", "
                    << mean_pad2 << ", "
                    << mean_lbp1 << ", "
                    << mean_lbp2 << ", ";
            sp_file << "\n";
            sp_file.flush();
        }
    }
    sp_file.close();
}