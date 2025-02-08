#ifndef LBP_SEQ_H
#define LBP_SEQ_H

#include <array>
#include <vector>

int _lbp_seq(int* image, int r, int c, int cols);
unsigned int* _histogram_seq(int* lbp_image, int rows, int cols);
unsigned int* get_LBP_hist_seq(int* image, int rows, int cols);
#endif