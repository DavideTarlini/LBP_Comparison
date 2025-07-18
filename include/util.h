#ifndef UTIL_H
#define UTIL_H

#include <chrono>

struct results {
    unsigned int* histogram;
    std::chrono::duration<double, std::milli> time_pad;
    std::chrono::duration<double, std::milli> time_lbp;
    int grid_size;
};

#endif