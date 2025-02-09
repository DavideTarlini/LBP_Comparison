#ifndef UTIL_H
#define UTIL_H

#include <chrono>

struct results {
    unsigned int* histogram;
    std::chrono::duration<double, std::milli> time;
};

#endif