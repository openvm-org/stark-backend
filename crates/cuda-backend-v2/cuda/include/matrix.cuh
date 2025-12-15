#pragma once

#include <cstdint>

template <typename T> struct MainMatrixPtrs {
    const T *__restrict__ data;
    uint32_t air_width;
};
