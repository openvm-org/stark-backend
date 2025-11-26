#pragma once

#include <cstdint>

template <typename T> struct MainMatrixPtrs {
    const T *data;
    uint32_t air_width;
};
