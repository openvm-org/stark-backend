#pragma once

#include "fp.h"

struct MainMatrixPtrs {
    const Fp *data;
    uint32_t air_width;
};

extern "C" int _batch_rotate_pad(
    Fp *out,
    const Fp *in,
    uint32_t width, 
    uint32_t num_x, // = (matrix.height() / domain_size)
    uint32_t domain_size,
    uint32_t padded_size
);


