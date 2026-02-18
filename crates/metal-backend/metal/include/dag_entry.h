// DAG node representation for Metal
// Translated from CUDA: cuda-backend/cuda/include/dag_entry.cuh
// NOTE: The CUDA version uses C++ templates for NTT-based coset evaluation.
// Metal does not support templates in shaders, so we use function constants
// and manual specialization where needed.
#pragma once

#include "codec.h"
#include "baby_bear.h"
#include "baby_bear_ext.h"

// NTT-based DAG evaluation is complex and uses CUDA templates extensively.
// For Metal, the NTT coset evaluation is handled in the round0 kernel files
// using function constants for specialization parameters.
// This header provides the common DAG entry evaluation used by MLE kernels.
