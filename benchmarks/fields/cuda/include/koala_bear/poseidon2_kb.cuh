/*
 * KoalaBear Poseidon2 Permutation
 *
 * Same structure as BabyBear poseidon2.cuh but with:
 * - Kb field type instead of Fp
 * - sbox_d3 (x^3, 2 muls) instead of sbox_d7 (x^7, 4 muls)
 * - 20 partial rounds instead of 13
 * - KoalaBear round constants from Plonky3 p3-koala-bear crate
 *
 * Constants source: p3-koala-bear-0.4.1/src/poseidon2.rs
 */

#pragma once

#include "kb.h"

namespace kb_poseidon2 {

// Round constants from Plonky3 KOALABEAR_RC16_EXTERNAL_INITIAL (4 rounds × 16)
static __device__ __constant__ Kb INITIAL_ROUND_CONSTANTS[64] = {
    2128964168, 288780357,  316938561,  2126233899, 426817493,  1714118888, 1045008582, 1738510837,
    889721787,  8866516,    681576474,  419059826,  1596305521, 1583176088, 1584387047, 1529751136,
    1863858111, 1072044075, 517831365,  1464274176, 1138001621, 428001039,  245709561,  1641420379,
    1365482496, 770454828,  693167409,  757905735,  136670447,  436275702,  525466355,  1559174242,
    1030087950, 869864998,  322787870,  267688717,  948964561,  740478015,  679816114,  113662466,
    2066544572, 1744924186, 367094720,  1380455578, 1842483872, 416711434,  1342291586, 1692058446,
    1493348999, 1113949088, 210900530,  1071655077, 610242121,  1136339326, 2020858841, 1019840479,
    678147278,  1678413261, 1361743414, 61132629,   1209546658, 64412292,   1936878279, 1980661727
};

// Round constants from Plonky3 KOALABEAR_RC16_EXTERNAL_FINAL (4 rounds × 16)
static __device__ __constant__ Kb TERMINAL_ROUND_CONSTANTS[64] = {
    1423960925, 2101391318, 1915532054, 275400051,  1168624859, 1141248885, 356546469,  1165250474,
    1320543726, 932505663,  1204226364, 1452576828, 1774936729, 926808140,  1184948056, 1186493834,
    843181003,  185193011,  452207447,  510054082,  1139268644, 630873441,  669538875,  462500858,
    876500520,  1214043330, 383937013,  375087302,  636912601,  307200505,  390279673,  1999916485,
    1518476730, 1606686591, 1410677749, 1581191572, 1004269969, 143426723,  1747283099, 1016118214,
    1749423722, 66331533,   1177761275, 1581069649, 1851371119, 852520128,  1499632627, 1820847538,
    150757557,  884787840,  619710451,  1651711087, 505263814,  212076987,  1482432120, 1458130652,
    382871348,  417404007,  2066495280, 1996518884, 902934924,  582892981,  1337064375, 1199354861
};

// Round constants from Plonky3 KOALABEAR_RC16_INTERNAL (20 partial rounds)
static __device__ __constant__ Kb INTERNAL_ROUND_CONSTANTS[20] = {
    2102596038, 1533193853, 1436311464, 2012303432,
    839997195,  1225781098, 2011967775, 575084315,
    1309329169, 786393545,  995788880,  1702925345,
    1444525226, 908073383,  1811535085, 1531002367,
    1635653662, 1585100155, 867006515,  879151050
};

// Internal diagonal: [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
// Computed mod p = 2^31 - 2^24 + 1 = 2130706433
static __device__ __constant__ Kb internal_diag16[16] = {
    2130706431, // -2
    1,          // 1
    2,          // 2
    1065353217, // 1/2
    3,          // 3
    4,          // 4
    1065353216, // -1/2
    2130706430, // -3
    2130706429, // -4
    2122383361, // 1/2^8
    1864368129, // 1/8
    2130706306, // 1/2^24
    8323072,    // -1/2^8
    266338304,  // -1/8
    133169152,  // -1/16
    127         // -1/2^24
};


#define KB_CELLS 16
#define KB_ROUNDS_FULL 8
#define KB_ROUNDS_HALF_FULL (KB_ROUNDS_FULL / 2)
#define KB_ROUNDS_PARTIAL 20

static __device__ Kb sbox_d3(Kb x) {
    Kb x2 = x * x;
    return x2 * x;
}

static __device__ void do_full_sboxes(Kb *cells) {
    for (uint i = 0; i < KB_CELLS; i++) {
        cells[i] = sbox_d3(cells[i]);
    }
}

static __device__ void do_partial_sboxes(Kb *cells) {
    cells[0] = sbox_d3(cells[0]);
}

// Multiply a 4-element vector x by circ(2, 3, 1, 1)
static __device__ void multiply_by_4x4_circulant(Kb *x) {
    Kb t01 = x[0] + x[1];
    Kb t23 = x[2] + x[3];
    Kb t0123 = t01 + t23;
    Kb t01123 = t0123 + x[1];
    Kb t01233 = t0123 + x[3];

    x[3] = t01233 + Kb(2) * x[0];
    x[1] = t01123 + Kb(2) * x[2];
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

static __device__ void multiply_by_m_ext(Kb *old_cells) {
    Kb cells[KB_CELLS];
    for (uint i = 0; i < KB_CELLS; i++) {
        cells[0] = 0;
    }
    Kb tmp_sums[4];
    for (uint i = 0; i < 4; i++) {
        tmp_sums[i] = 0;
    }
    for (uint i = 0; i < KB_CELLS / 4; i++) {
        multiply_by_4x4_circulant(old_cells + i * 4);
        for (uint j = 0; j < 4; j++) {
            Kb to_add = old_cells[i * 4 + j];
            tmp_sums[j] += to_add;
            cells[i * 4 + j] += to_add;
        }
    }
    for (uint i = 0; i < KB_CELLS; i++) {
        old_cells[i] = cells[i] + tmp_sums[i % 4];
    }
}

static __device__ void add_round_constants_full(const Kb *ROUND_CONSTANTS, Kb *cells, uint round) {
    for (uint i = 0; i < KB_CELLS; i++) {
        cells[i] += ROUND_CONSTANTS[round * KB_CELLS + i];
    }
}

static __device__ void add_round_constants_partial(const Kb *PARTIAL_ROUND_CONSTANTS, Kb *cells, uint round) {
    cells[0] += PARTIAL_ROUND_CONSTANTS[round];
}

static __device__ __forceinline__ void internal_layer_mat_mul(Kb* cells, Kb sum) {
    cells[1] += sum;
#pragma unroll
    for (int i = 2; i < KB_CELLS; i++) {
        cells[i] = sum + cells[i] * internal_diag16[i];
    }
}

static __device__ void full_round_half(const Kb *ROUND_CONSTANTS, Kb *cells, uint round) {
    add_round_constants_full(ROUND_CONSTANTS, cells, round);
    do_full_sboxes(cells);
    multiply_by_m_ext(cells);
}

static __device__ void partial_round(const Kb *PARTIAL_ROUND_CONSTANTS, Kb *cells, uint round) {
    add_round_constants_partial(PARTIAL_ROUND_CONSTANTS, cells, round);
    do_partial_sboxes(cells);
    Kb part_sum = Kb(0);
    for (uint i = 1; i < KB_CELLS; i++) {
        part_sum += cells[i];
    }
    Kb full_sum = part_sum + cells[0];
    cells[0] = part_sum - cells[0];
    internal_layer_mat_mul(cells, full_sum);
}

static __device__ void poseidon2_mix(Kb *cells) {
    // First linear layer.
    multiply_by_m_ext(cells);

    // perform initial full rounds (external)
    for (uint i = 0; i < KB_ROUNDS_HALF_FULL; i++) {
        full_round_half(INITIAL_ROUND_CONSTANTS, cells, i);
    }

    // perform partial rounds (internal)
    for (uint i = 0; i < KB_ROUNDS_PARTIAL; i++) {
        partial_round(INTERNAL_ROUND_CONSTANTS, cells, i);
    }

    // perform terminal full rounds (external)
    for (uint r = 0; r < KB_ROUNDS_HALF_FULL; r++) {
        full_round_half(TERMINAL_ROUND_CONSTANTS, cells, r);
    }
}

} // namespace kb_poseidon2
