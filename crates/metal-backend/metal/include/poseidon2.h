// Poseidon2 permutation for BabyBear (width=16, rate=8)
// Translated from CUDA: cuda-common/include/poseidon2.cuh
#pragma once

#include "baby_bear.h"

#define CELLS 16
#define CELLS_RATE 8
#define CELLS_OUT 8

#define ROUNDS_FULL 8
#define ROUNDS_HALF_FULL (ROUNDS_FULL / 2)
#define ROUNDS_PARTIAL 13

constant Fp INITIAL_ROUND_CONSTANTS[64] = {
    Fp(1774958255u), Fp(1185780729u), Fp(1621102414u), Fp(1796380621u), Fp(588815102u),  Fp(1932426223u), Fp(1925334750u), Fp(747903232u),
    Fp(89648862u),   Fp(360728943u),  Fp(977184635u),  Fp(1425273457u), Fp(256487465u),  Fp(1200041953u), Fp(572403254u),  Fp(448208942u),
    Fp(1215789478u), Fp(944884184u),  Fp(953948096u),  Fp(547326025u),  Fp(646827752u),  Fp(889997530u),  Fp(1536873262u), Fp(86189867u),
    Fp(1065944411u), Fp(32019634u),   Fp(333311454u),  Fp(456061748u),  Fp(1963448500u), Fp(1827584334u), Fp(1391160226u), Fp(1348741381u),
    Fp(88424255u),   Fp(104111868u),  Fp(1763866748u), Fp(79691676u),   Fp(1988915530u), Fp(1050669594u), Fp(359890076u),  Fp(573163527u),
    Fp(222820492u),  Fp(159256268u),  Fp(669703072u),  Fp(763177444u),  Fp(889367200u),  Fp(256335831u),  Fp(704371273u),  Fp(25886717u),
    Fp(51754520u),   Fp(1833211857u), Fp(454499742u),  Fp(1384520381u), Fp(777848065u),  Fp(1053320300u), Fp(1851729162u), Fp(344647910u),
    Fp(401996362u),  Fp(1046925956u), Fp(5351995u),    Fp(1212119315u), Fp(754867989u),  Fp(36972490u),   Fp(751272725u),  Fp(506915399u)
};

constant Fp TERMINAL_ROUND_CONSTANTS[64] = {
    Fp(1922082829u), Fp(1870549801u), Fp(1502529704u), Fp(1990744480u), Fp(1700391016u), Fp(1702593455u), Fp(321330495u),  Fp(528965731u),
    Fp(183414327u),  Fp(1886297254u), Fp(1178602734u), Fp(1923111974u), Fp(744004766u),  Fp(549271463u),  Fp(1781349648u), Fp(542259047u),
    Fp(1536158148u), Fp(715456982u),  Fp(503426110u),  Fp(340311124u),  Fp(1558555932u), Fp(1226350925u), Fp(742828095u),  Fp(1338992758u),
    Fp(1641600456u), Fp(1843351545u), Fp(301835475u),  Fp(43203215u),   Fp(386838401u),  Fp(1520185679u), Fp(1235297680u), Fp(904680097u),
    Fp(1491801617u), Fp(1581784677u), Fp(913384905u),  Fp(247083962u),  Fp(532844013u),  Fp(107190701u),  Fp(213827818u),  Fp(1979521776u),
    Fp(1358282574u), Fp(1681743681u), Fp(1867507480u), Fp(1530706910u), Fp(507181886u),  Fp(695185447u),  Fp(1172395131u), Fp(1250800299u),
    Fp(1503161625u), Fp(817684387u),  Fp(498481458u),  Fp(494676004u),  Fp(1404253825u), Fp(108246855u),  Fp(59414691u),   Fp(744214112u),
    Fp(890862029u),  Fp(1342765939u), Fp(1417398904u), Fp(1897591937u), Fp(1066647396u), Fp(1682806907u), Fp(1015795079u), Fp(1619482808u)
};

constant Fp INTERNAL_ROUND_CONSTANTS[13] = {
    Fp(1518359488u),
    Fp(1765533241u),
    Fp(945325693u),
    Fp(422793067u),
    Fp(311365592u),
    Fp(1311448267u),
    Fp(1629555936u),
    Fp(1009879353u),
    Fp(190525218u),
    Fp(786108885u),
    Fp(557776863u),
    Fp(212616710u),
    Fp(605745517u)
};

constant Fp internal_diag16[16] = {
    Fp(2013265919u), // -2
    Fp(1u),
    Fp(2u),
    Fp(1006632961u), // 1/2
    Fp(3u),
    Fp(4u),
    Fp(1006632960u), // -1/2
    Fp(2013265918u), // -3
    Fp(2013265917u), // -4
    Fp(2005401601u), // 1/2^8
    Fp(1509949441u), // 1/4
    Fp(1761607681u), // 1/8
    Fp(2013265906u), // 1/2^27
    Fp(7864320u),    // -1/2^8
    Fp(125829120u),  // -1/16
    Fp(15u)          // -1/2^27
};

inline Fp sbox_d7(Fp x) {
    Fp x2 = x * x;
    Fp x4 = x2 * x2;
    Fp x6 = x4 * x2;
    return x6 * x;
}

inline void do_full_sboxes(thread Fp *cells) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = sbox_d7(cells[i]);
    }
}

inline void do_partial_sboxes(thread Fp *cells) {
    cells[0] = sbox_d7(cells[0]);
}

// Multiply a 4-element vector x by:
// [ 2 3 1 1 ]
// [ 1 2 3 1 ]
// [ 1 1 2 3 ]
// [ 3 1 1 2 ]
inline void multiply_by_4x4_circulant(thread Fp *x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1];
    Fp t01233 = t0123 + x[3];

    x[3] = t01233 + fp_doubled(x[0]);
    x[1] = t01123 + fp_doubled(x[2]);
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

inline void multiply_by_m_ext(thread Fp *old_cells) {
    Fp tmp_sums[4];
    for (uint i = 0; i < 4; i++) {
        tmp_sums[i] = Fp(0u);
    }
    for (uint i = 0; i < CELLS / 4; i++) {
        multiply_by_4x4_circulant(old_cells + i * 4);
        for (uint j = 0; j < 4; j++) {
            tmp_sums[j] = tmp_sums[j] + old_cells[i * 4 + j];
        }
    }
    for (uint i = 0; i < CELLS; i++) {
        old_cells[i] = old_cells[i] + tmp_sums[i % 4];
    }
}

inline void add_round_constants_full(constant Fp *ROUND_CONSTANTS, thread Fp *cells, uint round) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = cells[i] + ROUND_CONSTANTS[round * CELLS + i];
    }
}

inline void add_round_constants_partial(constant Fp *PARTIAL_ROUND_CONSTANTS, thread Fp *cells, uint round) {
    cells[0] = cells[0] + PARTIAL_ROUND_CONSTANTS[round];
}

inline void internal_layer_mat_mul(thread Fp *cells) {
    Fp part_sum = cells[1];
    for (uint i = 2; i < CELLS; i++) {
        part_sum = part_sum + cells[i];
    }
    Fp sum = part_sum + cells[0];
    cells[0] = part_sum - cells[0];                          // -2
    cells[1] = cells[1] + sum;                               // 1
    cells[2] = sum + fp_doubled(cells[2]);                   // 2
    cells[3] = sum + fp_halve(cells[3]);                     // 1/2
    cells[4] = sum + fp_doubled(cells[4]) + cells[4];        // 3
    cells[5] = sum + fp_doubled(fp_doubled(cells[5]));       // 4
    cells[6] = sum - fp_halve(cells[6]);                     // -1/2
    cells[7] = sum - (fp_doubled(cells[7]) + cells[7]);      // -3
    cells[8] = sum - fp_doubled(fp_doubled(cells[8]));       // -4
    cells[9] = sum + cells[9] * internal_diag16[9];          // 1/2^8
    cells[10] = sum + fp_halve(fp_halve(cells[10]));         // 1/4
    cells[11] = sum + cells[11] * internal_diag16[11];       // 1/8
    cells[12] = sum + cells[12] * internal_diag16[12];       // 1/2^27
    cells[13] = sum + cells[13] * internal_diag16[13];       // -1/2^8
    cells[14] = sum + cells[14] * internal_diag16[14];       // -1/16
    cells[15] = sum + cells[15] * internal_diag16[15];       // -1/2^27
}

inline void full_round_half(constant Fp *ROUND_CONSTANTS, thread Fp *cells, uint round) {
    add_round_constants_full(ROUND_CONSTANTS, cells, round);
    do_full_sboxes(cells);
    multiply_by_m_ext(cells);
}

inline void partial_round(constant Fp *PARTIAL_ROUND_CONSTANTS, thread Fp *cells, uint round) {
    add_round_constants_partial(PARTIAL_ROUND_CONSTANTS, cells, round);
    do_partial_sboxes(cells);
    internal_layer_mat_mul(cells);
}

inline void poseidon2_mix(thread Fp *cells) {
    // First linear layer
    multiply_by_m_ext(cells);

    // perform initial full rounds (external)
    for (uint i = 0; i < ROUNDS_HALF_FULL; i++) {
        full_round_half(INITIAL_ROUND_CONSTANTS, cells, i);
    }

    // perform partial rounds (internal)
    for (uint i = 0; i < ROUNDS_PARTIAL; i++) {
        partial_round(INTERNAL_ROUND_CONSTANTS, cells, i);
    }

    // perform terminal full rounds (external)
    for (uint r = 0; r < ROUNDS_HALF_FULL; r++) {
        full_round_half(TERMINAL_ROUND_CONSTANTS, cells, r);
    }
}
