// Poseidon2 hash permutation for Metal Shading Language.
//
// Based on openvm-org cuda-common/include/poseidon2.cuh
// and risc0/risc0/sys/kernels/zkp/metal/poseidon2.metal
//
// Configuration for BabyBear:
//   WIDTH  = 16 (state size)
//   RATE   = 8
//   OUTPUT = 8 (digest size in field elements)
//   ROUNDS_FULL    = 8 (4 initial + 4 terminal)
//   ROUNDS_PARTIAL = 13 (internal rounds)
//   S-box degree   = 7 (x^7)

#pragma once

#include <metal_stdlib>

#include "baby_bear.h"

using namespace metal;

namespace poseidon2 {

// ============================================================================
// Constants
// ============================================================================

#define CELLS 16
#define CELLS_RATE 8
#define CELLS_OUT 8

#define ROUNDS_FULL 8
#define ROUNDS_HALF_FULL (ROUNDS_FULL / 2)
#define ROUNDS_PARTIAL 13

// Initial (external) round constants: 4 rounds x 16 elements = 64 values
constant Fp INITIAL_ROUND_CONSTANTS[64] = {
    Fp(1774958255), Fp(1185780729), Fp(1621102414), Fp(1796380621),
    Fp(588815102),  Fp(1932426223), Fp(1925334750), Fp(747903232),
    Fp(89648862),   Fp(360728943),  Fp(977184635),  Fp(1425273457),
    Fp(256487465),  Fp(1200041953), Fp(572403254),  Fp(448208942),
    Fp(1215789478), Fp(944884184),  Fp(953948096),  Fp(547326025),
    Fp(646827752),  Fp(889997530),  Fp(1536873262), Fp(86189867),
    Fp(1065944411), Fp(32019634),   Fp(333311454),  Fp(456061748),
    Fp(1963448500), Fp(1827584334), Fp(1391160226), Fp(1348741381),
    Fp(88424255),   Fp(104111868),  Fp(1763866748), Fp(79691676),
    Fp(1988915530), Fp(1050669594), Fp(359890076),  Fp(573163527),
    Fp(222820492),  Fp(159256268),  Fp(669703072),  Fp(763177444),
    Fp(889367200),  Fp(256335831),  Fp(704371273),  Fp(25886717),
    Fp(51754520),   Fp(1833211857), Fp(454499742),  Fp(1384520381),
    Fp(777848065),  Fp(1053320300), Fp(1851729162), Fp(344647910),
    Fp(401996362),  Fp(1046925956), Fp(5351995),    Fp(1212119315),
    Fp(754867989),  Fp(36972490),   Fp(751272725),  Fp(506915399)
};

// Terminal (external) round constants: 4 rounds x 16 elements = 64 values
constant Fp TERMINAL_ROUND_CONSTANTS[64] = {
    Fp(1922082829), Fp(1870549801), Fp(1502529704), Fp(1990744480),
    Fp(1700391016), Fp(1702593455), Fp(321330495),  Fp(528965731),
    Fp(183414327),  Fp(1886297254), Fp(1178602734), Fp(1923111974),
    Fp(744004766),  Fp(549271463),  Fp(1781349648), Fp(542259047),
    Fp(1536158148), Fp(715456982),  Fp(503426110),  Fp(340311124),
    Fp(1558555932), Fp(1226350925), Fp(742828095),  Fp(1338992758),
    Fp(1641600456), Fp(1843351545), Fp(301835475),  Fp(43203215),
    Fp(386838401),  Fp(1520185679), Fp(1235297680), Fp(904680097),
    Fp(1491801617), Fp(1581784677), Fp(913384905),  Fp(247083962),
    Fp(532844013),  Fp(107190701),  Fp(213827818),  Fp(1979521776),
    Fp(1358282574), Fp(1681743681), Fp(1867507480), Fp(1530706910),
    Fp(507181886),  Fp(695185447),  Fp(1172395131), Fp(1250800299),
    Fp(1503161625), Fp(817684387),  Fp(498481458),  Fp(494676004),
    Fp(1404253825), Fp(108246855),  Fp(59414691),   Fp(744214112),
    Fp(890862029),  Fp(1342765939), Fp(1417398904), Fp(1897591937),
    Fp(1066647396), Fp(1682806907), Fp(1015795079), Fp(1619482808)
};

// Internal (partial) round constants: 13 values (one per partial round)
constant Fp INTERNAL_ROUND_CONSTANTS[13] = {
    Fp(1518359488),
    Fp(1765533241),
    Fp(945325693),
    Fp(422793067),
    Fp(311365592),
    Fp(1311448267),
    Fp(1629555936),
    Fp(1009879353),
    Fp(190525218),
    Fp(786108885),
    Fp(557776863),
    Fp(212616710),
    Fp(605745517)
};

// Diagonal elements for the internal MDS matrix.
// These encode the diagonal d_i of M_int = 1 + diag(d_0, ..., d_15).
// The actual value used is: sum_of_all + d_i * cell_i (see internal_layer_mat_mul).
constant Fp INTERNAL_DIAG16[16] = {
    Fp(2013265919), // -2
    Fp(1),
    Fp(2),
    Fp(1006632961), // 1/2
    Fp(3),
    Fp(4),
    Fp(1006632960), // -1/2
    Fp(2013265918), // -3
    Fp(2013265917), // -4
    Fp(2005401601), // 1/2^8
    Fp(1509949441), // 1/4
    Fp(1761607681), // 1/8
    Fp(2013265906), // 1/2^27
    Fp(7864320),    // -1/2^8
    Fp(125829120),  // -1/16
    Fp(15)          // -1/2^27
};

// ============================================================================
// S-box: x^7
// ============================================================================

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

// ============================================================================
// MDS matrix operations
// ============================================================================

/// Multiply a 4-element vector by the circulant matrix:
///   [ 2 3 1 1 ]
///   [ 1 2 3 1 ]
///   [ 1 1 2 3 ]
///   [ 3 1 1 2 ]
///
/// This is the Plonky3 optimized version.
inline void multiply_by_4x4_circulant(thread Fp *x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1];
    Fp t01233 = t0123 + x[3];

    x[3] = t01233 + x[0].doubled();
    x[1] = t01123 + x[2].doubled();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

/// External MDS matrix: applies the width-16 M_ext matrix.
/// Uses the block-diagonal + low-rank structure from the Poseidon2 paper (Appendix B).
inline void multiply_by_m_ext(thread Fp *old_cells) {
    Fp tmp_sums[4];
    for (uint i = 0; i < 4; i++) {
        tmp_sums[i] = Fp(0);
    }
    for (uint i = 0; i < CELLS / 4; i++) {
        multiply_by_4x4_circulant(old_cells + i * 4);
        for (uint j = 0; j < 4; j++) {
            tmp_sums[j] += old_cells[i * 4 + j];
        }
    }
    for (uint i = 0; i < CELLS; i++) {
        old_cells[i] += tmp_sums[i % 4];
    }
}

// ============================================================================
// Round-constant addition
// ============================================================================

inline void add_round_constants_full(constant Fp *ROUND_CONSTANTS, thread Fp *cells, uint round) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] += ROUND_CONSTANTS[round * CELLS + i];
    }
}

inline void add_round_constants_partial(constant Fp *PARTIAL_ROUND_CONSTANTS, thread Fp *cells, uint round) {
    cells[0] += PARTIAL_ROUND_CONSTANTS[round];
}

// ============================================================================
// Internal MDS matrix multiplication (optimized diagonal form)
//
// The internal matrix is 1 + diag(d_0, ..., d_15) applied as:
//   sum = sum(cells[0..15])
//   cells[i] = sum + d_i * cells[i]
//
// For BabyBear width-16, the diagonal values are small fractions/negatives
// of powers of 2, allowing optimized multiplications using doubled()/halve().
// ============================================================================

inline void internal_layer_mat_mul(thread Fp *cells) {
    Fp part_sum = cells[1];
    for (uint i = 2; i < CELLS; i++) {
        part_sum += cells[i];
    }
    // sum = cells[0] + cells[1] + ... + cells[15]
    Fp sum = part_sum + cells[0];

    // Apply diagonal: cells[i] = sum + (d_i - 1) * cells[i]
    // where (d_i - 1) encodes the deviation from the identity.
    // The actual formula is: cells[i] = sum - cells[i] + d_i * cells[i] = sum + (d_i - 1)*cells[i]
    // But following Plonky3 convention: cells[i] = sum + diag_i * cells[i] where diag_i already
    // includes the -1 offset from the identity part. The computation below matches the Plonky3
    // Rust implementation exactly.
    cells[0]  = part_sum - cells[0];                       // diag = -2
    cells[1]  += sum;                                      // diag =  1
    cells[2]  = sum + cells[2].doubled();                  // diag =  2
    cells[3]  = sum + cells[3].halve();                    // diag =  1/2
    cells[4]  = sum + cells[4].doubled() + cells[4];       // diag =  3
    cells[5]  = sum + cells[5].doubled().doubled();        // diag =  4
    cells[6]  = sum - cells[6].halve();                    // diag = -1/2
    cells[7]  = sum - (cells[7].doubled() + cells[7]);     // diag = -3
    cells[8]  = sum - cells[8].doubled().doubled();        // diag = -4
    cells[9]  = sum + cells[9] * INTERNAL_DIAG16[9];       // diag =  1/2^8
    cells[10] = sum + cells[10].halve().halve();           // diag =  1/4
    cells[11] = sum + cells[11] * INTERNAL_DIAG16[11];     // diag =  1/8
    cells[12] = sum + cells[12] * INTERNAL_DIAG16[12];     // diag =  1/2^27
    cells[13] = sum + cells[13] * INTERNAL_DIAG16[13];     // diag = -1/2^8
    cells[14] = sum + cells[14] * INTERNAL_DIAG16[14];     // diag = -1/16
    cells[15] = sum + cells[15] * INTERNAL_DIAG16[15];     // diag = -1/2^27
}

// ============================================================================
// Round functions
// ============================================================================

inline void full_round_half(constant Fp *ROUND_CONSTANTS, thread Fp *cells, uint round) {
    add_round_constants_full(ROUND_CONSTANTS, cells, round);
    do_full_sboxes(cells);
    multiply_by_m_ext(cells);
}

inline void partial_round(thread Fp *cells, uint round) {
    add_round_constants_partial(INTERNAL_ROUND_CONSTANTS, cells, round);
    do_partial_sboxes(cells);
    internal_layer_mat_mul(cells);
}

// ============================================================================
// Full Poseidon2 permutation
// ============================================================================

/// Apply the complete Poseidon2 permutation to a state of CELLS (16) field elements.
/// The state is modified in-place.
inline void poseidon2_permute(thread Fp *cells) {
    // First linear layer
    multiply_by_m_ext(cells);

    // Initial full rounds (external)
    for (uint i = 0; i < ROUNDS_HALF_FULL; i++) {
        full_round_half(INITIAL_ROUND_CONSTANTS, cells, i);
    }

    // Partial rounds (internal)
    for (uint i = 0; i < ROUNDS_PARTIAL; i++) {
        partial_round(cells, i);
    }

    // Terminal full rounds (external)
    for (uint r = 0; r < ROUNDS_HALF_FULL; r++) {
        full_round_half(TERMINAL_ROUND_CONSTANTS, cells, r);
    }
}

} // namespace poseidon2
