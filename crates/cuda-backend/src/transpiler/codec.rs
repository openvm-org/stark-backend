// A simple custom codec for symbolic constraints that's easy to implement in CUDA

use openvm_stark_backend::air_builders::symbolic::symbolic_variable::{Entry, SymbolicVariable};
use p3_field::{Field, PrimeField32};

use super::{Constraint, ConstraintWithFlag, Source};

// Basic codec trait
pub trait Codec {
    type Encoded;

    fn encode(&self) -> Self::Encoded;

    fn decode(encoded: Self::Encoded) -> Self
    where
        Self: Sized;
}

const PREPROCESSED: u64 = 0;
const MAIN: u64 = 1;
const PERMUTATION: u64 = 2;
const PUBLIC: u64 = 3;
const CHALLENGE: u64 = 4;
const EXPOSED: u64 = 5;

// Entry is encoded in 16-bit little-endian
impl Codec for Entry {
    type Encoded = u64;

    fn encode(&self) -> u64 {
        let (src, part_index, offset) = match self {
            Entry::Preprocessed { offset } => (PREPROCESSED, 0, *offset),
            Entry::Main { part_index, offset } => (MAIN, *part_index, *offset),
            Entry::Permutation { offset } => (PERMUTATION, 0, *offset),
            Entry::Public => (PUBLIC, 0, 0),
            Entry::Challenge => (CHALLENGE, 0, 0),
            Entry::Exposed => (EXPOSED, 0, 0),
        };
        // 4-bit src | 8-bit part_index | 4-bit offset
        assert!(src < 16);
        assert!(part_index < 256);
        assert!(offset < 16);
        src | (part_index << 4) as u64 | (offset << 12) as u64
    }

    fn decode(encoded: u64) -> Self {
        // 4-bit src | 8-bit part_index | 4-bit offset
        let src = encoded & 0x0f;
        let part_index = ((encoded >> 4) & 0xff) as usize;
        let offset = ((encoded >> 12) & 0x0f) as usize;
        match src {
            PREPROCESSED => Entry::Preprocessed { offset },
            MAIN => Entry::Main { part_index, offset },
            PERMUTATION => Entry::Permutation { offset },
            PUBLIC => Entry::Public,
            CHALLENGE => Entry::Challenge,
            EXPOSED => Entry::Exposed,
            _ => panic!(
                "Invalid Entry: src={} part_index={} offset={}",
                src, part_index, offset
            ),
        }
    }
}

// SymbolicVariable is encoded in 32-bit little-endian
impl<F: Field> Codec for SymbolicVariable<F> {
    type Encoded = u64;

    fn encode(&self) -> u64 {
        let entry_code = self.entry.encode();
        let index = self.index as u64;

        assert!(entry_code <= 0xffff);
        assert!(index <= 0xffff);
        // 16-bit entry | 16-bit index
        const ENTRY_SHIFT: u64 = 16;
        entry_code | (index << ENTRY_SHIFT)
    }

    fn decode(encoded: u64) -> Self {
        // 16-bit entry | 16-bit index
        const ENTRY_MASK: u64 = 0xffff;
        const ENTRY_SHIFT: u64 = 16;
        let entry = Entry::decode(encoded & ENTRY_MASK);
        let index = (encoded >> ENTRY_SHIFT) as usize;
        Self::new(entry, index)
    }
}

// 0..=5 is reserved for Source::Var
const SOURCE_INTERMEDIATE: u64 = EXPOSED + 1;
const SOURCE_CONSTANT: u64 = SOURCE_INTERMEDIATE + 1;
const SOURCE_IS_FIRST: u64 = SOURCE_CONSTANT + 1;
const SOURCE_IS_LAST: u64 = SOURCE_IS_FIRST + 1;
const SOURCE_IS_TRANSITION: u64 = SOURCE_IS_LAST + 1;
const SOURCE_TERMINAL: u64 = SOURCE_IS_TRANSITION + 1;

// To add to PREPROCESSED, MAIN, and PERMUTATION to indicate that the variable is buffered
const BUFFERED_IDE_OFFSET: u64 = SOURCE_TERMINAL + 1;
const BUFFERED_PREPROCESSED: u64 = BUFFERED_IDE_OFFSET + PREPROCESSED;
// const BUFFERED_MAIN: u64 = BUFFERED_IDE_OFFSET + MAIN;
const BUFFERED_PERMUTATION: u64 = BUFFERED_IDE_OFFSET + PERMUTATION;

// Source is encoded in 48-bit little-endian and the enum discriminant is encoded by least
// significant 4-bits
impl<F: Field + PrimeField32> Codec for Source<F> {
    type Encoded = u64;

    fn encode(&self) -> u64 {
        match self {
            Source::IsFirst => SOURCE_IS_FIRST,
            Source::IsLast => SOURCE_IS_LAST,
            Source::IsTransition => SOURCE_IS_TRANSITION,
            Source::Constant(f) => {
                // 16-bit src | 32-bit value
                const CONSTANT_SHIFT: u64 = 16;
                // f.as_u32() is not implemented for this field => unsafe cast
                let f_u32 = f.as_canonical_u32();

                SOURCE_CONSTANT | ((f_u32 as u64) << CONSTANT_SHIFT)
            }
            Source::Var(v) => v.encode(),
            Source::BufferedVar((v, idx)) => {
                // 16-bit entry | 16-bit index | 16-bit buffer index
                (v.encode() + BUFFERED_IDE_OFFSET) | ((*idx as u64) << 32)
            }
            Source::Intermediate(idx) => {
                // 4-bit src | 20-bit index
                const INTERMEDIATE_SHIFT: u64 = 4;
                const INTERMEDIATE_MASK: u64 = 0xf_ffff; // 20-bit index
                SOURCE_INTERMEDIATE | (((*idx as u64) & INTERMEDIATE_MASK) << INTERMEDIATE_SHIFT)
            }
            Source::TerminalIntermediate => SOURCE_INTERMEDIATE,
        }
    }

    fn decode(encoded: u64) -> Self {
        const ENTRY_SRC_MASK: u64 = 0xf; // 4-bit
        match encoded & ENTRY_SRC_MASK {
            SOURCE_IS_FIRST => Source::IsFirst,
            SOURCE_IS_LAST => Source::IsLast,
            SOURCE_IS_TRANSITION => Source::IsTransition,
            SOURCE_CONSTANT => {
                // 16-bit src | 32-bit value
                const CONSTANT_SHIFT: u64 = 16;
                Source::Constant(F::from_canonical_u32((encoded >> CONSTANT_SHIFT) as u32))
            }
            PREPROCESSED..=EXPOSED => Source::Var(SymbolicVariable::decode(encoded)),
            SOURCE_INTERMEDIATE => {
                const INTERMEDIATE_SHIFT: u64 = 4;
                const INTERMEDIATE_MASK: u64 = 0xf_ffff; // 20-bit index
                Source::Intermediate(((encoded >> INTERMEDIATE_SHIFT) & INTERMEDIATE_MASK) as usize)
            }
            SOURCE_TERMINAL => Source::TerminalIntermediate,
            BUFFERED_PREPROCESSED..=BUFFERED_PERMUTATION => {
                let buffer_idx = ((encoded >> 32) & 0xffff) as usize;
                let v = SymbolicVariable::decode(encoded & 0xffffffff);
                Source::BufferedVar((v, buffer_idx))
            }
            _ => unreachable!(),
        }
    }
}

#[allow(unused)]
pub fn as_intermediate<F: Field>(source: &Source<F>) -> Option<&usize> {
    match source {
        Source::Intermediate(id) => Some(id),
        _ => None,
    }
}

const OP_ADD: u8 = 0;
const OP_SUB: u8 = 1;
const OP_MUL: u8 = 2;
const OP_NEG: u8 = 3;
const OP_VAR: u8 = 4;

const INPUT_OPERANDS_MASK: u64 = (1 << 48) - 1; // 48-bit mask
const OUTPUT_OPERAND_MASK: u64 = (1 << 24) - 1; // 24-bit mask

impl<F: Field + PrimeField32> Codec for ConstraintWithFlag<F> {
    type Encoded = u128;

    // 48-bit x | 48-bit y | 24-bit z | 7-bit op | 1-bit is_constraint
    fn encode(&self) -> u128 {
        let dummy_source = Source::Constant(F::ZERO);
        let (x, y, z, exp, write_buffer) = match &self.constraint {
            Constraint::Add(x, y, z) => (
                x.encode(),
                y.encode(),
                z.encode(),
                OP_ADD,
                !matches!(z, Source::TerminalIntermediate),
            ),
            Constraint::Sub(x, y, z) => (
                x.encode(),
                y.encode(),
                z.encode(),
                OP_SUB,
                !matches!(z, Source::TerminalIntermediate),
            ),
            Constraint::Mul(x, y, z) => (
                x.encode(),
                y.encode(),
                z.encode(),
                OP_MUL,
                !matches!(z, Source::TerminalIntermediate),
            ),
            // since y is not used, we just fill it with F::ZERO
            Constraint::Neg(x, z) => (
                x.encode(),
                dummy_source.encode(),
                z.encode(),
                OP_NEG,
                !matches!(z, Source::TerminalIntermediate),
            ),
            Constraint::Variable(v) => (
                v.encode(),
                dummy_source.encode(),
                dummy_source.encode(),
                OP_VAR,
                false,
            ),
        };

        let x = x & INPUT_OPERANDS_MASK; // 48bit
        let y = y & INPUT_OPERANDS_MASK; // 48bit
        let z = z & OUTPUT_OPERAND_MASK; // 24bit

        (x as u128)
            | ((y as u128) << 48)
            | ((z as u128) << 96)
            | ((exp as u128) << 120)
            | ((write_buffer as u128) << 126)
            | ((self.need_accumulate as u128) << 127)
    }

    fn decode(encoded: u128) -> Self {
        let coded_x = encoded & (INPUT_OPERANDS_MASK as u128); // 48bit
        let coded_y = (encoded >> 48) & (INPUT_OPERANDS_MASK as u128); // 48bit
        let coded_z = (encoded >> 96) & (OUTPUT_OPERAND_MASK as u128); // 24bit

        let x = Source::decode(coded_x as u64);
        let y = Source::decode(coded_y as u64);
        let write_buffer = (encoded >> 126) & 1 == 1;
        let z = if write_buffer {
            Source::decode(coded_z as u64)
        } else {
            Source::TerminalIntermediate
        };
        let exp = (encoded >> 120) as u8 & 0x3F; // 6bit
        let need_accumulate = (encoded >> 127) & 1 == 1;

        let constraint = match exp {
            OP_ADD => Constraint::Add(x, y, z),
            OP_SUB => Constraint::Sub(x, y, z),
            OP_MUL => Constraint::Mul(x, y, z),
            OP_NEG => Constraint::Neg(x, z),
            OP_VAR => Constraint::Variable(x),
            _ => panic!(
                "Invalid constraint decoding: exp={} coded_x={} coded_y={} coded_z={}, x={:?} y={:?} z={:?}",
                exp, coded_x, coded_y, coded_z, x, y, z
            ),
        };

        Self {
            constraint,
            need_accumulate,
        }
    }
}
