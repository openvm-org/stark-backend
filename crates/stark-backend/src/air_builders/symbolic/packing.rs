use super::symbolic_variable::{Entry, SymbolicVariable};

pub const ENTRY_TYPE_PREPROCESSED: u8 = 0;
pub const ENTRY_TYPE_MAIN: u8 = 1;
pub const ENTRY_TYPE_PERMUTATION: u8 = 2;
pub const ENTRY_TYPE_PUBLIC: u8 = 3;
pub const ENTRY_TYPE_CHALLENGE: u8 = 4;
pub const ENTRY_TYPE_EXPOSED: u8 = 5;

pub fn entry_parts(entry: &Entry) -> (u8, u8, u8) {
    let (entry_type, part_index, offset) = match *entry {
        Entry::Preprocessed { offset } => (ENTRY_TYPE_PREPROCESSED, 0, offset),
        Entry::Main { part_index, offset } => (ENTRY_TYPE_MAIN, part_index, offset),
        Entry::Permutation { offset } => (ENTRY_TYPE_PERMUTATION, 0, offset),
        Entry::Public => (ENTRY_TYPE_PUBLIC, 0, 0),
        Entry::Challenge => (ENTRY_TYPE_CHALLENGE, 0, 0),
        Entry::Exposed => (ENTRY_TYPE_EXPOSED, 0, 0),
    };

    assert!(entry_type < 16);
    assert!(part_index < 256);
    assert!(offset < 16);

    (entry_type, part_index as u8, offset as u8)
}

pub fn pack_entry(entry: &Entry) -> u16 {
    let (entry_type, part_index, offset) = entry_parts(entry);
    (entry_type as u16) | ((part_index as u16) << 4) | ((offset as u16) << 12)
}

pub fn pack_index(index: usize) -> u16 {
    assert!(index <= u16::MAX as usize);
    index as u16
}

pub fn pack_symbolic_var<F>(var: &SymbolicVariable<F>) -> u32 {
    let entry_code = pack_entry(&var.entry) as u32;
    let index = pack_index(var.index) as u32;
    entry_code | (index << 16)
}
