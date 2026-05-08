//! FNV-1a 32-bit hash used for `kernel_id` values in [`record::CtaRecord`].
//!
//! The NVBit tool computes the same hash on the C++ side
//! (`nvbit-tool/tool.cu::fnv1a`) so a `(kernel_id, name)` pair embedded as a
//! `Record::KernelName` always agrees with the `kernel_id` field of every
//! `Record::Cta` from the same producer. Keeping a tiny pure-Rust copy here
//! lets the assembler verify or recompute the mapping if the log is missing
//! a `KernelName` record (e.g. a truncated capture).
//!
//! ## Wire-format detail
//!
//! `id == 0` is the wire-level "uninitialized slot" sentinel; a kernel name
//! whose FNV-1a hash naturally lands on 0 is biased to `0xdeadbeef` so the
//! sentinel stays unambiguous.

/// FNV-1a, 32-bit. Bytes-only, no Unicode normalization — match the C++
/// implementation exactly.
pub fn fnv1a(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in name.as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    if h == 0 {
        0xdead_beef
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a known answers — keep this in lockstep with the C++
    /// `fnv1a` in `nvbit-tool/tool.cu`. If the algorithm ever changes,
    /// change both.
    #[test]
    fn known_fnv1a() {
        assert_eq!(fnv1a(""), 0x811c9dc5);
        assert_eq!(fnv1a("a"), 0xe40c292c);
        assert_eq!(fnv1a("foobar"), 0xbf9cf968);
    }
}
