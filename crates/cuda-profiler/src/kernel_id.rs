//! Kernel-id <-> name registry.
//!
//! On the device side, every probe call is parameterized by a `u32` kernel id
//! supplied via the `KID(...)` macro from `cta_probe.cuh`, which expands to a
//! constexpr FNV-1a hash of the kernel name. On the host side, we hash with
//! the *exact same* algorithm and store the (id, name) pair so the assembler
//! can label each CtaRecord on the timeline.
//!
//! Synchronization rule: every kernel that uses `SHADOW_KERNEL_BEGIN(KID(name))`
//! on the device side must have a matching `register_kernel(name)` call on
//! the host side. The id is implicitly the FNV-1a hash, so the two sides
//! cannot drift as long as the spelling matches.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use tracing::warn;

/// FNV-1a, 32-bit. Matches the `cta_kid_fnv1a` constexpr in `cta_probe.cuh`.
///
/// We deliberately bias the hash space away from 0: id = 0 is the wire value
/// for "uninitialized slot" in the device ring, so a kernel name that hashed
/// to 0 would be silently dropped by the drain thread. We perturb a 0 hash
/// to the nearby non-zero constant — this can only happen if a name is
/// crafted to collide, which we treat as a bug.
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

static REGISTRY: Lazy<RwLock<HashMap<u32, String>>> = Lazy::new(|| RwLock::new(HashMap::new()));

/// Register a kernel name, returning the id that was assigned.
///
/// Calling twice with the same name is fine. Calling twice with two different
/// names that hash to the same id is logged as a collision warning; the first
/// registered name wins.
pub fn register_kernel(name: &str) -> u32 {
    let id = fnv1a(name);
    let mut map = REGISTRY.write();
    if let Some(existing) = map.get(&id) {
        if existing != name {
            warn!(
                id,
                existing = %existing,
                colliding = %name,
                "cuda-profiler: kernel-id hash collision; first name wins",
            );
        }
        return id;
    }
    map.insert(id, name.to_string());
    id
}

/// Snapshot the registry for inclusion in the binary log header.
pub fn registered_kernels() -> Vec<(u32, String)> {
    REGISTRY
        .read()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a known answers — keep this in lockstep with the constexpr in
    /// `cta_probe.cuh`. If you ever change the algorithm, change both.
    #[test]
    fn known_fnv1a() {
        assert_eq!(fnv1a(""), 0x811c9dc5);
        assert_eq!(fnv1a("a"), 0xe40c292c);
        assert_eq!(fnv1a("foobar"), 0xbf9cf968);
    }
}
