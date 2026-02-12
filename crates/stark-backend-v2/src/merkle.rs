use p3_field::Field;

/// Trait abstracting Merkle tree hash operations.
///
/// The `MerkleHasher` is parameterized over the base field `F` used as leaves,
/// and provides a `Digest` type for internal nodes.
pub trait MerkleHasher: 'static {
    type F: Field;
    type Digest: Copy + Send + Sync + Eq + core::fmt::Debug;

    /// Hash a slice of field elements into a digest (used for leaf hashing).
    fn hash_slice(vals: &[Self::F]) -> Self::Digest;

    /// Compress two digests into one (used for internal Merkle tree nodes).
    fn compress(left: Self::Digest, right: Self::Digest) -> Self::Digest;

    /// Compress a vector of digests (must be power-of-two length) into a single digest
    /// by repeated binary compression.
    fn tree_compress(mut hashes: Vec<Self::Digest>) -> Self::Digest {
        debug_assert!(hashes.len().is_power_of_two());
        while hashes.len() > 1 {
            let mut next = Vec::with_capacity(hashes.len() / 2);
            for pair in hashes.chunks_exact(2) {
                next.push(Self::compress(pair[0], pair[1]));
            }
            hashes = next;
        }
        hashes.pop().unwrap()
    }
}
