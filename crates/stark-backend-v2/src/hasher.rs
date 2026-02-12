use std::{fmt::Debug, marker::PhantomData};

use derivative::Derivative;
use p3_field::Field;
use p3_symmetric::{CompressionFunction, CryptographicHasher};

/// Trait abstracting Merkle tree hash operations. This trait is used as part of the definition of
/// the [`StarkProtocolConfig`](crate::StarkProtocolConfig).
///
/// The `MerkleHasher` is parameterized over the base field `F` used as leaves,
/// and provides a `Digest` type for internal nodes.
pub trait MerkleHasher: 'static + Clone + Send + Sync {
    type F: Field;
    type Digest: 'static + Copy + Send + Sync;

    /// Hash a slice of field elements into a digest (used for leaf hashing).
    fn hash_slice(&self, vals: &[Self::F]) -> Self::Digest;

    /// Compress two digests into one (used for internal Merkle tree nodes).
    fn compress(&self, left: Self::Digest, right: Self::Digest) -> Self::Digest;

    /// Compress a vector of digests (must be power-of-two length) into a single digest
    /// by repeated binary compression.
    fn tree_compress(&self, mut hashes: Vec<Self::Digest>) -> Self::Digest {
        debug_assert!(hashes.len().is_power_of_two());
        while hashes.len() > 1 {
            let mut next = Vec::with_capacity(hashes.len() / 2);
            for pair in hashes.chunks_exact(2) {
                next.push(self.compress(pair[0], pair[1]));
            }
            hashes = next;
        }
        hashes.pop().unwrap()
    }
}

/// [MerkleHasher] implementation built from a cryptographic hash function and a compression
/// function. This struct is intended for use by the protocol verifier. Prover backends may use
/// independent but functionally equivalent implementations.
#[derive(Derivative, derive_new::new)]
#[derivative(
    Clone(bound = "H: Clone, C: Clone"),
    Debug(bound = "H: Debug, C: Debug")
)]
pub struct Hasher<F, Digest, H, C> {
    hash: H,
    compress: C,
    _phantom: PhantomData<(F, Digest)>,
}

impl<F, Digest, H, C> MerkleHasher for Hasher<F, Digest, H, C>
where
    F: Field,
    Digest: 'static + Copy + Send + Sync,
    H: 'static + Send + Sync + CryptographicHasher<F, Digest>,
    C: 'static + Send + Sync + CompressionFunction<Digest, 2>,
{
    type F = F;
    type Digest = Digest;

    fn hash_slice(&self, vals: &[Self::F]) -> Self::Digest {
        self.hash.hash_slice(vals)
    }

    fn compress(&self, left: Self::Digest, right: Self::Digest) -> Self::Digest {
        self.compress.compress([left, right])
    }
}
