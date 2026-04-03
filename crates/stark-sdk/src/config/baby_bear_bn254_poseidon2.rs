//! BabyBear + BN254 Poseidon2 STARK configuration.
//!
//! Two Poseidon2 permutations over BN254 are used:
//!
//! - **Width 3** (leaf hashing & transcript sponge): rF=8, rP=56, d=5. Round constants from [HorizenLabs/poseidon2](https://github.com/HorizenLabs/poseidon2)
//!   via the `zkhash` crate (`RC3`). Matches `p3-bn254`'s `Poseidon2Bn254<3>`.
//!
//! - **Width 2** (Merkle compression): rF=6, rP=50, d=5. Round constants from [gnark-crypto](https://github.com/Consensys/gnark-crypto).
//!   Matches `poseidon2.NewPermutation(2, 6, 50)`. See [`super::bn254_poseidon2`].

use std::{
    io::{self, Read, Write},
    marker::PhantomData,
};

use num_bigint::BigUint;
use openvm_stark_backend::{
    codec::{
        decode_extension_field32, decode_prime_field32, encode_extension_field32,
        encode_prime_field32, DecodableConfig, EncodableConfig,
    },
    hasher::{Hasher, MultiFieldHasher},
    p3_symmetric::TruncatedPermutation,
    prover::{Coordinator, CpuColMajorBackend, ReferenceDevice},
    transcript::multi_field::MultiFieldTranscript,
    FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams,
};
use p3_baby_bear::BabyBear;
// NOTE: plonky3's Bn254 is the type for scalar field of the BN254 curve. It is not the type
// for the curve itself.
pub use p3_bn254::Bn254 as Bn254Scalar;
use p3_field::{extension::BinomialExtensionField, PrimeField};

use super::bn254_poseidon2::{
    default_bn254_poseidon2_width2, default_bn254_poseidon2_width3, Poseidon2Bn254Width2,
    Poseidon2Bn254Width3,
};

/// Width of the Poseidon2 sponge permutation (leaf hashing & transcript).
pub const SPONGE_WIDTH: usize = 3;
/// Width of the Poseidon2 compression permutation (Merkle tree).
pub const COMPRESS_WIDTH: usize = 2;
/// Sponge rate in BabyBear elements: BN254 rate (2) * BabyBear elements per BN254 (8) = 16.
pub const BABY_BEAR_RATE: usize = 16;
/// Sponge rate in BN254 elements.
pub const BN254_RATE: usize = 2;
/// Digest width in BN254 elements.
pub const DIGEST_WIDTH: usize = 1;

type H = MultiFieldHasher<
    F,
    Bn254Scalar,
    Poseidon2Bn254Width3,
    SPONGE_WIDTH,
    BABY_BEAR_RATE,
    DIGEST_WIDTH,
>;
type Compress =
    TruncatedPermutation<Poseidon2Bn254Width2, COMPRESS_WIDTH, DIGEST_WIDTH, COMPRESS_WIDTH>;
type PermHasher = Hasher<F, Digest, H, Compress>;
// Defined below
type SC = BabyBearBn254Poseidon2Config;

// Convenience type aliases
pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, 4>;
pub const D_EF: usize = 4;
pub type Digest = [Bn254Scalar; DIGEST_WIDTH];

#[derive(Clone, Debug, derive_new::new)]
pub struct BabyBearBn254Poseidon2Config {
    params: SystemParams,
    hasher: PermHasher,
}

impl StarkProtocolConfig for BabyBearBn254Poseidon2Config {
    type F = F;
    type EF = EF;
    type Digest = Digest;
    type Hasher = PermHasher;

    fn params(&self) -> &SystemParams {
        &self.params
    }

    fn hasher(&self) -> &Self::Hasher {
        &self.hasher
    }
}

impl EncodableConfig for BabyBearBn254Poseidon2Config {
    fn encode_base_field<W: Write>(val: &F, writer: &mut W) -> io::Result<()> {
        encode_prime_field32(val, writer)
    }

    fn encode_extension_field<W: Write>(val: &EF, writer: &mut W) -> io::Result<()> {
        encode_extension_field32::<Self::F, _, _>(val, writer)
    }

    fn encode_digest<W: Write>(digest: &Self::Digest, writer: &mut W) -> io::Result<()> {
        for val in digest {
            // Bn254 is encoded as 32 big-endian bytes for compatibility with
            // CommitBytes in openvm and EVM conventions.
            let bytes = val.as_canonical_biguint().to_bytes_be();
            let mut buf = [0u8; 32];
            buf[32 - bytes.len()..].copy_from_slice(&bytes);
            writer.write_all(&buf)?;
        }
        Ok(())
    }
}

impl DecodableConfig for BabyBearBn254Poseidon2Config {
    fn decode_base_field<R: Read>(reader: &mut R) -> io::Result<F> {
        decode_prime_field32(reader)
    }

    fn decode_extension_field<R: Read>(reader: &mut R) -> io::Result<EF> {
        decode_extension_field32::<F, _, _>(reader)
    }

    fn decode_digest<R: Read>(reader: &mut R) -> io::Result<Digest> {
        let mut result = Digest::default();
        for val in &mut result {
            // Bn254 is encoded as 32 big-endian bytes for compatibility with
            // CommitBytes in openvm and EVM conventions.
            let mut buf = [0u8; 32];
            reader.read_exact(&mut buf)?;
            let big = BigUint::from_bytes_be(&buf);
            *val = Bn254Scalar::from_biguint(big)
                .ok_or_else(|| io::Error::other("invalid Bn254 element"))?;
        }
        Ok(result)
    }
}

impl BabyBearBn254Poseidon2Config {
    pub fn new_from_perms(
        params: SystemParams,
        hash_perm: Poseidon2Bn254Width3,
        compress_perm: Poseidon2Bn254Width2,
    ) -> Self {
        let hasher = Hasher::new(
            MultiFieldHasher::new(hash_perm),
            TruncatedPermutation::new(compress_perm),
        );
        Self { params, hasher }
    }

    pub fn default_from_params(params: SystemParams) -> Self {
        let hash_perm = default_bn254_poseidon2_width3();
        let compress_perm = default_bn254_poseidon2_width2();
        Self::new_from_perms(params, hash_perm, compress_perm)
    }
}

pub type Transcript =
    MultiFieldTranscript<F, Bn254Scalar, Poseidon2Bn254Width3, SPONGE_WIDTH, BN254_RATE>;

pub struct BabyBearBn254Poseidon2RefEngine<TS = Transcript> {
    device: ReferenceDevice<SC>,
    _transcript: PhantomData<TS>,
}

impl<TS> StarkEngine for BabyBearBn254Poseidon2RefEngine<TS>
where
    TS: FiatShamirTranscript<SC> + From<Poseidon2Bn254Width3>,
{
    type SC = SC;
    type PB = CpuColMajorBackend<SC>;
    type PD = ReferenceDevice<SC>;
    type TS = TS;

    fn new(params: SystemParams) -> Self {
        let config = BabyBearBn254Poseidon2Config::default_from_params(params);
        Self {
            device: ReferenceDevice::new(config),
            _transcript: PhantomData,
        }
    }

    fn config(&self) -> &SC {
        self.device.config()
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn initial_transcript(&self) -> Self::TS {
        TS::from(default_bn254_poseidon2_width3())
    }

    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(CpuColMajorBackend::new(), self.device.clone(), transcript)
    }
}

// ---- Optimized CPU engine (behind `cpu-backend` feature) ----

#[cfg(feature = "cpu-backend")]
mod cpu_engine {
    use openvm_cpu_backend::{CpuBackend, CpuDevice};

    use super::*;

    /// Row-major CPU engine for BabyBear + BN254 Poseidon2.
    ///
    /// Uses the standard [`Transcript`] (no special SIMD optimization for BN254).
    pub struct BabyBearBn254Poseidon2CpuEngine<TS = Transcript> {
        device: CpuDevice<SC>,
        _transcript: PhantomData<TS>,
    }

    impl<TS> StarkEngine for BabyBearBn254Poseidon2CpuEngine<TS>
    where
        TS: FiatShamirTranscript<SC> + From<Poseidon2Bn254Width3>,
    {
        type SC = SC;
        type PB = CpuBackend<SC>;
        type PD = CpuDevice<SC>;
        type TS = TS;

        fn new(params: SystemParams) -> Self {
            let config = BabyBearBn254Poseidon2Config::default_from_params(params);
            Self {
                device: CpuDevice::new(config),
                _transcript: PhantomData,
            }
        }

        fn config(&self) -> &SC {
            self.device.config()
        }

        fn device(&self) -> &Self::PD {
            &self.device
        }

        fn initial_transcript(&self) -> Self::TS {
            TS::from(default_bn254_poseidon2_width3())
        }

        fn prover_from_transcript(
            &self,
            transcript: TS,
        ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
            Coordinator::new(CpuBackend::new(), self.device.clone(), transcript)
        }
    }
}

#[cfg(feature = "cpu-backend")]
pub use cpu_engine::BabyBearBn254Poseidon2CpuEngine;

pub fn default_transcript() -> Transcript {
    Transcript::from(default_bn254_poseidon2_width3())
}
