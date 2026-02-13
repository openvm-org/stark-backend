use std::io::{self, Read, Write};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::extension::BinomialExtensionField;
use stark_backend_v2::{
    codec::{
        decode_extension_field32, decode_prime_field32, encode_extension_field32,
        encode_prime_field32, DecodableConfig, EncodableConfig,
    },
    hasher::Hasher,
    p3_symmetric::{PaddingFreeSponge, TruncatedPermutation},
    StarkProtocolConfig, SystemParams,
};

const RATE: usize = 8;
/// permutation width
const WIDTH: usize = 16; // rate + capacity
pub const CHUNK: usize = 8;
pub const DIGEST_SIZE: usize = CHUNK;

type Perm = Poseidon2BabyBear<WIDTH>;
// Generic over P: CryptographicPermutation<[F; WIDTH]>
type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_SIZE>;
type Compress<P> = TruncatedPermutation<P, 2, CHUNK, WIDTH>;
type PermHasher<P> = Hasher<F, Digest, Hash<P>, Compress<P>>;

// Convenience type aliases
pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, 4>;
pub const D_EF: usize = 4;
pub type Digest = [F; DIGEST_SIZE];

#[derive(Clone, Debug, derive_new::new)]
pub struct BabyBearPoseidon2ConfigV2 {
    params: SystemParams,
    hasher: PermHasher<Perm>,
}

impl StarkProtocolConfig for BabyBearPoseidon2ConfigV2 {
    type F = F;
    type EF = EF;
    type Digest = Digest;
    type Hasher = PermHasher<Perm>;

    fn params(&self) -> &SystemParams {
        &self.params
    }

    fn hasher(&self) -> &Self::Hasher {
        &self.hasher
    }
}

impl EncodableConfig for BabyBearPoseidon2ConfigV2 {
    fn encode_base_field<W: Write>(val: &F, writer: &mut W) -> io::Result<()> {
        encode_prime_field32(val, writer)
    }

    fn encode_extension_field<W: Write>(val: &EF, writer: &mut W) -> io::Result<()> {
        encode_extension_field32::<Self::F, _, _>(val, writer)
    }

    fn encode_digest<W: Write>(digest: &Self::Digest, writer: &mut W) -> io::Result<()> {
        for val in digest {
            encode_prime_field32(val, writer)?;
        }
        Ok(())
    }
}

impl DecodableConfig for BabyBearPoseidon2ConfigV2 {
    fn decode_base_field<R: Read>(reader: &mut R) -> io::Result<F> {
        decode_prime_field32(reader)
    }

    fn decode_extension_field<R: Read>(reader: &mut R) -> io::Result<EF> {
        decode_extension_field32::<F, _, _>(reader)
    }

    fn decode_digest<R: Read>(reader: &mut R) -> io::Result<Digest> {
        let mut result = Digest::default();
        for val in &mut result {
            *val = decode_prime_field32(reader)?;
        }
        Ok(result)
    }
}

pub fn poseidon2_compress_with_capacity(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> ([F; CHUNK], [F; CHUNK]) {
    let mut state = [F::ZERO; WIDTH];
    state[..CHUNK].copy_from_slice(&left);
    state[CHUNK..].copy_from_slice(&right);
    poseidon2_perm().permute_mut(&mut state);
    (
        state[..CHUNK].try_into().unwrap(),
        state[CHUNK..].try_into().unwrap(),
    )
}

#[cfg(test)]
mod poseidon2_constant_tests {
    use super::*;

    /// Uses HorizenLabs Poseidon2 round constants, but plonky3 Mat4 and also
    /// with a p3 Monty reduction factor.
    pub fn default_perm() -> Perm {
        let (external_constants, internal_constants) = horizen_round_consts_16();
        Perm::new(external_constants, internal_constants)
    }

    fn horizen_to_p3(horizen_babybear: HorizenBabyBear) -> BabyBear {
        BabyBear::from_u64(horizen_babybear.into_bigint().0[0])
    }

    pub fn horizen_round_consts_16() -> (ExternalLayerConstants<BabyBear, 16>, Vec<BabyBear>) {
        let p3_rc16: Vec<Vec<BabyBear>> = RC16
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|babybear| horizen_to_p3(*babybear))
                    .collect()
            })
            .collect();

        let rounds_f = 8;
        let rounds_p = 13;
        let rounds_f_beginning = rounds_f / 2;
        let p_end = rounds_f_beginning + rounds_p;
        let initial: Vec<[BabyBear; 16]> = p3_rc16[..rounds_f_beginning]
            .iter()
            .cloned()
            .map(|round| round.try_into().unwrap())
            .collect();
        let terminal: Vec<[BabyBear; 16]> = p3_rc16[p_end..]
            .iter()
            .cloned()
            .map(|round| round.try_into().unwrap())
            .collect();
        let internal_round_constants: Vec<BabyBear> = p3_rc16[rounds_f_beginning..p_end]
            .iter()
            .map(|round| round[0])
            .collect();
        (
            ExternalLayerConstants::new(initial, terminal),
            internal_round_constants,
        )
    }
}
