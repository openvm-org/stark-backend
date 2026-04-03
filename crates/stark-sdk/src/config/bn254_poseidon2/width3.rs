//! Width-3 Poseidon2 permutation over BN254 for leaf hashing and transcript sponge.
//!
//! Parameters: t=3, rF=8, rP=56, d=5.
//! Round constants sourced from `zkhash`'s `RC3`, matching `p3-bn254`'s `Poseidon2Bn254<3>`.
//!
//! Constants: <https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs#L32>
//! Plonky3 type: <https://github.com/Plonky3/Plonky3/blob/65bdaa3d9996516dfb80eabf26840ab80da137b0/bn254/src/poseidon2.rs#L26>
//!
//! See [`crate::config::baby_bear_bn254_poseidon2`] for how this fits into the full STARK config.

use std::sync::OnceLock;

use p3_bn254::{Bn254, Poseidon2Bn254};
use p3_field::PrimeCharacteristicRing;
use zkhash::{
    ark_ff::PrimeField as _, fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::RC3,
};

use super::common::{poseidon2_from_constants, split_row_round_constants, Poseidon2Bn254Constants};

const WIDTH: usize = 3;
const ROUNDS_F: usize = 8;
const ROUNDS_P: usize = 56;

/// Width-3 Poseidon2 permutation over BN254 (p3-bn254 / HorizenLabs compatible).
pub type Poseidon2Bn254Width3 = Poseidon2Bn254<WIDTH>;

pub type Poseidon2Bn254Width3Constants = Poseidon2Bn254Constants<WIDTH>;

/// Construct the shared width-3 Poseidon2 BN254 round constants.
pub fn default_bn254_poseidon2_width3_constants() -> &'static Poseidon2Bn254Width3Constants {
    static CONSTANTS: OnceLock<Poseidon2Bn254Width3Constants> = OnceLock::new();
    CONSTANTS.get_or_init(|| {
        let round_constants: Vec<[Bn254; WIDTH]> = RC3
            .iter()
            .map(|row| {
                row.iter()
                    .cloned()
                    .map(bn254_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
        split_row_round_constants(
            round_constants,
            ROUNDS_F,
            ROUNDS_P,
            // Must match p3-bn254's `bn254_matmul_internal`: M_I = 1 + diag([1, 1, 2]).
            [Bn254::ONE, Bn254::ONE, Bn254::TWO],
        )
    })
}

/// Construct the default width-3 Poseidon2 BN254 permutation.
pub fn default_bn254_poseidon2_width3() -> Poseidon2Bn254Width3 {
    static PERM: OnceLock<Poseidon2Bn254Width3> = OnceLock::new();
    PERM.get_or_init(|| {
        let constants = default_bn254_poseidon2_width3_constants();
        poseidon2_from_constants(constants)
    })
    .clone()
}

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254 {
    let limbs_le = input.into_bigint().0;
    let bytes = limbs_le
        .iter()
        .flat_map(|limb| limb.to_le_bytes())
        .collect::<Vec<_>>();
    let big = num_bigint::BigUint::from_bytes_le(&bytes);
    Bn254::from_biguint(big).expect("invalid BN254 element")
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use openvm_stark_backend::p3_symmetric::Permutation;
    use p3_field::{integers::QuotientMap, PrimeCharacteristicRing, PrimeField};
    use zkhash::{
        fields::bn256::FpBN256 as ark_FpBN256,
        poseidon2::{
            poseidon2::Poseidon2 as Poseidon2Ref,
            poseidon2_instance_bn256::POSEIDON2_BN256_PARAMS,
        },
    };

    use super::*;

    fn ark_ff_from_bn254(input: Bn254) -> ark_FpBN256 {
        let bigint = BigUint::from_bytes_le(&input.as_canonical_biguint().to_bytes_le());
        ark_FpBN256::from(bigint)
    }

    #[test]
    fn test_poseidon2_bn254_width3_matches_zkhash_reference() {
        let perm = default_bn254_poseidon2_width3();
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_BN256_PARAMS);

        let mut state = [Bn254::ONE, Bn254::TWO, Bn254::from_int(3u32)];
        let input_ark_ff = state.map(ark_ff_from_bn254);
        let expected: [ark_FpBN256; WIDTH] = poseidon2_ref
            .permutation(&input_ark_ff)
            .try_into()
            .unwrap();
        let expected = expected.map(bn254_from_ark_ff);

        perm.permute_mut(&mut state);

        assert_eq!(state, expected);
    }
}
