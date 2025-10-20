use std::sync::OnceLock;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::FieldAlgebra;
use p3_poseidon2::ExternalLayerConstants;

mod instance_babybear;
pub mod sponge;

pub use instance_babybear::*;

pub const WIDTH: usize = 16;
pub const CHUNK: usize = 8;

// Fixed Poseidon2 configuration
pub fn poseidon2_perm() -> &'static Poseidon2BabyBear<WIDTH> {
    static PERM: OnceLock<Poseidon2BabyBear<WIDTH>> = OnceLock::new();
    PERM.get_or_init(|| {
        let (external_constants, internal_constants) = horizen_round_consts_16();
        Poseidon2BabyBear::new(external_constants, internal_constants)
    })
}

pub fn horizen_round_consts_16() -> (ExternalLayerConstants<BabyBear, 16>, Vec<BabyBear>) {
    let p3_rc16: Vec<Vec<BabyBear>> = RC16
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|&u32_canonical| BabyBear::from_wrapped_u32(u32_canonical))
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
