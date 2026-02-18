/// STARK config where the base field is BabyBear, extension field is BabyBear^4, and the hasher is
/// `Poseidon2<Bn254>`.
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub mod baby_bear_bn254_poseidon2;
/// STARK config where the base field is BabyBear, extension field is BabyBear^4, and the hasher is
/// `Poseidon2<BabyBear>`.
pub mod baby_bear_poseidon2;
pub mod log_up_params;
