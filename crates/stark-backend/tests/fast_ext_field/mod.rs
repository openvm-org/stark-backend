use std::time::Instant;

use openvm_stark_backend::{
    fast_ext_field::FastBinomialExtensionField,
    p3_field::{extension::BinomialExtensionField, FieldExtensionAlgebra},
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use zkhash::ark_ff::UniformRand;

pub type OldBabyBearExt4 = BinomialExtensionField<BabyBear, 4>;
pub type BabyBearExt4 = FastBinomialExtensionField<BabyBear, 4>;

#[test]
pub fn arith_tests() {
    let mut rng = create_seeded_rng();
    let a = BabyBearExt4::rand(&mut rng);
    let b = BabyBearExt4::rand(&mut rng);
    let a1 = OldBabyBearExt4::from_base_slice(FieldExtensionAlgebra::<BabyBear>::as_base_slice(&a));
    let b1 = OldBabyBearExt4::from_base_slice(FieldExtensionAlgebra::<BabyBear>::as_base_slice(&b));
    println!("{:?}", a);
    println!("{:?}", a1);
    assert_same(a1 + b1, a + b);
    assert_same(a1 * b1, a * b);
    assert_same(a1 - b1, a - b);
    assert_same(a1 / b1, a / b);
}

#[test]
pub fn mul_time_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 1000000;
    let now = Instant::now();
    for _ in 0..NUM_OPS {
        let a = BabyBearExt4::rand(&mut rng);
        let b = BabyBearExt4::rand(&mut rng);
        let _ = a * b;
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    let now = Instant::now();
    for _ in 0..NUM_OPS {
        let a = OldBabyBearExt4::rand(&mut rng);
        let b = OldBabyBearExt4::rand(&mut rng);
        let _ = a * b;
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}

pub fn assert_same(a: OldBabyBearExt4, b: BabyBearExt4) {
    use openvm_stark_backend::p3_field::FieldExtensionAlgebra;
    for i in 0..4 {
        assert_eq!(
            FieldExtensionAlgebra::<BabyBear>::as_base_slice(&a)[i],
            FieldExtensionAlgebra::<BabyBear>::as_base_slice(&b)[i]
        )
    }
}
