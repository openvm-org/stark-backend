use std::sync::Arc;

use itertools::Itertools;
use openvm_circuit::arch::{
    testing::{TestChipHarness, VmChipTestBuilder, VmChipTester, BITWISE_OP_LOOKUP_BUS},
    VmChipWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_keccak256_circuit::{KeccakVmAir, KeccakVmExecutor, KeccakVmFiller};
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_stark_backend::prover::{
    hal::DeviceDataTransporter,
    types::{AirProvingContext, ProvingContext},
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing, FriParameters,
    },
    engine::StarkFriEngine,
    openvm_stark_backend::{engine::StarkEngine, p3_field::FieldAlgebra},
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use rand::Rng;

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 8192;

fn build_keccak256_test(inputs: Vec<Vec<u8>>) -> VmChipTester<BabyBearPoseidon2Config> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<8>::new(bitwise_bus));

    let mut tester = VmChipTestBuilder::default();
    let executor = KeccakVmExecutor::new(Rv32KeccakOpcode::CLASS_OFFSET, tester.address_bits());
    let air = KeccakVmAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_bus,
        tester.address_bits(),
        Rv32KeccakOpcode::CLASS_OFFSET,
    );
    let chip = VmChipWrapper::new(
        KeccakVmFiller::new(bitwise_chip.clone(), tester.address_bits()),
        tester.memory_helper(),
    );
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    let mut dst = 0;
    let src = 0;

    for input in inputs {
        let [a, b, c] = [0, 4, 8]; // space apart for register limbs
        let [d, e] = [1, 2];

        tester.write(d, a, (dst as u32).to_le_bytes().map(F::from_canonical_u8));
        tester.write(d, b, (src as u32).to_le_bytes().map(F::from_canonical_u8));
        tester.write(
            d,
            c,
            (input.len() as u32).to_le_bytes().map(F::from_canonical_u8),
        );
        input.chunks(4).enumerate().for_each(|(i, chunk)| {
            let chunk: [&u8; 4] = std::array::from_fn(|i| chunk.get(i).unwrap_or(&0));
            tester.write(
                e,
                src as usize + i * 4,
                chunk.map(|&x| F::from_canonical_u8(x)),
            );
        });

        tester.execute(
            &mut harness,
            &Instruction::from_isize(
                Rv32KeccakOpcode::KECCAK256.global_opcode(),
                a as isize,
                b as isize,
                c as isize,
                d as isize,
                e as isize,
            ),
        );
        // shift dst to not deal with timestamps for pranking
        dst += 32;
    }
    tester
        .build_babybear_poseidon2()
        .load(harness)
        .load_periphery((
            BitwiseOperationLookupAir::<8>::new(bitwise_bus),
            bitwise_chip,
        ))
        .finalize()
}

const LOG_BLOWUP: usize = 2;
const NUM_INPUTS: usize = 1 << 6;
const MAX_INPUT_LEN: usize = 1 << 10;

fn main() {
    setup_tracing();
    let mut rng = create_seeded_rng();
    let inputs = (0..NUM_INPUTS)
        .map(|_| {
            let len = rng.gen_range(0..MAX_INPUT_LEN);
            (0..len).map(|_| rng.gen()).collect::<Vec<u8>>()
        })
        .collect::<Vec<_>>();

    println!("\nStarting CPU proof");
    build_keccak256_test(inputs.clone())
        .test(|| {
            BabyBearPoseidon2Engine::new(
                FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
            )
        })
        .expect("CPU proving failed");

    println!("\nStarting GPU proof");
    let tester = build_keccak256_test(inputs);
    let engine = GpuBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();
    let air_id_and_inputs = tester
        .air_ctxs
        .into_iter()
        .map(|(air, input)| {
            let air_id = keygen_builder.add_air(air);
            debug_assert!(input.cached_mains.is_empty());
            let gpu_ctx = AirProvingContext::new(
                vec![],
                Some(
                    engine
                        .device()
                        .transport_matrix_to_device(input.common_main.as_ref().unwrap()),
                ),
                input.public_values,
            );
            (air_id, gpu_ctx)
        })
        .collect_vec();

    let pk_host = &keygen_builder.generate_pk();
    let vk = pk_host.get_vk();
    let pk_device = engine.device().transport_pk_to_device(pk_host);
    let proof_input = ProvingContext::new(air_id_and_inputs);
    let gpu_proof = engine.prove(&pk_device, proof_input);
    engine.verify(&vk, &gpu_proof).unwrap();
}
