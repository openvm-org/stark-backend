use itertools::Itertools;
use openvm_circuit::arch::testing::{VmChipTestBuilder, VmChipTester, BITWISE_OP_LOOKUP_BUS};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_keccak256_circuit::KeccakVmChip;
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_stark_backend::prover::types::ProofInput;
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
use stark_backend_gpu::engine::GpuBabyBearPoseidon2Engine;

type F = BabyBear;

fn build_keccak256_test(inputs: Vec<Vec<u8>>) -> VmChipTester<BabyBearPoseidon2Config> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<8>::new(bitwise_bus);

    let mut tester = VmChipTestBuilder::default();
    let mut chip = KeccakVmChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
        Rv32KeccakOpcode::CLASS_OFFSET,
        tester.offline_memory_mutex_arc(),
    );

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
        for (i, byte) in input.iter().enumerate() {
            tester.write_cell(e, src + i, F::from_canonical_u8(*byte));
        }

        tester.execute(
            &mut chip,
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
        .load(chip)
        .load(bitwise_chip)
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
    let tester = build_keccak256_test(inputs); // includes trace gen

    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();
    let air_id_and_inputs = tester
        .air_proof_inputs
        .into_iter()
        .map(|(air, input)| {
            let air_id = keygen_builder.add_air(air);
            (air_id, input)
        })
        .collect_vec();

    let pk = keygen_builder.generate_pk();
    let cpu_proof_input = ProofInput::new(air_id_and_inputs);
    let gpu_proof_input = cpu_proof_input.clone();

    // CPU
    // println!("\nStarting CPU proof");
    // let cpu_proof = engine.prove(&pk, cpu_proof_input);

    // engine.verify(&pk.get_vk(), &cpu_proof).unwrap();

    // GPU
    println!("\nStarting GPU proof");
    let engine = GpuBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let gpu_proof = engine.prove(&pk, gpu_proof_input);

    engine.verify(&pk.get_vk(), &gpu_proof).unwrap();
}
