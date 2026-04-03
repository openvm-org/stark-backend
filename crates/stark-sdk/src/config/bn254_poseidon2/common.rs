use p3_bn254::Bn254;
use p3_poseidon2::{
    ExternalLayerConstants, ExternalLayerConstructor, InternalLayerConstructor, Poseidon2,
};

#[derive(Clone, Debug)]
pub struct Poseidon2Bn254Constants<const WIDTH: usize> {
    initial_external_rc: Vec<[Bn254; WIDTH]>,
    internal_rc: Vec<Bn254>,
    terminal_external_rc: Vec<[Bn254; WIDTH]>,
    /// The diagonal entries (minus one) of the internal MDS matrix:
    /// `M_I = I + diag(mat_internal_diag_m_1)`.
    mat_internal_diag_m_1: [Bn254; WIDTH],
}

impl<const WIDTH: usize> Poseidon2Bn254Constants<WIDTH> {
    pub fn new(
        rounds_f: usize,
        rounds_p: usize,
        initial_external_rc: Vec<[Bn254; WIDTH]>,
        internal_rc: Vec<Bn254>,
        terminal_external_rc: Vec<[Bn254; WIDTH]>,
        mat_internal_diag_m_1: [Bn254; WIDTH],
    ) -> Self {
        assert_eq!(initial_external_rc.len(), rounds_f / 2);
        assert_eq!(internal_rc.len(), rounds_p);
        assert_eq!(terminal_external_rc.len(), rounds_f / 2);
        Self {
            initial_external_rc,
            internal_rc,
            terminal_external_rc,
            mat_internal_diag_m_1,
        }
    }

    pub fn initial_external_rc(&self) -> &[[Bn254; WIDTH]] {
        &self.initial_external_rc
    }

    pub fn internal_rc(&self) -> &[Bn254] {
        &self.internal_rc
    }

    pub fn terminal_external_rc(&self) -> &[[Bn254; WIDTH]] {
        &self.terminal_external_rc
    }

    pub fn mat_internal_diag_m_1(&self) -> &[Bn254; WIDTH] {
        &self.mat_internal_diag_m_1
    }
}

pub fn split_flat_round_constants<const WIDTH: usize>(
    all_rc: Vec<Bn254>,
    rounds_f: usize,
    rounds_p: usize,
    mat_internal_diag_m_1: [Bn254; WIDTH],
) -> Poseidon2Bn254Constants<WIDTH> {
    let half_f = rounds_f / 2;
    let initial_external_rc = all_rc[..half_f * WIDTH]
        .chunks_exact(WIDTH)
        .map(|chunk| chunk.try_into().unwrap())
        .collect();
    let internal_rc = all_rc[half_f * WIDTH..half_f * WIDTH + rounds_p].to_vec();
    let terminal_external_rc = all_rc[half_f * WIDTH + rounds_p..]
        .chunks_exact(WIDTH)
        .map(|chunk| chunk.try_into().unwrap())
        .collect();

    Poseidon2Bn254Constants::new(
        rounds_f,
        rounds_p,
        initial_external_rc,
        internal_rc,
        terminal_external_rc,
        mat_internal_diag_m_1,
    )
}

pub fn split_row_round_constants<const WIDTH: usize>(
    mut round_constants: Vec<[Bn254; WIDTH]>,
    rounds_f: usize,
    rounds_p: usize,
    mat_internal_diag_m_1: [Bn254; WIDTH],
) -> Poseidon2Bn254Constants<WIDTH> {
    let internal_end = (rounds_f / 2) + rounds_p;
    let terminal_external_rc = round_constants.split_off(internal_end);
    let internal_rc = round_constants
        .split_off(rounds_f / 2)
        .into_iter()
        .map(|row| row[0])
        .collect();
    let initial_external_rc = round_constants;

    Poseidon2Bn254Constants::new(
        rounds_f,
        rounds_p,
        initial_external_rc,
        internal_rc,
        terminal_external_rc,
        mat_internal_diag_m_1,
    )
}

pub fn poseidon2_from_constants<ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>(
    constants: &Poseidon2Bn254Constants<WIDTH>,
) -> Poseidon2<Bn254, ExternalPerm, InternalPerm, WIDTH, D>
where
    ExternalPerm: ExternalLayerConstructor<Bn254, WIDTH>,
    InternalPerm: InternalLayerConstructor<Bn254>,
{
    Poseidon2::new(
        ExternalLayerConstants::new(
            constants.initial_external_rc().to_vec(),
            constants.terminal_external_rc().to_vec(),
        ),
        constants.internal_rc().to_vec(),
    )
}
