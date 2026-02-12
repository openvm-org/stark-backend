use itertools::Itertools;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::Level;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

pub fn setup_tracing() {
    setup_tracing_with_log_level(Level::INFO);
}

pub fn setup_tracing_with_log_level(level: Level) {
    // Set up tracing:
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("{},p3_=warn", level)));
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();
}

/// Deterministic seeded RNG, for testing use
pub fn create_seeded_rng() -> StdRng {
    let seed = [42; 32];
    StdRng::from_seed(seed)
}

pub fn create_seeded_rng_with_seed(seed: u64) -> StdRng {
    let seed_be = seed.to_be_bytes();
    let mut seed = [0u8; 32];
    seed[24..32].copy_from_slice(&seed_be);
    StdRng::from_seed(seed)
}

// Returns row major matrix
pub fn generate_random_matrix<F: PrimeCharacteristicRing>(
    mut rng: impl Rng,
    height: usize,
    width: usize,
) -> Vec<Vec<F>> {
    (0..height)
        .map(|_| (0..width).map(|_| F::from_u32(rng.random())).collect_vec())
        .collect_vec()
}

pub fn to_field_vec<F: PrimeCharacteristicRing>(v: Vec<u32>) -> Vec<F> {
    v.into_iter().map(F::from_u32).collect()
}
