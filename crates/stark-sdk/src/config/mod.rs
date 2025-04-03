use tracing::Level;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

pub mod baby_bear_blake3;
pub mod baby_bear_bytehash;
pub mod baby_bear_keccak;
pub mod baby_bear_poseidon2;
/// Stark Config for root stark, which field is BabyBear but polynomials are committed in Bn254.
pub mod baby_bear_poseidon2_root;
pub mod fri_params;
pub mod goldilocks_poseidon;
pub mod instrument;
pub mod log_up_params;

pub use fri_params::FriParameters;

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
