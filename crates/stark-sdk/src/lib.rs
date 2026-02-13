pub use p3_baby_bear;
pub use stark_backend_v2;

pub mod bench;
pub mod config;
#[cfg(feature = "metrics")]
pub mod metrics_tracing;
pub mod utils;

// #[macro_export]
// macro_rules! assert_sc_compatible_with_serde {
//     ($sc:ty) => {
//         static_assertions::assert_impl_all!
// ($crate::openvm_stark_backend::keygen::types::MultiStarkProvingKey<$sc>: serde::Serialize,
// serde::de::DeserializeOwned);         static_assertions::assert_impl_all!
// ($crate::openvm_stark_backend::keygen::types::MultiStarkVerifyingKey<$sc>: serde::Serialize,
// serde::de::DeserializeOwned);         static_assertions::assert_impl_all!
// ($crate::openvm_stark_backend::Proof<$sc>: serde::Serialize, serde::de::DeserializeOwned);     };
// }
