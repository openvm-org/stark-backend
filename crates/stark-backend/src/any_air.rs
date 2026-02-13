//! Traits for [Air] trait objects.

use std::{
    any::{type_name, Any},
    sync::Arc,
};

// Re-export for backwards compatibility
pub use p3_air::BaseAirWithPublicValues;
use p3_air::{Air, BaseAir};

use crate::{
    air_builders::{debug::DebugConstraintBuilder, symbolic::SymbolicRapBuilder},
    config::StarkProtocolConfig,
};

/// An AIR with 1 or more main trace partitions.
pub trait PartitionedBaseAir<F>: BaseAir<F> {
    /// By default, an AIR has no cached main trace.
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![]
    }
    /// By default, an AIR has only one private main trace.
    fn common_main_width(&self) -> usize {
        self.width()
    }
}

/// Shared reference to any Interactive Air.
/// This type is the main interface for keygen.
pub type AirRef<SC> = Arc<dyn AnyAir<SC>>;

/// RAP trait for all-purpose dynamic dispatch use.
/// This trait is auto-implemented if you implement `Air` and `BaseAirWithPublicValues` and
/// `PartitionedBaseAir` traits.
pub trait AnyAir<SC: StarkProtocolConfig>:
Air<SymbolicRapBuilder<SC::F>> // for keygen to extract fixed data about the RAP
    + for<'a> Air<DebugConstraintBuilder<'a, SC>> // for debugging
    + BaseAirWithPublicValues<SC::F>
    + PartitionedBaseAir<SC::F>
    + Send + Sync
{
    fn as_any(&self) -> &dyn Any;
    /// Name for display purposes
    fn name(&self) -> String;
}

impl<SC, T> AnyAir<SC> for T
where
    SC: StarkProtocolConfig,
    T: Air<SymbolicRapBuilder<SC::F>>
        + for<'a> Air<DebugConstraintBuilder<'a, SC>>
        + BaseAirWithPublicValues<SC::F>
        + PartitionedBaseAir<SC::F>
        + Send
        + Sync
        + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> String {
        get_air_name(self)
    }
}

/// Automatically derives the AIR name from the type name for pretty display purposes.
pub fn get_air_name<T>(_rap: &T) -> String {
    let full_name = type_name::<T>().to_string();
    // Split the input by the first '<' to separate the main type from its generics
    if let Some((main_part, generics_part)) = full_name.split_once('<') {
        // Extract the last segment of the main type
        let main_type = main_part.split("::").last().unwrap_or("");

        // Remove the trailing '>' from the generics part and split by ", " to handle multiple
        // generics
        let generics: Vec<String> = generics_part
            .trim_end_matches('>')
            .split(", ")
            .map(|generic| {
                // For each generic type, extract the last segment after "::"
                generic.split("::").last().unwrap_or("").to_string()
            })
            .collect();

        // Join the simplified generics back together with ", " and format the result
        format!("{}<{}>", main_type, generics.join(", "))
    } else {
        // If there's no generic part, just return the last segment after "::"
        full_name.split("::").last().unwrap_or("").to_string()
    }
}
