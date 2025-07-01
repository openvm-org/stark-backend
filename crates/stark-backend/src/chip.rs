use std::{
    any::Any,
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::prover::{hal::ProverBackend, types::AirProvingContext};

/// A chip is a [ProverBackend]-specific construct used to generate the trace matrix of a specific AIR as a `DeviceMatrix`. A chip may be stateful and store state on either host or device. However the trait is also designed to support a common trace generation pattern where certain "records" (e.g., logs from a VM execution) are ingested to produce the trace matrix.
pub trait Chip<R, PB: ProverBackend> {
    /// Generate all necessary context for proving a single AIR.
    // The lifetime parameter `'b` is a placeholder for the lifetime of any cached trace. It should not be related to the lifetime of the borrow.
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB>;
}

/// Auto-implemented trait for downcasting of trait objects.
pub trait AnyChip<R, PB: ProverBackend>: Chip<R, PB> {
    fn as_any(&self) -> &dyn Any;
}

impl<R, PB: ProverBackend, C: Chip<R, PB> + 'static> AnyChip<R, PB> for C {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for RefCell<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.borrow().generate_proving_ctx(records)
    }
}
impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for Rc<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.as_ref().generate_proving_ctx(records)
    }
}
impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for Arc<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.as_ref().generate_proving_ctx(records)
    }
}
impl<R, PB: ProverBackend, C: Chip<R, PB>> Chip<R, PB> for Mutex<C> {
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB> {
        self.lock().unwrap().generate_proving_ctx(records)
    }
}

// TODO: consider deleting this
/// A trait to get chip usage information.
pub trait ChipUsageGetter {
    fn air_name(&self) -> String;
    /// If the chip has a state-independent trace height that is determined
    /// upon construction, return this height. This is used to distinguish
    /// "static" versus "dynamic" usage metrics.
    fn constant_trace_height(&self) -> Option<usize> {
        None
    }
    /// Height of used rows in the main trace.
    fn current_trace_height(&self) -> usize;
    /// Width of the main trace
    fn trace_width(&self) -> usize;
    /// For metrics collection
    fn current_trace_cells(&self) -> usize {
        self.trace_width() * self.current_trace_height()
    }
}

impl<C: ChipUsageGetter> ChipUsageGetter for Rc<C> {
    fn air_name(&self) -> String {
        self.as_ref().air_name()
    }
    fn constant_trace_height(&self) -> Option<usize> {
        self.as_ref().constant_trace_height()
    }
    fn current_trace_height(&self) -> usize {
        self.as_ref().current_trace_height()
    }
    fn trace_width(&self) -> usize {
        self.as_ref().trace_width()
    }
}

impl<C: ChipUsageGetter> ChipUsageGetter for RefCell<C> {
    fn air_name(&self) -> String {
        self.borrow().air_name()
    }
    fn constant_trace_height(&self) -> Option<usize> {
        self.borrow().constant_trace_height()
    }
    fn current_trace_height(&self) -> usize {
        self.borrow().current_trace_height()
    }
    fn trace_width(&self) -> usize {
        self.borrow().trace_width()
    }
}

impl<C: ChipUsageGetter> ChipUsageGetter for Mutex<C> {
    fn air_name(&self) -> String {
        self.lock().unwrap().air_name()
    }
    fn constant_trace_height(&self) -> Option<usize> {
        self.lock().unwrap().constant_trace_height()
    }
    fn current_trace_height(&self) -> usize {
        self.lock().unwrap().current_trace_height()
    }
    fn trace_width(&self) -> usize {
        self.lock().unwrap().trace_width()
    }
}
