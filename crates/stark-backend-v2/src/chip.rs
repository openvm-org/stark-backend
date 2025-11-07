// Copied from v1
// TODO[jpw]: remove duplication
use std::any::Any;

use crate::prover::{AirProvingContextV2, ProverBackendV2};

/// A chip is a [ProverBackend]-specific object that converts execution logs (also referred to as
/// records) into a trace matrix.
///
/// A chip may be stateful and store state on either host or device, although it is preferred that
/// all state is received through records.
pub trait ChipV2<R, PB: ProverBackendV2> {
    /// Generate all necessary context for proving a single AIR.
    fn generate_proving_ctx(&self, records: R) -> AirProvingContextV2<PB>;
}

/// Auto-implemented trait for downcasting of trait objects.
pub trait AnyChip<R, PB: ProverBackendV2>: ChipV2<R, PB> {
    fn as_any(&self) -> &dyn Any;
}

impl<R, PB: ProverBackendV2, C: ChipV2<R, PB> + 'static> AnyChip<R, PB> for C {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// impl<R, PB: ProverBackendV2, C: ChipV2<R, PB>> ChipV2<R, PB> for RefCell<C> {
//     fn generate_proving_ctx(&self, records: R) -> AirProvingContextV2<PB> {
//         self.borrow().generate_proving_ctx(records)
//     }
// }
// impl<R, PB: ProverBackendV2, C: ChipV2<R, PB>> ChipV2<R, PB> for Rc<C> {
//     fn generate_proving_ctx(&self, records: R) -> AirProvingContextV2<PB> {
//         self.as_ref().generate_proving_ctx(records)
//     }
// }
// impl<R, PB: ProverBackendV2, C: ChipV2<R, PB>> ChipV2<R, PB> for Arc<C> {
//     fn generate_proving_ctx(&self, records: R) -> AirProvingContextV2<PB> {
//         self.as_ref().generate_proving_ctx(records)
//     }
// }
// impl<R, PB: ProverBackendV2, C: ChipV2<R, PB>> ChipV2<R, PB> for Mutex<C> {
//     fn generate_proving_ctx(&self, records: R) -> AirProvingContextV2<PB> {
//         self.lock().unwrap().generate_proving_ctx(records)
//     }
// }
