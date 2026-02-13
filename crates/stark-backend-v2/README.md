# stark-backend

This crate provides the traits and CPU implementation of the SWIRL protocol. All traits and implementations in this crate **must** be generic to the field. Concrete specializations should be provided in the `openvm-stark-sdk` crate. The latter is imported as a dev-dependency for concrete instantiations during testing.
