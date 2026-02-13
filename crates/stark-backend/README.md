# stark-backend

This crate provides the traits and CPU implementation of the SWIRL protocol. All traits and implementations in this crate **must** be generic to the field. Concrete specializations should be provided in the `openvm-stark-sdk` crate. To avoid a circular dependency, `openvm-stark-sdk` is a dev-dependency that should only used in [integration tests](https://doc.rust-lang.org/book/ch11-03-test-organization.html). As such, most tests of the protocol are integration tests.
