# Changelog

All notable changes to STARK Backend will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## Unreleased

### Changed
- GPU prover: batched the per-round parameter uploads, kernel launches, and result readbacks of the batch-constraint, stacked-reduction, and round-0 sumcheck phases, and moved per-round transcript readbacks to pinned memory; the interactions round-0 DAG is now cached in the proving key instead of rebuilt per proof. Measured ~2.4% lower proving time on the reth benchmark on RTX Pro 6000.

## v2.0.0 (2026-07-06)

### Added
- Segment memory metering APIs for estimating segmented prover memory usage.
- AIR rotation metadata used by LogUp and stacked-opening proving to skip unused rotation work.
- Backend conformance test crate for shared prover/verifier behavior.
- Row-major CPU backend crate as part of the prover implementation.

### Changed
- Updated the STARK protocol to SWIRL: LogUp interactions are now proven through the GKR fractional sumcheck, constraint checking uses the batched constraint sumcheck, and trace openings reduce through the stacked PCS opening reduction and WHIR.
- Bumped the workspace MSRV from Rust 1.83 to Rust 1.90.

## v1.4.0 (2026-05-15)

### Changed
- Updated Plonky3 to crates.io `v0.4.3` to fix `MultiField32Challenger`.

## v1.3.0 (2026-02-05)

### Changed
- Updated Plonky3 to crates.io `v0.4.1`.
- Updated STARK protocol to 100 bits of provable security with FRI.

## v1.2.3 (2026-01-11)

### Changed
- (CUDA common) VPMM is now fully async and CPU does not block when GPU memory needs to be defragmented.

## v1.2.2 (2025-12-08)

### Changed
- (CUDA backend) The alignment of `FpExt` is changed to 4 bytes (from 16 bytes) to match the Rust Plonky3 alignment for the BabyBear degree-4 extension field.
- (CUDA common) Fix for VPMM to reuse unmapped VA regions and claim more VA ranges on-demand if needed. Lowers the default VA range to 8TB to support Windows.

## v1.2.1 (2025-10-26)

### Added
- (CUDA common) New memory manager with Virtual Pool ([VPMM Spec](./docs/vpmm_spec.md)) with multi-stream support built on top of the CUDA Virtual Memory Management [driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)

### Changed
- (CUDA common) Multi-arch build support
- (CUDA backend) Quotient values kernel optimization
- (CUDA backend) FRI reduced opening kernel optimization by removing bit reversal for better memory access patterns
