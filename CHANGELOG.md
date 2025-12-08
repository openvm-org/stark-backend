# Changelog

All notable changes to STARK Backend will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## v1.2.2 (2025-12-08)

### Changed
- (CUDA backend) The alignment of `FpExt` is changed to 4 bytes (from 16 bytes) to match the Rust Plonky3 alignment for the BabyBear degree-4 extension field.
- (CUDA common) Fix for VPMM to re-use unmapped VA regions and claim more VA ranges on-demand if needed. Lowers the default VA range to 8TB to support Windows.

## v1.2.1 (2025-10-26)

### Added
- (CUDA common) New memory manager with Virtual Pool ([VPMM Spec](./docs/vpmm_spec.md)) with multi-stream support built on top of the CUDA Virtual Memory Management [driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)

### Changed
- (CUDA common) Multi-arch build support
- (CUDA backend) Quotient values kernel optimization
- (CUDA backend) FRI reduced opening kernel optimization by removing bit reversal for better memory access patterns
