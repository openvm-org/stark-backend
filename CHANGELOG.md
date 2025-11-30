# Changelog

All notable changes to STARK Backend will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## v1.2.2-rc (Unreleased)

### Changed
- (CUDA common) Fix for VPMM to re-use unmapped VA regions and claim more VA ranges on-demand if needed. Lowers the default VA range to 8TB to support Windows.

## v1.2.1 (2025-10-26)

### Added
- (CUDA common) New memory manager with Virtual Pool ([VPMM Spec](./docs/vpmm_spec.md)) with multi-stream support built on top of the CUDA Virtual Memory Management [driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)

### Changed
- (CUDA common) Multi-arch build support
- (CUDA backend) Quotient values kernel optimization
- (CUDA backend) FRI reduced opening kernel optimization by removing bit reversal for better memory access patterns
