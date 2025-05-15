# OpenVM Stark Backend

[Contributor Docs](./docs)
| [Crate Docs](https://docs.openvm.dev/stark-backend)

A modular proof system backend for proving and verifying multi-chip circuits with inter-chip communication.

The backend is designed to be modular and compatible with different proof systems, with a focus on performance and extensibility. The aim is to support different circuit representations and permutation/lookup arguments.

## Crates

- [`openvm-stark-backend`](crates/stark-backend): General purpose STARK proving system with multi-trace and logup support, built on top of Plonky3.
- [`openvm-stark-sdk`](crates/stark-sdk): Low-level SDK for use with STARK backend to generate proofs for specific STARK configurations.

## Status

As of May 2025, STARK Backend v1.1.0 and later are recommended for production use. STARK Backend completed an external [audit](https://github.com/openvm-org/openvm/blob/main/audits/v1-cantina-report.pdf) on [Cantina](https://cantina.xyz/) from January to March 2025 as well as an internal [audit](https://github.com/openvm-org/openvm/blob/main/audits/v1-internal/README.md) by members of the [Axiom](https://axiom.xyz/) team during the same timeframe.

## Security

See [SECURITY.md](./SECURITY.md).

## Acknowledgements

We studied and built upon the work of other teams in our quest to design a modular and performant proving framework.
We would like to thank these teams for sharing their code for open source development:

- [Plonky3](https://github.com/Plonky3/Plonky3): This codebase is built on top of Plonky3, where we have heavily benefited from their modular design at the polynomial IOP level. We extend Plonky3 by providing higher level interfaces for proving multi-chip circuits.
- [Valida](https://github.com/valida-xyz/valida): Valida introduced the exceptionally elegant interactions interface for multi-chip communication via logup permutation arguments. We have found this interface quite well thought out and have built upon and extended it.
- [SP1](https://github.com/succinctlabs/sp1): We learned from SP1's `AirBuilder` designs, and the original design for the `InteractionBuilder` was inspired by them.
- [Stwo](https://github.com/starkware-libs/stwo): We studied Stwo's performant sumcheck implementations and have begun integrating them into our backend.
