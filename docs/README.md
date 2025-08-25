# Contributor Docs

- [STARK Backend](./stark-backend.md)
  - [AIR Interactions](./interactions.md)
  - [Metrics](./metrics.md): Guide to metrics collected by the prover.

## Development without CUDA

The repository is a Rust workspace which includes non-default member crates `openvm-cuda-*` which are only meant to compile on machines with CUDA toolkit installed. On machines without CUDA toolkit, ensure that your IDE does not check all crates in the workspace by adding the following to your IDE settings file (e.g., `.vscode/settings.json`):
```json
{
  "rust-analyzer.check.workspace": false,
}
```

## Development with CUDA
The CUDA crates in this repository should build via `cargo` on machines with [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation) 12.8 or later installed. In addition to Rust analyzer for linting Rust code, we recommend installing a clangd server linting CUDA code: in VS Code this comes with the C/C++ extension. For the clangd server to work properly, run
```bash
python3 scripts/generate_clangd.py
```
once to generate a local `.clangd` file. This file cannot be committed to the repository as it includes local paths.

To check the version of [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation) you have installed, run
```bash
nvcc --version
```
Lastly, ensure that your shell profile or startup script sets the proper [path variables for CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#environment-setup):
```bash
PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Note that `/usr/local/cuda` is a symlink for the latest version of CUDA installed.
